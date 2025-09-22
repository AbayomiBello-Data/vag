import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from openai import OpenAI

# --- Load environment variables ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

# --- Helper function to check for polite greetings ---
def is_polite_greeting(text):
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    return text.strip().lower() in greetings

# --- Streamlit Streaming Handler ---
class StreamHandler(BaseCallbackHandler):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.text = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.placeholder.markdown(
            f'<div class="assistant-message"><strong>📘 Vagaro Tutor:</strong><br>{self.text}</div>',
            unsafe_allow_html=True,
        )

# --- Load Vectorstore ---
@st.cache_resource
def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

# --- Content Moderation ---
def moderate_content(user_question):
    try:
        response = client.moderations.create(
            model="omni-moderation-latest",
            input=user_question
        )
        return response.results[0].flagged
    except Exception as e:
        st.error(f"Moderation error: {e}")
        return False

# --- Vagaro-related Question Classifier ---
def is_question_vagaro_related(question):
    """
    Returns True if the question is related to Vagaro support.
    """
    classifier_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=openai_api_key
    )

    prompt = f"""
You are a Vagaro support assistant. Decide if the following question is related
to Vagaro support.

Question:
{question}

Respond ONLY with 'YES' or 'NO', nothing else.
"""
    try:
        response = classifier_llm.invoke(prompt).content.strip().lower()
        return response.startswith("yes")  # catches variations like "Yes.", "YES!"
    except Exception as e:
        st.error(f"Error classifying question: {e}")
        return False

# --- Document Retrieval ---
def get_relevant_docs(question, vectorstore, threshold=0.75):
    results = vectorstore.similarity_search_with_score(question, k=3)
    return [doc for doc, score in results if score < threshold]

# --- Get LLM ---
def get_llm(callback):
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.4,
        streaming=True,
        openai_api_key=openai_api_key,
        callbacks=[callback],
    )

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template(""" 
You are a helpful and friendly help desk assistant for Vagaro.

You are given a question asked by customers.
Your job is to explain the step by step process to answer the customer question.

- Do NOT introduce answers not present in the pdf.
- If the pdf includes diagrams or visual hints, simulate them in Markdown.
- Stay aligned with the pdf step by step guide.

Pdf context:
{context}

Agent question:
{question}

Your explanation:
""")

# --- Streamlit UI ---
st.set_page_config(page_title="📘 Vagaro Support Tutor", layout="wide")
st.title("📘 Guided Vagaro Support Chatbot")

# --- CSS Styling ---
st.markdown("""
<style>
.user-message {
    background-color: #d1f5d3;
    padding: 10px;
    border-radius: 10px;
    margin: 10px 0;
    max-width: 80%;
    margin-left: auto;
    font-weight: 500;
}
.assistant-message {
    background-color: #e1e1e1;
    padding: 10px;
    border-radius: 10px;
    margin: 10px 0;
    max-width: 80%;
    margin-right: auto;
}
</style>
""", unsafe_allow_html=True)

# --- Session state setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Helper function to render assistant responses ---
def render_assistant_response(text):
    placeholder = st.empty()
    handler = StreamHandler(placeholder)
    with st.spinner("Thinking..."):
        for token in text:
            handler.on_llm_new_token(token)
        st.session_state.messages.append({
            "role": "assistant",
            "content": handler.text
        })

# --- Show chat history ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message"><strong>👤 You:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-message"><strong>📘 Vagaro Tutor:</strong><br>{msg["content"]}</div>', unsafe_allow_html=True)

# --- Chat input ---
user_question = st.chat_input("Type your question here...")

# --- Unified Assistant Message Rendering ---
if user_question:
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.markdown(f'<div class="user-message"><strong>👤 You:</strong><br>{user_question}</div>', unsafe_allow_html=True)

    # Handle polite greeting
    if is_polite_greeting(user_question):
        render_assistant_response("👋 Hello! I’m your guide for Vagaro support. Feel free to ask any question, and I’ll walk you through the steps.")
    # Moderation check
    elif moderate_content(user_question):
        render_assistant_response("❌ Your input contains harmful content. Please revise it.")
    # Vagaro question check
    elif not is_question_vagaro_related(user_question):
        render_assistant_response("❌ I can only help with **Vagaro support-related** questions.")
    else:
        vectorstore = load_vectorstore()
        docs = get_relevant_docs(user_question, vectorstore)

        if not docs:
            render_assistant_response("❌ I couldn't find a relevant answer in the documents.")
        else:
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = prompt_template.format(context=context, question=user_question)

            placeholder = st.empty()
            handler = StreamHandler(placeholder)
            llm = get_llm(handler)

            with st.spinner("Thinking..."):
                try:
                    _ = llm.invoke(prompt)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": handler.text
                    })
                except Exception as e:
                    render_assistant_response(f"❌ An error occurred: {e}")
