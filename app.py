
import os
import streamlit as st
from dotenv import load_dotenv
from data.loader import load_and_split_docs
from vectorstores.chroma_store import embed_and_store_in_chroma
from chains.rag_chain import build_qa_chain

load_dotenv()

# Load custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit page setup
st.set_page_config(page_title="Healthcare Chatbot", layout="wide")
st.title("🧬 AI Chatbot for Healthcare & Life Sciences")

# Initialize session states
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "reset_query" not in st.session_state:
    st.session_state.reset_query = False
if "chain" not in st.session_state:
    st.session_state.chain = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None

# Upload file
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Detect new file and rebuild chain if needed
if uploaded_file:
    current_file_name = uploaded_file.name
    if current_file_name != st.session_state.current_file_name:
        with st.spinner("Processing document..."):
            docs = load_and_split_docs(uploaded_file)
            vectordb = embed_and_store_in_chroma(docs)
            st.session_state.chain = build_qa_chain(vectordb)
            st.session_state.docs = docs
            st.session_state.current_file_name = current_file_name
            st.session_state.chat_history = []  # clear old chat

# Clear chat manually
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.query_input = ""
    st.session_state.reset_query = False
    st.session_state.docs = None
    st.session_state.chain = None
    st.session_state.current_file_name = None
    st.rerun()

# Show chat history
if st.session_state.chat_history:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for chat in st.session_state.chat_history:
        st.markdown(
            f"""
            <div class='chat-row'>
                <div class='chat-bubble user'>{chat['query']}</div>
                <div class='chat-bubble chatbot'>{chat['response']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Show source docs
    with st.expander("📄 Source Documents"):
        for idx, chat in enumerate(st.session_state.chat_history):
            for i, doc in enumerate(chat["sources"]):
                st.markdown(f"**Q{idx+1} - Document {i+1}:** {doc.metadata.get('source', 'N/A')}")
                st.write(doc.page_content)

# Reset query input state before re-showing it
if st.session_state.reset_query:
    st.session_state.query_input = ""
    st.session_state.reset_query = False

# Bottom input box using form
with st.form(key="chat_form", clear_on_submit=True):
    query = st.text_input("Ask a question about the document", key="query_input")
    submitted = st.form_submit_button("Send")

# Handle query submission
if submitted and query and st.session_state.chain:
    result = st.session_state.chain.invoke({"query": query})
    st.session_state.chat_history.append({
        "query": query,
        "response": result["result"],
        "sources": result["source_documents"]
    })
    st.session_state.reset_query = True
    st.rerun()








