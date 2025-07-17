# 🏥 RAG-Based Healthcare Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot designed for the healthcare and pharmaceutical domain. It uses hybrid search (semantic + keyword) with Chroma DB and Groq's LLaMA3-8B model for intelligent document-based question answering.

---

## 🚀 Features

- 💬 Conversational chatbot using **LangChain** and **Streamlit**
- 🔎 Hybrid search with **BM25 (keyword)** + **Vector search (Chroma DB)**
- 📚 Document ingestion from PDFs
- 🧠 Powered by **Groq’s LLaMA3-8B** for fast and cost-effective inference
- 🧾 Clean UI for uploading medical documents and chatting with them

---

## 🧰 Tech Stack

- **Python**
- **LangChain**
- **Chroma DB**
- **Groq API** (for LLaMA3)
- **Streamlit** (for UI)
- **Sentence Transformers** (Hugging Face embeddings)

---

## 🗂️ Project Structure


RAG_CHATBOT/
│
├── chains/
│ └── rag_chain.py # RAG pipeline with Groq LLM
│
├── chroma_db/ # (Optional) Persistent Chroma DB folder
│
├── data/
│ └── loader.py # PDF loader and chunking logic
│
├── prompts/
│ └── prompt_template.py # Custom prompt templates
│
├── vectorstores/
│ └── chroma_store.py # Vectorstore setup using Chroma
│
├── .env # Secrets like GROQ_API_KEY
├── .gitignore
├── app.py # Main Streamlit interface
├── config.py # Configuration values (paths, model name, etc.)
├── requirements.txt # Python dependencies
├── style.css # Custom Streamlit styling
└── venv/ # Python virtual environment

