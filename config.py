import os

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME="llama3-8b-8192"
GROQ_API_KEY =os.getenv("GROQ_API_KEY")
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT_ID = os.getenv("CHROMA_TENANT_ID")
CHROMA_DB_NAME = os.getenv("CHROMA_DB_NAME", "vectorstore")
