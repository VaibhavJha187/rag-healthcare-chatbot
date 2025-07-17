
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document
from config import EMBED_MODEL_NAME
import os
import hashlib

def get_doc_hash(docs):
    all_text = ''.join([doc.page_content for doc in docs])
    return hashlib.md5(all_text.encode()).hexdigest()

def get_hybrid_retriever(vectorstore, docs):
    # BM25 Retriever (Keyword-based)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    # Chroma Retriever (Semantic)
    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Ensemble Retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]
    )

    return hybrid_retriever

def embed_and_store_in_chroma(docs):
    doc_hash = get_doc_hash(docs)
    persist_directory = os.path.join("chroma_db", doc_hash)

    if os.path.exists(persist_directory):
        print(f"Using existing Chroma DB at: {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        )
    else:
        print(f"Creating new Chroma DB at: {persist_directory}")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
        )
        vectorstore.add_documents(docs)
        vectorstore.persist()

    return get_hybrid_retriever(vectorstore, docs)

