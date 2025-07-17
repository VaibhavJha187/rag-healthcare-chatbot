from langchain.chains import RetrievalQA
from prompts.prompt_template import QA_PROMPT
from langchain_groq import ChatGroq
import os

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")

def build_qa_chain(retriever):
    """
    Build QA chain using the provided hybrid retriever and Groq's LLM.
    """

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL_NAME,
    )

    # Build QA chain using "stuff" method
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )

    return chain
