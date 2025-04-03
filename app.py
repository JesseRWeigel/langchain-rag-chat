import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

# Load and preprocess documents
def load_docs(file):
    if file.type == "application/pdf":
        loader = PyPDFLoader(file.name)
    else:
        loader = TextLoader(file.name)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

# Initialize vector database with valid client settings
def init_db(docs):
    embeddings = OpenAIEmbeddings()
    from chromadb.config import Settings
    # Removed unsupported fields "in_memory" and "tenant"
    client_settings = Settings(
        persist_directory="./chroma_db"  # Use this directory to persist data; omit for in-memory operation
    )
    db = Chroma.from_documents(docs, embeddings, client_settings=client_settings)
    return db

# Streamlit UI
st.title("Chat with Your Docs (LangChain + RAG)")

uploaded_file = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])
question = st.text_input("Ask something about your document:")

if uploaded_file and question:
    with open(uploaded_file.name, "wb") as f:
        f.write(uploaded_file.getbuffer())

    docs = load_docs(uploaded_file)
    db = init_db(docs)

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", temperature=0),
        chain_type="stuff",
        retriever=db.as_retriever()
    )

    response = qa_chain.run(question)
    st.write(response)