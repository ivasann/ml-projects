import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

# 1. Configuration
load_dotenv()
os.environ["GROQ_API_KEY"] = "Enter your api key!" # Or use a .env file

st.set_page_config(page_title="Groq PDF Chatbot", layout="wide")
st.title("⚡ Groq-Powered PDF RAG Chatbot")

# 2. File Upload & Processing
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    # Save file locally to process
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Analyzing PDF..."):
        # Load and Split Text
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(documents)

        # Create Embeddings (Local & Free)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create Vector Store (FAISS)
        vectorstore = FAISS.from_documents(final_docs, embeddings)
        st.success("PDF Indexed Successfully!")

    # 3. Chat Interface
    st.divider()
    user_question = st.text_input("Ask a question about your document:")

    if user_question:
        # Initialize Groq LLM
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile", # High-performance model
            temperature=0
        )

        # Create QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        with st.spinner("Groq is thinking..."):
            response = qa_chain.invoke(user_question)
            st.markdown("### Answer:")
            st.write(response["result"])