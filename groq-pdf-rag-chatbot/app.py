import os
import streamlit as st
import tempfile
import gc

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# 🔑 GROQ API KEY
# Best practice: use st.secrets or a .env file for security
os.environ["GROQ_API_KEY"] = "Enter your API key!"

# Page setup
st.set_page_config(page_title="AI Document Analyst", layout="wide", page_icon="🧬")
st.title("🤖 Multi-PDF Intelligence Dashboard")

# Sidebar for controls and stats
with st.sidebar:
    st.header("⚙️ System Status")
    if "vectorstore" in st.session_state:
        st.success("Database: Online")
    else:
        st.warning("Database: Awaiting Data")

    if st.button("🗑️ Clear Chat History"):
        if "memory" in st.session_state:
            st.session_state.memory.clear()
        st.session_state.messages = []
        st.rerun()

# 1. File Upload
uploaded_files = st.file_uploader(
    "Upload PDFs (NVIDIA Reports, Technical Docs, etc.)",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    # Process documents once
    if "vectorstore" not in st.session_state:
        with st.status("🛠️ Building Vector Knowledge Base...") as status:
            all_docs = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    st.write(f"Parsing: {uploaded_file.name}")
                    loader = PyPDFLoader(tmp.name)
                    all_docs.extend(loader.load())

            # Optimized for financial tables and technical context
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(all_docs)

            st.write("Generating Embeddings...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

            # Critical for secondary laptop RAM
            gc.collect()
            status.update(label="✅ Documents Indexed!", state="complete", expanded=False)

    # 2. Memory Setup (Explicit Output Key Fix)
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # FIXED: Tells memory which key to save
        )

    # 3. LLM + Chain Setup
    if "qa_chain" not in st.session_state:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)

        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=st.session_state.memory,
            return_source_documents=True,
            verbose=False
        )

    st.divider()

    # 4. Chat UI Logic
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input
    if user_question := st.chat_input("Ask a cross-document question..."):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing across documents..."):
                # Use the chain from session state
                response = st.session_state.qa_chain.invoke({"question": user_question})

                answer = response["answer"]
                st.markdown(answer)

                # Expandable Sources
                with st.expander("📄 View Source Context"):
                    for i, doc in enumerate(response["source_documents"]):
                        st.write(f"**Source {i+1} (Page {doc.metadata.get('page', 'N/A')}):**")
                        st.caption(doc.page_content[:400] + "...")
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👋 System Ready. Please upload at least one PDF to begin.")