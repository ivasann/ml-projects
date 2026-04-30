# ⚡ Groq-Powered PDF RAG Chatbot

A fast Retrieval-Augmented Generation (RAG) chatbot that lets you chat with your PDF documents using Groq-powered LLMs and local embeddings.

---

## 🚀 Demo

Upload a PDF → Ask questions → Get accurate, context-aware answers instantly.

---

## 🧠 Features

* ⚡ Ultra-fast responses using Groq LLMs
* 📄 Chat with any PDF document
* 🧩 Semantic search with FAISS
* 🆓 Local embeddings (no OpenAI cost)
* 🖥 Simple Streamlit UI
* 🔍 Context-aware answers (RAG pipeline)

---

## 🏗 Architecture

```id="e3aqkw"
User Query
   ↓
Embedding Model
   ↓
FAISS Vector Search
   ↓
Relevant Chunks Retrieved
   ↓
Groq LLM (LLaMA 3)
   ↓
Final Answer
```

---

## 📁 Project Structure

```id="m7tq6l"
app/            # Core app logic
data/           # Input documents
embeddings/     # Vector database
scripts/        # Ingestion + evaluation
```

---

## ⚙️ Setup

### 1. Clone Repository

```id="s4kkp2"
git clone https://github.com/yourusername/groq-pdf-rag-chatbot.git
cd groq-pdf-rag-chatbot
```

### 2. Create Virtual Environment

```id="48u4wx"
python -m venv venv
source venv/bin/activate
# Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```id="syjs9d"
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Create a `.env` file:

```id="9nq0af"
GROQ_API_KEY=your_api_key_here
```

---

## ▶️ Run the App

```id="p3x3qp"
streamlit run app/main.py
```

---

## 📥 Usage

1. Upload a PDF
2. Wait for indexing
3. Ask questions about the document

---

## 🧪 Example Queries

* "Summarize this document"
* "What are the key points?"
* "Explain section 3 in simple terms"

---

## 🛠 Tech Stack

* Python
* Streamlit
* FAISS (Vector Database)
* HuggingFace Embeddings
* Groq LLM (LLaMA 3)

---

## 📈 Future Improvements

* Chat history / memory
* Multi-PDF support
* Source citations
* API deployment
* Hybrid search (BM25 + vector)

---

## 📄 License

MIT License
