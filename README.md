# 🤖 HR Policy Q&A Bot

> An AI-powered question answering system for HR policy documents.  
> Upload any company HR PDF and ask questions in plain English —  
> get instant answers with source page citations.

## 🔴 Live Demo
👉 **[Try it here](https://hr-qna-bot-bpsyeudn7ch5zen7bknbpi.streamlit.app/)**

---

## 💡 What Problem Does It Solve?

Employees waste hours manually searching through lengthy HR policy documents.
This bot makes any HR document instantly queryable in plain English — like a
smart search engine that actually understands meaning, not just keywords.

**Ask questions like:**
- *"What is the casual leave policy?"*
- *"How many days of probation are required?"*
- *"What are the maternity leave benefits?"*
- *"What happens if I work on a public holiday?"*

---

## 🏗️ Architecture — RAG Pipeline

PDF Upload
↓
Text Extraction      → PyMuPDF
↓
Text Chunking        → LangChain RecursiveCharacterTextSplitter
↓
Semantic Embedding   → all-MiniLM-L6-v2 (runs locally, free)
↓
Vector Storage       → FAISS (Facebook AI Similarity Search)
↓
Similarity Search    → cosine similarity, top-3 chunks
↓
Answer Generation    → Groq LLaMA 3.1 (sub-second inference)
↓
Web Interface        → Streamlit

This pattern is called **RAG — Retrieval-Augmented Generation**.
Instead of relying on the LLM's training knowledge, the system
retrieves relevant document sections first and uses them as
context. This prevents hallucination and keeps answers grounded
in your actual document.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Web UI | Streamlit | Interactive interface |
| PDF Reading | PyMuPDF | Text extraction |
| Text Splitting | LangChain | Overlapping chunks |
| Embeddings | HuggingFace (local) | Text → vectors |
| Vector Store | FAISS | Similarity search |
| LLM | Groq (LLaMA 3.1) | Answer generation |
| Orchestration | LangChain LCEL | Pipeline management |

---

## 📁 Project Structure

hr_qna_bot/
│
├── app.py                  # Streamlit UI — entry point
├── config.py               # All settings and constants
├── requirements.txt        # Python dependencies
├── runtime.txt             # Python version for deployment
├── .env                    # API keys (never committed)
├── .gitignore              # Git ignore rules
│
└── src/
├── init.py         # Package marker
├── pdf_processor.py    # PDF text extraction
├── text_chunker.py     # Text splitting with overlap
├── embeddings.py       # HuggingFace embedding model
├── vector_store.py     # FAISS build, save, search
└── qa_chain.py         # Prompt engineering + Groq LLM

---

## ⚙️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Haru-bit-code/hr-qna-bot.git
cd hr-qna-bot
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up API key
Create a `.env` file in the root folder:

GROQ_API_KEY="your_groq_api_key_here"

Get your free Groq API key at: https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔑 Key Concepts

**RAG (Retrieval-Augmented Generation)**  
Retrieves relevant document sections before generating answers.
Prevents hallucination — answers come only from your document.

**Semantic Search**  
Finds chunks by meaning, not keywords.
"Sick leave" and "medical leave" return the same results.

**Chunk Overlap**  
200-character overlap between chunks prevents answers
from being cut off at chunk boundaries.

**Prompt Engineering**  
Structured prompt with explicit rules forces the LLM
to stay grounded and admit when it cannot find an answer.

---

## 📊 Project Stats

| Metric | Value |
|---|---|
| Embedding dimensions | 384 |
| Default chunk size | 1000 chars |
| Chunk overlap | 200 chars |
| Chunks retrieved per query | 3 |
| LLM temperature | 0 (fully deterministic) |
| Average response time | < 2 seconds |

---

## 🚀 Future Improvements

- [ ] Support multiple PDF uploads simultaneously
- [ ] Add conversation memory across questions
- [ ] Support scanned PDFs using OCR
- [ ] Add user authentication
- [ ] Export Q&A session as PDF report

---

## 👨‍💻 Author

**Haru-bit-code**  
Built as a portfolio project demonstrating end-to-end
RAG pipeline implementation using open-source tools.

⭐ Star this repo if you found it useful!

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).