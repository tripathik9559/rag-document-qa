# 📄 RAG Document Q&A System

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-CONTAINERIZED-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![RAG](https://img.shields.io/badge/RAG-ENABLED-00C853?style=for-the-badge)
![License](https://img.shields.io/badge/LICENSE-EDUCATIONAL-7E57C2?style=for-the-badge)

---

**A GenAI-powered document intelligence system** — upload any PDF and ask questions in natural
language. Built on a full Retrieval-Augmented Generation (RAG) pipeline using Gemini + ChromaDB,
capable of grounding every answer in the source document with page-level citations.

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- A Google Gemini API key ([get one here](https://aistudio.google.com/app/apikey))

### Steps

```bash
# 1. Clone and enter the project
git clone https://github.com/tripathik9559/rag-document-qa.git
cd rag-document-qa

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and paste your GOOGLE_API_KEY

# 5. Start the backend (FastAPI)
uvicorn backend.api:app --reload

# 6. Start the frontend (in a new terminal)
streamlit run frontend/app.py
```

Open **http://localhost:8501** in your browser.

---

## 🐳 Docker Deployment

```bash
# 1. Copy and edit environment variables
cp .env.example .env
# Edit .env with your GOOGLE_API_KEY

# 2. Build and run
docker build -t rag-doc-qa .
docker run -p 8000:8000 --env-file .env rag-doc-qa
```

App will be available at **http://localhost:8000**

---

## 🏗️ Project Structure

```
rag-document-qa/
├── backend/
│   ├── __init__.py
│   ├── ingestor.py           # PDF → text extraction → cleaning → chunking
│   ├── embedder.py           # Chunks → Gemini embeddings → stored in ChromaDB
│   ├── retriever.py          # Question → embedding → semantic search → relevant chunks
│   ├── generator.py          # Retrieved chunks + question → prompt → Gemini → answer
│   └── api.py                # FastAPI endpoints wiring the full pipeline together
│
├── frontend/
│   └── app.py                 # Streamlit UI — upload, ask, view answers + sources
│
├── vectorstore/
│   └── chroma_data/           # ChromaDB's persistent vector storage (auto-created)
│
├── uploaded_docs/              # Uploaded PDFs saved here (auto-created)
│
├── .env.example                 # Template for GOOGLE_API_KEY
├── .gitignore
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🧠 How It Works

Instead of feeding an entire document into an LLM (expensive, slow, and inaccurate at scale),
this system retrieves only the most relevant pieces of the document before generating an answer.

```
📄 PDF Upload
     ↓
📖 Text Extraction
     ↓
🧹 Cleaning
     ↓
✂️ Chunking (with overlap)
     ↓
🔢 Embedding (Gemini)
     ↓
🗄️ ChromaDB (vector storage)


❓ User Question
     ↓
🔢 Question Embedding
     ↓
🔍 Semantic Search (top-k chunks)
     ↓
🤖 Gemini (context + question)
     ↓
💬 Grounded Answer + 📌 Source Page(s)
```

| Module | RAG Stage | Responsibility |
|---|---|---|
| `ingestor.py` | Document Processing | Extract text from PDF, clean it, split into overlapping chunks with page metadata |
| `embedder.py` | Document Processing | Convert chunks to vectors (Gemini embeddings), store in ChromaDB |
| `retriever.py` | **Retrieval** | Convert question to vector, semantic search in ChromaDB, return top-k chunks |
| `generator.py` | **Augmentation + Generation** | Build a prompt from retrieved chunks + question, call Gemini, return grounded answer |
| `api.py` | Orchestration | FastAPI endpoints — `/upload`, `/ask`, `/documents` — wire all of the above together |

---

## ✨ Features

### Document Handling
- PDF upload with automatic text extraction, cleaning, and chunking
- Overlapping chunks to preserve context across boundaries
- Multiple document support via unique `doc_id` per upload

### Retrieval & Answering
- Semantic (meaning-based) search — not just keyword matching
- Top-k relevant chunk retrieval from ChromaDB before every answer
- Answers grounded strictly in retrieved context — no hallucinated facts
- Page-level **source citation** on every answer, so it can be verified against the original PDF

### Interface
- Simple Streamlit UI: upload a PDF, ask a question, see the answer + source pages
- FastAPI backend with interactive Swagger docs (`/docs`) for direct API testing

---

## ⚙️ Environment Variables (`.env`)

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 🧪 Testing Each Stage Independently (Recommended Order)

Test each module standalone before running the full app — this makes it easy to isolate exactly
where a problem is.

**1. Ingestion**
```bash
python backend/ingestor.py path/to/sample.pdf
```
Check: are the chunks readable and reasonably sized?

**2. Embedding + storage**
```bash
python backend/embedder.py path/to/sample.pdf
```
Check: does it report chunks stored without errors?

**3. Retrieval** — ⚠️ do this before touching generation
```bash
python backend/retriever.py "What is IAM Role?" <doc_id>
```
Check: are the returned chunks *actually* relevant, or just keyword noise?
If retrieval is weak, adjust `chunk_size` / `chunk_overlap` in `ingestor.py`, or the `k` value —
**do not proceed** until this looks right. Bad retrieval means bad answers, no matter how good the prompt is.

**4. Generation**
```bash
python backend/generator.py "What is IAM Role?" <doc_id>
```
Check: is the answer grounded in the retrieved context, and are the source pages correct?

**5. Full API**
```bash
curl -X POST http://localhost:8000/upload -F "file=@sample.pdf"
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "What is IAM Role?"}'
```

**6. Full UI** — run Streamlit and test end-to-end.

---

## 📌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/upload` | Upload a PDF — processes and stores it in ChromaDB |
| `POST` | `/ask` | Ask a question — `{"question": "...", "doc_id": "optional"}` |
| `GET` | `/documents` | List all uploaded documents |

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Uvicorn, Pydantic |
| GenAI / RAG | LangChain, Google Gemini (LLM + Embeddings) |
| Vector Database | ChromaDB |
| Frontend | Streamlit |
| Containerization | Docker |

---

## 🔮 Roadmap

- [ ] DOCX / TXT file support
- [ ] Conversation memory (multi-turn chat)
- [ ] Streaming answers
- [ ] Hybrid search (keyword + semantic) and re-ranking
- [ ] PostgreSQL for document/user/chat metadata
- [ ] React + TypeScript + Tailwind frontend
- [ ] Production deployment (Docker Compose / cloud)

---

## 🎤 Key Concepts

- **RAG (Retrieval-Augmented Generation)** — retrieve relevant context first, then let the LLM generate an answer grounded in that context.
- **Embeddings** — numerical vectors that capture semantic meaning, enabling similarity search beyond exact keyword matches.
- **Chunking** — splitting large documents into smaller retrievable pieces, with overlap to preserve context across boundaries.
- **ChromaDB** — vector database used to store embeddings and perform similarity search.
- **Hallucination control** — the LLM is instructed to answer only from retrieved context; source page citations let users verify every answer.

---

## 📝 License
For educational and portfolio use.