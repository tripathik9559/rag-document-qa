# RAG-Based Document Q&A System

Upload a PDF, ask questions in natural language, get grounded answers with source page citations.

## Folder Structure

rag-document-qa/
- backend/ -> ingestor.py, embedder.py, retriever.py, generator.py, api.py
- frontend/ -> app.py (Streamlit UI)
- vectorstore/chroma_data/ -> ChromaDB storage (auto-filled)
- uploaded_docs/ -> uploaded PDFs

## Setup

1. python -m venv venv
2. venv\Scripts\activate
3. pip install -r requirements.txt
4. copy .env.example .env   (then paste your GOOGLE_API_KEY)
5. uvicorn backend.api:app --reload
6. streamlit run frontend/app.py

## Test order

python backend\ingestor.py sample.pdf
python backend\embedder.py sample.pdf
python backend\retriever.py "your question" <doc_id>
python backend\generator.py "your question" <doc_id>
