"""
embedder.py
-----------
Responsible for:
1. Converting text chunks into embeddings (Gemini embedding model)
2. Storing embeddings + chunk text + metadata into ChromaDB

This is Stage 2 of the pipeline, and it also sets up the store
that retriever.py will later query from.
"""

import os
import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "vectorstore/chroma_data"
COLLECTION_NAME = "documents"


def get_embedding_model():
    """Initializes and returns the Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_chroma_client():
    """Initializes a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection(client):
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_and_store(chunks: list[dict], doc_id: str) -> int:
    """
    Embeds each chunk and stores it in ChromaDB with metadata.
    Returns number of chunks stored.
    """
    embedder = get_embedding_model()
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    texts = [c["text"] for c in chunks]
    ids = [f"{doc_id}_{c['chunk_id']}" for c in chunks]
    metadatas = [{"doc_id": doc_id, "page_num": c["page_num"], "chunk_id": c["chunk_id"]} for c in chunks]

    vectors = embedder.embed_documents(texts)

    collection.add(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )

    return len(chunks)


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend.ingestor import process_pdf

    if len(sys.argv) < 2:
        print("Usage: python embedder.py <path_to_pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    doc_id = os.path.basename(pdf_path).replace(".pdf", "")

    chunks = process_pdf(pdf_path)
    count = embed_and_store(chunks, doc_id)
    print(f"Stored {count} chunks in ChromaDB under doc_id='{doc_id}'")
