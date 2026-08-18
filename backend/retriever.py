"""
retriever.py
------------
Responsible for:
1. Embedding the user's question
2. Searching ChromaDB for the most semantically similar chunks

This is Stage 3 (RETRIEVAL) - the "R" in RAG.

IMPORTANT: Always test this file in isolation before moving to generator.py.
If retrieval quality is bad, no prompt to Gemini will fix it.
"""

import os
from backend.embedder import get_embedding_model, get_chroma_client, get_or_create_collection


def get_relevant_chunks(query: str, doc_id: str = None, k: int = 5) -> list[dict]:
    """
    Given a user query, returns the top-k most relevant chunks from ChromaDB.
    """
    embedder = get_embedding_model()
    client = get_chroma_client()
    collection = get_or_create_collection(client)

    query_vector = embedder.embed_query(query)

    where_filter = {"doc_id": doc_id} if doc_id else None

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        where=where_filter,
    )

    chunks = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, meta, dist in zip(documents, metadatas, distances):
        chunks.append({
            "text": text,
            "page_num": meta.get("page_num"),
            "doc_id": meta.get("doc_id"),
            "distance": dist,
        })

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python retriever.py "<query>" [doc_id]')
        sys.exit(1)

    test_query = sys.argv[1]
    test_doc_id = sys.argv[2] if len(sys.argv) > 2 else None

    results = get_relevant_chunks(test_query, doc_id=test_doc_id, k=5)

    print(f"Query: {test_query}\n")
    for i, chunk in enumerate(results):
        print(f"--- Chunk {i+1} | Page {chunk['page_num']} | distance={chunk['distance']:.4f} ---")
        print(chunk["text"][:300])
        print()

    print("Manually verify: do these chunks actually answer the query?")
    print("If not, fix chunk_size/overlap in ingestor.py or increase k before touching generator.py.")
