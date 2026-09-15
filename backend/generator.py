"""
generator.py
------------
Responsible for:
1. Building a prompt from retrieved chunks + user question (AUGMENTATION)
2. Calling Gemini to generate a grounded answer (GENERATION)

This is Stage 4 - the "AG" in RAG.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from backend.retriever import get_relevant_chunks

load_dotenv()


def get_llm():
    """Initializes and returns the Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.2,
    )


def build_prompt(query: str, chunks: list[dict]) -> str:
    """Combines retrieved chunks into a context block and builds the final prompt."""
    context_block = "\n\n".join(
        f"[Page {c['page_num']}]\n{c['text']}" for c in chunks
    )

    prompt = f"""You are a helpful assistant that answers questions ONLY using the provided context.
If the answer is not present in the context, say "I couldn't find this in the document."
Always be concise and accurate. Do not make up information.

Context:
{context_block}

Question: {query}

Answer:"""
    return prompt


def generate_answer(query: str, doc_id: str = None, k: int = 5) -> dict:
    """
    Full RAG answer generation:
    1. Retrieve relevant chunks
    2. Build augmented prompt
    3. Call Gemini
    4. Return answer + source pages
    """
    chunks = get_relevant_chunks(query, doc_id=doc_id, k=k)

    if not chunks:
        return {"answer": "No relevant content found in the document.", "sources": []}

    prompt = build_prompt(query, chunks)
    llm = get_llm()

    response = llm.invoke(prompt)
    answer_text = response.content

    sources = sorted(set(c["page_num"] for c in chunks))

    return {"answer": answer_text, "sources": sources}


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print('Usage: python generator.py "<query>" [doc_id]')
        sys.exit(1)

    test_query = sys.argv[1]
    test_doc_id = sys.argv[2] if len(sys.argv) > 2 else None

    result = generate_answer(test_query, doc_id=test_doc_id)
    print(f"Answer:\n{result['answer']}\n")
    print(f"Sources: Page(s) {result['sources']}")
