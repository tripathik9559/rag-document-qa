"""
ingestor.py
-----------
Responsible for:
1. Extracting text from a PDF
2. Cleaning the extracted text
3. Splitting the text into overlapping chunks (with page metadata)

This is Stage 1 of the RAG pipeline: DOCUMENT PROCESSING.
"""

import re
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Reads a PDF and extracts text page by page.

    Returns:
        List of dicts: [{"page_num": 1, "text": "..."}, {"page_num": 2, "text": "..."}, ...]
    """
    reader = PdfReader(file_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append({"page_num": i + 1, "text": text})

    return pages


def clean_text(text: str) -> str:
    """
    Cleans raw extracted PDF text:
    - Removes extra whitespace/newlines
    - Removes common header/footer noise (page numbers, repeated dashes)
    """
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)
    return text.strip()


def chunk_text(pages: list[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[dict]:
    """
    Splits cleaned page text into overlapping chunks.
    Each chunk keeps track of which page it came from -> needed for source citation.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    chunk_counter = 0

    for page in pages:
        cleaned = clean_text(page["text"])
        if not cleaned:
            continue

        splits = splitter.split_text(cleaned)
        for split in splits:
            all_chunks.append({
                "chunk_id": f"chunk_{chunk_counter}",
                "text": split,
                "page_num": page["page_num"],
            })
            chunk_counter += 1

    return all_chunks


def process_pdf(file_path: str) -> list[dict]:
    """Full ingestion pipeline: PDF -> extracted pages -> cleaned -> chunked."""
    pages = extract_text_from_pdf(file_path)
    chunks = chunk_text(pages)
    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingestor.py <path_to_pdf>")
        sys.exit(1)

    result_chunks = process_pdf(sys.argv[1])
    print(f"Total chunks created: {len(result_chunks)}\n")
    for c in result_chunks[:3]:
        print(f"--- {c['chunk_id']} (Page {c['page_num']}) ---")
        print(c["text"][:300])
        print()
