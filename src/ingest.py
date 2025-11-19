# src/ingest.py

import os
from pathlib import Path
import json

# FIXED: use relative import
from .utils import DATA_DIR


def load_txt_documents():
    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".txt"):
            path = os.path.join(DATA_DIR, file)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            docs.append({"doc_id": file, "text": text})
    return docs


def docs_to_chunks(docs, chunk_size=500, overlap=100):
    chunks = []
    meta = []
    chunk_id = 0

    for d in docs:
        text = d["text"]
        doc_id = d["doc_id"]

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append(chunk_text)

            meta.append({
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": len(meta),
                "text": chunk_text
            })

            chunk_id += 1
            start += (chunk_size - overlap)

    return chunks, meta
