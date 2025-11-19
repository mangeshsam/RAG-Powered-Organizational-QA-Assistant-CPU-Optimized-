# src/embed.py

from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

# IMPORTANT: use relative imports inside src package
from .utils import (
    EMB_MODEL_NAME, EMB_BATCH, EMB_NPY_PATH, IDS_NPY_PATH,
    DOCS_JSON_PATH, CHUNK_SIZE, CHUNK_OVERLAP
)
from .ingest import load_txt_documents, docs_to_chunks

import csv
from pathlib import Path


def compute_and_save_embeddings(recompute=False):
    # Load docs & chunk
    docs = load_txt_documents()
    chunks, meta = docs_to_chunks(docs, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    print(f"Chunks: {len(chunks)}")

    # Load embedding model
    model = SentenceTransformer(EMB_MODEL_NAME)

    # encode in batches
    all_embs = []
    for i in tqdm(range(0, len(chunks), EMB_BATCH), desc="Embedding batches"):
        batch_texts = chunks[i:i+EMB_BATCH]
        embs = model.encode(batch_texts, convert_to_numpy=True, show_progress_bar=False)
        all_embs.append(embs)

    all_embs = np.vstack(all_embs).astype('float32')

    # Save embeddings and ids mapping
    np.save(EMB_NPY_PATH, all_embs)
    ids = np.array([m["chunk_id"] for m in meta], dtype=int)
    np.save(IDS_NPY_PATH, ids)

    # Save metadata CSV
    Path(DOCS_JSON_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(DOCS_JSON_PATH, "w", encoding="utf-8", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["chunk_id", "doc_id", "chunk_index", "text"])
        writer.writeheader()
        writer.writerows(meta)

    print(f"Saved embeddings ({all_embs.shape}) to {EMB_NPY_PATH}")
    print(f"Saved meta to {DOCS_JSON_PATH}")


if __name__ == "__main__":
    compute_and_save_embeddings()
