# src/retrieve.py

import numpy as np
import csv

# FIXED: relative imports for package execution
from .utils import (
    EMB_NPY_PATH, IDS_NPY_PATH, DOCS_JSON_PATH, EMB_MODEL_NAME
)
from .index_faiss import load_index

from sentence_transformers import SentenceTransformer
import faiss


# Load embedding model once
MODEL = SentenceTransformer(EMB_MODEL_NAME)


# Load metadata CSV into a dictionary
def load_meta():
    meta = {}
    with open(DOCS_JSON_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            chunk_id = int(r["chunk_id"])
            meta[chunk_id] = r
    return meta


META = load_meta()
IDS = np.load(IDS_NPY_PATH)


def embed_query(q):
    vec = MODEL.encode([q], convert_to_numpy=True)
    vec = vec.astype('float32')
    faiss.normalize_L2(vec)
    return vec


def retrieve(query, k=5):
    qvec = embed_query(query)
    index = load_index()

    # FAISS search
    D, I = index.search(qvec, k)

    results = []
    for idx in I[0]:
        if idx == -1:
            continue

        chunk_id = int(IDS[idx])
        meta = META.get(chunk_id, {})

        results.append({
            "chunk_id": chunk_id,
            "doc_id": meta.get("doc_id"),
            "text": meta.get("text")
        })

    return results, D[0].tolist()
