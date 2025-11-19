# src/index_faiss.py

import numpy as np
import faiss
from pathlib import Path

# FIXED: Correct relative import for package execution
from .utils import FAISS_INDEX_PATH, EMB_NPY_PATH


def build_faiss_index():
    vectors = np.load(EMB_NPY_PATH).astype("float32")
    dim = vectors.shape[1]
    print("Vectors shape:", vectors.shape)

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # Ensure index directory exists
    Path(FAISS_INDEX_PATH.parent).mkdir(parents=True, exist_ok=True)

    # Save index
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"FAISS index written to {FAISS_INDEX_PATH}, ntotal={index.ntotal}")


def load_index():
    if not Path(FAISS_INDEX_PATH).exists():
        raise FileNotFoundError("FAISS index not found. Build it first.")
    return faiss.read_index(str(FAISS_INDEX_PATH))


if __name__ == "__main__":
    build_faiss_index()
