# src/utils.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
EMB_DIR = ROOT / "embeddings"
INDEX_DIR = ROOT / "index"
MODEL_DIR = ROOT / "models"

# Ensure dirs exist
for d in (EMB_DIR, INDEX_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# Configurable params (can also set via env vars)
EMB_MODEL_NAME = os.environ.get("EMB_MODEL_NAME", "all-MiniLM-L6-v2")
EMB_BATCH = int(os.environ.get("EMB_BATCH", 32))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))   # chars per chunk
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
EMB_NPY_PATH = EMB_DIR / "embeddings.npy"
IDS_NPY_PATH = EMB_DIR / "ids.npy"
DOCS_JSON_PATH = EMB_DIR / "docs_meta.csv"  # metadata mapping
LLAMA_MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", str(MODEL_DIR / "model.gguf"))
