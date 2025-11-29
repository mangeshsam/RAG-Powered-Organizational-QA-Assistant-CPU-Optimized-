#  **README.md — RAG-Powered Organizational QA Assistant (CPU-Optimized)
CPU-Based RAG LLM Inference Pipeline (GPT4All + FAISS + Streamlit GUI)**

---

# CPU-Based RAG LLM Inference Pipeline

*A fully offline Retrieval-Augmented Generation (RAG) system running on CPU using GPT4All + FAISS + Sentence Transformers, with a ChatGPT-style Streamlit GUI.*

This project lets you build a **local AI chatbot** that answers questions **based only on your own documents**, without using the internet.

Perfect for:
✔ Students
✔ Data Science Practicals
✔ Portfolio Projects
✔ Offline Intelligent Search Applications
✔ Company Knowledge-Base Chatbots

---

# Features

* **FAISS vector search**
* **MiniLM text embeddings**
* **GPT4All LLM (offline CPU model)**
* **TXT document ingestion**
* **ChatGPT-style UI (Streamlit)**
* **100% offline — data privacy guaranteed**
* Works on Windows CPU (no GPU required)

---

#  Project Structure

```
CPU_based_RAG_LLM_inference_pipeline/
│
├── data/                         # Your text files
├── embeddings/                   # Generated embeddings + metadata
├── index/                        # FAISS index
│
├── models/
│    └── GPT4All/                 # GPT4All .gguf models
│         └── Llama-3.2-3B-Instruct-uncensored-Q4_0.gguf
│
├── src/
│    ├── utils.py
│    ├── ingest.py
│    ├── embed.py
│    ├── index_faiss.py
│    ├── retrieve.py
│    ├── llm_infer.py
│    ├── query.py
│    └── __init__.py
│
├── app.py                        # Streamlit ChatGPT-like UI
├── requirements.txt
└── README.md
```

---

#  File Responsibilities

### `ingest.py`

* Load TXT documents
* Split into chunks with overlap
* Assign chunk IDs and metadata

### `embed.py`

* Use MiniLM encoder
* Generate vector embeddings
* Save `.npy` and metadata CSV

### `index_faiss.py`

* Build FAISS index
* Use cosine similarity via normalized vectors

### `retrieve.py`

* Embed the query
* Search top-k similar chunks
* Return text + metadata

### `llm_infer.py`

* Load GPT4All model (gguf)
* Generate answers using context
* Only answers from retrieved text (no hallucination)

### `query.py`

* Complete RAG pipeline
* `ask(question)` returns:

  * answer
  * context
  * sources

### `app.py`

* Full Streamlit UI
* ChatGPT-like interface
* Sidebar source display

---

#  Installation & Setup

## **1️ Create Conda environment**

```
conda create -n rag_evn python=3.11 -y
conda activate rag_evn
```

## **2️ Install dependencies**

```
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```
faiss-cpu
sentence-transformers
gpt4all
streamlit
numpy<2
tqdm
```

---

#  **3️ Add your documents**

Place your `.txt` files inside:

```
data/
```

Example:

```
data/immverse_company_overview.txt
data/student_faq.txt
```

---

#  **4️ Add your GPT4All model**

Download from GPT4All model browser.

Place inside:

```
models/GPT4All/
```

Example:

```
models/GPT4All/Llama-3.2-3B-Instruct-uncensored-Q4_0.gguf
```

---

#  **5️ Generate embeddings**

```
python -m src.embed
```

Output:

```
embeddings/embeddings.npy
embeddings/ids.npy
embeddings/docs_meta.csv
```

---

#  **6️ Build FAISS index**

```
python -m src.index_faiss
```

Output:

```
index/faiss.index
```

---

#  **7️ Run the ChatGPT-Style Web UI**

**Use Python inside the conda env:**

```
python -m streamlit run app.py
```

Open browser:

```
http://localhost:8501
```

---

# GUI Features

* Chat bubbles
* Chat history
* Bot responses grounded in document context
* Sources listed in sidebar
* Beautiful GPT-like interface

Example:

```
User: What does Immverse AI do?

Bot:
Immverse AI provides immersive AI-powered learning experiences
based on digital avatars and personalized training modules.

Sources:
- immverse_company_overview.txt (chunk 3)
- student_faq.txt (chunk 5)
```

---

# RAG Pipeline Flow (Architecture)

```
   User Question
          |
          v
  SentenceTransformer (MiniLM)
          |
          v
       FAISS
  (semantic search)
          |
          v
  Top-k relevant chunks
          |
          v
  GPT4All LLM (offline CPU)
          |
          v
  Final Answer + Sources
```

---

#  Performance Optimization

To speed up the model:

###  Use smaller model (recommended):

```
phi-2.Q4_0.gguf
```

###  In llm_infer.py set threads:

```python
LLM = GPT4All(
    model_name=MODEL_NAME,
    model_path=MODEL_DIR,
    n_threads=8,
    allow_download=False
)
```

###  Reduce tokens

```
max_tokens = 128
```

###  Limit retrieval

```
k = 2
```

---

# Common Errors & Fixes

### FAISS not found

Install inside environment:

```
pip install faiss-cpu
```

### Streamlit using wrong Python (3.12)

Run using:

```
python -m streamlit run app.py
```

### Model not found

Check your model folder:

```
models/GPT4All/
```

---
