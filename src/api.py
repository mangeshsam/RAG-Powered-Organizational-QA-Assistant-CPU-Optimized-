# src/api.py
from fastapi import FastAPI, Query
from query import ask
import uvicorn
from pydantic import BaseModel

app = FastAPI(title="CPU RAG - Immverse demo")

class QueryRequest(BaseModel):
    question: str
    k: int = 4

@app.post("/ask")
def api_ask(req: QueryRequest):
    return ask(req.question, k=req.k)

@app.get("/health")
def health():
    return {"status":"ok"}

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)
