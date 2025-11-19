# src/query.py

# FIXED relative imports
from .retrieve import retrieve
from .llm_infer import generate_answer


def ask(question, k=4):
    # Retrieve top-k chunks
    results, scores = retrieve(question, k=k)

    # Extract only text from retrieved chunks
    texts = [r["text"] for r in results]

    # Generate answer using GPT4All LLM
    answer = generate_answer(texts, question)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {"doc_id": r["doc_id"], "chunk_id": r["chunk_id"]} for r in results
        ],
        "scores": scores
    }


if __name__ == "__main__":
    q = input("Enter question: ")
    out = ask(q)
    print("\nAnswer:\n", out["answer"])
    print("\nSources:", out["sources"])
