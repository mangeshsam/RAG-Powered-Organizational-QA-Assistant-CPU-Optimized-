# src/llm_infer.py

import os
from gpt4all import GPT4All

# -----------------------------------------------------
# CORRECT MODEL PATH (use exact folder name!)
# -----------------------------------------------------

MODEL_DIR = r"C:\Users\SHREE\CPU_based_RAG_LLM_inference_pipeline\models\GPT4All"
MODEL_NAME = "Llama-3.2-3B-Instruct-uncensored-Q4_0.gguf"

MODEL_PATH = MODEL_DIR + "\\" + MODEL_NAME

print("\n=== MODEL PATH CONFIGURATION ===")
print("MODEL_DIR:  ", MODEL_DIR)
print("MODEL_PATH: ", MODEL_PATH)
print("================================\n")

# Check if file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"MODEL FILE NOT FOUND:\n{MODEL_PATH}")

# -----------------------------------------------------
# Load GPT4All model CORRECTLY
# -----------------------------------------------------

LLM = GPT4All(
    model_name=MODEL_NAME,
    model_path=MODEL_DIR,
    allow_download=False
)

# -----------------------------------------------------
# Generate Answer
# -----------------------------------------------------

def generate_answer(context_chunks, question, max_tokens=256, temp=0.1):
    context = "\n\n".join(context_chunks)

    prompt = f"""
Use ONLY the following context to answer the question.
If the answer is NOT in the context, reply: "I don't know."

---------------------
Context:
{context}
---------------------

Question:
{question}

Answer:
"""

    response = LLM.generate(
        prompt,
        max_tokens=max_tokens,
        temp=temp
    )

    return response.strip()


# Debug test
if __name__ == "__main__":
    print(generate_answer(
        ["Immverse AI is an immersive learning company."],
        "What is Immverse AI?"
    ))
