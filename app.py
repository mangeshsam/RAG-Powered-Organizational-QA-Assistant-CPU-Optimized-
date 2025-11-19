# app.py 

import streamlit as st
import time
from src.query import ask

st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖", layout="wide")

# ------------------ STYLES ------------------
st.markdown("""
    <style>
        .chat-message {
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .user-msg {
            background-color: #DCF8C6;
            text-align: right;
        }
        .bot-msg {
            background-color: #F1F0F0;
            text-align: left;
        }
    </style>
""", unsafe_allow_html=True)


# ------------------ TITLE ------------------
st.title("🤖 Local RAG LLM Chatbot")
st.caption("Powered by GPT4All + FAISS + Sentence Transformers")


# ------------------ SESSION STATE ------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ------------------ CHAT HISTORY ------------------
for msg in st.session_state.messages:
    role, text = msg
    css = "user-msg" if role == "user" else "bot-msg"
    st.markdown(f"<div class='chat-message {css}'>{text}</div>", unsafe_allow_html=True)


# ------------------ USER INPUT ------------------
user_input = st.text_input("Ask your question:", placeholder="Type here...")

if user_input:
    # Save user message
    st.session_state.messages.append(("user", user_input))

    with st.spinner("Thinking..."):
        result = ask(user_input)

    bot_answer = result["answer"]

    # Save assistant message
    st.session_state.messages.append(("bot", bot_answer))

    st.rerun()


# ------------------ SIDEBAR ------------------
st.sidebar.title("Sources")

if st.session_state.messages:
    if len(st.session_state.messages) >= 2:  # answer exists
        latest_question = st.session_state.messages[-2][1]
        latest_answer = st.session_state.messages[-1][1]
        result = ask(latest_question)

        for src in result["sources"]:
            st.sidebar.write(f"📄 **{src['doc_id']}** — chunk {src['chunk_id']}")
