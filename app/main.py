from __future__ import annotations
import os
from typing import Any, Dict, List, Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from pathlib import Path 
import sys
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
from rag.pipeline import answer_question
from rag.config import get_settings







st.set_page_config(page_title="Healthcare RAG Chatbot", page_icon="🩺", layout="wide")

settings = get_settings()

st.title("🩺 Healthcare & Life Sciences RAG Chatbot")
st.caption("Grounded Q&A over healthcare governance / compliance / system-design PDFs (HIPAA, GDPR, ISO/IEC).")

with st.sidebar:
    st.subheader("Settings")
    namespace = st.text_input("Pinecone namespace", value="healthcare")
    top_k = st.slider("Top-K passages", min_value=2, max_value=12, value=settings.top_k)

    st.markdown("---")
    st.subheader("Metadata filter (optional)")
    st.write("Filter retrieval to a specific document name (exact match).")
    doc_filter = st.text_input("Document name (e.g., ISO_27001.pdf)", value="").strip()

    st.markdown("---")
    st.subheader("Memory")
    mem_turns = st.slider("Keep last N turns", min_value=2, max_value=12, value=6)
    
    st.markdown("---")
    st.subheader("Conversation")

    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()


    st.markdown("---")
    st.subheader("Status")
    st.write(f"Model: `{settings.chat_model}`")
    st.write(f"Embeddings: `{settings.embed_model}`")
    st.write(f"Pinecone index: `{settings.pinecone_index_name}`")

    if not settings.openai_api_key:
        st.warning("OPENAI_API_KEY is not set (answers will fail).")
    if not settings.pinecone_api_key:
        st.warning("PINECONE_API_KEY is not set (retrieval will fail).")


def get_history() -> List[BaseMessage]:
    if "history" not in st.session_state:
        st.session_state.history = []
    return st.session_state.history


def push_history(role: str, content: str) -> None:
    hist = get_history()
    if role == "user":
        hist.append(HumanMessage(content=content))
    else:
        hist.append(AIMessage(content=content))

 
    # Each turn is user+assistant, so keep 2*mem_turns messages
    max_msgs = 2 * mem_turns
    if len(hist) > max_msgs:
        st.session_state.history = hist[-max_msgs:]


# Render existing chat
for msg in get_history():
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    else:
        with st.chat_message("assistant"):
            st.markdown(msg.content)


user_q = st.chat_input("Ask about HIPAA, GDPR, ISO/IEC, governance, system design...")

if user_q:
    push_history("user", user_q)
    with st.chat_message("user"):
        st.markdown(user_q)

    metadata_filter: Optional[Dict[str, Any]] = None
    if doc_filter:
        metadata_filter = {"source": {"$eq": doc_filter}}

    with st.chat_message("assistant"):
        with st.spinner("Retrieving sources and drafting a grounded answer..."):
            result = answer_question(
                question=user_q,
                chat_history=get_history()[:-1],  # exclude the current user message from history placeholder
                namespace=namespace,
                top_k=top_k,
                metadata_filter=metadata_filter,
            )

        st.markdown(result["answer"])

        # Optional: show retrieved chunks
        with st.expander("Retrieved evidence (debug)"):
            if not result["docs"]:
                st.write("No passages retrieved.")
            else:
                for i, d in enumerate(result["docs"], start=1):
                    st.markdown(f"**{i}. {d.get('source')} — p.{d.get('page')}**  \n`{d.get('chunk_id')}`")
                    st.write(d.get("text", "")[:1500] + ("..." if len(d.get("text", "")) > 1500 else ""))

        if result.get("blocked"):
            st.warning(f"Guardrail triggered: {result.get('block_reason','')}")

    push_history("assistant", result["answer"])
