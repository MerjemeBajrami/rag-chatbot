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



import re

IDK_TEXT = "I don't know based on the provided documents."

def strip_trailing_sources_block(text: str) -> str:
    """
    Removes any trailing 'Sources:' section that the model might output.
    Keeps inline citations like (Doc.pdf, p.10) intact.
    """
    if not text:
        return text

    # Common variants: "Sources:", "Sources (5)", "Sources\n\n..."
    pattern = r"\n\s*Sources\s*(?:\(\d+\))?\s*:\s*\n.*$"
    pattern2 = r"\n\s*Sources\s*\(\d+\)\s*\n.*$"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(pattern2, "", text, flags=re.IGNORECASE | re.DOTALL)

    return text.strip()




st.set_page_config(page_title="Healthcare RAG Chatbot", page_icon="🩺", layout="wide")
st.markdown(
    """
    <style>
  
    /* Add horizontal padding to chat area (messages + input) */
    [data-testid="stChatMessage"],
    [data-testid="stChatInput"] {
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Slightly tighten vertical spacing so it doesn't feel too airy */
    [data-testid="stChatMessage"] {
        padding-top: 0.25rem;
        padding-bottom: 0.25rem;
    }
    [data-testid="stChatMessage"],
    [data-testid="stChatInput"] {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Responsive: reduce padding on small screens */
    @media (max-width: 900px) {
        [data-testid="stChatMessage"],
        [data-testid="stChatInput"],
        .user-wrap {
            padding-left: 1.25rem;
            padding-right: 1.25rem;
        }
    }

    .user-wrap {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    .user-bubble {
        display: flex;
        justify-content: flex-end;
        margin: 0.35rem 0;
    }

    .user-bubble > div {
        max-width: 80%;
        padding: 0.6rem 0.85rem;
        border-radius: 1rem;
        border-top-right-radius: 0.25rem;
        line-height: 1.4;
        word-wrap: break-word;
        color: var(--text-color);
    }

    /* LIGHT MODE */
    @media (prefers-color-scheme: light) {
        .user-bubble > div {
            background: rgba(0, 0, 0, 0.06);
            border: 1px solid rgba(0, 0, 0, 0.10);
        }
    }

    /* DARK MODE */
    @media (prefers-color-scheme: dark) {
        .user-bubble > div {
            background: rgba(255, 255, 255, 0.07);
            border: 1px solid rgba(255, 255, 255, 0.10);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)




settings = get_settings()
def render_user_right(text: str) -> None:
    st.markdown(
        f"""
        <div class="user-wrap">
            <div class="user-bubble">
                <div>{text}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )



st.title("🩺 Healthcare & Life Sciences RAG Chatbot")
st.caption("Grounded Q&A over healthcare governance / compliance / system-design PDFs (HIPAA, GDPR, ISO/IEC).")
def get_available_documents(data_dir: str = "data") -> List[str]:
    """
    Return a sorted list of PDF filenames used for ingestion.
    Used to populate the document filter dropdown.
    """
    if not os.path.isdir(data_dir):
        return []

    return sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    )

with st.sidebar:
    st.subheader("⚙️ Controls")

    with st.expander("Retrieval settings", expanded=True):
        namespace = st.text_input("Pinecone namespace", value="healthcare")
        top_k = st.slider("Top-K passages", min_value=2, max_value=12, value=settings.top_k)

    with st.expander("Document filter"):
        st.caption("Restrict retrieval to a single document (exact filename match).")

        documents = get_available_documents()
        options = ["All documents"] + documents

        selected_doc = st.selectbox(
            "Select a document",
            options=options,
            index=0,
        )

        doc_filter = ""
        if selected_doc != "All documents":
            doc_filter = selected_doc


    with st.expander("Conversation", expanded=True):
        mem_turns = st.slider("Keep last N turns", min_value=2, max_value=12, value=6)

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    with st.expander("System status"):
        st.code(
            f"Model: {settings.chat_model}\n"
            f"Embeddings: {settings.embed_model}\n"
            f"Pinecone index: {settings.pinecone_index_name}"
        )

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
        render_user_right(msg.content)
    else:
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(msg.content)



user_q = st.chat_input("Ask about HIPAA, GDPR, ISO/IEC, governance, system design...")

if user_q:
    push_history("user", user_q)
    render_user_right(user_q)


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

        answer_text = strip_trailing_sources_block(result.get("answer", "")).strip()
        st.markdown(answer_text)

        
        
        docs = result.get("docs", []) or []

        is_idk = (answer_text == IDK_TEXT)

# Only show sources if the assistant actually answered (not IDK)
        if not is_idk:
            def fmt_page(p):
                try:
                    if isinstance(p, float) and p.is_integer():
                        return str(int(p))
                    return str(int(p)) if str(p).isdigit() else str(p)
                except Exception:
                    return str(p)

            seen = set()
            sources = []
            for d in docs:
                src = d.get("source", "Unknown")
                page = fmt_page(d.get("page", "?"))
                key = (src, page)
                if key not in seen:
                    seen.add(key)
                    sources.append((src, page))

            with st.expander(f"Retrieved passages ({len(sources)})", expanded=False):
                if not sources:
                    st.write("No sources retrieved.")
                else:
                    for i, (src, page) in enumerate(sources, start=1):
                        st.markdown(f"{i}. **{src}** — p.{page}")

        if result.get("blocked"):
            st.warning(f"Guardrail triggered: {result.get('block_reason','')}")

    push_history("assistant", answer_text)

