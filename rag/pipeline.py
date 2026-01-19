from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from rag.config import get_settings
from rag.retriever import retrieve_top_k
from rag.prompting import build_prompt, format_context
from rag.safety import detect_prompt_injection


def _format_citations(docs: List[Dict[str, Any]]) -> List[str]:
    cites = []
    for d in docs:
        src = d.get("source", "Unknown")
        page = d.get("page", "?")
        cites.append(f"({src}, p.{page})")
    # de-duplicate while preserving order
    seen = set()
    uniq = []
    for c in cites:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq


def answer_question(
    question: str,
    chat_history: List[BaseMessage],
    namespace: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "answer": str,
        "docs": [ {text, source, page, chunk_id}, ... ],
        "blocked": bool,
        "block_reason": str
      }
    """
    settings = get_settings()
    if top_k is None:
        top_k = settings.top_k

    # Guardrail: injection detection
    safety = detect_prompt_injection(question)
    if safety.is_suspicious:
        return {
            "answer": (
                "I can’t comply with instruction-override requests. "
                "Please ask a normal question about the healthcare standards in the documents."
            ),
            "docs": [],
            "blocked": True,
            "block_reason": safety.reason,
        }

    docs = retrieve_top_k(
        question=question,
        namespace=namespace,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )

    # If retrieval fails, return safe "I don't know"
    if not docs:
        return {
            "answer": "I don't know based on the provided documents.",
            "docs": [],
            "blocked": False,
            "block_reason": "",
        }

    prompt = build_prompt()
    context_str = format_context(docs)

    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key or None,
        temperature=0.2,
    )
    # Initialize Gemini 3 Flash
    '''llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=0.1,  # Low temperature = more factual, less creative
        max_tokens=None,
        timeout=None,
        version="v1beta",  # Force v1beta if the stable v1 doesn't have Gemini 3 yet
        max_retries=2,
    )'''
   

    messages = prompt.format_messages(
        chat_history=chat_history,
        question=question,
        context=context_str,
    )

    resp = llm.invoke(messages)
    text = resp.content if hasattr(resp, "content") else str(resp)

    # Ensure citations are present even if model forgets 
    citations = _format_citations(docs)
    if "Sources" not in text:
        text = text.strip() + "\n\nSources:\n- " + "\n- ".join(citations)

    return {
        "answer": text,
        "docs": docs,
        "blocked": False,
        "block_reason": "",
    }
