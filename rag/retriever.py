from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_openai import OpenAIEmbeddings

from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from rag.config import get_settings


def _require_non_empty(value: str, name: str) -> None:
    if not value:
        raise ValueError(
            f"{name} is empty. Set it via environment variables (e.g. in .env)."
        )


def get_vectorstore(namespace: str) -> PineconeVectorStore:
    settings = get_settings()

    _require_non_empty(settings.pinecone_api_key, "PINECONE_API_KEY")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    embeddings = OpenAIEmbeddings(
        model=settings.embed_model,
        api_key=settings.openai_api_key or None,
    )
  

    return PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)


def retrieve_top_k(
    question: str,
    namespace: str,
    top_k: int,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Returns normalized docs containing:
    - text
    - source (filename)
    - page (1-indexed)
    - chunk_id
    """
    vs = get_vectorstore(namespace)

    results = vs.similarity_search(
        query=question,
        k=top_k,
        filter=metadata_filter,
    )

    normalized: List[Dict[str, Any]] = []
    for d in results:
        md = d.metadata or {}
        normalized.append(
            {
                "text": d.page_content,
                "source": md.get("source", "Unknown"),
                "page": md.get("page", "?"),
                "chunk_id": md.get("chunk_id", ""),
            }
        )
    return normalized
