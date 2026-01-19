from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embed_model: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    #chat_model: str = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    #embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    #ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Pinecone
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "healthcare-rag")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))


def get_settings() -> Settings:
    return Settings()
