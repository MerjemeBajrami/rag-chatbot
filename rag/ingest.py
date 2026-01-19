from __future__ import annotations

import argparse
import glob

import os
import uuid
from dataclasses import dataclass
from typing import Dict, List, Tuple
from pypdf.errors import PdfReadError, PdfStreamError

from tqdm import tqdm
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from rag.config import get_settings


@dataclass
class PageDoc:
    text: str
    source: str
    page: int  # 1-indexed


def load_pdf_pages(pdf_path: str) -> List[PageDoc]:
    reader = PdfReader(pdf_path)
    source = os.path.basename(pdf_path)

    pages: List[PageDoc] = []
    for idx, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = clean_text(txt)
        if txt.strip():
            pages.append(PageDoc(text=txt, source=source, page=idx + 1))
    return pages


def clean_text(text: str) -> str:
    # Minimal cleaning. Keep it conservative for standards documents.
    text = text.replace("\u00a0", " ")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text.strip()


def chunk_pages(pages: List[PageDoc]) -> List[Tuple[str, Dict]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: List[Tuple[str, Dict]] = []
    for p in pages:
        splits = splitter.split_text(p.text)
        for j, chunk in enumerate(splits):
            chunks.append(
                (
                    chunk,
                    {
                        "source": p.source,
                        "page": p.page,
                        "chunk_id": f"{p.source}-p{p.page}-c{j}",
                    },
                )
            )
    return chunks


def ensure_pinecone_index(index_name: str, dimension: int, cloud: str, region: str) -> None:
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is empty.")

    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = [idx["name"] for idx in pc.list_indexes()]

    if index_name in existing:
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )


def ingest_directory(data_dir: str, namespace: str, recreate: bool = False) -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is empty.")
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is empty.")

    pc = Pinecone(api_key=settings.pinecone_api_key)

    # Build embeddings first so we can know dimension
    embeddings = OpenAIEmbeddings(
        model=settings.embed_model,
        api_key=settings.openai_api_key,
    )
    #embeddings = GoogleGenerativeAIEmbeddings(
     #   model="models/text-embedding-004", 
     #   task_type="retrieval_document"
#)  # New Ollama Embedding setup
    
    # Determine embedding dimension with a tiny probe
    probe = embeddings.embed_query("dimension probe")
    dim = len(probe)

    if recreate:
        # DANGEROUS: delete and recreate
        existing = [idx["name"] for idx in pc.list_indexes()]
        if settings.pinecone_index_name in existing:
            pc.delete_index(settings.pinecone_index_name)

    ensure_pinecone_index(
        index_name=settings.pinecone_index_name,
        dimension=dim,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )

    index = pc.Index(settings.pinecone_index_name)
    vs = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

    pdf_paths = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in: {data_dir}")

    all_texts: List[str] = []
    all_metas: List[Dict] = []
    all_ids: List[str] = []

    '''for path in tqdm(pdf_paths, desc="Loading PDFs"):
        pages = load_pdf_pages(path)
        chunks = chunk_pages(pages)

        for chunk_text, meta in chunks:
            all_texts.append(chunk_text)
            all_metas.append(meta)
            # stable-ish unique id; can also include uuid to avoid collisions
            all_ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, meta["chunk_id"])))'''
    bad_files = []

    for path in tqdm(pdf_paths, desc="Loading PDFs"):
        try:
            pages = load_pdf_pages(path)
        except (PdfReadError, PdfStreamError, ValueError, OSError) as e:
            bad_files.append((os.path.basename(path), str(e)))
            continue
        except Exception as e:
        # last-resort: don't let one bad file kill ingestion
            bad_files.append((os.path.basename(path), f"Unknown error: {e}"))
            continue

        chunks = chunk_pages(pages)

        for chunk_text, meta in chunks:
            all_texts.append(chunk_text)
            all_metas.append(meta)
            all_ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, meta["chunk_id"])))

    if bad_files:
        print("\n⚠️ Skipped invalid/corrupted PDFs:")
        for fn, err in bad_files:
            print(f" - {fn}: {err}")
        print("👉 Re-download these files as real PDFs.\n")

    # Upsert in batches
    batch_size = 100
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Upserting to Pinecone"):
        bt = all_texts[i : i + batch_size]
        bm = all_metas[i : i + batch_size]
        bi = all_ids[i : i + batch_size]
        vs.add_texts(texts=bt, metadatas=bm, ids=bi)

    print(f"✅ Ingested {len(pdf_paths)} PDFs into index='{settings.pinecone_index_name}', namespace='{namespace}'.")
    print(f"✅ Total chunks upserted: {len(all_texts)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing PDFs.")
    parser.add_argument("--namespace", type=str, default="healthcare", help="Pinecone namespace.")
    parser.add_argument("--recreate", type=str, default="false", help="true/false: Recreate Pinecone index.")
    args = parser.parse_args()

    recreate = args.recreate.strip().lower() in ("1", "true", "yes", "y")
    ingest_directory(args.data_dir, args.namespace, recreate=recreate)


if __name__ == "__main__":
    main()
