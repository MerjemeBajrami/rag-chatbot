# Project #4 — RAG Chatbot (Healthcare & Life Sciences)

This project implements a **Retrieval-Augmented Generation (RAG) chatbot** that answers governance, compliance (HIPAA, GDPR, ISO/IEC), and system-design questions using a curated set of healthcare and life sciences standards PDFs.

The system is designed to be **grounded, auditable, and safe**, aligning with best practices for AI systems in regulated domains such as healthcare.

---

## Project Scope & Features

This implementation satisfies the assignment requirements:

* **Ingestion pipeline**: load → clean → chunk → embed → index
* **Vector search retriever** using Pinecone (Top-K retrieval)
* **Answer generation grounded strictly in retrieved sources**
* **Citations** included in every answer (document name + page number)
* **Safe fallback**: explicit *“I don’t know”* when evidence is missing
* **Simple UI**: Streamlit-based chat interface
* **Nice-to-haves included**:

  * Short-term conversational memory
  * Metadata-based document filtering
  * Prompt-injection guardrails
  * Transparent retrieval debugging view

(See the Project 4 RAG-Chatbot PDF for full requirements.)

---

## 1) Setup

### Prerequisites

* Python **3.10+**
* A **Pinecone** account and API key
* An **OpenAI** API key (chat + embeddings)

---

### Install dependencies

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

---

### Environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key

PINECONE_INDEX_NAME=healthcare-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small
```

> **Note**
> For Pinecone free plans, only specific regions are supported.
> `us-east-1` is recommended.

---

## 2) Data Ingestion

Place all healthcare standards PDFs into a `data/` directory:

```text
data/
 ├─ OECD_Health_Data_Governance_2015.pdf
 ├─ OECD_Health_Data_Governance_Digital_Age_2022.pdf
 ├─ FRA_CoE_EDPS_Handbook_Data_Protection_Law_2018.pdf
 └─ ...
```

Run ingestion:

```bash
python -m rag.ingest --data_dir ./data --namespace healthcare
```

What ingestion does:

1. Loads PDFs page by page
2. Cleans extracted text conservatively
3. Splits text into overlapping chunks
4. Generates embeddings (OpenAI)
5. Creates or reuses a Pinecone index
6. Upserts vectors with metadata:

   * `source` (document name)
   * `page` (page number)
   * `chunk_id` (stable identifier)

Invalid or corrupted PDFs are safely skipped.

---

## 3) Run the Chatbot UI

Start the Streamlit app:

```bash
streamlit run app/main.py
```

Open your browser at:

```
http://localhost:8501
```

---

## 4) Using the Chatbot

### Sidebar Controls

* **Pinecone namespace**
  Selects which logical dataset to query.

* **Top-K passages**
  Controls how many document chunks are retrieved per question.

* **Metadata filter (optional)**
  Restricts retrieval to a single document (exact filename match).

* **Memory (last N turns)**
  Controls short-term conversational context without affecting retrieval.

* **Status section**
  Displays the active chat model, embedding model, and Pinecone index.

* **Clear conversation**
  Resets chat history and memory.

---

### Retrieved Evidence (Debug)

Each answer includes a collapsible **“Retrieved evidence (debug)”** section showing:

* Retrieved document chunks
* Document names and page numbers
* Chunk identifiers

This supports **auditability, transparency, and evaluation** and is intentionally retained even when the model refuses to answer.

---

## 5) Safety & Guardrails

* The system **never answers outside retrieved evidence**
* Explicit refusal for:

  * Out-of-scope questions
  * Missing evidence
  * Prompt-injection attempts
* Lightweight prompt-injection detection blocks instruction overrides
* Citations are mandatory; answers without evidence result in *“I don’t know”*

---

## 6) Example Questions

* *What governance measures are recommended for managing health information systems?*
* *How does GDPR treat personal health data, and what obligations does it impose?*
* *What technical security measures are recommended for EHR exchange?*
* *What is pseudonymisation and how does it differ from anonymisation?*

Out-of-scope example:

* *How do I make a pizza using HIPAA guidelines?* → safely refused

---

## 7) Project Structure

```text
rag-chatbot/
├─ app/
│  └─ main.py              # Streamlit UI
├─ rag/
│  ├─ ingest.py            # PDF ingestion & indexing
│  ├─ retriever.py         # Vector search logic
│  ├─ pipeline.py          # RAG orchestration
│  ├─ prompting.py         # System & grounding prompts
│  ├─ safety.py            # Guardrails
│  └─ config.py            # Environment-based settings
├─ data/                   # Healthcare PDFs
├─ requirements.txt
├─ README.md
└─ .env                    # (not committed)
```

---

## 8) Notes

* This project is intended for **educational and demonstration purposes**
* It is not a substitute for legal or compliance advice
* The design emphasizes **explainability, safety, and grounded AI**, which are critical in healthcare settings




