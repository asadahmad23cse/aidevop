"""
rag_server.py — Real RAG Pipeline & Multi-Service API Backend (FastAPI)
========================================================================
Implements full Assignment Architecture:
  - Exercise 1: LLM Application endpoint (/api/v1/llm/generate) via Ollama + Code Llama
  - Exercise 2: Document Ingestion, Chunking, Embeddings, FAISS indexing (/ingest)
  - Exercise 3: Vector similarity search (/query), RAG context generation (/generate),
                and Side-by-side RAG vs No-RAG comparison (/api/v1/rag/compare)
  - Exercise 4: Orchestration & Service health checking (/api/v1/services/status)

Run locally:
    python src/rag_server.py
"""

import json
import os
import re
import time
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ollama_client import (
    query_ollama,
    generate_with_rag,
    check_ollama_status,
    DEFAULT_MODEL,
    OLLAMA_HOST,
)

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install sentence-transformers faiss-cpu fastapi uvicorn python-multipart")
    sys.exit(1)

# ── App Setup ────────────────────────────────────────────────
app = FastAPI(
    title="Medical RAG & LLM Orchestration API",
    description="Multi-tier RAG architecture with FAISS vector retrieval and Ollama (Code Llama) integration.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global State ─────────────────────────────────────────────
MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
INDEX_DIR = os.environ.get("RAG_INDEX_DIR", "outputs/rag_index")

print(f"\nLoading embedding model: {MODEL_NAME}...")
_t = time.time()
EMBEDDER = SentenceTransformer(MODEL_NAME)
print(f"Model loaded in {time.time()-_t:.1f}s")

FAISS_INDEX = None
CHUNK_META = []
INDEX_STATS = {}


def load_existing_index():
    """Try to load pre-built index from disk on startup."""
    global FAISS_INDEX, CHUNK_META, INDEX_STATS
    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "chunk_metadata.json")
    if os.path.exists(idx_path) and os.path.exists(meta_path):
        FAISS_INDEX = faiss.read_index(idx_path)
        with open(meta_path, encoding="utf-8") as f:
            CHUNK_META = json.load(f)
        INDEX_STATS = {
            "vectors": FAISS_INDEX.ntotal,
            "dimensions": FAISS_INDEX.d,
            "chunks": len(CHUNK_META),
            "source": "pre-built index",
        }
        print(f"Loaded existing index: {FAISS_INDEX.ntotal} vectors")


load_existing_index()


# ── Helper: Parse uploaded file ───────────────────────────────
def parse_file_content(text: str, filename: str) -> List[dict]:
    records = []
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext in ("jsonl", "json"):
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                instruction = obj.get("instruction") or obj.get("question") or obj.get("input") or ""
                output = obj.get("output") or obj.get("answer") or obj.get("text") or obj.get("content") or ""
                if instruction or output:
                    records.append({"instruction": instruction, "output": output})
            except Exception:
                pass

        if not records:
            try:
                arr = json.loads(text)
                if isinstance(arr, list):
                    for obj in arr:
                        instruction = obj.get("instruction") or obj.get("question") or ""
                        output = obj.get("output") or obj.get("answer") or ""
                        if instruction or output:
                            records.append({"instruction": instruction, "output": output})
            except Exception:
                pass

    if not records:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for p in paragraphs:
            records.append({"instruction": "", "output": p})

    return records


# ── Helper: Chunking ──────────────────────────────────────────
def chunk_text(text: str, chunk_words: int = 200, overlap_words: int = 38) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks


# ── Helper: Extractive Fallback Generation ────────────────────
def generate_extractive_answer(query: str, chunks: List[dict]) -> str:
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    stopwords = {"what", "how", "why", "when", "where", "is", "are", "the", "a", "an",
                 "of", "in", "for", "to", "do", "does", "i", "my", "and", "or", "with"}
    query_keywords = query_words - stopwords

    best_sentences = []
    for chunk in chunks:
        text = chunk["text"]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 40:
                continue
            sent_words = set(re.findall(r'\b\w+\b', sent.lower()))
            overlap = len(query_keywords & sent_words)
            score = overlap / max(len(query_keywords), 1)
            best_sentences.append((score, sent, chunk.get("score", 0)))

    best_sentences.sort(key=lambda x: x[0], reverse=True)
    top_sentences = best_sentences[:5]

    if not top_sentences or top_sentences[0][0] == 0:
        return f"Based on the retrieved medical knowledge:\n\n{chunks[0]['text'][:400]}..."

    lines = [f"[RAG-GROUNDED CLINICAL SUMMARY — {len(chunks)} Verified Sources Retrieved]\n"]
    seen = set()
    for score, sent, _ in top_sentences:
        if sent not in seen and score > 0:
            lines.append(f"• {sent}")
            seen.add(sent)

    lines.append("\n─── SOURCE EVIDENCE ───")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk["text"][:120].replace("\n", " ")
        lines.append(f"[{i}] (Score: {chunk.get('score', 0):.4f}) {preview}...")

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# REQUEST SCHEMAS
# ══════════════════════════════════════════════════════════════

class LLMRequest(BaseModel):
    prompt: str
    model: Optional[str] = DEFAULT_MODEL
    temperature: Optional[float] = 0.2


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class GenerateRequest(BaseModel):
    query: str
    chunks: List[dict]
    model: Optional[str] = DEFAULT_MODEL
    force_llm: Optional[bool] = False


class CompareRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    model: Optional[str] = DEFAULT_MODEL


# ══════════════════════════════════════════════════════════════
# EXERCISE 4 & SYSTEM ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/status")
@app.get("/api/v1/services/status")
def get_system_status():
    """
    Exercise 4: Multi-Service Architecture Status.
    Monitors Application Service, Vector/RAG Service, and Ollama LLM Service.
    """
    ollama_info = check_ollama_status()
    return {
        "ready": FAISS_INDEX is not None,
        "services": {
            "application_service": {"status": "healthy", "description": "Dashboard Web UI on port 8000"},
            "retrieval_rag_service": {
                "status": "healthy" if FAISS_INDEX is not None else "no_index",
                "vectors": FAISS_INDEX.ntotal if FAISS_INDEX else 0,
                "chunks": len(CHUNK_META),
                "embedding_model": MODEL_NAME,
                "dimensions": FAISS_INDEX.d if FAISS_INDEX else 0
            },
            "llm_service_ollama": {
                "status": "connected" if ollama_info["available"] else "disconnected",
                "host": ollama_info["host"],
                "target_model": DEFAULT_MODEL,
                "available_models": ollama_info["models"],
                "has_target_model": ollama_info["has_default_model"]
            }
        },
        "vectors": FAISS_INDEX.ntotal if FAISS_INDEX else 0,
        "chunks": len(CHUNK_META),
        "model": MODEL_NAME,
        "stats": INDEX_STATS
    }


# ══════════════════════════════════════════════════════════════
# EXERCISE 1: BASIC LLM APPLICATION API
# ══════════════════════════════════════════════════════════════

@app.post("/api/v1/llm/generate")
def llm_direct_generate(req: LLMRequest):
    """
    Exercise 1 Endpoint: User -> Application -> API -> Ollama -> Code Llama -> Response
    Direct LLM generation without retrieval context.
    """
    res = query_ollama(
        prompt=req.prompt,
        model=req.model or DEFAULT_MODEL,
        temperature=req.temperature or 0.2
    )
    return res


# ══════════════════════════════════════════════════════════════
# EXERCISE 2: KNOWLEDGE BASE INGESTION & VECTOR INDEXING
# ══════════════════════════════════════════════════════════════

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Exercise 2: Document Ingestion -> Chunking -> Embeddings -> FAISS Index.
    """
    global FAISS_INDEX, CHUNK_META, INDEX_STATS

    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    fname = file.filename

    # Step 1: Parse
    t0 = time.time()
    records = parse_file_content(text, fname)
    if not records:
        raise HTTPException(400, "No records found. Ensure file is valid JSONL or TXT.")
    t_parse = time.time() - t0

    # Step 2: Chunk
    t0 = time.time()
    all_chunks = []
    chunk_meta = []
    for i, rec in enumerate(records):
        combined = f"Q: {rec['instruction']}\nA: {rec['output']}" if rec["instruction"] else rec["output"]
        chunks = chunk_text(combined)
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_meta.append({
                "doc_id": i + 1,
                "chunk_idx": j,
                "text": chunk,
                "instruction": rec["instruction"],
            })
    t_chunk = time.time() - t0

    # Step 3: Embed
    t0 = time.time()
    embeddings = EMBEDDER.encode(
        all_chunks,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    t_embed = time.time() - t0
    dim = embeddings.shape[1]

    # Step 4: Index in FAISS
    t0 = time.time()
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    t_idx = time.time() - t0

    os.makedirs(INDEX_DIR, exist_ok=True)
    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "chunk_metadata.json")
    faiss.write_index(index, idx_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunk_meta, f, ensure_ascii=False)

    FAISS_INDEX = index
    CHUNK_META = chunk_meta
    INDEX_STATS = {
        "filename": fname,
        "documents": len(records),
        "chunks": len(all_chunks),
        "vectors": index.ntotal,
        "dimensions": dim,
        "index_size_kb": round(os.path.getsize(idx_path) / 1024, 1),
        "t_parse_ms": round(t_parse * 1000),
        "t_chunk_ms": round(t_chunk * 1000),
        "t_embed_ms": round(t_embed * 1000),
        "t_index_ms": round(t_idx * 1000),
        "sample_chunks": [c["text"][:120] for c in chunk_meta[:4]],
        "sample_vector": embeddings[0][:16].round(4).tolist(),
    }

    return INDEX_STATS


# ══════════════════════════════════════════════════════════════
# EXERCISE 3: RETRIEVAL, VECTOR SIMILARITY & RAG GENERATION
# ══════════════════════════════════════════════════════════════

@app.post("/query")
def query_vector_index(req: QueryRequest):
    """
    Exercise 3: Question -> Query Embedding -> FAISS Vector Similarity -> Relevant Chunks.
    """
    if FAISS_INDEX is None or not CHUNK_META:
        raise HTTPException(400, "No index found. Please ingest a document first.")

    t0 = time.time()
    q_vec = EMBEDDER.encode(
        [req.query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    t_embed = time.time() - t0

    t0 = time.time()
    scores, indices = FAISS_INDEX.search(q_vec, k=req.top_k)
    t_search = time.time() - t0

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(CHUNK_META):
            continue
        chunk = CHUNK_META[idx]
        results.append({
            "chunk_id": int(idx),
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "instruction": chunk.get("instruction", ""),
            "score": round(float(score), 4),
        })

    return {
        "query": req.query,
        "query_vector": q_vec[0][:16].round(4).tolist(),
        "query_dims": int(q_vec.shape[1]),
        "results": results,
        "t_embed_ms": round(t_embed * 1000),
        "t_search_ms": round(t_search * 1000, 2),
        "total_vectors": FAISS_INDEX.ntotal,
    }


@app.post("/generate")
def generate_rag_answer(req: GenerateRequest):
    """
    Exercise 3: Context + Question -> Ollama (Code Llama) -> Response.
    Falls back gracefully to extractive generation if Ollama is offline.
    """
    context_parts = [
        f"[Context {i}] (similarity: {c.get('score', 0):.4f}):\n{c['text']}"
        for i, c in enumerate(req.chunks, 1)
    ]

    augmented_prompt = (
        "### Instruction:\n"
        "You are an expert medical assistant. Answer the following query using ONLY the provided verified context.\n\n"
        "### Context:\n"
        + "\n\n".join(context_parts)
        + f"\n\n### Query:\n{req.query}\n\n### Response:"
    )

    ollama_status = check_ollama_status()
    generation_source = "extractive_fallback"
    answer = ""

    if ollama_status["available"]:
        ollama_res = generate_with_rag(
            query=req.query,
            context_chunks=req.chunks,
            model=req.model or DEFAULT_MODEL
        )
        if ollama_res.get("success"):
            answer = ollama_res["response"]
            generation_source = f"ollama/{req.model or DEFAULT_MODEL}"

    if not answer:
        answer = generate_extractive_answer(req.query, req.chunks)

    return {
        "augmented_prompt": augmented_prompt,
        "prompt_tokens": len(augmented_prompt.split()),
        "answer": answer,
        "chunks_used": len(req.chunks),
        "avg_similarity": round(sum(c.get("score", 0) for c in req.chunks) / max(len(req.chunks), 1), 4),
        "generation_source": generation_source,
        "ollama_available": ollama_status["available"]
    }


@app.post("/api/v1/rag/compare")
def compare_rag_vs_baseline(req: CompareRequest):
    """
    Exercise 3 Deliverable: Demonstrate how the response differs when relevant
    information is provided through RAG compared with asking the LLM without retrieval.
    """
    if FAISS_INDEX is None or not CHUNK_META:
        raise HTTPException(400, "Vector index not ready.")

    # 1. Retrieval
    q_vec = EMBEDDER.encode([req.query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    scores, indices = FAISS_INDEX.search(q_vec, k=req.top_k or 3)
    chunks = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(CHUNK_META):
            chunk = CHUNK_META[idx]
            chunks.append({
                "chunk_id": int(idx),
                "text": chunk["text"],
                "score": round(float(score), 4)
            })

    # 2. Without RAG (Exercise 1 Pure LLM)
    raw_res = query_ollama(prompt=req.query, model=req.model or DEFAULT_MODEL)
    baseline_answer = raw_res.get("response") if raw_res.get("success") else "Generic ungrounded response (Ollama offline fallback)."

    # 3. With RAG (Exercise 3 Augmented)
    rag_res = generate_rag_answer(GenerateRequest(query=req.query, chunks=chunks, model=req.model))

    return {
        "query": req.query,
        "retrieved_chunks": chunks,
        "without_rag": {
            "answer": baseline_answer,
            "source": f"ollama/{req.model or DEFAULT_MODEL} (Pure Parametric Memory)",
            "grounded": False
        },
        "with_rag": {
            "answer": rag_res["answer"],
            "source": rag_res["generation_source"],
            "grounded": True,
            "avg_similarity": rag_res["avg_similarity"]
        }
    }


# ── Run Server ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001))
    print("\n" + "=" * 60)
    print("  Medical RAG & LLM Orchestration Service Starting...")
    print(f"  API Port: {port}")
    print(f"  Swagger Docs: http://localhost:{port}/docs")
    print(f"  Ollama Host:  {OLLAMA_HOST}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
