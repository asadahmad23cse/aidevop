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
import threading
from collections import deque
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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

REQUESTED_BACKEND = os.environ.get("RETRIEVAL_BACKEND", "semantic").lower()
try:
    if REQUESTED_BACKEND != "semantic":
        raise ImportError("semantic dependencies intentionally disabled")
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    SEMANTIC_DEPS_AVAILABLE = True
except ImportError:
    np = None
    SentenceTransformer = None
    faiss = None
    SEMANTIC_DEPS_AVAILABLE = False

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
RETRIEVAL_BACKEND = "semantic" if REQUESTED_BACKEND == "semantic" and SEMANTIC_DEPS_AVAILABLE else "lexical"

EMBEDDER = None
if RETRIEVAL_BACKEND == "semantic":
    print(f"\nLoading embedding model: {MODEL_NAME}...")
    _t = time.time()
    EMBEDDER = SentenceTransformer(MODEL_NAME)
    print(f"Model loaded in {time.time()-_t:.1f}s")
else:
    print("\nUsing memory-efficient lexical retrieval backend")

FAISS_INDEX = None
CHUNK_META = []
INDEX_STATS = {}


def load_existing_index():
    """Try to load pre-built index from disk on startup."""
    global FAISS_INDEX, CHUNK_META, INDEX_STATS
    idx_path = os.path.join(INDEX_DIR, "faiss.index")
    meta_path = os.path.join(INDEX_DIR, "chunk_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            CHUNK_META = json.load(f)
    if RETRIEVAL_BACKEND == "semantic" and os.path.exists(idx_path) and CHUNK_META:
        FAISS_INDEX = faiss.read_index(idx_path)
        INDEX_STATS = {
            "vectors": FAISS_INDEX.ntotal,
            "dimensions": FAISS_INDEX.d,
            "chunks": len(CHUNK_META),
            "source": "pre-built index",
        }
        print(f"Loaded existing index: {FAISS_INDEX.ntotal} vectors")
    elif CHUNK_META:
        INDEX_STATS = {
            "vectors": len(CHUNK_META),
            "dimensions": 0,
            "chunks": len(CHUNK_META),
            "source": "pre-built chunk metadata",
            "retrieval_backend": "lexical",
        }
        print(f"Loaded existing metadata: {len(CHUNK_META)} searchable chunks")


load_existing_index()


def index_ready() -> bool:
    return bool(CHUNK_META) and (RETRIEVAL_BACKEND == "lexical" or FAISS_INDEX is not None)


def total_vectors() -> int:
    return int(FAISS_INDEX.ntotal) if FAISS_INDEX is not None else len(CHUNK_META)


def retrieve_chunks(query: str, top_k: int) -> tuple[List[dict], List[float], int, float, float]:
    """Search via FAISS locally or a low-memory lexical scorer in constrained cloud runtimes."""
    if not index_ready():
        raise HTTPException(400, "No index found. Please ingest a document first.")

    top_k = max(1, min(top_k, len(CHUNK_META)))
    if RETRIEVAL_BACKEND == "semantic":
        t0 = time.time()
        q_vec = EMBEDDER.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        t_embed = time.time() - t0
        t0 = time.time()
        scores, indices = FAISS_INDEX.search(q_vec, k=top_k)
        t_search = time.time() - t0
        valid = [(float(score), int(idx)) for score, idx in zip(scores[0], indices[0]) if 0 <= idx < len(CHUNK_META)]
        return [CHUNK_META[idx] for _, idx in valid], [score for score, _ in valid], int(q_vec.shape[1]), t_embed, t_search

    t0 = time.time()
    query_terms = set(re.findall(r"\b[a-z0-9_]+\b", query.lower()))
    stopwords = {"a", "an", "and", "are", "does", "for", "how", "in", "is", "of", "or", "the", "to", "what", "which", "with"}
    query_terms -= stopwords
    scored = []
    for idx, chunk in enumerate(CHUNK_META):
        text = chunk.get("text", "")
        terms = set(re.findall(r"\b[a-z0-9_]+\b", text.lower()))
        overlap = len(query_terms & terms)
        coverage = overlap / max(len(query_terms), 1)
        phrase_bonus = 0.15 if query.lower() in text.lower() else 0.0
        scored.append((coverage + phrase_bonus, idx))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[:top_k]
    t_search = time.time() - t0
    return [CHUNK_META[i] for _, i in selected], [float(s) for s, _ in selected], 0, 0.0, t_search


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


class PulseMirrorRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3
    model: Optional[str] = DEFAULT_MODEL


# Lightweight, process-local operational telemetry. It intentionally contains
# no patient identifiers or raw question text.
OBSERVABILITY_EVENTS = deque(maxlen=200)
OBSERVABILITY_LOCK = threading.Lock()
DEPLOYMENT_VERSION = os.environ.get("RENDER_GIT_COMMIT", os.environ.get("APP_VERSION", "local"))[:7]

EMERGENCY_PATTERNS = {
    "chest pain": "possible cardiac emergency",
    "difficulty breathing": "possible breathing emergency",
    "can't breathe": "possible breathing emergency",
    "severe bleeding": "possible severe bleeding",
    "unconscious": "loss of consciousness",
    "overdose": "possible overdose",
    "suicidal": "self-harm risk",
    "face drooping": "possible stroke warning sign",
}
HIGH_RISK_PATTERNS = {
    "dosage": "medication dosing request",
    "dose": "medication dosing request",
    "stop taking": "medication change request",
    "pregnant": "pregnancy-related query",
    "insulin": "high-risk medication query",
    "diagnose": "diagnostic request",
    "prescription": "prescription request",
}
MEDIUM_RISK_PATTERNS = {
    "symptom": "symptom assessment",
    "pain": "pain-related query",
    "fever": "acute symptom query",
    "blood pressure": "clinical measurement query",
    "heart rate": "clinical measurement query",
    "stress": "mental wellness query",
}


def classify_medical_risk(query: str) -> dict:
    """Deterministic routing guardrail; it is a safety policy, not a diagnosis."""
    lowered = query.lower()
    matches = [(term, reason) for term, reason in EMERGENCY_PATTERNS.items() if term in lowered]
    if matches:
        return {"level": "emergency", "score": 1.0, "reasons": sorted({reason for _, reason in matches})}
    matches = [(term, reason) for term, reason in HIGH_RISK_PATTERNS.items() if term in lowered]
    if matches:
        return {"level": "high", "score": 0.8, "reasons": sorted({reason for _, reason in matches})}
    matches = [(term, reason) for term, reason in MEDIUM_RISK_PATTERNS.items() if term in lowered]
    if matches:
        return {"level": "medium", "score": 0.5, "reasons": sorted({reason for _, reason in matches})}
    return {"level": "low", "score": 0.2, "reasons": ["general wellness or informational query"]}


def apply_medical_safety_gate(answer: str, risk: dict) -> tuple[str, List[str], bool]:
    """Block emergency generation and label higher-risk informational output."""
    flags = []
    blocked = False
    level = risk["level"]
    if level == "emergency":
        blocked = True
        flags.append("emergency_output_blocked")
        answer = (
            "This may be a medical emergency. Contact your local emergency service now or go to the nearest "
            "emergency department. Do not rely on this application for emergency diagnosis or treatment. "
            "If it is safe to do so, stay with the affected person and follow instructions from emergency professionals."
        )
    elif level == "high":
        flags.append("clinical_review_recommended")
        answer += "\n\nSafety note: This is general information, not a diagnosis or medication instruction. Confirm decisions with a qualified clinician."
    elif level == "medium":
        flags.append("medical_context_detected")
        answer += "\n\nSafety note: Seek professional care if symptoms are severe, worsening, or persistent."
    return answer, flags, blocked


def record_observability_event(event: dict) -> None:
    with OBSERVABILITY_LOCK:
        OBSERVABILITY_EVENTS.append(event)


def observability_snapshot() -> dict:
    with OBSERVABILITY_LOCK:
        events = list(OBSERVABILITY_EVENTS)
    if not events:
        return {
            "requests": 0, "avg_latency_ms": 0, "p95_latency_ms": 0,
            "fallback_rate": 0, "safety_flag_rate": 0, "routes": {},
            "risk_levels": {}, "recent": [],
        }
    latencies = sorted(float(event["latency_ms"]) for event in events)
    p95_index = min(len(latencies) - 1, max(0, round(0.95 * len(latencies)) - 1))
    routes, risks = {}, {}
    for event in events:
        routes[event["route"]] = routes.get(event["route"], 0) + 1
        risks[event["risk_level"]] = risks.get(event["risk_level"], 0) + 1
    return {
        "requests": len(events),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        "p95_latency_ms": round(latencies[p95_index], 1),
        "fallback_rate": round(sum(bool(e["fallback_used"]) for e in events) / len(events), 3),
        "safety_flag_rate": round(sum(bool(e["safety_flags"]) for e in events) / len(events), 3),
        "routes": routes,
        "risk_levels": risks,
        "recent": events[-8:][::-1],
    }


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
        "ready": index_ready(),
        "services": {
            "application_service": {"status": "healthy", "description": "Dashboard Web UI on port 8000"},
            "retrieval_rag_service": {
                "status": "healthy" if index_ready() else "no_index",
                "vectors": total_vectors(),
                "chunks": len(CHUNK_META),
                "embedding_model": MODEL_NAME,
                "dimensions": FAISS_INDEX.d if FAISS_INDEX else 0,
                "backend": RETRIEVAL_BACKEND,
            },
            "llm_service_ollama": {
                "status": "connected" if ollama_info["available"] else "disconnected",
                "host": ollama_info["host"],
                "target_model": DEFAULT_MODEL,
                "available_models": ollama_info["models"],
                "has_target_model": ollama_info["has_default_model"]
            }
        },
        "vectors": total_vectors(),
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

    if RETRIEVAL_BACKEND != "semantic":
        os.makedirs(INDEX_DIR, exist_ok=True)
        meta_path = os.path.join(INDEX_DIR, "chunk_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunk_meta, f, ensure_ascii=False)
        CHUNK_META = chunk_meta
        FAISS_INDEX = None
        INDEX_STATS = {
            "filename": fname, "documents": len(records), "chunks": len(all_chunks),
            "vectors": len(all_chunks), "dimensions": 0, "retrieval_backend": "lexical",
            "t_parse_ms": round(t_parse * 1000), "t_chunk_ms": round(t_chunk * 1000),
        }
        return INDEX_STATS

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
    chunks, scores, dims, t_embed, t_search = retrieve_chunks(req.query, req.top_k)
    results = []
    for chunk, score in zip(chunks, scores):
        idx = CHUNK_META.index(chunk)
        results.append({
            "chunk_id": int(idx),
            "doc_id": chunk["doc_id"],
            "text": chunk["text"],
            "instruction": chunk.get("instruction", ""),
            "score": round(float(score), 4),
        })

    return {
        "query": req.query,
        "query_vector": [],
        "query_dims": dims,
        "retrieval_backend": RETRIEVAL_BACKEND,
        "results": results,
        "t_embed_ms": round(t_embed * 1000),
        "t_search_ms": round(t_search * 1000, 2),
        "total_vectors": total_vectors(),
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
    if not index_ready():
        raise HTTPException(400, "Vector index not ready.")

    # 1. Retrieval
    retrieved, scores, _, _, _ = retrieve_chunks(req.query, req.top_k or 3)
    chunks = []
    for chunk, score in zip(retrieved, scores):
        chunks.append({
            "chunk_id": CHUNK_META.index(chunk),
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


@app.post("/api/v1/pulsemirror/ask")
def pulsemirror_risk_aware_ask(req: PulseMirrorRequest):
    """Risk-aware RAG orchestration with a deterministic safety gate and fallback path."""
    started = time.perf_counter()
    query = req.query.strip()
    if not query:
        raise HTTPException(400, "Query must not be empty.")

    risk = classify_medical_risk(query)
    chunks = []
    generation_source = "emergency_rule_based"
    fallback_used = False

    if risk["level"] != "emergency":
        retrieved, scores, _, _, _ = retrieve_chunks(query, req.top_k or 3)
        for chunk, score in zip(retrieved, scores):
            chunks.append({
                "chunk_id": CHUNK_META.index(chunk),
                "doc_id": chunk.get("doc_id"),
                "text": chunk.get("text", ""),
                "instruction": chunk.get("instruction", ""),
                "score": round(float(score), 4),
            })
        generated = generate_rag_answer(GenerateRequest(query=query, chunks=chunks, model=req.model))
        answer = generated["answer"]
        generation_source = generated["generation_source"]
        fallback_used = generation_source == "extractive_fallback"
    else:
        answer = ""

    answer, safety_flags, blocked = apply_medical_safety_gate(answer, risk)
    if risk["level"] == "emergency":
        route = "emergency_safety_pipeline"
    elif risk["level"] == "high":
        route = "strict_grounded_pipeline"
    elif fallback_used:
        route = "grounded_fallback_pipeline"
    else:
        route = "standard_rag_pipeline"

    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    event = {
        "timestamp": int(time.time()),
        "risk_level": risk["level"],
        "route": route,
        "latency_ms": latency_ms,
        "fallback_used": fallback_used,
        "safety_flags": safety_flags,
        "generation_source": generation_source,
    }
    record_observability_event(event)

    return {
        "answer": answer,
        "risk": risk,
        "routing": {
            "selected_route": route,
            "generation_source": generation_source,
            "fallback_used": fallback_used,
            "output_blocked": blocked,
        },
        "safety_flags": safety_flags,
        "retrieved_chunks": chunks,
        "latency_ms": latency_ms,
        "deployment_version": DEPLOYMENT_VERSION,
        "disclaimer": "Research and wellness prototype; not a diagnostic medical device.",
    }


@app.get("/api/v1/observability")
def get_observability():
    ollama_info = check_ollama_status()
    snapshot = observability_snapshot()
    snapshot.update({
        "deployment_version": DEPLOYMENT_VERSION,
        "retrieval_backend": RETRIEVAL_BACKEND,
        "knowledge_chunks": len(CHUNK_META),
        "primary_model_available": ollama_info["available"],
        "fallback_service_ready": index_ready(),
        "policy": {
            "emergency": "block model output and show urgent-care guidance",
            "high": "strict grounded retrieval plus clinical-review warning",
            "medium": "grounded retrieval plus symptom safety warning",
            "low": "standard RAG; fall back to extractive evidence when needed",
        },
    })
    return snapshot


@app.get("/api/v1/week4/results")
def get_week4_results():
    """Expose the checked-in, reproducible Week 4 summary to the dashboard."""
    summary_path = Path(__file__).resolve().parent.parent / "outputs" / "week4" / "medical_evaluation_summary.json"
    if not summary_path.exists():
        raise HTTPException(404, "Week 4 evaluation summary is not available.")
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    model_names = {
        "codellama_7b": "Code Llama 7B",
        "starcoder2_3b": "StarCoder2 3B",
        "qwen25_coder_3b": "Qwen2.5-Coder 3B",
    }
    models = []
    for key in ("codellama_7b", "starcoder2_3b", "qwen25_coder_3b"):
        values = summary["models"][key]
        quality = values["quality"]
        manual = quality["manual_adjudication"]
        performance = values["performance"]
        models.append({
            "id": key,
            "name": model_names[key],
            "correctness": manual["correctness_accuracy"],
            "relevance": manual["relevance"],
            "hallucination_rate": manual["hallucination_rate"],
            "retrieval_recall_at_3": quality["retrieval"]["recall_at_k"],
            "latency_ms_mean": performance["latency_ms_mean"],
            "total_tokens": performance["total_tokens"],
            "cpu_percent_mean": performance["system_cpu_percent_mean"],
            "ram_mb_peak_mean": performance["system_memory_used_mb_peak_mean"],
            "vram_mb_mean": performance["ollama_model_vram_mb_mean"],
        })
    return {
        "validity": summary["validity"],
        "models": models,
        "test_pass_rate": None,
        "test_pass_rate_note": "N/A: the benchmark evaluates medical QA and repository understanding, not executable code generation.",
        "conclusion": "StarCoder2 leads medical correctness and hallucination safety; Qwen is fastest; retrieval is fixed across models.",
    }


# Serve the dashboard from the same origin as the API in cloud deployments.
# Register this catch-all mount after all API routes so /status, /query, etc.
# continue to resolve before static files.
DASHBOARD_DIR = Path(__file__).resolve().parent.parent / "dashboard"
if DASHBOARD_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


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
