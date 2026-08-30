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
    list_installed_models,
    model_status,
    DEFAULT_MODEL,
    OLLAMA_HOST,
)
from difficulty_classifier import classify as classify_difficulty
import guardrails

# ── LLM model routing config (PRD sections 3, 5) ──────────────
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_llm_routing() -> Dict[str, Any]:
    path = _CONFIG_DIR / "llm_models.json"
    data: Dict[str, Any] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    routing = data.get("routing", {})
    # Env overrides: LLM_MODEL_EASY / _MEDIUM / _COMPLEX
    for tier in ("easy", "medium", "complex"):
        env = os.environ.get(f"LLM_MODEL_{tier.upper()}")
        if env:
            routing.setdefault(tier, {})["ollama_tag"] = env
    data["routing"] = routing
    data.setdefault("fallback_order", ["medium", "easy", "complex"])
    data.setdefault("generation", {"temperature": 0.2, "num_predict": 320})
    data.setdefault(
        "grounded_system_prompt",
        "You are HealthRAG AI. Answer using ONLY the provided retrieved context. "
        "If the context is insufficient, say the knowledge base does not contain "
        "enough information. Do not invent facts. Cite context as [C1], [C2].",
    )
    return data


LLM_ROUTING = _load_llm_routing()


def resolve_llm_model(requested: Optional[str] = None) -> Optional[str]:
    """Return a model tag that Ollama actually has installed, or None.

    Order: an explicitly requested+installed tag -> the configured medium tier ->
    any installed model. Returning None tells callers to use the extractive
    fallback instead of hanging on an unresolvable tag (which can trigger
    Ollama's auto-pull - PRD forbids downloading models from the app)."""
    info = list_installed_models()
    installed = info["models"]
    if not info["online"] or not installed:
        return None

    def actual(tag):
        """Return the exact installed model name for a tag, or None.
        Ollama's /api/generate needs the precise name it reports in /api/tags."""
        if not tag:
            return None
        if tag in installed:
            return tag
        base = tag.split(":")[0]
        for name in installed:
            if name == tag or name.startswith(tag + ":") or name.split(":")[0] == base:
                return name
        return None

    candidates = [requested]
    candidates.append(LLM_ROUTING["routing"].get("medium", {}).get("ollama_tag"))
    for tier in LLM_ROUTING["fallback_order"]:
        candidates.append(LLM_ROUTING["routing"].get(tier, {}).get("ollama_tag"))
    for cand in candidates:
        resolved = actual(cand)
        if resolved:
            return resolved
    return installed[0]


def select_model(difficulty: str) -> Dict[str, Any]:
    """Pick an Ollama model for a difficulty tier, walking the fallback order
    when the preferred model is not installed. Always returns a target tag."""
    from ollama_client import _tag_matches
    routing = LLM_ROUTING["routing"]
    installed = list_installed_models()
    order = [difficulty] + [t for t in LLM_ROUTING["fallback_order"] if t != difficulty]

    preferred = routing.get(difficulty, {})
    for tier in order:
        cfg = routing.get(tier)
        if not cfg:
            continue
        tag = cfg.get("ollama_tag")
        if installed["online"] and _tag_matches(tag, installed["models"]):
            return {
                "tier": tier,
                "tag": tag,
                "name": cfg.get("name", tag),
                "developer": cfg.get("developer", "Unknown"),
                "reason": cfg.get("reason", ""),
                "fell_back": tier != difficulty,
                "preferred_tag": preferred.get("ollama_tag"),
                "ollama_online": True,
                "tag_status": "available",
            }

    # Nothing installed / Ollama offline - still report the preferred target.
    return {
        "tier": difficulty,
        "tag": preferred.get("ollama_tag", DEFAULT_MODEL),
        "name": preferred.get("name", preferred.get("ollama_tag", DEFAULT_MODEL)),
        "developer": preferred.get("developer", "Unknown"),
        "reason": preferred.get("reason", ""),
        "fell_back": False,
        "preferred_tag": preferred.get("ollama_tag"),
        "ollama_online": installed["online"],
        "tag_status": "not_installed" if installed["online"] else "ollama_offline",
    }

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

    lines = [f"[RAG-GROUNDED SUMMARY - {len(chunks)} verified sources retrieved]\n"]
    seen = set()
    for score, sent, _ in top_sentences:
        if sent not in seen and score > 0:
            lines.append(f"- {sent}")
            seen.add(sent)

    lines.append("\n--- SOURCE EVIDENCE ---")
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


class SuggestRequest(BaseModel):
    partial: str
    limit: Optional[int] = 4


class RouterRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


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
            },
            "llm_router": {
                "status": "ready",
                "tiers": {
                    tier: {
                        "configured_tag": cfg.get("ollama_tag"),
                        "name": cfg.get("name"),
                        "developer": cfg.get("developer"),
                        "available": (
                            model_status(cfg.get("ollama_tag", "")) == "available"
                            if ollama_info["available"] else False
                        ),
                    }
                    for tier, cfg in LLM_ROUTING["routing"].items()
                },
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

    usable_model = resolve_llm_model(req.model)
    if usable_model:
        ollama_res = generate_with_rag(
            query=req.query,
            context_chunks=req.chunks,
            model=usable_model,
        )
        if ollama_res.get("success"):
            answer = ollama_res["response"]
            generation_source = f"ollama/{usable_model}"

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
    usable_model = resolve_llm_model(req.model)
    if usable_model:
        raw_res = query_ollama(prompt=req.query, model=usable_model)
        if raw_res.get("success"):
            baseline_answer = raw_res.get("response")
            baseline_source = f"ollama/{usable_model} (pure parametric memory)"
        else:
            baseline_answer = "The language model could not be reached for an ungrounded baseline answer."
            baseline_source = "unavailable (Ollama call failed)"
    else:
        baseline_answer = (
            "No routed Ollama model is installed, so no ungrounded baseline was generated. "
            "Pull a model with 'ollama pull gemma3:1b' to enable this comparison."
        )
        baseline_source = "unavailable (no model installed)"

    # 3. With RAG (Exercise 3 Augmented)
    rag_res = generate_rag_answer(GenerateRequest(query=req.query, chunks=chunks, model=req.model))

    return {
        "query": req.query,
        "retrieved_chunks": chunks,
        "without_rag": {
            "answer": baseline_answer,
            "source": baseline_source,
            "grounded": False
        },
        "with_rag": {
            "answer": rag_res["answer"],
            "source": rag_res["generation_source"],
            "grounded": True,
            "avg_similarity": rag_res["avg_similarity"]
        }
    }


# ══════════════════════════════════════════════════════════════
# QUESTION SUGGESTION SERVICE
# ══════════════════════════════════════════════════════════════

_QS_TEMPLATES = [
    (re.compile(r"what does (the )?(doc(ument)?|guideline|report)s? say about\s*$", re.I),
     lambda t: f"What does the document say about {t}?"),
    (re.compile(r"what are the symptoms of\s*$", re.I),
     lambda t: f"What are the symptoms of {t}?"),
    (re.compile(r"how is\s*$", re.I),
     lambda t: f"How is {t} treated?"),
    (re.compile(r"what is the (recommended )?(dose|dosage) (of|for)\s*$", re.I),
     lambda t: f"What is the recommended dose of {t}?"),
    (re.compile(r"(tell me|explain) about\s*$", re.I),
     lambda t: f"Explain {t} from the uploaded document."),
]

# Common English words to exclude when mining topics from KB text.
_QS_STOPWORDS = {
    "about", "after", "again", "against", "along", "also", "always", "another",
    "around", "because", "been", "before", "being", "below", "between", "both",
    "could", "doctor", "document", "doing", "during", "each", "either", "every",
    "explain", "first", "found", "further", "guideline", "having", "however",
    "including", "into", "itself", "just", "least", "level", "like", "likely",
    "made", "many", "may", "might", "more", "most", "much", "must", "need",
    "often", "only", "other", "over", "please", "possible", "provide", "recommend",
    "recommended", "report", "result", "said", "same", "several", "should", "show",
    "since", "some", "such", "take", "tell", "than", "that", "their", "them",
    "then", "there", "these", "they", "this", "those", "through", "under", "until",
    "using", "very", "were", "what", "when", "where", "which", "while", "will",
    "with", "within", "without", "would", "your",
}

# Curated medical seed topics — always available even before ingestion.
_QS_SEED_TOPICS = [
    "hypertension", "diabetes", "asthma", "treatment", "diagnosis", "dosage",
    "side effects", "symptoms", "prevention", "risk factors", "amoxicillin",
    "paracetamol", "blood pressure", "cholesterol", "vaccination",
    "otitis media", "anticoagulants", "chronic kidney disease", "pregnancy",
    "pediatric dosing",
]


def _kb_topics(limit: int = 40) -> List[str]:
    """Derive candidate topics from indexed KB chunks, then append curated seeds."""
    freq: Dict[str, int] = {}
    for chunk in CHUNK_META[:2000]:
        text = (chunk.get("instruction") or "") + " " + chunk.get("text", "")
        for term in re.findall(r"\b[a-zA-Z][a-zA-Z\-]{5,}\b", text.lower()):
            if term in _QS_STOPWORDS:
                continue
            freq[term] = freq.get(term, 0) + 1
    ranked = [t for t, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])) if c > 1]
    # Curated seeds first (clean, predictable), then KB-mined terms for niche queries.
    merged = list(_QS_SEED_TOPICS)
    for term in ranked[:limit]:
        if term not in merged:
            merged.append(term)
    return merged


@app.post("/api/suggest/questions")
def suggest_questions(req: SuggestRequest):
    """
    Question Suggestion Service — returns 0..limit autocomplete suggestions for a
    partial question, grounded in the current knowledge-base topics. Never calls
    the LLM; safe to invoke on every keystroke. The frontend also has a local
    mock fallback when this endpoint is unavailable.
    """
    partial = (req.partial or "").strip()
    limit = max(1, min(req.limit or 4, 8))
    if len(partial) < 3:
        return {"partial": partial, "suggestions": [], "source": "kb" if CHUNK_META else "empty"}

    low = partial.lower()
    topics = _kb_topics()
    out: List[str] = []
    seen = set()

    def add(s: str):
        key = s.lower()
        if s and key not in seen and key != low:
            seen.add(key)
            out.append(s)

    # 1. Template completion using KB topics.
    for pattern, make in _QS_TEMPLATES:
        if pattern.search(low):
            for topic in topics:
                add(make(topic))
                if len(out) >= limit:
                    break
        if len(out) >= limit:
            break

    # 2. Topic match on the trailing token(s) of the partial query.
    if len(out) < limit:
        tail = low.split()[-1] if low.split() else ""
        for topic in topics:
            if tail and (topic.startswith(tail) or tail in topic):
                add(f"What does the document say about {topic}?")
                if len(out) >= limit:
                    break

    return {"partial": partial, "suggestions": out[:limit], "source": "kb"}


# ══════════════════════════════════════════════════════════════
# MODEL ROUTER + FULL GUARDRAIL PIPELINE (PRD sections 5-11)
# ══════════════════════════════════════════════════════════════

def _sources_from_chunks(chunks: List[dict]) -> List[dict]:
    out = []
    for i, c in enumerate(chunks, 1):
        out.append({
            "label": f"C{i}",
            "doc_id": c.get("doc_id"),
            "score": c.get("score"),
            "preview": (c.get("text", "")[:180] + "...") if c.get("text") else "",
        })
    return out


@app.get("/api/v1/models/status")
def models_status():
    """Which configured models are actually available locally (PRD section 13)."""
    info = list_installed_models()
    routing = LLM_ROUTING["routing"]
    tiers = []
    for tier in ("easy", "medium", "complex"):
        cfg = routing.get(tier, {})
        tag = cfg.get("ollama_tag", "")
        if not info["online"]:
            status = "ollama_offline"
        else:
            status = model_status(tag)
        tiers.append({
            "tier": tier,
            "tag": tag,
            "name": cfg.get("name", tag),
            "developer": cfg.get("developer", "Unknown"),
            "status": status,
        })
    return {
        "ollama_online": info["online"],
        "installed_models": info["models"],
        "tiers": tiers,
    }


@app.post("/api/v1/router/answer")
def router_answer(req: RouterRequest):
    """
    Full pipeline: prompt-injection guard -> domain guard -> RAG retrieval ->
    difficulty classification -> model routing -> Ollama generation ->
    grounding + medical-safety checks -> final answer (PRD section 24).

    Out-of-scope and prompt-injection queries are answered WITHOUT calling Ollama.
    """
    t_start = time.time()
    query = (req.query or "").strip()
    result: Dict[str, Any] = {
        "query": query,
        "guardrails": {},
        "retrieval": None,
        "difficulty": None,
        "reason": None,
        "model": None,
        "answer": None,
        "llm_source": None,
        "latency_ms": 0,
        "status": "ok",
        "llm_called": False,
    }

    if not query:
        raise HTTPException(400, "Query must not be empty.")

    # ── Input guardrail 1: prompt injection ──
    injection = guardrails.check_prompt_injection(query)
    result["guardrails"]["prompt_injection"] = injection
    if not injection["passed"]:
        result["status"] = injection["status"]
        result["answer"] = (
            "This request looks like an attempt to override the assistant's "
            "instructions, so it was not sent to the language model. Please ask a "
            "health or knowledge-base question."
        )
        result["latency_ms"] = round((time.time() - t_start) * 1000)
        return result

    # ── RAG retrieval (also feeds the domain check below) ──
    try:
        retrieved, scores, _, _, t_search = retrieve_chunks(query, req.top_k or 3)
    except HTTPException as exc:
        result["status"] = "no_index"
        result["retrieval"] = {"count": 0, "backend": RETRIEVAL_BACKEND, "error": exc.detail}
        result["answer"] = "The knowledge base is not indexed yet. Please ingest a document first."
        result["latency_ms"] = round((time.time() - t_start) * 1000)
        return result

    chunks = []
    for chunk, score in zip(retrieved, scores):
        chunks.append({
            "chunk_id": CHUNK_META.index(chunk),
            "doc_id": chunk.get("doc_id"),
            "text": chunk["text"],
            "score": round(float(score), 4),
        })
    result["retrieval"] = {
        "count": len(chunks),
        "backend": RETRIEVAL_BACKEND,
        "search_ms": round(t_search * 1000, 2),
        "sources": _sources_from_chunks(chunks),
    }

    # ── Input guardrail 2: domain relevance ──
    # Rejects anything that is not a health / medical / knowledge-base question.
    # Only the semantic backend produces scores meaningful enough to use as a
    # topic signal; lexical word-overlap scores are not passed.
    domain_scores = [c["score"] for c in chunks] if RETRIEVAL_BACKEND == "semantic" else None
    domain = guardrails.check_domain(query, retrieval_scores=domain_scores)
    result["guardrails"]["domain"] = domain
    if not domain["passed"]:
        result["status"] = domain["status"]
        result["answer"] = domain["reason"]
        result["latency_ms"] = round((time.time() - t_start) * 1000)
        return result

    # ── Difficulty classification + model routing ──
    diff = classify_difficulty(query)
    result["difficulty"] = diff["difficulty"]
    result["reason"] = diff["reason"]
    result["difficulty_signals"] = diff["signals"]

    model = select_model(diff["difficulty"])
    result["model"] = {
        "tag": model["tag"],
        "name": model["name"],
        "developer": model["developer"],
        "tier": model["tier"],
        "fell_back": model["fell_back"],
        "preferred_tag": model["preferred_tag"],
    }

    # ── Generation ──
    # Only call Ollama when the routed model is actually installed. This keeps
    # the app from triggering Ollama's auto-pull (PRD: never download models).
    # tag_status comes from select_model's single /api/tags call - no extra pings.
    answer = ""
    llm_source = "offline"
    tag_status = model.get("tag_status", "ollama_offline")

    if tag_status == "available":
        gen = generate_with_rag(query=query, context_chunks=chunks, model=model["tag"])
        if gen.get("success"):
            answer = gen["response"]
            llm_source = f"ollama:{model['tag']}"
            result["llm_called"] = True

    if not answer:
        answer = generate_extractive_answer(query, chunks)
        if tag_status == "ollama_offline":
            llm_source = "extractive_fallback (Ollama offline)"
            result["status"] = "ollama_offline"
        elif tag_status == "not_installed":
            llm_source = f"extractive_fallback ({model['tag']} not installed - run: ollama pull {model['tag']})"
            result["status"] = "model_not_installed"
        else:
            llm_source = "extractive_fallback"

    result["model"]["status"] = tag_status

    result["llm_source"] = llm_source

    # ── Output guardrails ──
    grounding = guardrails.check_grounding(answer, chunks)
    safety = guardrails.check_medical_safety(answer, query)
    result["guardrails"]["grounding"] = grounding
    result["guardrails"]["medical_safety"] = {k: v for k, v in safety.items() if k != "annotated_answer"}
    result["answer"] = safety.get("annotated_answer", answer)

    result["latency_ms"] = round((time.time() - t_start) * 1000)
    return result


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
