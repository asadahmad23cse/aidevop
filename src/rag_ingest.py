"""
rag_ingest.py — Document Ingestion Script for RAG Pipeline
============================================================
Injects medical_knowledge_base.jsonl through the complete
RAG pipeline: Read → Chunk → Embed → Index (FAISS)

Usage:
    python src/rag_ingest.py
    python src/rag_ingest.py --file data/raw/medical_knowledge_base.jsonl
    python src/rag_ingest.py --file data/raw/medical_knowledge_base.jsonl --chunk-size 256 --overlap 50
"""

import json
import os
import time
import argparse
import numpy as np

# ── Step 1: Check dependencies ──────────────────────────────
print("\n" + "="*60)
print("  RAG PIPELINE — Document Ingestion")
print("="*60)

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    DEPS_OK = True
except ImportError:
    DEPS_OK = False
    print("\n❌ Missing dependencies. Run first:")
    print("   pip install sentence-transformers faiss-cpu numpy")
    exit(1)


# ── Argument Parser ──────────────────────────────────────────
parser = argparse.ArgumentParser(description="RAG Document Ingestion")
parser.add_argument("--file",       default="data/raw/medical_knowledge_base.jsonl",
                    help="Path to JSONL file to ingest")
parser.add_argument("--chunk-size", type=int, default=256,
                    help="Tokens per chunk (approx words x 1.3)")
parser.add_argument("--overlap",    type=int, default=50,
                    help="Overlap tokens between chunks")
parser.add_argument("--model",      default="all-MiniLM-L6-v2",
                    help="Sentence transformer model name")
parser.add_argument("--out-dir",    default="outputs/rag_index",
                    help="Directory to save FAISS index + metadata")
args = parser.parse_args()


# ══════════════════════════════════════════════════════════════
# STEP 1 — DOCUMENT INGESTION
# ══════════════════════════════════════════════════════════════
print("\n[STEP 1] Document Ingestion")
print(f"   File : {args.file}")

if not os.path.exists(args.file):
    print(f"   ERROR: File not found: {args.file}")
    exit(1)

documents = []
with open(args.file, "r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            instruction = record.get("instruction", "")
            output      = record.get("output", "")
            text = f"Q: {instruction}\nA: {output}"
            documents.append({
                "id":   line_num,
                "text": text,
                "meta": {"instruction": instruction, "source": args.file}
            })
        except json.JSONDecodeError as e:
            print(f"   WARNING: Skipping line {line_num}: {e}")

print(f"   Loaded {len(documents)} documents")
for i, doc in enumerate(documents[:3]):
    preview = doc['meta']['instruction'][:70]
    print(f"      [{i+1}] {preview}...")
if len(documents) > 3:
    print(f"      ... and {len(documents)-3} more")


# ══════════════════════════════════════════════════════════════
# STEP 2 — TEXT CHUNKING
# ══════════════════════════════════════════════════════════════
print(f"\n[STEP 2] Text Chunking")
print(f"   Chunk size : ~{args.chunk_size} tokens  |  Overlap : {args.overlap} tokens")

def chunk_text(text, chunk_words=200, overlap_words=38):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_words - overlap_words
    return chunks

chunk_words   = int(args.chunk_size * 0.75)
overlap_words = int(args.overlap * 0.75)

all_chunks = []
chunk_meta = []

for doc in documents:
    chunks = chunk_text(doc["text"], chunk_words, overlap_words)
    for j, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        chunk_meta.append({
            "doc_id":      doc["id"],
            "chunk_idx":   j,
            "text":        chunk,
            "instruction": doc["meta"]["instruction"]
        })

print(f"   Created {len(all_chunks)} chunks from {len(documents)} documents")
print(f"   Sample chunk #1: \"{all_chunks[0][:100]}...\"")


# ══════════════════════════════════════════════════════════════
# STEP 3 — EMBEDDING GENERATION
# ══════════════════════════════════════════════════════════════
print(f"\n[STEP 3] Embedding Generation")
print(f"   Model : sentence-transformers/{args.model}")
print(f"   Loading model...", end=" ", flush=True)

t0 = time.time()
model = SentenceTransformer(args.model)
print(f"done ({time.time()-t0:.1f}s)")

print(f"   Encoding {len(all_chunks)} chunks...", end=" ", flush=True)
t0 = time.time()
embeddings = model.encode(
    all_chunks,
    batch_size=32,
    show_progress_bar=False,
    convert_to_numpy=True,
    normalize_embeddings=True
)
elapsed = time.time() - t0
print(f"done ({elapsed:.1f}s)")

dim = embeddings.shape[1]
print(f"   Generated {len(embeddings)} embeddings — shape ({len(embeddings)}, {dim})")
print(f"   Sample vector (first 8 dims): {embeddings[0][:8].round(4).tolist()}")


# ══════════════════════════════════════════════════════════════
# STEP 4 — VECTOR STORE INDEXING (FAISS)
# ══════════════════════════════════════════════════════════════
print(f"\n[STEP 4] Vector Store Indexing (FAISS)")
print(f"   Index type : IndexFlatIP (cosine similarity via normalized vectors)")

index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype(np.float32))
print(f"   FAISS index built — {index.ntotal} vectors stored")

os.makedirs(args.out_dir, exist_ok=True)
index_path = os.path.join(args.out_dir, "faiss.index")
meta_path  = os.path.join(args.out_dir, "chunk_metadata.json")

faiss.write_index(index, index_path)
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(chunk_meta, f, indent=2, ensure_ascii=False)

index_size_kb = os.path.getsize(index_path) / 1024
print(f"   Saved: {index_path}  ({index_size_kb:.1f} KB)")
print(f"   Saved: {meta_path}")


# ══════════════════════════════════════════════════════════════
# QUICK TEST — Retrieval Check
# ══════════════════════════════════════════════════════════════
print(f"\n[TEST] Retrieval Check")
test_query = "What are the symptoms of a heart attack?"
print(f"   Query: \"{test_query}\"")

q_vec = model.encode([test_query], normalize_embeddings=True).astype(np.float32)
scores, indices = index.search(q_vec, k=3)

print(f"   Top-3 results:")
for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
    snippet = chunk_meta[idx]["text"][:90].replace("\n", " ")
    print(f"      [{rank}] score={score:.4f}  \"{snippet}...\"")


# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("  INGESTION COMPLETE")
print("="*60)
print(f"  Documents : {len(documents)}")
print(f"  Chunks    : {len(all_chunks)}")
print(f"  Dimensions: {dim}")
print(f"  Vectors   : {index.ntotal}")
print(f"  Index     : {index_path}")
print(f"  Metadata  : {meta_path}")
print("="*60 + "\n")
