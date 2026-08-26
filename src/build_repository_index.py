"""Build an isolated FAISS index for Week 4 repository-understanding probes.

Default mode validates the source selection without loading an embedding model or
writing files. Use ``--build --confirm-build`` to create the index explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "week4" / "repository_index"
INCLUDED_FILES = (
    "README.md",
    "requirements.txt",
    "docker-compose.yml",
    "Dockerfile.api",
    "Dockerfile.dashboard",
    "src/basic_llm_app.py",
    "src/evaluate.py",
    "src/generate_dataset.py",
    "src/inference.py",
    "src/ollama_client.py",
    "src/preprocess.py",
    "src/rag_demo.py",
    "src/rag_ingest.py",
    "src/rag_server.py",
    "src/train.py",
    "dashboard/app.js",
    "dashboard/index.html",
    "dashboard/styles.css",
    "data/raw/medical_knowledge_base.jsonl",
)


def chunk_file(
    path: Path,
    source: str,
    lines_per_chunk: int,
    overlap_lines: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    chunks: List[Dict[str, Any]] = []
    step = lines_per_chunk - overlap_lines
    for start in range(0, len(lines), step):
        selected = lines[start : start + lines_per_chunk]
        if not selected:
            break
        line_start = start + 1
        line_end = start + len(selected)
        header = f"SOURCE: {source}\nLINES: {line_start}-{line_end}\n"
        body = "\n".join(selected)
        body_budget = max_chars - len(header)
        # Line-count chunking alone is unsafe for JSONL and minified assets,
        # where a single physical line may contain many thousands of chars.
        # Hard-cap every embedded unit so top-k retrieval cannot unexpectedly
        # exhaust the LLM context window or GPU KV-cache allocation.
        for segment_index, offset in enumerate(range(0, len(body) or 1, body_budget)):
            segment = body[offset : offset + body_budget]
            chunks.append({
                "source": source,
                "line_start": line_start,
                "line_end": line_end,
                "segment_idx": segment_index,
                "text": header + segment,
            })
        if start + lines_per_chunk >= len(lines):
            break
    return chunks


def collect_chunks(lines_per_chunk: int, overlap_lines: int, max_chars: int) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    for doc_id, source in enumerate(INCLUDED_FILES, start=1):
        path = PROJECT_ROOT / source
        if not path.is_file():
            raise FileNotFoundError(f"Required repository source is missing: {source}")
        for chunk_index, chunk in enumerate(
            chunk_file(path, source, lines_per_chunk, overlap_lines, max_chars)
        ):
            chunk.update({"doc_id": doc_id, "chunk_idx": chunk_index})
            chunks.append(chunk)
    return chunks


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--lines-per-chunk", type=int, default=80)
    parser.add_argument("--overlap-lines", type=int, default=20)
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--confirm-build", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.lines_per_chunk <= 0 or args.overlap_lines < 0 or args.overlap_lines >= args.lines_per_chunk:
        raise SystemExit("Invalid chunk settings: require 0 <= overlap < lines-per-chunk")
    if args.max_chars < 500:
        raise SystemExit("Invalid chunk settings: --max-chars must be at least 500")
    chunks = collect_chunks(args.lines_per_chunk, args.overlap_lines, args.max_chars)
    if any(len(chunk["text"]) > args.max_chars for chunk in chunks):
        raise RuntimeError("Chunk-size invariant failed")
    print(f"Validated {len(INCLUDED_FILES)} source files and prepared {len(chunks)} in-memory chunks.")
    if not args.build:
        print("Check only; no embedding model was loaded and no index was written.")
        return 0
    if not args.confirm_build:
        print("ERROR: --build also requires --confirm-build.")
        return 2

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.embedding_model)
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    args.output.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.output / "faiss.index"))
    (args.output / "chunk_metadata.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    manifest = {
        "embedding_model": args.embedding_model,
        "files": [
            {"source": source, "sha256": file_sha256(PROJECT_ROOT / source)}
            for source in INCLUDED_FILES
        ],
        "chunks": len(chunks),
        "dimensions": int(embeddings.shape[1]),
        "lines_per_chunk": args.lines_per_chunk,
        "overlap_lines": args.overlap_lines,
        "max_chars": args.max_chars,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote isolated repository index to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
