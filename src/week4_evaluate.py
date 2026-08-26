"""Guarded, reproducible Week 4 multi-model RAG benchmark.

The default action is preflight only. A full benchmark requires both ``--run``
and ``--confirm-full-run`` so missing models, a stale index, or an accidental
invocation cannot silently create misleading results.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests

from week4_metrics import (
    abstention_detected,
    answer_relevance_proxy,
    concept_coverage,
    retrieval_metrics,
    unsupported_claim_proxy,
)

try:
    import psutil
except ImportError:  # Resource collection remains explicit rather than silently faked.
    psutil = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "week4_models.json"
DEFAULT_DATASET = PROJECT_ROOT / "data" / "evaluation" / "week4_medical_eval.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "week4" / "raw_results.jsonl"
SYSTEM_PROMPT = (
    "You are a careful medical question-answering assistant in an educational RAG experiment. "
    "Use only the retrieved context. If the context does not contain enough information, say "
    "that the verified knowledge base does not contain the answer. Do not invent doses, diagnoses, "
    "or treatment recommendations. Give a concise answer and cite supporting chunks as [C1], [C2], etc."
)
WARMUP_PROMPT = "Reply with exactly: READY"


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_model_name(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


def ollama_tags(host: str, timeout: float = 3.0) -> Tuple[Optional[List[str]], Optional[str]]:
    try:
        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        return [item.get("name", "") for item in response.json().get("models", [])], None
    except Exception as exc:
        return None, str(exc)


def validate_dataset(
    rows: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    size_range: Sequence[int] = (20, 30),
) -> List[str]:
    errors: List[str] = []
    minimum, maximum = int(size_range[0]), int(size_range[1])
    if not minimum <= len(rows) <= maximum:
        errors.append(f"Evaluation dataset must contain {minimum}-{maximum} questions; found {len(rows)}")
    ids = [str(row.get("id", "")) for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("Evaluation question IDs are not unique")
    required = {"id", "category", "question", "reference_answer", "required_facts", "should_abstain"}
    for row in rows:
        missing = sorted(required - set(row))
        if missing:
            errors.append(f"{row.get('id', '<unknown>')} missing fields: {', '.join(missing)}")
    available_doc_ids = {int(item["doc_id"]) for item in metadata if "doc_id" in item}
    available_sources = {str(item["source"]).replace("\\", "/") for item in metadata if "source" in item}
    for row in rows:
        if "relevant_doc_ids" in row:
            missing_docs = set(int(value) for value in row.get("relevant_doc_ids", [])) - available_doc_ids
            if missing_docs:
                errors.append(f"{row['id']} references absent knowledge-base document IDs: {sorted(missing_docs)}")
        elif "relevant_sources" in row:
            missing_sources = {str(value).replace("\\", "/") for value in row.get("relevant_sources", [])} - available_sources
            if missing_sources:
                errors.append(f"{row['id']} references absent repository sources: {sorted(missing_sources)}")
        else:
            errors.append(f"{row['id']} needs relevant_doc_ids or relevant_sources")
    return errors


def preflight(config_path: Path, dataset_path: Path, host: str) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    if not config_path.exists():
        add("configuration", False, f"Missing {config_path}")
        return {"ready": False, "checks": checks}
    config = load_json(config_path)
    models = config.get("models", [])
    add("three-model configuration", len(models) >= 3, f"configured={len(models)}")
    unique_models = {item.get("ollama_name") for item in models}
    add("unique model names", len(unique_models) == len(models), ", ".join(sorted(str(value) for value in unique_models)))

    index_path = PROJECT_ROOT / config["retrieval"]["index_path"]
    metadata_path = PROJECT_ROOT / config["retrieval"]["metadata_path"]
    add("FAISS index", index_path.is_file(), str(index_path))
    add("retrieval metadata", metadata_path.is_file(), str(metadata_path))
    add("evaluation dataset", dataset_path.is_file(), str(dataset_path))

    if index_path.is_file() and metadata_path.is_file() and importlib.util.find_spec("faiss") is not None:
        try:
            import faiss

            index = faiss.read_index(str(index_path))
            metadata_count = len(load_json(metadata_path))
            add(
                "index/metadata integrity",
                index.ntotal == metadata_count and index.d > 0,
                f"vectors={index.ntotal}, metadata={metadata_count}, dimensions={index.d}",
            )
        except Exception as exc:
            add("index/metadata integrity", False, str(exc))

    if metadata_path.is_file() and dataset_path.is_file():
        metadata = load_json(metadata_path)
        rows = load_jsonl(dataset_path)
        dataset_errors = validate_dataset(rows, metadata, config.get("dataset_size_range", (20, 30)))
        add("dataset schema and labels", not dataset_errors, "; ".join(dataset_errors) if dataset_errors else f"{len(rows)} valid tasks")

    dependencies = {
        "requests": True,
        "psutil": psutil is not None,
    }
    dependencies["faiss"] = importlib.util.find_spec("faiss") is not None
    dependencies["sentence_transformers"] = importlib.util.find_spec("sentence_transformers") is not None
    add("Python dependencies", all(dependencies.values()), json.dumps(dependencies, sort_keys=True))

    installed, error = ollama_tags(host)
    add("Ollama service", installed is not None, f"host={host}" if installed is not None else error or "unreachable")
    if installed is not None:
        installed_canonical = {canonical_model_name(name) for name in installed}
        missing = [item["ollama_name"] for item in models if canonical_model_name(item["ollama_name"]) not in installed_canonical]
        add("configured models installed", not missing, f"missing={missing}" if missing else f"available={installed}")

    return {"ready": all(item["passed"] for item in checks), "checks": checks}


def build_prompt(question: str, chunks: Sequence[Mapping[str, Any]]) -> str:
    context = "\n\n".join(
        f"[C{index}] source={chunk.get('source', 'doc:' + str(chunk['doc_id']))} similarity={chunk['score']:.4f}\n{chunk['text']}"
        for index, chunk in enumerate(chunks, start=1)
    )
    return f"Retrieved context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


def query_ollama(
    host: str,
    model: str,
    prompt: str,
    generation: Mapping[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
    timeout: int = 300,
    keep_alive: Any = "10m",
) -> Dict[str, Any]:
    options = {
        "temperature": generation["temperature"],
        "seed": generation["seed"],
        "top_p": generation["top_p"],
        "num_predict": generation["num_predict"],
        "num_ctx": generation["context_window"],
    }
    start = time.perf_counter()
    response = requests.post(
        f"{host.rstrip('/')}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "system": system_prompt,
            "stream": False,
            "keep_alive": keep_alive,
            "options": options,
        },
        timeout=timeout,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    payload = response.json()
    return {
        "answer": payload.get("response", "").strip(),
        "latency_ms": latency_ms,
        "prompt_tokens": int(payload.get("prompt_eval_count", 0)),
        "completion_tokens": int(payload.get("eval_count", 0)),
        "load_duration_ms": float(payload.get("load_duration", 0)) / 1_000_000,
        "prompt_eval_duration_ms": float(payload.get("prompt_eval_duration", 0)) / 1_000_000,
        "eval_duration_ms": float(payload.get("eval_duration", 0)) / 1_000_000,
    }


def ollama_runtime_snapshot(host: str, model: str) -> Dict[str, Optional[float]]:
    """Read Ollama's model allocation, including VRAM on non-NVIDIA devices."""
    try:
        response = requests.get(f"{host.rstrip('/')}/api/ps", timeout=5)
        response.raise_for_status()
        wanted = canonical_model_name(model)
        for item in response.json().get("models", []):
            if canonical_model_name(item.get("name", "")) == wanted:
                return {
                    "ollama_model_size_mb": float(item.get("size", 0)) / (1024 * 1024),
                    "ollama_model_vram_mb": float(item.get("size_vram", 0)) / (1024 * 1024),
                    "ollama_context_length": float(item.get("context_length", 0)),
                }
    except Exception:
        pass
    return {
        "ollama_model_size_mb": None,
        "ollama_model_vram_mb": None,
        "ollama_context_length": None,
    }


def unload_ollama_model(host: str, model: str) -> None:
    """Explicitly unload a model so the next model receives comparable resources."""
    try:
        requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": 0},
            timeout=30,
        ).raise_for_status()
    except Exception as exc:
        print(f"WARNING: failed to unload {model}: {exc}", file=sys.stderr)


@dataclass
class ResourceMonitor:
    """Sample system, Ollama process, and NVIDIA GPU utilization during a request."""

    interval_seconds: float = 0.20
    samples: List[Dict[str, Optional[float]]] = field(default_factory=list)
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    def __enter__(self) -> "ResourceMonitor":
        if psutil is not None:
            psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if not self.samples:
            self.samples.append(self._sample())

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            self.samples.append(self._sample())
            self._stop.wait(self.interval_seconds)

    def _sample(self) -> Dict[str, Optional[float]]:
        sample: Dict[str, Optional[float]] = {
            "system_cpu_percent": None,
            "system_memory_used_mb": None,
            "ollama_rss_mb": None,
            "gpu_utilization_percent": None,
            "gpu_memory_used_mb": None,
        }
        if psutil is not None:
            sample["system_cpu_percent"] = float(psutil.cpu_percent(interval=None))
            memory = psutil.virtual_memory()
            sample["system_memory_used_mb"] = float(memory.used / (1024 * 1024))
            rss = 0
            for process in psutil.process_iter(["name", "memory_info"]):
                try:
                    if "ollama" in (process.info["name"] or "").lower():
                        rss += process.info["memory_info"].rss
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    continue
            sample["ollama_rss_mb"] = float(rss / (1024 * 1024))
        if shutil.which("nvidia-smi"):
            try:
                command = [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ]
                output = subprocess.run(command, capture_output=True, text=True, timeout=2, check=True).stdout
                rows = [row.split(",") for row in output.strip().splitlines() if row.strip()]
                sample["gpu_utilization_percent"] = max(float(row[0].strip()) for row in rows)
                sample["gpu_memory_used_mb"] = sum(float(row[1].strip()) for row in rows)
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        return sample

    def summary(self) -> Dict[str, Optional[float]]:
        def values(key: str) -> List[float]:
            return [float(item[key]) for item in self.samples if item.get(key) is not None]

        cpu = values("system_cpu_percent")
        return {
            "system_cpu_percent_mean": sum(cpu) / len(cpu) if cpu else None,
            "system_memory_used_mb_peak": max(values("system_memory_used_mb"), default=None),
            "ollama_rss_mb_peak": max(values("ollama_rss_mb"), default=None),
            "gpu_utilization_percent_peak": max(values("gpu_utilization_percent"), default=None),
            "gpu_memory_used_mb_peak": max(values("gpu_memory_used_mb"), default=None),
            "resource_sample_count": len(self.samples),
        }


class Retriever:
    def __init__(self, config: Mapping[str, Any]) -> None:
        import faiss
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        retrieval = config["retrieval"]
        self.index = faiss.read_index(str(PROJECT_ROOT / retrieval["index_path"]))
        self.metadata = load_json(PROJECT_ROOT / retrieval["metadata_path"])
        self.embedder = SentenceTransformer(retrieval["embedding_model"])

    def search(self, question: str, top_k: int) -> Tuple[List[Dict[str, Any]], float]:
        start = time.perf_counter()
        vector = self.embedder.encode([question], normalize_embeddings=True, convert_to_numpy=True).astype(self.np.float32)
        scores, indices = self.index.search(vector, k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        chunks: List[Dict[str, Any]] = []
        for score, index in zip(scores[0], indices[0]):
            if 0 <= int(index) < len(self.metadata):
                item = self.metadata[int(index)]
                chunks.append({
                    "chunk_id": int(index),
                    "doc_id": int(item["doc_id"]),
                    "score": float(score),
                    "text": item["text"],
                    "source": item.get("source"),
                })
        return chunks, latency_ms


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def completed_keys(path: Path, experiment_id: str) -> set[Tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for record in load_jsonl(path):
        if record.get("experiment", {}).get("id") == experiment_id:
            keys.add((record["model"]["id"], record["task"]["id"]))
    return keys


def benchmark(args: argparse.Namespace, config: Mapping[str, Any], tasks: List[Dict[str, Any]]) -> None:
    experiment_id = hashlib.sha256(
        (sha256_file(args.config) + sha256_file(args.dataset) + json.dumps(config, sort_keys=True)).encode("utf-8")
    ).hexdigest()[:16]
    random.Random(config["generation"]["seed"]).shuffle(tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    retriever = Retriever(config)
    completed = completed_keys(args.output, experiment_id) if args.resume else set()
    trace_ids = set(config.get("rag_analysis_question_ids", []))
    top_k = int(config["retrieval"]["top_k"])

    retrieval_cache: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}
    for task in tasks:
        retrieval_cache[task["id"]] = retriever.search(task["question"], top_k)

    for model in config["models"]:
        print(f"Warming model {model['ollama_name']}...")
        warmup = query_ollama(args.host, model["ollama_name"], WARMUP_PROMPT, config["generation"], system_prompt="Follow the instruction exactly.")
        print(f"  warmup/load latency={warmup['latency_ms']:.1f} ms")
        for position, task in enumerate(tasks, start=1):
            key = (model["id"], task["id"])
            if key in completed:
                print(f"[{model['id']} {position}/{len(tasks)}] {task['id']} already complete; skipping")
                continue
            chunks, retrieval_latency_ms = retrieval_cache[task["id"]]
            prompt = build_prompt(task["question"], chunks)
            system_prompt = config.get("system_prompt", SYSTEM_PROMPT)
            with ResourceMonitor() as resources:
                generation_result = query_ollama(
                    args.host, model["ollama_name"], prompt, config["generation"], system_prompt=system_prompt
                )
            allocation = ollama_runtime_snapshot(args.host, model["ollama_name"])
            answer = generation_result.pop("answer")
            retrieval_uses_sources = "relevant_sources" in task
            retrieved_doc_ids = [chunk.get("source") if retrieval_uses_sources else chunk["doc_id"] for chunk in chunks]
            relevant_doc_ids = task.get("relevant_sources", task.get("relevant_doc_ids", []))
            contexts = [chunk["text"] for chunk in chunks]

            baseline = None
            if task["id"] in trace_ids:
                baseline_prompt = f"Question: {task['question']}\n\nAnswer:"
                with ResourceMonitor() as baseline_resources:
                    baseline_generation = query_ollama(
                        args.host,
                        model["ollama_name"],
                        baseline_prompt,
                        config["generation"],
                        system_prompt=config.get("baseline_system_prompt", "Answer the question accurately and concisely."),
                    )
                baseline = {"answer": baseline_generation.pop("answer"), **baseline_generation, **baseline_resources.summary()}

            record = {
                "schema_version": 1,
                "experiment": {
                    "id": experiment_id,
                    "dataset_sha256": sha256_file(args.dataset),
                    "config_sha256": sha256_file(args.config),
                    "knowledge_base_metadata_sha256": sha256_file(PROJECT_ROOT / config["retrieval"]["metadata_path"]),
                    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "conditions": {
                        "same_prompt_template": True,
                        "same_question_order": True,
                        "same_retrieved_context_per_question": True,
                        "same_generation_options": dict(config["generation"]),
                        "top_k": top_k,
                    },
                },
                "model": dict(model),
                "task": task,
                "prompt": {
                    "system": system_prompt,
                    "augmented": prompt,
                    "sha256": hashlib.sha256((system_prompt + "\n" + prompt).encode()).hexdigest(),
                },
                "retrieval": {"chunks": chunks, "latency_ms": retrieval_latency_ms},
                "response": answer,
                "baseline_without_rag": baseline,
                "quality": {
                    "fact_coverage_proxy": concept_coverage(answer, task["required_facts"]),
                    "relevance_proxy": answer_relevance_proxy(answer, task["question"], task["reference_answer"]),
                    "abstained": abstention_detected(answer),
                    "unsupported_claim_proxy": unsupported_claim_proxy(answer, contexts),
                    "retrieval": retrieval_metrics(retrieved_doc_ids, relevant_doc_ids, top_k),
                },
                "performance": {
                    **generation_result,
                    **resources.summary(),
                    **allocation,
                    "retrieval_latency_ms": retrieval_latency_ms,
                },
                "manual_review": None,
            }
            append_jsonl(args.output, record)
            print(f"[{model['id']} {position}/{len(tasks)}] {task['id']} saved ({record['performance']['latency_ms']:.1f} ms)")
        unload_ollama_model(args.host, model["ollama_name"])
        print(f"Unloaded {model['ollama_name']} before the next model.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--run", action="store_true", help="Run generation after successful preflight")
    parser.add_argument("--confirm-full-run", action="store_true", help="Required acknowledgement for model execution")
    parser.add_argument("--limit", type=int, default=None, help="Limit questions per model for an explicit smoke run")
    parser.add_argument("--resume", action="store_true", help="Skip completed model/question pairs for this experiment")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = preflight(args.config, args.dataset, args.host)
    print("Week 4 benchmark preflight")
    for check in report["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        print(f"[{marker}] {check['name']}: {check['detail']}")
    if not args.run:
        print("\nPreflight only; no models were executed. Use --run --confirm-full-run after every check passes.")
        return 0 if report["ready"] else 2
    if not args.confirm_full_run:
        print("ERROR: --run also requires --confirm-full-run.", file=sys.stderr)
        return 2
    if not report["ready"]:
        print("ERROR: preflight failed; benchmark was not started.", file=sys.stderr)
        return 2
    config = load_json(args.config)
    tasks = load_jsonl(args.dataset)
    benchmark(args, config, tasks)
    print(f"Benchmark records saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
