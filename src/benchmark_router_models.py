"""
benchmark_router_models.py - Quantitative evaluation of the three ROUTED models
=============================================================================
Runs the same medical evaluation set through each of the deployed Ollama models
(qwen2.5:0.5b / gemma3:1b / smollm2:1.7b), holding the application, prompt,
questions, knowledge base and retrieved context constant.

    python src/benchmark_router_models.py            # all 25 questions
    python src/benchmark_router_models.py --limit 10 # quick run

Outputs:
    outputs/benchmark/router_models_summary.json
    dashboard/benchmark_data.js   (window.BENCHMARK_DATA, read by the dashboard)

Every number is measured from a real Ollama call. Nothing is invented.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ollama_client import query_ollama, list_installed_models, _tag_matches  # noqa: E402
from week4_metrics import (  # noqa: E402
    concept_coverage, abstention_detected, answer_relevance_proxy,
    unsupported_claim_proxy, percentile,
)

EVAL_FILE = ROOT / "data" / "evaluation" / "week4_medical_eval.jsonl"
INDEX_META = ROOT / "outputs" / "rag_index" / "chunk_metadata.json"
CONFIG = ROOT / "config" / "llm_models.json"
OUT_JSON = ROOT / "outputs" / "benchmark" / "router_models_summary.json"
OUT_JS = ROOT / "dashboard" / "benchmark_data.js"

_STOP = {"a", "an", "and", "are", "does", "for", "how", "in", "is", "of", "or",
         "the", "to", "what", "which", "with"}


def load_config():
    data = json.loads(CONFIG.read_text(encoding="utf-8"))
    r = data["routing"]
    tiers = []
    for tier in ("easy", "medium", "complex"):
        c = r[tier]
        tiers.append({"tier": tier, "tag": os.environ.get(f"LLM_MODEL_{tier.upper()}", c["ollama_tag"]),
                      "name": c["name"], "developer": c["developer"]})
    return tiers, data.get("grounded_system_prompt", "")


def lexical_retrieve(query, chunks, top_k=3):
    q_terms = set(re.findall(r"\b[a-z0-9_]+\b", query.lower())) - _STOP
    scored = []
    for i, ch in enumerate(chunks):
        terms = set(re.findall(r"\b[a-z0-9_]+\b", ch.get("text", "").lower()))
        overlap = len(q_terms & terms)
        cov = overlap / max(len(q_terms), 1)
        scored.append((cov + (0.15 if query.lower() in ch.get("text", "").lower() else 0.0), i))
    scored.sort(key=lambda p: (-p[0], p[1]))
    return [chunks[i] for _, i in scored[:top_k]]


def build_prompt(system_prompt, context_chunks, question):
    ctx = "\n\n".join(f"[C{i+1}]\n{c['text']}" for i, c in enumerate(context_chunks))
    return (f"### Retrieved Context:\n{ctx}\n\n### User Question:\n{question}\n\n### Answer:"), system_prompt


def run_model(model_tag, questions, chunks, system_prompt):
    records = []
    for q in questions:
        ctx = lexical_retrieve(q["question"], chunks, 3)
        retrieved_doc_ids = [c.get("doc_id") for c in ctx]
        prompt, sysp = build_prompt(system_prompt, ctx, q["question"])

        t0 = time.time()
        res = query_ollama(prompt=prompt, system_prompt=sysp, model=model_tag, temperature=0.2)
        latency_ms = (time.time() - t0) * 1000
        answer = res.get("response", "") if res.get("success") else ""
        ok = res.get("success", False)

        ctx_texts = [c["text"] for c in ctx]
        cov = concept_coverage(answer, q.get("required_facts", [])) if answer else 0.0
        rel = answer_relevance_proxy(answer, q["question"], q.get("reference_answer", "")) if answer else 0.0
        unsup = unsupported_claim_proxy(answer, ctx_texts).get("unsupported_sentence_rate", 0.0) if answer else 0.0
        abstained = abstention_detected(answer) if answer else False
        hit = 1.0 if (q.get("relevant_doc_ids") and set(q["relevant_doc_ids"]) & set(retrieved_doc_ids)) else 0.0

        records.append({
            "id": q["id"], "should_abstain": q.get("should_abstain", False),
            "ok": ok, "latency_ms": latency_ms,
            "completion_tokens": res.get("eval_count", 0),
            "prompt_tokens": len(prompt.split()),
            "answer_chars": len(answer),
            "fact_coverage": cov, "relevance": rel, "unsupported_rate": unsup,
            "abstained": abstained, "retrieval_hit": hit,
        })
        status = "ok" if ok else "FAIL"
        print(f"    {q['id']:<8} {latency_ms/1000:6.1f}s  cov={cov:.2f}  {status}", flush=True)
    return records


def aggregate(records):
    answerable = [r for r in records if not r["should_abstain"]]
    oos = [r for r in records if r["should_abstain"]]
    lat = [r["latency_ms"] for r in records if r["ok"]]
    return {
        "questions": len(records),
        "ok_count": sum(1 for r in records if r["ok"]),
        "fact_coverage_mean": round(statistics.fmean(r["fact_coverage"] for r in answerable), 3) if answerable else None,
        "relevance_mean": round(statistics.fmean(r["relevance"] for r in answerable), 3) if answerable else None,
        "unsupported_rate_mean": round(statistics.fmean(r["unsupported_rate"] for r in answerable), 3) if answerable else None,
        "abstention_accuracy": round(statistics.fmean(1.0 if r["abstained"] else 0.0 for r in oos), 3) if oos else None,
        "retrieval_hit_rate": round(statistics.fmean(r["retrieval_hit"] for r in answerable), 3) if answerable else None,
        "latency_ms_mean": round(statistics.fmean(lat), 1) if lat else None,
        "latency_ms_p95": round(percentile(lat, 95), 1) if lat else None,
        "completion_tokens_mean": round(statistics.fmean(r["completion_tokens"] for r in records), 1),
        "total_completion_tokens": int(sum(r["completion_tokens"] for r in records)),
        "answer_chars_mean": round(statistics.fmean(r["answer_chars"] for r in records), 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max questions (0 = all)")
    args = ap.parse_args()

    tiers, system_prompt = load_config()
    installed = list_installed_models()
    if not installed["online"]:
        sys.exit("Ollama is not running. Start it with 'ollama serve'.")
    missing = [t["tag"] for t in tiers if not _tag_matches(t["tag"], installed["models"])]
    if missing:
        sys.exit(f"Missing models: {missing}. Pull them with 'ollama pull <tag>'.")

    questions = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        questions = questions[:args.limit]
    chunks = json.loads(INDEX_META.read_text(encoding="utf-8"))
    print(f"Benchmark: {len(questions)} questions x {len(tiers)} models, {len(chunks)} indexed chunks\n")

    models_out = {}
    t_start = time.time()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "questions": len(questions),
            "chunks_indexed": len(chunks),
            "retrieval_backend": "lexical",
            "system_prompt": system_prompt,
            "elapsed_s": round(time.time() - t_start, 1),
            "complete": len(models_out) == len(tiers),
            "models": models_out,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        OUT_JS.write_text("window.BENCHMARK_DATA = " + json.dumps(payload) + ";\n", encoding="utf-8")
        return payload

    for t in tiers:
        print(f"  {t['name']} ({t['tag']}) ...", flush=True)
        recs = run_model(t["tag"], questions, chunks, system_prompt)
        models_out[t["tier"]] = {
            "tier": t["tier"], "tag": t["tag"], "name": t["name"], "developer": t["developer"],
            "summary": aggregate(recs), "records": recs,
        }
        checkpoint()  # write after every model so a slow/partial run is still usable
        print(flush=True)

    payload = checkpoint()
    print(f"Wrote {OUT_JSON}\nWrote {OUT_JS}\nTotal {payload['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
