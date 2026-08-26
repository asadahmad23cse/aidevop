"""Validate Week 4 benchmark records and generate quantitative reports/traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from week4_metrics import aggregate_records, concept_coverage, answer_relevance_proxy


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "week4" / "raw_results.jsonl"
DEFAULT_REVIEW = PROJECT_ROOT / "outputs" / "week4" / "manual_review.csv"
DEFAULT_JSON = PROJECT_ROOT / "outputs" / "week4" / "evaluation_summary.json"
DEFAULT_MARKDOWN = PROJECT_ROOT / "outputs" / "week4" / "WEEK4_RESULTS.md"
REVIEW_FIELDS = (
    "model_id",
    "task_id",
    "correctness_0_to_2",
    "relevance_0_to_2",
    "factual_claims",
    "hallucinated_claims",
    "reviewer_notes",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_review_template(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: Dict[Tuple[str, str], Dict[str, str]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                existing[(row["model_id"], row["task_id"])] = row
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item["model"]["id"], item["task"]["id"])):
            key = (record["model"]["id"], record["task"]["id"])
            writer.writerow(existing.get(key, {"model_id": key[0], "task_id": key[1]}))


def load_reviews(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path.exists():
        return {}
    reviews: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), start=2):
            numeric = ("correctness_0_to_2", "relevance_0_to_2", "factual_claims", "hallucinated_claims")
            if not any((row.get(field) or "").strip() for field in numeric):
                continue
            if not all((row.get(field) or "").strip() for field in numeric):
                raise ValueError(f"Partial manual review at {path}:{row_number}")
            correctness = float(row["correctness_0_to_2"])
            relevance = float(row["relevance_0_to_2"])
            factual = int(row["factual_claims"])
            hallucinated = int(row["hallucinated_claims"])
            if correctness not in (0, 1, 2) or relevance not in (0, 1, 2):
                raise ValueError(f"Review scores must be 0, 1, or 2 at {path}:{row_number}")
            if factual < 0 or not 0 <= hallucinated <= factual:
                raise ValueError(f"Invalid claim counts at {path}:{row_number}")
            reviews[(row["model_id"], row["task_id"])] = {
                "correctness_0_to_2": correctness,
                "relevance_0_to_2": relevance,
                "factual_claims": factual,
                "hallucinated_claims": hallucinated,
                "reviewer_notes": row.get("reviewer_notes", ""),
            }
    return reviews


def validate_matrix(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    models = sorted({record["model"]["id"] for record in records})
    tasks = sorted({record["task"]["id"] for record in records})
    counts = defaultdict(int)
    prompt_hashes = defaultdict(set)
    context_hashes = defaultdict(set)
    for record in records:
        key = (record["model"]["id"], record["task"]["id"])
        counts[key] += 1
        prompt_hashes[record["task"]["id"]].add(record["prompt"]["sha256"])
        context = json.dumps(record["retrieval"]["chunks"], sort_keys=True)
        context_hashes[record["task"]["id"]].add(context)
    missing = [(model, task) for model in models for task in tasks if counts[(model, task)] == 0]
    duplicates = [(*key, count) for key, count in counts.items() if count > 1]
    prompt_mismatches = [task for task, hashes in prompt_hashes.items() if len(hashes) > 1]
    context_mismatches = [task for task, contexts in context_hashes.items() if len(contexts) > 1]
    return {
        "models": models,
        "tasks": tasks,
        "expected_records": len(models) * len(tasks),
        "actual_records": len(records),
        "missing_pairs": missing,
        "duplicate_pairs": duplicates,
        "prompt_mismatch_tasks": prompt_mismatches,
        "retrieval_context_mismatch_tasks": context_mismatches,
        "complete_and_comparable": not (missing or duplicates or prompt_mismatches or context_mismatches),
    }


def fmt(value: Optional[float], digits: int = 3) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def clean_markdown_text(value: str) -> str:
    """Remove model-emitted trailing whitespace before embedding text in Markdown."""
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def rank_models(summaries: Mapping[str, Mapping[str, Any]], path: Sequence[str], lower_is_better: bool = False) -> List[str]:
    def get(model: str) -> float:
        value: Any = summaries[model]
        for key in path:
            value = value[key]
        if value is None:
            return math.inf if lower_is_better else -math.inf
        return float(value)

    return sorted(summaries, key=get, reverse=not lower_is_better)


def create_analysis(summaries: Mapping[str, Mapping[str, Any]], complete: bool) -> List[str]:
    if not complete:
        return ["The model/question matrix is incomplete or conditions differ, so no winner is declared."]
    fact_rank = rank_models(summaries, ("quality", "automated_fact_coverage_proxy"))
    latency_rank = rank_models(summaries, ("performance", "latency_ms_mean"), lower_is_better=True)
    unsupported_rank = rank_models(summaries, ("quality", "automated_unsupported_sentence_rate_proxy"), lower_is_better=True)
    memory_rank = rank_models(summaries, ("performance", "ollama_model_vram_mb_mean"), lower_is_better=True)
    best_fact = fact_rank[0]
    best_latency = latency_rank[0]
    lines = [
        f"Highest automated required-fact coverage: {best_fact} ({fmt(summaries[best_fact]['quality']['automated_fact_coverage_proxy'])}).",
        f"Lowest unsupported-sentence proxy: {unsupported_rank[0]} ({fmt(summaries[unsupported_rank[0]]['quality']['automated_unsupported_sentence_rate_proxy'])}).",
        f"Lowest mean response latency: {best_latency} ({fmt(summaries[best_latency]['performance']['latency_ms_mean'], 1)} ms).",
    ]
    if all(summary["quality"]["manual_adjudication"] is not None for summary in summaries.values()):
        accuracy_rank = rank_models(summaries, ("quality", "manual_adjudication", "correctness_accuracy"))
        relevance_rank = rank_models(summaries, ("quality", "manual_adjudication", "relevance"))
        hallucination_rank = rank_models(
            summaries, ("quality", "manual_adjudication", "hallucination_rate"), lower_is_better=True
        )
        lines.extend([
            f"Highest evidence-adjudicated correctness: {accuracy_rank[0]} "
            f"({fmt(summaries[accuracy_rank[0]]['quality']['manual_adjudication']['correctness_accuracy'])}).",
            f"Highest evidence-adjudicated relevance: {relevance_rank[0]} "
            f"({fmt(summaries[relevance_rank[0]]['quality']['manual_adjudication']['relevance'])}).",
            f"Lowest evidence-adjudicated hallucination rate: {hallucination_rank[0]} "
            f"({fmt(summaries[hallucination_rank[0]]['quality']['manual_adjudication']['hallucination_rate'])}).",
        ])
    if summaries[memory_rank[0]]["performance"]["ollama_model_vram_mb_mean"] is not None:
        lines.append(
            f"Lowest Ollama model allocation on the Intel Arc iGPU: {memory_rank[0]} "
            f"({fmt(summaries[memory_rank[0]]['performance']['ollama_model_vram_mb_mean'], 1)} MB)."
        )
    else:
        lines.append("Model-allocation memory was unavailable on this platform, so no memory winner is declared.")
    if best_fact == best_latency:
        lines.append(f"{best_fact} leads both the automated quality proxy and mean latency in this run; no quality-latency trade-off is observed on those measures.")
    else:
        fact_delta = summaries[best_fact]["quality"]["automated_fact_coverage_proxy"] - summaries[best_latency]["quality"]["automated_fact_coverage_proxy"]
        latency_delta = summaries[best_fact]["performance"]["latency_ms_mean"] - summaries[best_latency]["performance"]["latency_ms_mean"]
        lines.append(
            f"A quality-latency trade-off is observed: {best_fact} gains {fact_delta:.3f} fact-coverage points over {best_latency}, "
            f"while taking {latency_delta:.1f} ms more per response on average."
        )
    if all(summary["quality"]["manual_adjudication"] is None for summary in summaries.values()):
        lines.append("Evidence adjudication is incomplete; correctness, relevance, and hallucination conclusions must not be inferred from automated proxies alone.")
    return lines


def rag_traces(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    for record in records:
        baseline = record.get("baseline_without_rag")
        if not baseline:
            continue
        task = record["task"]
        baseline_answer = baseline["answer"]
        baseline_coverage = concept_coverage(baseline_answer, task["required_facts"])
        rag_coverage = record["quality"]["fact_coverage_proxy"]
        retrieval_hit = record["quality"]["retrieval"].get("hit_at_k")
        if retrieval_hit == 1 and rag_coverage >= 0.60:
            relationship = "Relevant evidence retrieved; the RAG response covers most required facts."
        elif retrieval_hit == 1:
            relationship = "Relevant evidence was retrieved, but the response did not use it completely."
        elif task.get("should_abstain") and record["quality"]["abstained"]:
            relationship = "No labelled answer exists; the model correctly abstained despite forced top-k retrieval."
        elif task.get("should_abstain"):
            relationship = "No labelled answer exists, but the model failed to abstain; forced retrieval likely supplied misleading context."
        else:
            relationship = "Important evidence was missed; response quality should be interpreted as retrieval-limited."
        traces.append({
            "model": record["model"]["id"],
            "task_id": task["id"],
            "question": task["question"],
            "retrieved_context": record["retrieval"]["chunks"],
            "response_with_rag": record["response"],
            "response_without_rag": baseline_answer,
            "rag_fact_coverage_proxy": rag_coverage,
            "baseline_fact_coverage_proxy": baseline_coverage,
            "fact_coverage_delta": rag_coverage - baseline_coverage,
            "relationship_analysis": relationship,
        })
    return traces


def markdown_report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Week 4 Quantitative Evaluation Results",
        "",
        "> Generated only from recorded benchmark artifacts. Automated and rule-based evidence scores are explicitly identified; they are not blind human judgments.",
        "",
        "## Validity",
        "",
        f"- Complete comparable matrix: **{summary['validity']['complete_and_comparable']}**",
        f"- Records: {summary['validity']['actual_records']} / {summary['validity']['expected_records']}",
        f"- Models: {', '.join(summary['validity']['models'])}",
        f"- Tasks: {len(summary['validity']['tasks'])}",
        "",
        "## Model comparison",
        "",
        "| Model | Correctness | Relevance | Hallucination rate | Fact coverage proxy | Retrieval hit@k | Abstention accuracy | Mean latency ms | Mean completion tokens | Ollama RSS MB | Ollama VRAM MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, result in sorted(summary["models"].items()):
        quality = result["quality"]
        performance = result["performance"]
        adjudication = quality["manual_adjudication"] or {}
        lines.append(
            f"| {model} | {fmt(adjudication.get('correctness_accuracy'))} | {fmt(adjudication.get('relevance'))} | "
            f"{fmt(adjudication.get('hallucination_rate'))} | {fmt(quality['automated_fact_coverage_proxy'])} | "
            f"{fmt(quality['retrieval']['hit_at_k'])} | "
            f"{fmt(quality['out_of_scope_abstention_accuracy'])} | {fmt(performance['latency_ms_mean'], 1)} | "
            f"{fmt(performance['completion_tokens_mean'], 1)} | {fmt(performance['ollama_rss_mb_peak_mean'], 1)} | "
            f"{fmt(performance['ollama_model_vram_mb_mean'], 1)} |"
        )
    lines.extend(["", "## Evidence-based analysis", ""])
    lines.extend(f"- {line}" for line in summary["analysis"])
    lines.extend([
        "",
        "## Metric interpretation",
        "",
        "- Correctness, relevance, and hallucination rate use the completed evidence-adjudication CSV; its deterministic rubric is recorded in every row and can be replaced by blind human review.",
        "- Retrieval quality uses labelled relevant documents/files: Precision@k, Recall@k, MRR, Hit@k, and nDCG@k.",
        "- Test-pass rate is N/A because these suites ask QA/code-understanding questions and do not execute model-generated code.",
        "- Latency is wall-clock request time after one warm-up; token counts and model durations come from Ollama.",
        "- CPU, RAM, Ollama RSS, and NVIDIA GPU measures are sampled during each request; unavailable sensors remain N/A.",
        "",
        "## Selected RAG traces",
        "",
    ])
    for trace in summary["rag_traces"]:
        lines.extend([
            f"### {trace['task_id']} — {trace['model']}",
            "",
            f"**Question:** {trace['question']}",
            "",
            f"**Retrieved sources:** {', '.join(str(chunk.get('source') or ('doc:' + str(chunk['doc_id']))) for chunk in trace['retrieved_context'])}",
            "",
            f"**With RAG:** {clean_markdown_text(trace['response_with_rag'])}",
            "",
            f"**Without RAG:** {clean_markdown_text(trace['response_without_rag'])}",
            "",
            f"**Relationship:** {trace['relationship_analysis']} Fact-coverage delta = {trace['fact_coverage_delta']:.3f}.",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--create-review-template", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        print(f"No benchmark records found at {args.input}; run the guarded benchmark first.")
        return 2
    records = load_jsonl(args.input)
    if args.create_review_template:
        write_review_template(records, args.review)
        print(f"Review template written to {args.review}")
    reviews = load_reviews(args.review)
    for record in records:
        record["manual_review"] = reviews.get((record["model"]["id"], record["task"]["id"]))
    validity = validate_matrix(records)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["model"]["id"]].append(record)
    model_summaries = {model: aggregate_records(items) for model, items in grouped.items()}
    summary = {
        "validity": validity,
        "models": model_summaries,
        "analysis": create_analysis(model_summaries, validity["complete_and_comparable"]),
        "rag_traces": rag_traces(records),
        "manual_review_records": len(reviews),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_markdown.write_text(markdown_report(summary), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")
    return 0 if validity["complete_and_comparable"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
