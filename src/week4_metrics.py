"""Pure metric functions for the Week 4 LLM/RAG benchmark.

The automated quality measures in this module are deterministic proxies.  The
benchmark report labels them as such.  Human review fields, when supplied, are
reported separately as the definitive correctness, relevance, and hallucination
measures required by the assignment.
"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set


TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?%?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
ABSTENTION_PATTERNS = (
    "does not contain",
    "do not have enough",
    "insufficient information",
    "cannot answer",
    "can't answer",
    "not provided",
    "no information",
    "unable to",
)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "how", "in", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "was", "were", "what", "when", "which", "with", "you", "your",
}


def normalize(text: str) -> str:
    """Lowercase and collapse punctuation/whitespace for phrase matching."""
    return " ".join(TOKEN_RE.findall(text.lower()))


def content_tokens(text: str) -> Set[str]:
    """Return normalized non-stopword tokens."""
    return {token for token in TOKEN_RE.findall(text.lower()) if token not in STOPWORDS}


def percentile(values: Sequence[float], percentile_value: float) -> Optional[float]:
    """Calculate a linearly interpolated percentile without NumPy."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def concept_coverage(answer: str, required_facts: Sequence[Sequence[str]]) -> float:
    """Fraction of required concepts for which at least one accepted phrase occurs."""
    if not required_facts:
        return 1.0
    normalized_answer = normalize(answer)
    matched = 0
    for aliases in required_facts:
        if any(normalize(alias) in normalized_answer for alias in aliases):
            matched += 1
    return matched / len(required_facts)


def abstention_detected(answer: str) -> bool:
    """Detect an explicit statement that the supplied evidence is insufficient."""
    lowered = answer.lower()
    return any(pattern in lowered for pattern in ABSTENTION_PATTERNS)


def answer_relevance_proxy(answer: str, question: str, reference: str) -> float:
    """Estimate topical relevance as the share of substantive sentences on topic.

    A sentence is on topic when at least 20% of its content tokens overlap the
    union of question and reference tokens. Empty answers score zero.
    """
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(answer) if sentence.strip()]
    if not sentences:
        return 0.0
    topic = content_tokens(question) | content_tokens(reference)
    if not topic:
        return 0.0
    relevant = 0
    for sentence in sentences:
        tokens = content_tokens(sentence)
        if tokens and len(tokens & topic) / len(tokens) >= 0.20:
            relevant += 1
    return relevant / len(sentences)


def unsupported_claim_proxy(answer: str, retrieved_contexts: Sequence[str]) -> Dict[str, Any]:
    """Estimate unsupported factual sentences using lexical evidence overlap.

    This is deliberately named a proxy: paraphrases can be under-counted and a
    shared phrase does not prove a claim. Definitive hallucination rate comes
    from manual atomic-claim review.
    """
    sentences = [sentence.strip() for sentence in SENTENCE_RE.split(answer) if len(content_tokens(sentence)) >= 4]
    contexts = [content_tokens(context) for context in retrieved_contexts]
    unsupported: List[str] = []
    for sentence in sentences:
        if abstention_detected(sentence):
            continue
        tokens = content_tokens(sentence)
        best_support = max((len(tokens & context) / len(tokens) for context in contexts), default=0.0)
        if best_support < 0.35:
            unsupported.append(sentence)
    factual_count = sum(1 for sentence in sentences if not abstention_detected(sentence))
    rate = len(unsupported) / factual_count if factual_count else 0.0
    return {
        "unsupported_sentence_count": len(unsupported),
        "factual_sentence_count": factual_count,
        "unsupported_sentence_rate": rate,
        "unsupported_sentences": unsupported,
    }


def retrieval_metrics(retrieved_doc_ids: Sequence[Any], relevant_doc_ids: Sequence[Any], k: int) -> Dict[str, Optional[float]]:
    """Compute Precision@k, Recall@k, reciprocal rank, hit rate, and nDCG@k.

    Queries with no relevant document are out-of-scope abstention tests. Standard
    IR metrics are undefined for them and are returned as ``None``.
    """
    relevant = set(relevant_doc_ids)
    retrieved = list(retrieved_doc_ids[:k])
    if not relevant:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "reciprocal_rank": None,
            "hit_at_k": None,
            "ndcg_at_k": None,
        }

    hits = [1 if doc_id in relevant else 0 for doc_id in retrieved]
    precision = sum(hits) / k
    recall = len(set(retrieved) & relevant) / len(relevant)
    first_rank = next((index + 1 for index, hit in enumerate(hits) if hit), None)
    reciprocal_rank = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits, start=1))
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return {
        "precision_at_k": precision,
        "recall_at_k": recall,
        "reciprocal_rank": reciprocal_rank,
        "hit_at_k": 1.0 if first_rank else 0.0,
        "ndcg_at_k": dcg / idcg if idcg else 0.0,
    }


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    """Mean of non-null numeric values."""
    usable = [float(value) for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def aggregate_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate completed per-question model records into assignment metrics."""
    latencies = [float(record["performance"]["latency_ms"]) for record in records]
    prompt_tokens = [float(record["performance"].get("prompt_tokens", 0)) for record in records]
    completion_tokens = [float(record["performance"].get("completion_tokens", 0)) for record in records]
    answerable = [record for record in records if not record["task"].get("should_abstain", False)]
    out_of_scope = [record for record in records if record["task"].get("should_abstain", False)]
    manual = [record.get("manual_review") for record in records if record.get("manual_review")]

    retrieval_keys = ("precision_at_k", "recall_at_k", "reciprocal_rank", "hit_at_k", "ndcg_at_k")
    retrieval = {
        key: safe_mean(record["quality"]["retrieval"].get(key) for record in answerable)
        for key in retrieval_keys
    }

    manual_metrics: Optional[Dict[str, Any]] = None
    if manual and len(manual) == len(records):
        factual_claims = sum(int(item.get("factual_claims", 0)) for item in manual)
        hallucinated_claims = sum(int(item.get("hallucinated_claims", 0)) for item in manual)
        manual_metrics = {
            "correctness_accuracy": safe_mean(float(item["correctness_0_to_2"]) / 2 for item in manual),
            "relevance": safe_mean(float(item["relevance_0_to_2"]) / 2 for item in manual),
            "hallucination_rate": hallucinated_claims / factual_claims if factual_claims else 0.0,
            "reviewed_records": len(manual),
        }

    return {
        "record_count": len(records),
        "quality": {
            "automated_fact_coverage_proxy": safe_mean(record["quality"]["fact_coverage_proxy"] for record in records),
            "automated_relevance_proxy": safe_mean(record["quality"]["relevance_proxy"] for record in records),
            "automated_unsupported_sentence_rate_proxy": safe_mean(
                record["quality"]["unsupported_claim_proxy"]["unsupported_sentence_rate"] for record in records
            ),
            "out_of_scope_abstention_accuracy": safe_mean(
                1.0 if record["quality"]["abstained"] else 0.0 for record in out_of_scope
            ),
            "retrieval": retrieval,
            "manual_adjudication": manual_metrics,
            "test_pass_rate": None,
            "test_pass_rate_note": "Not applicable: this benchmark contains question-answering and code-understanding tasks, not generated-code tasks.",
        },
        "performance": {
            "latency_ms_mean": safe_mean(latencies),
            "latency_ms_median": statistics.median(latencies) if latencies else None,
            "latency_ms_p95": percentile(latencies, 95),
            "prompt_tokens_mean": safe_mean(prompt_tokens),
            "completion_tokens_mean": safe_mean(completion_tokens),
            "total_tokens": int(sum(prompt_tokens) + sum(completion_tokens)),
            "system_cpu_percent_mean": safe_mean(record["performance"].get("system_cpu_percent_mean") for record in records),
            "system_memory_used_mb_peak_mean": safe_mean(
                record["performance"].get("system_memory_used_mb_peak") for record in records
            ),
            "ollama_rss_mb_peak_mean": safe_mean(record["performance"].get("ollama_rss_mb_peak") for record in records),
            "gpu_utilization_percent_peak_mean": safe_mean(
                record["performance"].get("gpu_utilization_percent_peak") for record in records
            ),
            "gpu_memory_used_mb_peak_mean": safe_mean(
                record["performance"].get("gpu_memory_used_mb_peak") for record in records
            ),
            "ollama_model_vram_mb_mean": safe_mean(
                record["performance"].get("ollama_model_vram_mb") for record in records
            ),
        },
    }
