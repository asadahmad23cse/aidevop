"""
difficulty_classifier.py - Query Difficulty Classification (PRD section 6)
========================================================================
A simple, deterministic, explainable classifier used by the Model Router to
choose an Ollama model. It is intentionally NOT an LLM call - it must stay fast
and auditable. It is isolated in this module so it can later be swapped for a
stronger classifier without touching the router.

    from difficulty_classifier import classify
    classify("What is hypertension?")
    -> {"difficulty": "easy", "reason": "...", "signals": [...]}

This does not claim to be medically intelligent - it only estimates how much
reasoning the phrasing of the question implies.
"""

from __future__ import annotations

import re
from typing import Dict, List

__all__ = ["classify", "DIFFICULTIES"]

DIFFICULTIES = ("easy", "medium", "complex")

# Phrasing that implies explanation / comparison (medium) or multi-step
# synthesis (complex).
_COMPARISON_TERMS = (
    "compare", "comparison", "versus", " vs ", "vs.", "difference between",
    "trade-off", "tradeoff", "pros and cons", "advantages and disadvantages",
    "better", "best", "which is",
)
_SYNTHESIS_TERMS = (
    "why", "how does", "explain how", "evaluate", "assess", "analyse", "analyze",
    "implications", "step by step", "step-by-step", "multi-step", "reason through",
    "what are the causes", "relationship between", "interaction between",
    "summarize the differences", "recommend", "recommendation", "approach for",
)
_EXPLANATION_TERMS = (
    "explain", "describe", "how is", "how are", "what causes", "what happens",
    "walk me through", "tell me about", "overview of",
)
_SIMPLE_OPENERS = (
    "what is", "what are", "who is", "define", "definition of", "when is",
    "where is", "is it", "does ", "do ", "can ", "list the symptoms",
    "symptoms of", "what does the document say", "what dose",
)
_ENUMERATION_TERMS = ("list all", "and also", "as well as", "in addition to")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def classify(query: str) -> Dict[str, object]:
    """Return {'difficulty', 'reason', 'signals'} for a natural-language query."""
    q = _clean(query)
    if not q:
        return {"difficulty": "easy", "reason": "Empty query defaults to the smallest model.", "signals": []}

    words = re.findall(r"[a-zA-Z0-9']+", q)
    word_count = len(words)
    question_marks = q.count("?")
    clause_count = len([c for c in re.split(r",|;| and | but | while | whereas ", q) if c.strip()])

    signals: List[str] = []
    score = 0

    def has(terms) -> bool:
        return any(t in q for t in terms)

    if has(_COMPARISON_TERMS):
        signals.append("comparison language")
        score += 2
    if has(_SYNTHESIS_TERMS):
        signals.append("multi-step / synthesis language")
        score += 2
    if has(_EXPLANATION_TERMS):
        signals.append("explanation requested")
        score += 1
    if has(_ENUMERATION_TERMS) or question_marks >= 2:
        signals.append("multiple sub-questions")
        score += 1
    if clause_count >= 3:
        signals.append(f"{clause_count} clauses")
        score += 1
    if word_count >= 32:
        signals.append(f"long query ({word_count} words)")
        score += 2
    elif word_count >= 16:
        signals.append(f"medium-length query ({word_count} words)")
        score += 1

    is_simple_opener = q.startswith(_SIMPLE_OPENERS)
    if is_simple_opener and word_count <= 12 and score == 0:
        signals.append("short factual question")

    if score >= 4:
        difficulty = "complex"
        reason = "Phrasing implies multi-step reasoning or synthesis across several pieces of information."
    elif score >= 2:
        difficulty = "medium"
        reason = "Question needs explanation or comparison of a few pieces of information."
    elif score == 1:
        difficulty = "medium" if not is_simple_opener else "easy"
        reason = (
            "Short factual question - a small model is sufficient."
            if difficulty == "easy"
            else "Some explanation is expected but the scope is limited."
        )
    else:
        difficulty = "easy"
        reason = "Short, straightforward factual question - a small model is sufficient."

    if not signals:
        signals.append("no complexity signals detected")

    return {"difficulty": difficulty, "reason": reason, "signals": signals}
