"""Create a reproducible evidence-based adjudication CSV from benchmark records.

This is intentionally not described as human review. It converts the benchmark's
pre-recorded deterministic proxies into the assignment's 0-2 scoring rubric so
every record is scored consistently and can later be replaced by blind human
adjudication using the same CSV schema.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


FIELDS = (
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


def band(value: float) -> int:
    """Map a 0-1 evidence score to 0=poor, 1=partial, 2=strong."""
    if value >= 0.75:
        return 2
    if value >= 0.40:
        return 1
    return 0


def adjudicate(record: Dict[str, Any]) -> Dict[str, Any]:
    quality = record["quality"]
    task = record["task"]
    if task.get("should_abstain", False):
        correctness = 2 if quality["abstained"] else 0
    else:
        correctness = band(float(quality["fact_coverage_proxy"]))
    unsupported = quality["unsupported_claim_proxy"]
    return {
        "model_id": record["model"]["id"],
        "task_id": task["id"],
        "correctness_0_to_2": correctness,
        "relevance_0_to_2": band(float(quality["relevance_proxy"])),
        "factual_claims": int(unsupported["factual_sentence_count"]),
        "hallucinated_claims": int(unsupported["unsupported_sentence_count"]),
        "reviewer_notes": (
            "Rule-based evidence adjudication: correctness uses required-fact coverage "
            "(>=0.75 strong, >=0.40 partial) or abstention accuracy; relevance uses the "
            "same bands; hallucination counts are sentences with <0.35 lexical overlap "
            "to every retrieved chunk. Not a human review."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_jsonl(args.input)
    rows = sorted((adjudicate(record) for record in records), key=lambda row: (row["model_id"], row["task_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} evidence-adjudication rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
