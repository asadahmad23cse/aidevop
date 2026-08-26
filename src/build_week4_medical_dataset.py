"""Build the Week 4 evaluation set from the actual Week 3 RAG index.

The first 21 tasks are real indexed patient questions with their indexed answers
as references and exact document IDs as relevance labels. Four preserved
out-of-scope tasks test abstention and hallucination behaviour.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{4,}")
EXCLUDED = {
    "about", "after", "again", "answer", "because", "before", "being", "better",
    "could", "doctor", "doing", "during", "further", "health", "hello", "help",
    "however", "might", "please", "question", "should", "suggest", "thanks", "their",
    "there", "these", "thing", "think", "through", "usually", "which", "would", "your",
}


def answer_from_chunk(text: str) -> str:
    if " A: " not in text:
        raise ValueError("Indexed chunk does not contain the expected ' A: ' separator")
    return text.split(" A: ", 1)[1].strip()


def required_facts(answer: str, count: int = 6) -> List[List[str]]:
    """Select stable, distinctive answer terms for deterministic coverage scoring."""
    chosen: List[str] = []
    for token in TOKEN_RE.findall(answer.lower()):
        if token in EXCLUDED or token in chosen:
            continue
        chosen.append(token)
        if len(chosen) == count:
            break
    if len(chosen) < 3:
        raise ValueError("Reference answer has too few substantive terms")
    return [[token] for token in chosen]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "rag_index" / "chunk_metadata.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "evaluation" / "week4_medical_eval.jsonl",
    )
    parser.add_argument("--confirm-write", action="store_true")
    args = parser.parse_args()
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    first_chunks = {int(item["doc_id"]): item for item in metadata if int(item["chunk_idx"]) == 0}
    existing = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_of_scope = [item for item in existing if item.get("should_abstain")]
    if len(out_of_scope) != 4:
        raise ValueError("Expected exactly four preserved out-of-scope tasks")

    tasks: List[Dict[str, Any]] = []
    for doc_id in range(1, 22):
        item = first_chunks[doc_id]
        answer = answer_from_chunk(item["text"])
        tasks.append({
            "id": f"MED-{doc_id:03d}",
            "category": "indexed_patient_qa",
            "question": item["instruction"].strip(),
            "reference_answer": answer,
            "required_facts": required_facts(answer),
            "relevant_doc_ids": [doc_id],
            "should_abstain": False,
        })
    tasks.extend(out_of_scope)
    print(f"Prepared {len(tasks)} tasks: 21 indexed Q/A + {len(out_of_scope)} out-of-scope.")
    if not args.confirm_write:
        print("Check only; pass --confirm-write to replace the evaluation dataset.")
        return 0
    args.output.write_text(
        "".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks),
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
