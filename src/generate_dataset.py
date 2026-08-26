"""Dataset generation and curation for medical QA fine-tuning."""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

LOGGER = logging.getLogger(__name__)

DEFAULT_TOPICS = [
    "diabetes",
    "hypertension",
    "asthma",
    "depression",
    "arthritis",
    "fever",
    "migraine",
    "thyroid",
    "anemia",
    "anxiety",
]


@dataclass
class SyntheticConfig:
    """Configuration for synthetic example generation."""

    model: str
    api_key: Optional[str]
    api_base: str
    temperature: float
    max_tokens: int
    retries: int


def setup_logging(log_level: str) -> None:
    """Configure root logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate curated medical QA data from HF and synthetic examples.",
    )
    parser.add_argument(
        "--base_dataset",
        type=str,
        default="lavita/ChatDoctor-HealthCareMagic-100k",
        help="Hugging Face dataset id. Recommended: lavita/ChatDoctor-HealthCareMagic-100k or medalpaca/medical_meadow_medqa.",
    )
    parser.add_argument(
        "--base_split",
        type=str,
        default="train",
        help="Dataset split to load from the base dataset.",
    )
    parser.add_argument(
        "--base_sample_size",
        type=int,
        default=1500,
        help="Number of curated base examples to keep.",
    )
    parser.add_argument(
        "--synthetic_count",
        type=int,
        default=200,
        help="Number of synthetic examples to generate with OpenAI API.",
    )
    parser.add_argument(
        "--topics",
        type=str,
        nargs="*",
        default=DEFAULT_TOPICS,
        help="Medical topics for synthetic data generation.",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model used for synthetic example generation.",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=None,
        help="OpenAI API key. If omitted, script will read OPENAI_API_KEY env var.",
    )
    parser.add_argument(
        "--openai_api_base",
        type=str,
        default="https://api.openai.com/v1/chat/completions",
        help="OpenAI Chat Completions endpoint.",
    )
    parser.add_argument(
        "--raw_dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for raw exports.",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed jsonl outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling/shuffling.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    return parser.parse_args()


def normalize_example(example: Dict) -> Optional[Dict[str, str]]:
    """Normalize heterogeneous medical QA rows to {input, output}."""
    question_keys = ["input", "instruction", "question", "query", "prompt"]
    answer_keys = ["output", "response", "answer", "completion"]

    question = None
    answer = None

    for key in question_keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            question = value.strip()
            break

    for key in answer_keys:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            answer = value.strip()
            break

    if question and answer:
        return {"input": question, "output": answer}

    return None


def quality_filter(example: Dict[str, str]) -> bool:
    """Keep high-quality rows with practical length constraints."""
    q = example["input"].strip()
    a = example["output"].strip()
    if len(q) < 8 or len(a) < 20:
        return False
    if len(q) > 600 or len(a) > 4000:
        return False
    return True


def load_and_curate_base_data(
    dataset_name: str,
    split: str,
    sample_size: int,
    seed: int,
) -> List[Dict[str, str]]:
    """Load base dataset from HF, normalize, quality-filter, and sample examples."""
    LOGGER.info("Loading base dataset: %s (%s)", dataset_name, split)
    dataset = load_dataset(dataset_name, split=split)

    normalized: List[Dict[str, str]] = []
    for row in dataset:
        parsed = normalize_example(row)
        if parsed and quality_filter(parsed):
            normalized.append(parsed)

    if len(normalized) < sample_size:
        raise ValueError(
            f"Not enough curated examples ({len(normalized)}) for requested sample_size={sample_size}."
        )

    random.seed(seed)
    sampled = random.sample(normalized, sample_size)
    LOGGER.info("Selected %d curated base examples.", len(sampled))
    return sampled


def call_openai_chat_completion(
    topic: str,
    config: SyntheticConfig,
) -> Optional[Dict[str, str]]:
    """Call OpenAI chat completion API and parse a single synthetic example."""
    system_prompt = (
        "You are a senior physician creating educational but realistic medical QA pairs. "
        "Keep advice safe, accurate, and concise."
    )
    user_prompt = (
        f"Generate a realistic patient question about {topic} and a detailed, accurate "
        "doctor-style answer. Return JSON with keys: input, output."
    )

    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }

    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    req = urllib.request.Request(config.api_base, data=data, headers=headers, method="POST")

    for attempt in range(1, config.retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read().decode("utf-8")
                body = json.loads(raw)
                content = body["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                input_text = parsed.get("input", "").strip()
                output_text = parsed.get("output", "").strip()
                if not input_text or not output_text:
                    return None
                example = {"input": input_text, "output": output_text}
                return example if quality_filter(example) else None
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            LOGGER.warning("OpenAI call failed for topic=%s (attempt %d/%d): %s", topic, attempt, config.retries, exc)
            time.sleep(min(2 * attempt, 8))

    return None


def generate_synthetic_examples(
    count: int,
    topics: List[str],
    config: SyntheticConfig,
    seed: int,
) -> List[Dict[str, str]]:
    """Generate synthetic medical QA examples distributed across topics."""
    if count <= 0:
        return []
    if not config.api_key:
        raise ValueError("OpenAI API key is required for synthetic generation.")

    random.seed(seed)
    per_topic = count // len(topics)
    remainder = count % len(topics)

    topic_plan: List[Tuple[str, int]] = []
    for idx, topic in enumerate(topics):
        topic_count = per_topic + (1 if idx < remainder else 0)
        topic_plan.append((topic, topic_count))

    synthetic: List[Dict[str, str]] = []
    for topic, topic_count in topic_plan:
        LOGGER.info("Generating %d synthetic examples for topic=%s", topic_count, topic)
        for _ in range(topic_count):
            example = call_openai_chat_completion(topic, config)
            if example:
                synthetic.append(example)

    if len(synthetic) < count:
        LOGGER.warning("Generated %d/%d synthetic examples after filtering.", len(synthetic), count)

    return synthetic[:count]


def split_data(
    examples: List[Dict[str, str]],
    seed: int,
) -> Dict[str, List[Dict[str, str]]]:
    """Shuffle and split into train/val/test with 80/10/10 ratio."""
    random.seed(seed)
    shuffled = examples.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val :]

    return {"train": train, "val": val, "test": test}


def save_jsonl(rows: List[Dict[str, str]], path: Path) -> None:
    """Save rows to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def export_raw_snapshot(base_rows: List[Dict[str, str]], synthetic_rows: List[Dict[str, str]], raw_dir: Path) -> None:
    """Save raw snapshots for traceability."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(base_rows, raw_dir / "base_curated.jsonl")
    save_jsonl(synthetic_rows, raw_dir / "synthetic_generated.jsonl")


def main() -> None:
    """Run full dataset generation pipeline."""
    args = parse_args()
    setup_logging(args.log_level)

    api_key = args.openai_api_key
    if not api_key:
        api_key = __import__("os").environ.get("OPENAI_API_KEY")

    synthetic_cfg = SyntheticConfig(
        model=args.openai_model,
        api_key=api_key,
        api_base=args.openai_api_base,
        temperature=0.7,
        max_tokens=500,
        retries=3,
    )

    base_rows = load_and_curate_base_data(
        dataset_name=args.base_dataset,
        split=args.base_split,
        sample_size=args.base_sample_size,
        seed=args.seed,
    )

    synthetic_rows = generate_synthetic_examples(
        count=args.synthetic_count,
        topics=args.topics,
        config=synthetic_cfg,
        seed=args.seed,
    )

    merged = base_rows + synthetic_rows
    splits = split_data(merged, seed=args.seed)

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(splits["train"], args.processed_dir / "train.jsonl")
    save_jsonl(splits["val"], args.processed_dir / "val.jsonl")
    save_jsonl(splits["test"], args.processed_dir / "test.jsonl")

    export_raw_snapshot(base_rows, synthetic_rows, args.raw_dir)

    summary = {
        "base_dataset": args.base_dataset,
        "base_sample_size": len(base_rows),
        "synthetic_count": len(synthetic_rows),
        "total": len(merged),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "topics": args.topics,
        "seed": args.seed,
    }
    summary_path = args.processed_dir / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    LOGGER.info("Saved processed splits to %s", args.processed_dir.resolve())
    LOGGER.info("Dataset summary saved to %s", summary_path.resolve())


if __name__ == "__main__":
    main()
