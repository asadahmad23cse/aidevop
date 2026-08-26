"""Preprocess medical QA data into Alpaca format and tokenize."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict

from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

LOGGER = logging.getLogger(__name__)

INSTRUCTION = (
    "You are a helpful medical assistant. Answer the following patient question accurately and concisely."
)


def setup_logging(log_level: str) -> None:
    """Configure root logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Format and tokenize train/val/test medical QA data for SFT.",
    )
    parser.add_argument("--processed_dir", type=Path, default=Path("data/processed"), help="Directory containing train/val/test JSONL files.")
    parser.add_argument(
        "--tokenized_output_dir",
        type=Path,
        default=Path("data/processed/tokenized_dataset"),
        help="Output directory for tokenized dataset saved with datasets.save_to_disk().",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Tokenizer source model name.",
    )
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for tokenization.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def format_example(example: Dict[str, str]) -> Dict[str, str]:
    """Format one row in Gemma instruction-chat template."""
    user_content = f"{INSTRUCTION}\n\nPatient question: {example['input']}"
    text = (
        "<start_of_turn>user\n"
        f"{user_content}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
        f"{example['output']}"
        "<end_of_turn>"
    )
    return {"text": text}


def tokenize_example(example: Dict[str, str], tokenizer: AutoTokenizer, max_length: int) -> Dict:
    """Tokenize one formatted row and create labels for causal LM training."""
    tokenized = tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def build_collator(tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
    """Create a padding collator with attention mask support."""
    return DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


def main() -> None:
    """Run formatting/tokenization pipeline and save dataset to disk."""
    args = parse_args()
    setup_logging(args.log_level)

    data_files = {
        "train": str(args.processed_dir / "train.jsonl"),
        "validation": str(args.processed_dir / "val.jsonl"),
        "test": str(args.processed_dir / "test.jsonl"),
    }

    LOGGER.info("Loading processed JSONL files from %s", args.processed_dir.resolve())
    dataset = load_dataset("json", data_files=data_files)

    LOGGER.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    LOGGER.info("Applying template formatting")
    formatted: DatasetDict = dataset.map(format_example)

    LOGGER.info("Tokenizing with max_length=%d", args.max_length)
    tokenized: DatasetDict = formatted.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=[],
    )

    _ = build_collator(tokenizer)
    LOGGER.info("Data collator initialized with dynamic padding and attention masks.")

    args.tokenized_output_dir.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(args.tokenized_output_dir))
    LOGGER.info("Saved tokenized dataset to %s", args.tokenized_output_dir.resolve())


if __name__ == "__main__":
    main()


