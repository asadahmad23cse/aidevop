"""Evaluate base vs fine-tuned medical QA model on test data."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import sacrebleu
import torch
from bert_score import score as bertscore_score
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
    parser = argparse.ArgumentParser(description="Evaluate base and fine-tuned medical QA models.")
    parser.add_argument("--test_file", type=Path, default=Path("data/processed/test.jsonl"), help="Path to test JSONL.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Base model for comparison.",
    )
    parser.add_argument(
        "--fallback_model",
        type=str,
        default="google/gemma-2b-it",
        help="Fallback base model if primary fails.",
    )
    parser.add_argument(
        "--adapter_path",
        type=Path,
        default=Path("outputs/checkpoints/final_adapter_v2"),
        help="Path to fine-tuned LoRA adapter.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory for metrics and samples.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Max new tokens for generation.")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam size for deterministic decoding.")
    parser.add_argument("--report_file", type=str, default="evaluation_report_v2.json", help="Evaluation report filename.")
    parser.add_argument("--sample_file", type=str, default="sample_outputs_v2.txt", help="Sample outputs filename.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    """Load JSONL rows into memory."""
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_prompt(question: str) -> str:
    """Build generation prompt using Gemma chat turn template."""
    return (
        "<start_of_turn>user\n"
        f"{INSTRUCTION}\n\nPatient question: {question}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def load_base_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a base model with quantized GPU path and CPU fallback."""
    use_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_cuda:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None)
    return model, tokenizer


def resolve_backbone_and_tokenizer(
    base_model_name: str,
    fallback_model_name: str,
) -> Tuple[str, AutoTokenizer]:
    """Resolve accessible backbone name and tokenizer with fallback handling."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return base_model_name, tokenizer
    except Exception as exc:
        LOGGER.exception("Failed to load base model %s: %s", base_model_name, exc)
        tokenizer = AutoTokenizer.from_pretrained(fallback_model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return fallback_model_name, tokenizer


def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
) -> Tuple[str, float]:
    """Generate one response and return text + latency (ms)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    response = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()
    return response, latency_ms


def compute_metrics(predictions: List[str], references: List[str], latencies_ms: List[float]) -> Dict[str, float]:
    """Compute ROUGE, BLEU, BERTScore, average response length, and latency."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1, rouge2, rougel = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1.append(scores["rouge1"].fmeasure)
        rouge2.append(scores["rouge2"].fmeasure)
        rougel.append(scores["rougeL"].fmeasure)

    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    _, _, bert_f1 = bertscore_score(predictions, references, lang="en", verbose=False)

    avg_len = sum(len(p.split()) for p in predictions) / max(1, len(predictions))
    avg_latency = sum(latencies_ms) / max(1, len(latencies_ms))

    return {
        "rouge_1": float(sum(rouge1) / max(1, len(rouge1))),
        "rouge_2": float(sum(rouge2) / max(1, len(rouge2))),
        "rouge_l": float(sum(rougel) / max(1, len(rougel))),
        "bleu": float(bleu),
        "bertscore_f1": float(bert_f1.mean().item()),
        "average_response_length_words": float(avg_len),
        "inference_latency_ms": float(avg_latency),
    }


def save_sample_outputs(
    path: Path,
    rows: List[Dict[str, str]],
    base_responses: List[str],
    tuned_responses: List[str],
) -> None:
    """Save 10 side-by-side qualitative examples."""
    samples = min(10, len(rows))
    lines: List[str] = []
    for idx in range(samples):
        lines.append(f"Example {idx + 1}")
        lines.append("Prompt:")
        lines.append(rows[idx]["input"])
        lines.append("Base Model Response:")
        lines.append(base_responses[idx])
        lines.append("Fine-Tuned Model Response:")
        lines.append(tuned_responses[idx])
        lines.append("=" * 80)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run full model-vs-model evaluation and save outputs."""
    args = parse_args()
    setup_logging(args.log_level)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.test_file)
    references = [r["output"] for r in rows]

    model_name, tokenizer = resolve_backbone_and_tokenizer(
        args.base_model,
        args.fallback_model,
    )
    LOGGER.info("Evaluating models based on backbone: %s", model_name)

    base_model, _ = load_base_model(model_name)
    base_model.eval()

    base_predictions: List[str] = []
    tuned_predictions: List[str] = []
    base_latencies: List[float] = []
    tuned_latencies: List[float] = []

    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(row["input"])
        base_resp, base_ms = generate_one(base_model, tokenizer, prompt, args.max_new_tokens, args.num_beams)
        base_predictions.append(base_resp)
        base_latencies.append(base_ms)
        if idx % 20 == 0:
            LOGGER.info("Evaluated base model %d/%d test examples", idx, len(rows))

    tuned_model = PeftModel.from_pretrained(base_model, str(args.adapter_path))
    tuned_model.eval()

    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(row["input"])
        tuned_resp, tuned_ms = generate_one(tuned_model, tokenizer, prompt, args.max_new_tokens, args.num_beams)
        tuned_predictions.append(tuned_resp)
        tuned_latencies.append(tuned_ms)
        if idx % 20 == 0:
            LOGGER.info("Evaluated fine-tuned model %d/%d test examples", idx, len(rows))

    base_metrics = compute_metrics(base_predictions, references, base_latencies)
    tuned_metrics = compute_metrics(tuned_predictions, references, tuned_latencies)

    report = {
        "model_backbone": model_name,
        "base_model": base_metrics,
        "fine_tuned_model": tuned_metrics,
        "num_test_examples": len(rows),
    }

    report_path = args.results_dir / args.report_file
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    sample_path = args.results_dir / args.sample_file
    save_sample_outputs(sample_path, rows, base_predictions, tuned_predictions)

    LOGGER.info("Saved evaluation report to %s", report_path.resolve())
    LOGGER.info("Saved sample outputs to %s", sample_path.resolve())


if __name__ == "__main__":
    main()
