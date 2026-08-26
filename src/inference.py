"""CLI inference for the fine-tuned medical QA model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
from peft import PeftModel
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
    parser = argparse.ArgumentParser(description="Run medical QA inference using a LoRA fine-tuned model.")
    parser.add_argument("--question", type=str, required=True, help="Patient question text.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Primary base model name.",
    )
    parser.add_argument(
        "--fallback_model",
        type=str,
        default="google/gemma-2b-it",
        help="Fallback model if primary fails.",
    )
    parser.add_argument(
        "--adapter_path",
        type=Path,
        default=Path("outputs/checkpoints/final_adapter"),
        help="Path to final LoRA adapter.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=160, help="Maximum new tokens to generate.")
    parser.add_argument("--compare", action="store_true", help="Also generate answer from base model for side-by-side comparison.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def build_prompt(question: str) -> str:
    """Build generation prompt in assignment format."""
    return (
        "Instruction:\n"
        f"{INSTRUCTION}\n"
        "Input:\n"
        f"{question}\n"
        "Response:\n"
    )


def load_backbone(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load backbone model/tokenizer with quantization on GPU and fallback on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
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
        LOGGER.warning("CUDA not available; loading model on CPU without 4-bit quantization.")
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, device_map=None)

    model.eval()
    return model, tokenizer


def load_models(
    base_model_name: str,
    fallback_model_name: str,
    adapter_path: Path,
) -> Tuple[str, AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM]:
    """Load base model and merged fine-tuned model."""
    selected_model = base_model_name
    try:
        base_model, tokenizer = load_backbone(base_model_name)
    except Exception as exc:
        LOGGER.exception("Failed loading base model %s: %s", base_model_name, exc)
        selected_model = fallback_model_name
        base_model, tokenizer = load_backbone(fallback_model_name)

    try:
        tuned_backbone, _ = load_backbone(selected_model)
        peft_model = PeftModel.from_pretrained(tuned_backbone, str(adapter_path))
        merged_model = peft_model.merge_and_unload()
        merged_model.eval()
    except Exception as exc:
        LOGGER.exception("Failed loading or merging adapter from %s: %s", adapter_path, exc)
        raise

    return selected_model, tokenizer, base_model, merged_model


def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """Generate an answer string for a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()


def main() -> None:
    """Run CLI inference and print answer(s)."""
    args = parse_args()
    setup_logging(args.log_level)

    selected_model, tokenizer, base_model, tuned_model = load_models(
        args.base_model,
        args.fallback_model,
        args.adapter_path,
    )
    LOGGER.info("Using backbone model: %s", selected_model)

    prompt = build_prompt(args.question)
    tuned_answer = generate_answer(tuned_model, tokenizer, prompt, args.max_new_tokens)

    if args.compare:
        base_answer = generate_answer(base_model, tokenizer, prompt, args.max_new_tokens)
        print("Base model answer:\n")
        print(base_answer)
        print("\n" + "-" * 80 + "\n")
        print("Fine-tuned model answer:\n")
        print(tuned_answer)
    else:
        print(tuned_answer)


if __name__ == "__main__":
    main()
