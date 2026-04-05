"""Train a medical QA assistant using QLoRA + PEFT with TRL SFTTrainer."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

LOGGER = logging.getLogger(__name__)


def setup_logging(log_level: str) -> None:
    """Configure root logging."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train QLoRA adapters for medical QA.")
    parser.add_argument(
        "--tokenized_dataset_dir",
        type=Path,
        default=Path("data/processed/tokenized_dataset"),
        help="Path to tokenized dataset saved via datasets.save_to_disk().",
    )
    parser.add_argument(
        "--primary_model",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="Primary base model to load.",
    )
    parser.add_argument(
        "--fallback_model",
        type=str,
        default="google/gemma-2b-it",
        help="Fallback model if primary fails.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/checkpoints"),
        help="Checkpoint output directory.",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory to save training loss summaries.",
    )
    parser.add_argument(
        "--final_adapter_name",
        type=str,
        default="final_adapter_v2",
        help="Final adapter folder name under output_dir.",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    return parser.parse_args()


def load_model_and_tokenizer(primary_model: str, fallback_model: str) -> Tuple[str, AutoModelForCausalLM, AutoTokenizer]:
    """Load base model and tokenizer with fallback handling."""
    use_cuda = torch.cuda.is_available()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    for model_name in [primary_model, fallback_model]:
        try:
            LOGGER.info("Attempting model load: %s", model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            if use_cuda:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quant_config,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=False,
                )
            else:
                LOGGER.warning("CUDA not available. Falling back to CPU model loading without 4-bit quantization.")
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None,
                    trust_remote_code=False,
                )

            # Greedy generation config requested.
            model.generation_config.do_sample = False
            model.generation_config.num_beams = 1
            if hasattr(model.generation_config, "temperature"):
                model.generation_config.temperature = None
            if hasattr(model.generation_config, "top_p"):
                model.generation_config.top_p = None

            return model_name, model, tokenizer
        except Exception as exc:
            LOGGER.exception("Failed to load model %s: %s", model_name, exc)

    raise RuntimeError("Could not load either primary or fallback model.")


def build_lora_config() -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def build_training_args(output_dir: Path) -> TrainingArguments:
    """Create training arguments with requested hyperparameters."""
    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
    )


def extract_epoch_losses(log_history: list) -> Dict[str, list]:
    """Extract train and validation losses from trainer log history."""
    train_losses = []
    eval_losses = []

    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_losses.append({"epoch": float(entry["epoch"]), "loss": float(entry["loss"])})
        if "eval_loss" in entry and "epoch" in entry:
            eval_losses.append({"epoch": float(entry["epoch"]), "loss": float(entry["eval_loss"])})

    return {"train_loss": train_losses, "val_loss": eval_losses}


def main() -> None:
    """Run QLoRA supervised fine-tuning."""
    args = parse_args()
    setup_logging(args.log_level)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading tokenized dataset from %s", args.tokenized_dataset_dir.resolve())
    dataset: DatasetDict = load_from_disk(str(args.tokenized_dataset_dir))

    model_name, model, tokenizer = load_model_and_tokenizer(args.primary_model, args.fallback_model)
    model.config.use_cache = False

    lora_config = build_lora_config()
    training_args = build_training_args(args.output_dir)

    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]

    LOGGER.info("Initializing SFTTrainer with model=%s", model_name)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    LOGGER.info("Starting training")
    trainer.train()

    loss_history = extract_epoch_losses(trainer.state.log_history)
    history_path = args.results_dir / "training_history.json"
    history_path.write_text(json.dumps(loss_history, indent=2), encoding="utf-8")

    final_adapter_dir = args.output_dir / args.final_adapter_name
    final_adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_adapter_dir))
    tokenizer.save_pretrained(str(final_adapter_dir))

    LOGGER.info("Saved final adapter to %s", final_adapter_dir.resolve())
    LOGGER.info("Saved training history to %s", history_path.resolve())


if __name__ == "__main__":
    main()
