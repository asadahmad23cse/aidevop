# Fine-Tuning an LLM for Domain-Specific Medical Question Answering

This project fine-tunes an instruction-tuned LLM for healthcare question answering using QLoRA + PEFT. It supports `mistralai/Mistral-7B-Instruct-v0.2` as the primary model and falls back to `google/gemma-2b-it` when needed.

## Project Structure

```text
llm-finetune/
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataset_card.md
├── src/
│   ├── generate_dataset.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── checkpoints/
│   └── results/
├── requirements.txt
└── README.md
```

## Overview

The pipeline includes:
1. Dataset curation from medical QA sources on Hugging Face.
2. Synthetic augmentation (200 examples) using OpenAI API topic prompts.
3. Alpaca-style instruction formatting and tokenization (`max_length=512`).
4. QLoRA fine-tuning with LoRA adapters (`q_proj`, `v_proj`).
5. Evaluation of base vs fine-tuned models using ROUGE, BLEU, BERTScore, response length, and latency.
6. CLI inference with optional side-by-side base-model comparison.
7. Notebook analysis for data, training curves, metrics, and qualitative outputs.

## Setup

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Optional environment variables

Synthetic data generation requires OpenAI credentials:

```bash
export OPENAI_API_KEY="your_api_key"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

## Run Order

Run from the `llm-finetune/` project root.

### Step 1: Generate and split dataset

```bash
python src/generate_dataset.py \
  --base_dataset lavita/ChatDoctor-HealthCareMagic-100k \
  --base_sample_size 1500 \
  --synthetic_count 200
```

Outputs:
- `data/raw/base_curated.jsonl`
- `data/raw/synthetic_generated.jsonl`
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`
- `data/processed/dataset_summary.json`

### Step 2: Preprocess and tokenize

```bash
python src/preprocess.py \
  --processed_dir data/processed \
  --tokenized_output_dir data/processed/tokenized_dataset \
  --model_name mistralai/Mistral-7B-Instruct-v0.2 \
  --max_length 512
```

Output:
- `data/processed/tokenized_dataset/` (via `datasets.save_to_disk`)

### Step 3: Train with QLoRA + PEFT

```bash
python src/train.py \
  --tokenized_dataset_dir data/processed/tokenized_dataset \
  --output_dir outputs/checkpoints \
  --results_dir outputs/results
```

Outputs:
- Epoch checkpoints in `outputs/checkpoints/`
- Final adapter in `outputs/checkpoints/final_adapter/`
- Training history in `outputs/results/training_history.json`

### Step 4: Evaluate base vs fine-tuned

```bash
python src/evaluate.py \
  --test_file data/processed/test.jsonl \
  --adapter_path outputs/checkpoints/final_adapter \
  --results_dir outputs/results
```

Outputs:
- `outputs/results/evaluation_report.json`
- `outputs/results/sample_outputs.txt`

### Step 5: Inference

```bash
python src/inference.py --question "What are symptoms of diabetes?"
```

Compare mode:

```bash
python src/inference.py --question "What are symptoms of diabetes?" --compare
```

### Step 6: Analysis notebook

Open and run:
- `notebooks/analysis.ipynb`

## Hardware Requirements

- Recommended: NVIDIA A100 or T4 GPU with 16GB+ VRAM.
- Minimum practical: T4 16GB using QLoRA 4-bit quantization.
- CPU fallback is supported in scripts, but training will be very slow.

## Google Colab Tip

Use a T4/A100 runtime and run:

```python
!pip install -r requirements.txt
```

If Mistral loading fails due memory limits, training scripts automatically allow fallback to `google/gemma-2b-it`.

## Notes

- All scripts support `--help` via `argparse`.
- Logging uses Python `logging` module for status reporting.
- Paths are managed with `pathlib.Path` for cross-platform compatibility.
- The project is intended for educational/research use and not clinical diagnosis.

## Dashboard (Presentation UI)

A polished dashboard is available at:
- `dashboard/index.html`

Run from project root:

```bash
python -m http.server 8000
```

Then open:
- `http://localhost:8000/dashboard/index.html`

The dashboard reads live artifacts from:
- `outputs/results/evaluation_report.json`
- `outputs/results/training_history.json`
- `outputs/results/sample_outputs.txt`
- `data/processed/dataset_summary.json`

Inside the dashboard:
- `Analytics Mode` = technical metrics/charts
- `Teacher Mode` = viva-ready storyline slides (Methodology, Results, Qualitative Example, Limitations & Next Steps)
