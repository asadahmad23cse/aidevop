# Dataset Card: Medical Domain QA Fine-Tuning Set

## Source

This dataset is built from:
- Hugging Face base dataset (configurable):
  - `lavita/ChatDoctor-HealthCareMagic-100k` (default), or
  - `medalpaca/medical_meadow_medqa`
- Synthetic augmentation generated with OpenAI API prompts across these topics:
  - diabetes, hypertension, asthma, depression, arthritis, fever, migraine, thyroid, anemia, anxiety

## Size

Target composition:
- Curated base examples: 1500
- Synthetic examples: 200
- Total: 1700

Split ratio:
- Train: 80%
- Validation: 10%
- Test: 10%

Exact counts are written to `data/processed/dataset_summary.json` after generation.

## Format

Each sample is JSON with keys:

```json
{
  "input": "Patient question text",
  "output": "Doctor-style answer text"
}
```

Files:
- `data/processed/train.jsonl`
- `data/processed/val.jsonl`
- `data/processed/test.jsonl`

One JSON object per line (`.jsonl`).

## Preprocessing Steps

1. Load base dataset split from Hugging Face.
2. Normalize schema to `input` and `output` fields.
3. Apply quality filtering:
   - Remove empty rows.
   - Enforce minimum/maximum text length bounds.
4. Randomly sample 1500 curated base examples.
5. Generate 200 synthetic examples using a structured JSON prompt with OpenAI API.
6. Merge curated and synthetic data.
7. Shuffle with fixed seed.
8. Split into train/validation/test (80/10/10).
9. Save `.jsonl` files to `data/processed/`.
10. During tokenization (`src/preprocess.py`), transform each row to the exact Alpaca instruction template and tokenize to `max_length=512`.

## Intended Use

- Research and educational experiments on domain adaptation for medical QA.
- Demonstrating QLoRA/PEFT fine-tuning and model evaluation workflows.

## Limitations and Safety

- Data can contain medical simplifications, omissions, or outdated practices.
- Model outputs are not clinical advice.
- Human expert review is required before any real-world healthcare use.
