# Experimental Report (Gemma-2B-it QLoRA)

## 1. Objective
Fine-tune `google/gemma-2b-it` on 1500 medical QA examples and compare against zero-shot baseline.

## 2. Dataset
- Total examples: 1500
- Train/Val/Test: 1200 / 150 / 150
- Domain: medical QA

## 3. Preprocessing
- Prompt format converted to Gemma instruction-chat turns:
  - `<start_of_turn>user ... <end_of_turn>`
  - `<start_of_turn>model ... <end_of_turn>`
- Tokenizer/model source: `google/gemma-2b-it`
- Sequence length used in final successful run: 256 (hardware-limited run)

## 4. Training Setup
- Method: QLoRA (4-bit)
- LoRA rank/alpha: 16 / 32
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Learning rate: `2e-4` with cosine scheduler
- Epochs: 5
- Hardware notes: 4GB VRAM required memory-safe adjustments (batch size 1, grad accumulation 16, gradient checkpointing, paged optimizer).

## 5. Quantitative Results (Test=150)
| Metric | Base Gemma-2B-it | Fine-tuned Gemma-2B-it |
|---|---:|---:|
| ROUGE-1 | 0.2538 | 0.1597 |
| ROUGE-2 | 0.0284 | 0.0166 |
| ROUGE-L | 0.1216 | 0.0796 |
| BLEU | 1.1392 | 0.5961 |
| BERTScore F1 | 0.8222 | 0.7865 |
| Avg response length (words) | 136.83 | 177.55 |
| Avg latency (ms) | 4304.23 | 12363.23 |

## 6. Training Behavior
Observed training loss trajectory increased after early stage:
- 0.67 epoch: 4.38
- 2.00 epoch: 4.71
- 3.33 epoch: 7.99
- 4.67 epoch: 14.59

This indicates instability/overfitting under the constrained hardware configuration.

## 7. Conclusion
- End-to-end Gemma pipeline ran successfully: preprocessing, training, and evaluation completed.
- In this hardware-constrained run, fine-tuned adapter underperformed the base model on automatic metrics.
- Recommended next iteration: run on higher-VRAM GPU (>=16GB), restore `max_length=512`, and re-enable validation-based checkpoint selection.

## 8. Artifacts
- Training history: `outputs/results_gemma/training_history.json`
- Evaluation report: `outputs/results_gemma/evaluation_report_gemma.json`
- Sample generations: `outputs/results_gemma/sample_outputs_gemma.txt`
- Adapter: `outputs/checkpoints_gemma/final_adapter_gemma`
