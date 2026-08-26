# Week 4 Results: Multi-Model, RAG, and Repository Evaluation

## Completion and validity

All six exercises were executed locally on the same Windows 11 laptop (Intel Core Ultra 5 125H, 15.4 GB RAM, Intel Arc integrated GPU) using Ollama 0.32.15. The exact models were `codellama:7b`, `starcoder2:3b`, and `qwen2.5-coder:3b`. Temperature, seed, prompts, task order, retrieved context, knowledge base, and generation limits were held constant within each experiment.

The medical matrix has 75/75 unique records (3 models × 25 tasks), and the repository matrix has 24/24 (3 × 8). Both have zero errors, one experiment hash each, identical per-question prompt/context across models, and complete latency, token, CPU, RAM, and Ollama VRAM fields. Test-pass rate is N/A because neither suite asks the models to generate executable code.

## Exercises 1–4: quantitative model comparison

### Medical RAG application (25 tasks)

| Model | Correctness | Relevance | Hallucination | Retrieval P@3 / Recall@3 / MRR | Abstention | Mean / median / p95 latency | Total tokens | Mean CPU | Mean peak RAM | Model VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Code Llama 7B | 18.0% | 72.0% | 44.4% | .333 / 1.000 / 1.000 | 0% | 42.32 / 41.37 / 55.66 s | 27,209 | 23.0% | 15,150 MB | 5,710 MB |
| StarCoder2 3B | **58.0%** | **80.0%** | **11.1%** | .333 / 1.000 / 1.000 | 0% | 19.44 / 20.76 / 25.69 s | 24,491 | **8.5%** | **11,291 MB** | **1,789 MB** |
| Qwen2.5-Coder 3B | 18.0% | **80.0%** | 27.1% | .333 / 1.000 / 1.000 | **50%** | **9.26 / 8.84 / 13.99 s** | **21,201** | 9.5% | 12,011 MB | 2,055 MB |

StarCoder2 is the best medical-quality model under the recorded evidence rubric: it leads correctness by 40 percentage points over both alternatives and has the lowest hallucination rate. Qwen is 10.19 seconds faster per answer than StarCoder2 and uses 3,290 fewer total tokens, but loses 40 correctness points. StarCoder2 is also the lightest model allocation and has the lowest sampled CPU/RAM. The most accurate model is therefore not the fastest, but it is the most resource-efficient; a clear quality–latency trade-off remains.

Retrieval is identical across models by design. Every in-domain question retrieved its single labelled document at rank 1, so Recall@3, MRR, Hit@3, and nDCG@3 are 1.0. Precision@3 is 1/3 because only one of three returned documents was labelled relevant.

### Repository understanding (8 multi-file tasks)

| Model | Correctness | Relevance | Hallucination | Retrieval P@3 / Recall@3 / MRR / nDCG | Mean / median / p95 latency | Total tokens | Mean CPU | Mean peak RAM | Model VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Code Llama 7B | **56.2%** | **62.5%** | 17.6% | .708 / .338 / .875 / .748 | 78.89 / 77.86 / 89.50 s | 16,636 | 29.6% | 15,114 MB | 5,710 MB |
| StarCoder2 3B | 6.2% | 31.2% | **3.6%** | .708 / .338 / .875 / .748 | **21.95 / 17.80 / 39.46 s** | 13,543 | 27.0% | **9,431 MB** | **1,789 MB** |
| Qwen2.5-Coder 3B | 37.5% | 50.0% | 10.1% | .708 / .338 / .875 / .748 | 26.23 / 25.28 / 34.03 s | **13,447** | **21.3%** | 13,070 MB | 2,055 MB |

Code Llama is strongest on cross-file correctness, but it costs 56.94 seconds more per response than StarCoder2 and uses 3.19× its model VRAM. StarCoder2's low hallucination number is partly explained by much shorter answers (87.6 completion tokens on average versus Code Llama's 256.6), and those answers omit many required facts. Qwen is the middle-quality option and uses the fewest total tokens/CPU, while StarCoder2 remains fastest and lightest in VRAM/RAM.

## Exercise 5: retrieval → context → response analysis

The full `QUESTION → RETRIEVED CONTEXT → WITH-RAG RESPONSE → WITHOUT-RAG RESPONSE` traces are recorded in `outputs/week4/MEDICAL_RESULTS.md`.

- **Relevant retrieval and strong use:** MED-010 retrieves document 10 at rank 1. StarCoder2's RAG fact coverage is 1.0 versus 0.0 without RAG, demonstrating a direct retrieval benefit.
- **Relevant retrieval but incomplete use:** MED-017 retrieves document 17 at rank 1, yet Code Llama covers only 0.167 of required facts versus 0.333 without RAG. Correct retrieval did not guarantee context use.
- **Irrelevant forced retrieval:** MED-022 has no labelled answer, but top-k still returns documents 1350, 950, and 581. All three models failed to abstain, showing how irrelevant context can encourage unsupported answers.
- **Correct abstention despite irrelevant context:** For MED-025, Qwen explicitly abstains even though documents 1362, 1386, and 453 were forced into context; Code Llama and StarCoder2 do not.
- **Important information missed:** Repository recall@3 is only 0.338. REP-007 retrieves three chunks from `src/ollama_client.py`, omitting other labelled files needed for the full relationship, and all three models reach only 0.286 fact coverage.
- **Correct response after multi-file retrieval:** REP-001 retrieves `README.md`, `src/rag_ingest.py`, and `src/rag_server.py`; Code Llama reaches 0.833 fact coverage versus 0.0 without RAG.

Thus, retrieval quality controls the evidence ceiling, context selection controls whether key files/facts are visible, and model behavior controls whether visible evidence is used or ignored. RAG is not a binary feature: rank-1 medical retrieval still produced weak answers, while repository top-3 retrieval often omitted necessary files.

## Exercise 6: repository-level understanding and limitations

The repository index covers 19 files in 235 chunks with a hard 1,800-character cap and top-3 retrieval. Eight tasks ask about ingestion-to-serving flow, API/dashboard interactions, model configuration, preprocessing/training relationships, deployment files, and change impact across modules.

The system can answer multi-file questions when the needed files appear together (REP-001), but mean Recall@3 of 0.338 shows that a flat embedding index is insufficient for complete repository understanding. It does not build a call graph, resolve imports/symbol references, or perform rename/impact analysis. Duplicate chunks from one file can consume all retrieval slots, as REP-007 demonstrates. These are precisely the limitations that repository-aware search and Sourcegraph-style navigation should address next.

## Metric definitions and limitations

- Correctness and relevance are 0–2 evidence scores normalized to 0–1. Required-fact/topical coverage ≥0.75 scores 2, ≥0.40 scores 1, otherwise 0; correct out-of-scope abstention scores 2.
- Hallucination rate is unsupported factual sentences divided by factual sentences; support requires at least 35% content-token overlap with a retrieved chunk. This is a lexical proxy, not blind expert review.
- Retrieval metrics use labelled document/file sets. Out-of-scope questions are excluded from IR denominators.
- Latency is wall-clock Ollama request time after warm-up. Token counts come from Ollama. CPU and RAM were sampled every 200 ms. VRAM is Ollama's model allocation on Intel Arc; utilization was unavailable because `nvidia-smi` does not support this GPU.
- This is one controlled run, not repeated trials. Small differences should not be treated as statistically significant.
