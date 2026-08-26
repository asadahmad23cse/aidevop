# Week 4: Multi-Model LLM, RAG, and Repository Evaluation

This implementation completes the six Week 4 exercises as a reproducible experiment. The recorded results are in `outputs/week4/`; no model score is invented or entered before execution.

## Experimental controls

The medical benchmark uses the same 25 questions, knowledge base, FAISS index, embedding model, top-k value, system/prompt templates, retrieved chunks, question order, decoding values, and hardware session for all three models. The default comparison is:

1. `codellama:7b`
2. `starcoder2:3b`
3. `qwen2.5-coder:3b`

Their sizes are not identical, so conclusions apply to these deployable model variants, not to model families at a matched parameter count. Temperature is zero, seed is 42, maximum generation is 220 tokens, and top-k is 3. A single warm-up request is made before timed questions so model loading is recorded separately from steady-state response latency.

The 25-question medical set is `data/evaluation/week4_medical_eval.jsonl`. It contains 21 real patient questions and reference answers taken directly from indexed Week 3 documents plus four deliberately out-of-scope questions. Every answerable task is labelled with its exact knowledge-base document ID and required facts. All three models receive the exact same retrieved chunks for each question.

## Metric definitions

### Quality metrics and evidence adjudication

`src/week4_adjudicate.py` completed every CSV row using the following recorded deterministic rubric. This is explicitly **not** blind human review; the CSV schema permits a later blind human adjudication to replace it.

- **Correctness / Accuracy** = sum of per-answer correctness points / (2 × reviewed answers). Required-fact coverage ≥0.75 scores 2, ≥0.40 scores 1, otherwise 0; out-of-scope answers score 2 only when they explicitly abstain.
- **Relevance** = sum of relevance points / (2 × reviewed answers). The share of substantive answer sentences with ≥20% content-token overlap to the question/reference is banded at ≥0.75 (2), ≥0.40 (1), otherwise 0.
- **Hallucination rate** = unsupported factual sentences / factual sentences. A sentence is flagged when fewer than 35% of its content tokens overlap every retrieved chunk. It is a lexical evidence proxy and can under-score paraphrases or over-score copied but incorrect text.
- **Test-pass rate** = generated-code tasks passing all tests / generated-code tasks attempted. It is **N/A** here: the main application is medical QA, and Exercise 6 evaluates repository understanding rather than generated code. No denominator is manufactured.

The scripts also calculate clearly labelled deterministic proxies—required-fact phrase coverage, topical sentence relevance, and unsupported-sentence lexical overlap. These support debugging and initial comparison but do not replace human assessment.

### Retrieval metrics

For answerable questions, with `R` the labelled relevant documents and the first `k` retrieved documents:

- **Precision@k** = relevant documents retrieved in top-k / k.
- **Recall@k** = relevant documents retrieved in top-k / number of labelled relevant documents.
- **Hit@k** = 1 if at least one relevant document appears in top-k, otherwise 0.
- **MRR** = mean of `1 / rank of first relevant document`.
- **nDCG@k** = discounted cumulative gain divided by the ideal discounted gain using binary relevance.

These IR metrics are undefined for deliberately out-of-scope questions and are excluded rather than assigned a misleading zero. For those cases, report abstention accuracy and inspect whether forced top-k retrieval supplied irrelevant context.

### Performance metrics

- **Response latency** = wall-clock milliseconds from sending an Ollama generation request until its complete non-streamed response; report mean, median, and p95 after warm-up.
- **Token usage** = Ollama `prompt_eval_count` + `eval_count`; report prompt mean, completion mean, and total.
- **CPU consumption** = mean system CPU percentage sampled every 200 ms during each request.
- **Memory consumption** = peak system used memory and peak combined RSS of processes whose names contain `ollama`, sampled during each request.
- **GPU consumption** = peak utilization percentage and used memory reported by `nvidia-smi` during each request. On non-NVIDIA systems, Ollama's `/api/ps` VRAM allocation is recorded as a fallback; utilization remains N/A when no supported sensor exists.

Resource readings are observational system-level measurements. Close unrelated workloads and use the same machine/session for all models. A model server may retain allocated memory between calls; this limitation must be stated with results.

## Safe run order

All commands are run from the repository root.

### 1. Static preflight only

```powershell
python src/week4_evaluate.py
```

This validates configuration, the 25-task schema, ground-truth document IDs, index files, Python dependencies, Ollama availability, and all three installed models. It does not run generation. Install/start Ollama separately and pull the exact configured tags only after checking available RAM/VRAM and disk space.

### 2. Optional one-question smoke run

Only after preflight passes:

```powershell
python src/week4_evaluate.py --run --confirm-full-run --limit 1
```

Use a separate output path for a smoke run if you do not want it mixed with the full matrix.

### 3. Full, resumable medical benchmark

```powershell
python src/week4_evaluate.py --run --confirm-full-run --resume
```

Raw append-only records go to `outputs/week4/raw_results.jsonl`. Six selected questions also receive a no-RAG baseline so the report can show `QUESTION → RETRIEVED CONTEXT → WITH-RAG RESPONSE → WITHOUT-RAG RESPONSE` and quantify the required-fact coverage delta.

### 4. Evidence adjudication and report

```powershell
python src/week4_adjudicate.py --input outputs/week4/raw_results.jsonl --output outputs/week4/manual_review.csv
python src/week4_report.py --input outputs/week4/raw_results.jsonl --review outputs/week4/manual_review.csv
```

The report validates one record for every model/question pair and verifies that prompt hashes and retrieved contexts match across models. It then creates `evaluation_summary.json` and `WEEK4_RESULTS.md`. If the matrix is incomplete, it reports the problem and does not declare a winner.

## Exercise 5: RAG analysis method

The configured trace set includes answerable, emergency, medication, comparison, and out-of-scope questions. Analyse each trace at three linked stages:

1. **Retrieval quality:** Was a labelled relevant source in top-k? Were high-ranked chunks irrelevant? Was an important source absent?
2. **Context quality:** Did chunk boundaries preserve the complete fact? Did several chunks conflict or distract? For out-of-scope questions, did forced retrieval introduce misleading evidence?
3. **Response quality:** Did the model use the evidence, ignore it, distort it, or hallucinate despite it? Compare with the no-RAG answer and report both positive and negative deltas.

A relevant hit does not guarantee a good answer, and a fluent answer does not prove good retrieval. The generated report assigns relationship descriptions from labelled retrieval and fact coverage; the rule-based claim review remains a limitation.

## Exercise 6: repository-level understanding

The codebase probe is separate from medical evidence, preventing cross-domain contamination.

```powershell
python src/build_repository_index.py
python src/build_repository_index.py --build --confirm-build
python src/week4_evaluate.py `
  --config config/week4_repository_models.json `
  --dataset data/evaluation/week4_repository_eval.jsonl `
  --output outputs/week4/repository_raw_results.jsonl
```

The first command checks 19 included files and chunk settings without loading an embedding model or writing an index. The second explicitly builds the repository index. The third is still preflight-only; add `--run --confirm-full-run` only after it passes.

The eight probes require evidence across multiple files/modules and are labelled with relevant source paths. Generate a separate report by passing repository input/review/output paths to `week4_report.py`. This lexical chunk-index approach cannot resolve call graphs, symbol references, or rename impact as reliably as repository-aware tools; those are explicit limitations to discuss before next week's Sourcegraph work.

## Evidence-based conclusion checklist

The final conclusion must quote numbers from the generated summary and answer:

- highest human correctness and relevance;
- lowest human hallucination rate;
- best retrieval metrics (expected to be identical across models because retrieval is fixed);
- lowest mean/median/p95 latency and token use;
- lowest observed CPU/RAM/GPU use;
- whether the most accurate model is also the fastest or lightest;
- absolute quality differences alongside latency and memory differences;
- RAG cases where retrieval helped, failed, distracted, or was ignored;
- repository questions that failed because relevant files were not retrieved versus failures after adequate retrieval.

Do not call a small difference meaningful without repeated runs or uncertainty analysis. This framework records one controlled run; repeating the full experiment three times is recommended if time and hardware allow.
