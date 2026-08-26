# PulseMirror risk-aware AI operations

PulseMirror extends the Week 4 medical RAG application with a small, working AI-operations layer. It does not claim clinical diagnosis, autonomous treatment, Kubernetes scaling, or production-grade rollback.

## Request path

`POST /api/v1/pulsemirror/ask` runs this policy:

1. Classify operational risk from the question using deterministic safety patterns.
2. Block generative output for emergency patterns and return urgent-care guidance.
3. Retrieve evidence for non-emergency questions.
4. Use the configured LLM when available; otherwise use the extractive grounded fallback.
5. Add a clinical-review warning to high-risk medication or diagnostic requests.
6. Record route, latency, fallback use, and safety flags without storing the question text.

The risk label controls routing behavior. It is not a prediction of disease severity.

## Routes

| Risk | Operational route | Behavior |
|---|---|---|
| Low | Standard or grounded fallback RAG | Answer from retrieved evidence. |
| Medium | Grounded RAG with safety warning | Advise professional review for persistent or worsening symptoms. |
| High | Strict grounded pipeline | Require retrieved evidence and add a clinician-review warning. |
| Emergency | Emergency safety pipeline | Skip model generation and show deterministic urgent-care guidance. |

## Observability

`GET /api/v1/observability` exposes a bounded in-memory window containing request count, average and p95 latency, route distribution, risk distribution, fallback rate, safety-flag rate, deployment version, and service readiness. Raw questions and patient identifiers are not recorded.

The in-memory counters reset when the service restarts. A production system would export structured, de-identified telemetry to a durable monitoring platform with access controls and retention policies.

## Week 4 evidence

`GET /api/v1/week4/results` serves the checked-in evaluation summary used by the dashboard. It does not generate or alter benchmark scores. Full methodology and limitations remain in `docs/WEEK4_EVALUATION.md` and `docs/WEEK4_RESULTS.md`.

## Future production work

- Replace lexical safety patterns with a validated medical-safety classifier and clinician-reviewed policies.
- Add authenticated audit trails, privacy controls, and de-identification.
- Export metrics to an observability backend and define alert thresholds from repeated experiments.
- Add genuine shadow/canary deployments before enabling automated rollback.
- Run external clinical, security, bias, and failure-mode evaluation before any real healthcare use.
