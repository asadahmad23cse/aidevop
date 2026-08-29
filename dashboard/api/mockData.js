// Frontend mock data for HealthRAG AI — dashboard-only (no backend calls)
// Clearly labelled as frontend/demo data. Replace with real API adapters later.
(function () {
  window.HEALTHRAG_MOCK = {
    OVERVIEW_STATS: {
      knowledgeDocuments: 12,
      indexedChunks: 842,
      retrievalStatus: "Mock",
      llmStatus: "Pending Backend",
      evaluationQuestions: 25,
      services: 7,
    },

    SERVICES: [
      { name: "API Gateway", purpose: "Orchestrates requests between frontend and services.", endpoint: "/api/gateway/status", status: "Backend Pending", dependencies: ["Auth Service"] },
      { name: "RAG Service", purpose: "Retrieves relevant health information from the knowledge base.", endpoint: "/api/rag/query", status: "Backend Pending", dependencies: ["Vector Store"] },
      { name: "Knowledge Service", purpose: "Document ingestion & indexing.", endpoint: "/api/knowledge", status: "Backend Pending", dependencies: ["Storage"] },
      { name: "LLM Service", purpose: "Model inference (router + inference).", endpoint: "/api/llm/infer", status: "Backend Pending", dependencies: ["Model Pool"] },
      { name: "Evaluation Service", purpose: "Runs benchmarks and metrics aggregation.", endpoint: "/api/eval", status: "Backend Pending", dependencies: ["RAG Service", "LLM Service"] },
      { name: "Repository Service", purpose: "Repository / codebase analysis API.", endpoint: "/api/repo/analyze", status: "Backend Pending", dependencies: ["Git Reader"] },
      { name: "Guardrails", purpose: "Safety checks and prompt-injection defenses.", endpoint: "/api/guardrails/check", status: "Backend Pending", dependencies: ["LLM Service"] },
    ],

    DOCUMENTS: [
      { name: "medical_guidelines.json", type: "JSON", size: "1.2 MB", chunks: 120, status: "Indexed" },
      { name: "clinical_study_2023.pdf", type: "PDF", size: "3.4 MB", pages: 18, chunks: 240, status: "Indexed" },
    ],

    MODELS: [
      { id: "qwen", name: "Qwen2.5", developer: "Alibaba", size: "0.5B", status: "Waiting for API", latency: "—", accuracy: "—" },
      { id: "gemma", name: "Gemma 3", developer: "Google", size: "1B", status: "Waiting for API", latency: "—", accuracy: "—" },
      { id: "smol", name: "SmolLM2", developer: "Hugging Face", size: "1.7B", status: "Waiting for API", latency: "—", accuracy: "—" },
    ],

    EVAL_CATEGORIES: [
      "Health Information Retrieval",
      "Policy/Guideline Retrieval",
      "Symptom Information",
      "Medication Information",
      "Document Understanding",
      "RAG Grounding",
      "Context Relevance",
    ],

    EVAL_QUESTIONS: (function () {
      const qs = [];
      for (let i = 1; i <= 25; i++) {
        qs.push({ id: `Q${String(i).padStart(3, "0")}`, category: ["Health Information Retrieval", "Symptom Information", "Medication Information", "RAG Grounding"][i % 4], question: `Example evaluation question ${i}: Describe the recommended approach for example case ${i}.`, status: "Mock" });
      }
      return qs;
    })(),

    DEMO_METRICS: {
      quality: {
        accuracy: { qwen: 72, gemma: 81, smollm: 88 },
        relevance: { qwen: 68, gemma: 79, smollm: 85 },
        retrievalQuality: { qwen: 74, gemma: 82, smollm: 87 },
        hallucinationRate: { qwen: 18, gemma: 12, smollm: 8 },
        testPassRate: { qwen: 65, gemma: 78, smollm: 84 },
      },
      performance: {
        latency: { qwen: 1.2, gemma: 2.8, smollm: 4.5 },
        tokenUsage: { qwen: 800, gemma: 900, smollm: 1100 },
        cpuUsage: { qwen: 22, gemma: 38, smollm: 55 },
        gpuUsage: { qwen: 0, gemma: 0, smollm: 0 },
        memoryUsage: { qwen: 890, gemma: 1450, smollm: 2100 },
      },
    },

    TRADEOFFS: [
      { icon: "🏆", label: "Best Accuracy", value: "SmolLM2 (88%)", detail: "Highest accuracy in demo data" },
      { icon: "⚡", label: "Fastest Response", value: "Qwen2.5 (1.2s)", detail: "Lowest latency in demo" },
      { icon: "🧠", label: "Lowest Hallucination", value: "SmolLM2 (8%)", detail: "Lowest hallucination rate in demo" },
      { icon: "💾", label: "Lowest Memory", value: "Qwen2.5 (890 MB)", detail: "Lowest memory usage" },
      { icon: "⚖️", label: "Best Trade-off", value: "Gemma 3", detail: "Balanced accuracy and latency" },
    ],

    GUARDRAILS: [
      { name: "Prompt Injection Detection", description: "Detects malicious prompt constructs.", status: "Backend Pending" },
      { name: "Input Safety Check", description: "Identifies unsafe or disallowed requests.", status: "Backend Pending" },
      { name: "Unsupported Health Request", description: "Flags requests requiring clinician escalation.", status: "Backend Pending" },
      { name: "Grounding Check", description: "Verifies model answers against retrieved context.", status: "Backend Pending" },
      { name: "Medical Safety Check", description: "Performs final verification of medical safety.", status: "Backend Pending" },
    ],

    REPO_ANALYSIS: {
      files: ["src/rag_ingest.py", "src/rag_server.py", "src/ollama_client.py"],
      components: ["RAG Ingest", "Vector Store", "RAG API", "Dashboard UI"],
      dependencies: ["faiss", "sentence_transformers", "ollama-client"],
      functionCalls: ["ingest()", "chunk()", "embed()", "index()", "query()"],
      impactAnalysis: ["Changing chunk size requires reindexing", "Updating embedder affects all vectors"],
      relatedTests: ["tests/test_week4_metrics.py"],
      sampleQuestions: ["Which files are involved in ingestion?", "What happens after POST /api/v1/rag/compare?"],
    },

    MODEL_ROUTER_EXAMPLES: [
      { query: "What is hypertension?", difficulty: "Easy", model: "Qwen2.5 0.5B", reason: "Simple factual query", fallback: "Gemma 3 1B" },
      { query: "Compare treatment options for multi-morbidity in elderly with renal impairment.", difficulty: "Complex", model: "SmolLM2 1.7B", reason: "Requires nuanced synthesis", fallback: "Gemma 3 1B" },
    ],
  };
})();
