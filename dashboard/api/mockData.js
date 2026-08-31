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
      services: 8,
    },

    // `key` maps a card to live health from /api/v1/services/status (see renderServices).
    SERVICES: [
      { key: "gateway", name: "API Gateway / Orchestrator", purpose: "FastAPI app that runs the full pipeline and serves the dashboard.", endpoint: "/api/v1/services/status", status: "Checking", dependencies: ["RAG", "LLM Service"] },
      { key: "rag", name: "RAG Service", purpose: "Retrieves relevant passages from the knowledge base (FAISS or lexical).", endpoint: "/query · /api/v1/rag/compare", status: "Checking", dependencies: ["Vector Index"] },
      { key: "knowledge", name: "Knowledge Service", purpose: "JSON / PDF ingestion, chunking and indexing (append mode).", endpoint: "/ingest", status: "Checking", dependencies: ["Storage"] },
      { key: "llm", name: "LLM Service (Router)", purpose: "Difficulty classification, model routing and grounded generation via Ollama.", endpoint: "/api/v1/router/answer · /api/v1/models/status", status: "Checking", dependencies: ["Ollama"] },
      { key: "guardrails", name: "Guardrails", purpose: "Prompt-injection + domain scope (input) and grounding + medical-safety (output).", endpoint: "runs inside /api/v1/router/answer", status: "Active", dependencies: [] },
      { key: "suggest", name: "Question Suggestion Service", purpose: "Autocomplete questions from the partial query and knowledge-base topics.", endpoint: "/api/suggest/questions", status: "Checking", dependencies: ["Knowledge Base"] },
      { key: "eval", name: "Evaluation Service", purpose: "Week 4 multi-model benchmark — executed offline; results shipped in the repo.", endpoint: "outputs/week4/ · scripts/week4_evaluate.py", status: "Executed (offline)", dependencies: ["RAG", "LLM Service"] },
      { key: "repo", name: "Repository Service", purpose: "Week 4 repository-understanding benchmark — executed offline.", endpoint: "outputs/week4/repository_* · build_repository_index.py", status: "Executed (offline)", dependencies: ["Repo Index"] },
    ],

    // ── Question Suggestion Service (frontend mock config) ──────────────
    // Local suggestion logic. Structured so getSuggestions() can later POST the
    // partial query to /api/suggest/questions and receive KB/LLM-ranked results.
    QUESTION_SUGGESTIONS: {
      // Topics assumed to exist in the knowledge base. Replace with a live
      // list from the Knowledge Service when the backend is connected.
      kbTopics: [
        "hypertension", "diabetes", "asthma", "treatment", "diagnosis", "dosage",
        "side effects", "symptoms", "prevention", "risk factors", "amoxicillin",
        "paracetamol", "blood pressure", "cholesterol", "vaccination",
        "bone bruise", "diarrhea in toddlers", "otitis media", "anticoagulants",
        "chronic kidney disease", "pregnancy", "pediatric dosing",
      ],
      // Partial-query templates → completed with each kbTopic.
      templates: [
        { match: /what does (the )?(doc(ument)?|guideline|report)s? say about\s*$/i, make: (t) => `What does the document say about ${t}?` },
        { match: /what are the symptoms of\s*$/i, make: (t) => `What are the symptoms of ${t}?` },
        { match: /how is\s*$/i, make: (t) => `How is ${t} treated?` },
        { match: /what is the (recommended )?(dose|dosage) (of|for)\s*$/i, make: (t) => `What is the recommended dose of ${t}?` },
        { match: /(tell me|explain) about\s*$/i, make: (t) => `Explain ${t} from the uploaded document.` },
        { match: /what are the risk factors for\s*$/i, make: (t) => `What are the risk factors for ${t}?` },
      ],
      // Fallback bank of full questions, ranked by token overlap with the query.
      bank: [
        "What does the document say about hypertension?",
        "What does the document say about diabetes?",
        "What does the document say about treatment?",
        "What are the symptoms of diabetes?",
        "What are the symptoms of hypertension?",
        "Explain hypertension from the uploaded document.",
        "What does the guideline say about treatment?",
        "Summarize this health document.",
        "What is the recommended dose of amoxicillin for children?",
        "What are the side effects of paracetamol?",
        "How is asthma diagnosed and managed?",
        "What are the risk factors for chronic kidney disease?",
        "What does the document say about blood pressure targets?",
        "How is a deep bone bruise treated?",
        "What are the symptoms of viral diarrhea in toddlers?",
        "What does the guideline say about vaccination schedules?",
      ],
    },

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
      { name: "Prompt Injection Detection", description: "Blocks attempts to override or reveal the system instructions before any LLM call.", status: "Active" },
      { name: "Domain Relevance Guardrail", description: "Only health, medical and knowledge-base questions are allowed. Anything else is rejected as Out of Domain and never reaches the LLM.", status: "Active" },
      { name: "Grounding Check", description: "Verifies the answer is supported by the retrieved context (lexical proxy).", status: "Active" },
      { name: "Medical Safety Check", description: "Flags absolute clinical claims and appends a professional-advice note.", status: "Active" },
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
