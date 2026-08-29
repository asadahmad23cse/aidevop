// Frontend service adapters / placeholders for HealthRAG AI
// These provide a clean interface the UI uses. Replace with real API adapters later.
(function () {
  const MOCK = window.HEALTHRAG_MOCK || {};

  function statusBadgeClass(status) {
    if (!status) return "status-pending";
    const s = String(status).toLowerCase();
    if (s.includes("connected") || s.includes("live") || s.includes("connected")) return "status-connected";
    if (s.includes("mock")) return "status-mock";
    if (s.includes("waiting") || s.includes("pending") || s.includes("backend")) return "status-pending";
    return "status-pending";
  }

  // Sanitize user-facing text to remove training/fine-tuning references.
  // Keeps backend artifacts intact but prevents exposing training instructions in the UI.
  function sanitizeText(str) {
    if (!str || typeof str !== "string") return str;
    // Terms to redact from frontend display
    const banned = [
      /fine[- ]?tun(e|ing)/ig,
      /qlo?ra/ig,
      /lora/ig,
      /peft/ig,
      /adapter(s)?/ig,
      /epoch(s)?/ig,
      /learning[- ]?rate/ig,
      /optimizer(s)?/ig,
      /checkpoint(s)?/ig,
      /train(ing)?\b/ig,
      /dataset(s)?\b/ig,
      /qlora/ig,
      /fine[- ]?tuned/ig,
    ];

    let out = str;
    for (const r of banned) out = out.replace(r, "[redacted]");
    return out;
  }

  function deepSanitize(obj) {
    if (obj == null) return obj;
    if (typeof obj === "string") return sanitizeText(obj);
    if (Array.isArray(obj)) return obj.map(deepSanitize);
    if (typeof obj === "object") {
      const copy = {};
      for (const k of Object.keys(obj)) copy[k] = deepSanitize(obj[k]);
      return copy;
    }
    return obj;
  }

  // Lightweight service facade consumed by app.js
  window.HealthRAGServices = {
    statusBadge: (s) => statusBadgeClass(s),

    overviewService: {
      getStats: () => deepSanitize(MOCK.OVERVIEW_STATS || {}),
      getServices: () => deepSanitize(MOCK.SERVICES || []),
    },

    knowledgeService: {
      getDocuments: () => deepSanitize(MOCK.DOCUMENTS || []),
    },

    ragService: {
      // compare(query, topK) -> returns { source: 'mock'|'backend', data: { with_rag, without_rag }}
      compare: async (query, topK = 3) => {
        // Placeholder: return consistent mock structure. Backend integration will replace implementation.
        const with_rag = { answer: sanitizeText(`[Backend Pending] With-RAG demo answer for: ${query}`) };
        const without_rag = { answer: sanitizeText(`[Backend Pending] Baseline demo answer for: ${query}`) };
        return Promise.resolve({ source: "mock", data: deepSanitize({ with_rag, without_rag }) });
      },
    },

    modelRouterService: {
      getExamples: () => deepSanitize(MOCK.MODEL_ROUTER_EXAMPLES || []),
      classifyQuery: (q) => {
        // Simple frontend-only heuristic using keywords (NOT a real classifier)
        const text = (q || "").toLowerCase();
        if (text.length < 30 || /what|who|when|where|symptom|dose/.test(text)) {
          return { difficulty: "Easy", model: "Qwen2.5 0.5B", reason: "Short factual query", fallback: "Gemma 3 1B" };
        }
        if (/compare|recommend|best|evidence|systematic/.test(text)) {
          return { difficulty: "Complex", model: "SmolLM2 1.7B", reason: "Requires synthesis", fallback: "Gemma 3 1B" };
        }
        return deepSanitize({ difficulty: "Medium", model: "Gemma 3 1B", reason: "Moderate complexity", fallback: "Qwen2.5 0.5B" });
      },
    },

    modelService: {
      getModels: () => deepSanitize(MOCK.MODELS || []),
    },

    evaluationService: {
      getCategories: () => deepSanitize(MOCK.EVAL_CATEGORIES || []),
      getQuestions: () => deepSanitize(MOCK.EVAL_QUESTIONS || []),
      getDemoMetrics: () => deepSanitize(MOCK.DEMO_METRICS || {}),
      getTradeOffs: () => deepSanitize(MOCK.TRADEOFFS || []),
    },

    guardrailsService: {
      getChecks: () => deepSanitize(MOCK.GUARDRAILS || []),
    },

    repositoryService: {
      getAnalysis: () => deepSanitize(MOCK.REPO_ANALYSIS || {}),
    },
  };
})();
