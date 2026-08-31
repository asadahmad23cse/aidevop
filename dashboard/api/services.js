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

  // ── Backend transport ────────────────────────────────────────────────
  // The dashboard is served by the FastAPI app at the same origin, so calls
  // are relative. Every call has a timeout and falls back to mock data, and
  // tags its result with `source: "backend" | "mock"` so the UI can show
  // an honest connection state (never a fake "connected").
  const API_BASE = window.HEALTHRAG_API_BASE || "";

  async function apiFetch(path, opts = {}, timeoutMs = 8000) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const resp = await fetch(API_BASE + path, {
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        ...opts,
      });
      clearTimeout(timer);
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      return await resp.json();
    } catch (err) {
      clearTimeout(timer);
      console.warn("[HealthRAG] backend call failed for " + path + ":", err.message);
      return null;
    }
  }
  const apiGet = (p, t) => apiFetch(p, {}, t);
  const apiPost = (p, body, t) => apiFetch(p, { method: "POST", body: JSON.stringify(body || {}) }, t);

  // Lightweight service facade consumed by app.js
  window.HealthRAGServices = {
    statusBadge: (s) => statusBadgeClass(s),
    apiFetch,

    // Live system status (PRD sections 4, 12, 13, 16). Null when backend is down.
    apiService: {
      _cache: null,
      getStatus: async () => {
        const self = window.HealthRAGServices.apiService;
        const data = await apiGet("/api/v1/services/status", 4000);
        self._cache = data ? deepSanitize(data) : null;
        return self._cache;
      },
      cached: () => window.HealthRAGServices.apiService._cache,
    },

    overviewService: {
      getStats: () => deepSanitize(MOCK.OVERVIEW_STATS || {}),
      getServices: () => deepSanitize(MOCK.SERVICES || []),
    },

    knowledgeService: {
      getDocuments: () => deepSanitize(MOCK.DOCUMENTS || []),
    },

    ragService: {
      // compare(query, topK) -> { source: 'backend'|'mock', data: { with_rag, without_rag, retrieved_chunks } }
      compare: async (query, topK = 3) => {
        const data = await apiPost("/api/v1/rag/compare", { query, top_k: topK }, 300000);
        if (data && (data.with_rag || data.without_rag)) {
          return { source: "backend", data: deepSanitize(data) };
        }
        const with_rag = { answer: sanitizeText(`[Backend unavailable] Could not reach the RAG service for: ${query}`) };
        const without_rag = { answer: sanitizeText(`[Backend unavailable] Baseline unavailable for: ${query}`) };
        return { source: "mock", data: deepSanitize({ with_rag, without_rag, retrieved_chunks: [] }) };
      },
    },

    modelRouterService: {
      getExamples: () => deepSanitize(MOCK.MODEL_ROUTER_EXAMPLES || []),

      // route(query, topK, history) -> { source: 'backend'|'mock', data: <full pipeline result> }
      // `history` (optional): prior turns [{role:'user'|'assistant', content}] for multi-turn chat.
      // Long timeout: small local models can still take a while on modest CPU-only
      // hardware, especially on the first call while the model loads into memory.
      route: async (query, topK = 3, history = null, focusDocIds = null) => {
        const body = { query, top_k: topK };
        if (Array.isArray(history) && history.length) body.history = history;
        if (Array.isArray(focusDocIds) && focusDocIds.length) body.focus_doc_ids = focusDocIds;
        const data = await apiPost("/api/v1/router/answer", body, 300000);
        if (data && data.guardrails) {
          return { source: "backend", data: deepSanitize(data) };
        }
        return { source: "mock", data: window.HealthRAGServices.modelRouterService.classifyQuery(query) };
      },

      // Offline fallback only — simple keyword heuristic (NOT a real classifier).
      classifyQuery: (q) => {
        const text = (q || "").toLowerCase();
        let difficulty, model, reason;
        if (/compare|recommend|best|evidence|synthes|trade-?off|why |evaluate/.test(text) || text.length > 140) {
          difficulty = "Complex"; model = "SmolLM2 1.7B"; reason = "Requires synthesis (offline heuristic)";
        } else if (text.length < 40 && /^(what is|what are|who|when|where|define)/.test(text)) {
          difficulty = "Easy"; model = "Qwen2.5 0.5B"; reason = "Short factual query (offline heuristic)";
        } else {
          difficulty = "Medium"; model = "Gemma 3 1B"; reason = "Moderate complexity (offline heuristic)";
        }
        return deepSanitize({
          query: q, difficulty, reason,
          model: { name: model, tag: "", developer: "" },
          guardrails: {}, retrieval: null, answer: null,
          status: "offline", llm_source: "offline",
        });
      },
    },

    modelService: {
      // getModels() -> merges static specs with live availability from the backend.
      getModels: async () => {
        const base = deepSanitize(MOCK.MODELS || []);
        const live = await apiGet("/api/v1/models/status", 4000);
        if (!live || !Array.isArray(live.tiers)) {
          return base.map((m) => ({ ...m, availability: "checking" }));
        }
        const byTier = { easy: "qwen", medium: "gemma", complex: "smol" };
        const tierFor = {};
        live.tiers.forEach((t) => { tierFor[byTier[t.tier]] = t; });
        return base.map((m) => {
          const t = tierFor[m.id];
          return {
            ...m,
            tag: t ? t.tag : m.tag,
            developer: t ? t.developer : m.developer,
            availability: t ? t.status : "checking", // available | not_installed | ollama_offline
            ollamaOnline: !!live.ollama_online,
          };
        });
      },
      getModelsSync: () => deepSanitize(MOCK.MODELS || []),
    },

    evaluationService: {
      getCategories: () => deepSanitize(MOCK.EVAL_CATEGORIES || []),
      getQuestions: () => deepSanitize(MOCK.EVAL_QUESTIONS || []),
      getDemoMetrics: () => deepSanitize(MOCK.DEMO_METRICS || {}),
      getTradeOffs: () => deepSanitize(MOCK.TRADEOFFS || []),
    },

    // ── Week 4 executed-evaluation data (real, from outputs/week4/) ────────
    // Reads window.WEEK4_DATA (dashboard/week4_data.js). Falls back to the
    // demo mock when that file is absent so the dashboard still renders.
    week4Service: {
      _w4: () => window.WEEK4_DATA || null,
      hasRealData: () => {
        const w = window.WEEK4_DATA;
        return !!(w && w.med_summary && w.med_summary.models);
      },
      modelKeys: () => ["codellama_7b", "starcoder2_3b", "qwen25_coder_3b"],
      modelLabel: (k) => ({
        codellama_7b: "Code Llama 7B",
        starcoder2_3b: "StarCoder2 3B",
        qwen25_coder_3b: "Qwen2.5-Coder 3B",
      }[k] || k),

      // Per-model metric rows for Task 3 (real numbers, %/s/MB as given).
      getMedicalModels: () => {
        const w = window.WEEK4_DATA;
        const m = w && w.med_summary && w.med_summary.models;
        if (!m) return null;
        const out = {};
        for (const [k, d] of Object.entries(m)) {
          const q = d.quality || {}, p = d.performance || {}, adj = q.manual_adjudication || {}, r = q.retrieval || {};
          out[k] = {
            correctness: adj.correctness_accuracy != null ? +(adj.correctness_accuracy * 100).toFixed(1) : null,
            relevance: adj.relevance != null ? +(adj.relevance * 100).toFixed(1) : null,
            hallucination: adj.hallucination_rate != null ? +(adj.hallucination_rate * 100).toFixed(1) : null,
            retrievalPk: r.precision_at_k != null ? +(r.precision_at_k * 100).toFixed(1) : null,
            retrievalRecall: r.recall_at_k != null ? +(r.recall_at_k * 100).toFixed(1) : null,
            abstention: q.out_of_scope_abstention_accuracy != null ? +(q.out_of_scope_abstention_accuracy * 100).toFixed(0) : null,
            latencyS: p.latency_ms_mean != null ? +(p.latency_ms_mean / 1000).toFixed(1) : null,
            latencyP95S: p.latency_ms_p95 != null ? +(p.latency_ms_p95 / 1000).toFixed(1) : null,
            totalTokens: p.total_tokens ?? null,
            cpu: p.system_cpu_percent_mean != null ? +p.system_cpu_percent_mean.toFixed(1) : null,
            ramMB: p.system_memory_used_mb_peak_mean != null ? Math.round(p.system_memory_used_mb_peak_mean) : null,
            vramMB: p.ollama_model_vram_mb_mean != null ? Math.round(p.ollama_model_vram_mb_mean) : null,
          };
        }
        return deepSanitize(out);
      },

      // Task 4 analysis bullets (medical + repository).
      getAnalysis: () => {
        const w = window.WEEK4_DATA;
        if (!w) return null;
        return deepSanitize({
          medical: (w.med_summary && w.med_summary.analysis) || [],
          repository: (w.repo_summary && w.repo_summary.analysis) || [],
        });
      },

      // Task 5 real retrieval -> context -> response traces.
      getTraces: () => {
        const w = window.WEEK4_DATA;
        const t = w && w.med_summary && w.med_summary.rag_traces;
        return Array.isArray(t) ? deepSanitize(t) : null;
      },

      // Task 2 real evaluation dataset.
      getDataset: () => {
        const w = window.WEEK4_DATA;
        return Array.isArray(w && w.med_eval) ? deepSanitize(w.med_eval) : null;
      },
    },

    guardrailsService: {
      getChecks: () => deepSanitize(MOCK.GUARDRAILS || []),
    },

    repositoryService: {
      getAnalysis: () => deepSanitize(MOCK.REPO_ANALYSIS || {}),
    },

    // ── Question Suggestion Service ──────────────────────────────────────
    // Provides dynamic autocomplete suggestions for the question inputs.
    // Currently uses local mock logic (template + token-overlap ranking).
    // To connect the backend later, implement `fetchRemote` to POST
    // { partial, limit } to /api/suggest/questions and return string[].
    questionSuggestionService: {
      _cfg: () => MOCK.QUESTION_SUGGESTIONS || { kbTopics: [], templates: [], bank: [] },

      // Backend hook. Calls the Question Suggestion Service when the dashboard
      // is served by the API (same origin). Returns null on any failure so the
      // caller falls back to local mock logic — safe to call on every keystroke.
      _endpoint: (window.HEALTHRAG_API_BASE || "") + "/api/suggest/questions",
      _remoteOk: true,
      fetchRemote: async (partial, limit = 4) => {
        const self = window.HealthRAGServices.questionSuggestionService;
        if (!self._remoteOk || !partial || partial.trim().length < 3) return null;
        try {
          const ctrl = new AbortController();
          const timer = setTimeout(() => ctrl.abort(), 1200);
          const resp = await fetch(self._endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ partial, limit }),
            signal: ctrl.signal,
          });
          clearTimeout(timer);
          if (!resp.ok) throw new Error("bad status " + resp.status);
          const data = await resp.json();
          return Array.isArray(data.suggestions) ? data.suggestions.map(sanitizeText) : null;
        } catch (_) {
          self._remoteOk = false; // stop retrying this session once unavailable
          return null;
        }
      },

      // getSuggestions(partial, limit) -> string[]  (0..limit items, min 2 when any)
      getSuggestions: (partial, limit = 4) => {
        const cfg = window.HealthRAGServices.questionSuggestionService._cfg();
        const raw = (partial || "").trim();
        if (raw.length < 3) return [];
        const q = raw.toLowerCase();
        const out = [];
        const seen = new Set();
        const push = (s) => {
          const clean = sanitizeText(s);
          const key = clean.toLowerCase();
          if (!clean || seen.has(key)) return;
          seen.add(key);
          out.push(clean);
        };

        // 1. Template completion — "what does the doc say about" → topic list.
        for (const tpl of cfg.templates || []) {
          if (tpl.match.test(q)) {
            for (const topic of cfg.kbTopics || []) {
              push(tpl.make(topic));
              if (out.length >= limit) break;
            }
          }
          if (out.length >= limit) break;
        }

        // 2. Topic-aware bank ranking — token overlap with the partial query.
        if (out.length < limit) {
          const tokens = q.split(/\s+/).filter((t) => t.length > 2);
          const scored = (cfg.bank || [])
            .map((question) => {
              const lc = question.toLowerCase();
              let score = 0;
              tokens.forEach((t) => {
                if (lc.includes(t)) score += t.length >= 5 ? 2 : 1;
              });
              if (lc.startsWith(q)) score += 5;
              return { question, score };
            })
            .filter((x) => x.score > 0)
            .sort((a, b) => b.score - a.score);
          for (const s of scored) {
            push(s.question);
            if (out.length >= limit) break;
          }
        }

        // 3. Topic fallback — surface KB topics that appear in the query.
        if (out.length < 2) {
          for (const topic of cfg.kbTopics || []) {
            if (q.includes(topic.split(" ")[0])) {
              push(`What does the document say about ${topic}?`);
              if (out.length >= limit) break;
            }
          }
        }

        return out.slice(0, limit);
      },
    },
  };
})();
