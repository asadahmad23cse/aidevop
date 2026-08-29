/* ==========================================================================
   HealthRAG AI — Dashboard Application Logic
   ========================================================================== */

(function () {
  "use strict";

  const SVC = window.HealthRAGServices || {};
  const MOCK = window.HEALTHRAG_MOCK || {};
  const W4 = window.WEEK4_DATA || {};
  const charts = {};

  document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    renderOverview();
    renderServices();
    initKnowledgeBase();
    initRAGPipeline();
    initModelRouter();
    renderGuardrails();
    renderModelCards();
    renderEvalDataset();
    renderMetrics();
    initCharts();
    renderTradeOffs();
    initRagAnalysis();
    renderRepoExplorer();
    renderRepoProbes();
    renderW4Summaries();
    initModal();
  });

  /* ── Navigation ── */
  function initNavigation() {
    const links = document.querySelectorAll(".nav-link-main");
    const toggle = document.getElementById("navMobileToggle");
    const nav = document.getElementById("navLinksMain");

    links.forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const id = link.getAttribute("data-section");
        const el = document.getElementById(id);
        if (el) {
          const offset = 80;
          const top = el.getBoundingClientRect().top + window.scrollY - offset;
          window.scrollTo({ top, behavior: "smooth" });
        }
        links.forEach((l) => l.classList.remove("active"));
        link.classList.add("active");
        if (nav) nav.classList.remove("open");
      });
    });

    if (toggle && nav) {
      toggle.addEventListener("click", () => nav.classList.toggle("open"));
    }

    const sections = document.querySelectorAll(".dashboard-section[id]");
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const id = entry.target.id;
            links.forEach((l) => {
              l.classList.toggle("active", l.getAttribute("data-section") === id);
            });
          }
        });
      },
      { rootMargin: "-90px 0px -60% 0px", threshold: 0.1 }
    );
    sections.forEach((s) => observer.observe(s));
  }

  /* ── Overview ── */
  function renderOverview() {
    const stats = SVC.overviewService?.getStats() || {};
    const el = document.getElementById("overviewStats");
    if (!el) return;

    const items = [
      { label: "Knowledge Documents", val: stats.knowledgeDocuments, sub: "Indexed sources" },
      { label: "Indexed Chunks", val: stats.indexedChunks, sub: "Vector embeddings" },
      { label: "Retrieval Status", val: stats.retrievalStatus, sub: "Vector search", cls: "highlight-amber" },
      { label: "LLM Status", val: stats.llmStatus, sub: "Model inference", cls: "highlight-amber" },
      { label: "Evaluation Questions", val: stats.evaluationQuestions, sub: "Benchmark set" },
      { label: "Services", val: stats.services, sub: "Microservices" },
    ];

    el.innerHTML = items
      .map(
        (i) => `
      <div class="stat-box">
        <div class="stat-box-label">${i.label}</div>
        <div class="stat-box-val ${i.cls || ""}">${i.val}</div>
        <div class="stat-box-sub">${i.sub}</div>
      </div>`
      )
      .join("");
  }

  /* ── Services ── */
  function renderServices() {
    const grid = document.getElementById("servicesGrid");
    const services = SVC.overviewService?.getServices() || [];
    if (!grid) return;

    grid.innerHTML = services
      .map(
        (s) => `
      <article class="service-card">
        <div class="service-card-head">
          <h3>${escapeHtml(s.name)}</h3>
          <span class="status-badge ${SVC.statusBadge(s.status)}">${s.status}</span>
        </div>
        <p class="service-purpose">${escapeHtml(s.purpose)}</p>
        <div class="service-meta">
          <div><span>API</span><code>${escapeHtml(s.endpoint)}</code></div>
          <div><span>Dependencies</span><strong>${s.dependencies.map(escapeHtml).join(", ")}</strong></div>
        </div>
      </article>`
      )
      .join("");
  }

  /* ── Knowledge Base ── */
  function initKnowledgeBase() {
    setupUpload("json", "jsonDropZone", "jsonFileInput", "btnBrowseJson", "jsonUploadList", "JSON");
    setupUpload("pdf", "pdfDropZone", "pdfFileInput", "btnBrowsePdf", "pdfUploadList", "PDF");
    renderKnowledgeDocs();
  }

  function setupUpload(type, zoneId, inputId, btnId, listId, label) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    const btn = document.getElementById(btnId);
    const list = document.getElementById(listId);
    if (!zone || !input) return;

    btn?.addEventListener("click", () => input.click());

    input.addEventListener("change", () => handleFiles(input.files, list, type, label));

    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
      handleFiles(e.dataTransfer.files, list, type, label);
    });
  }

  function handleFiles(files, listEl, type, label) {
    if (!listEl || !files.length) return;
    Array.from(files).forEach((file) => {
      const size = formatSize(file.size);
      const pages = type === "pdf" ? Math.floor(Math.random() * 40) + 5 : "—";
      const item = document.createElement("div");
      item.className = "upload-item";
      item.innerHTML = `
        <div class="upload-item-info">
          <strong>${escapeHtml(file.name)}</strong>
          <span>${size}${type === "pdf" ? " · " + pages + " pages" : ""}</span>
        </div>
        <span class="status-badge status-mock">Mock Upload</span>
        <span class="status-badge status-pending">Indexing Pending</span>`;
      listEl.appendChild(item);
    });
    renderKnowledgeDocs();
  }

  function renderKnowledgeDocs() {
    const tbody = document.getElementById("knowledgeDocTable");
    if (!tbody) return;
    const docs = SVC.knowledgeService?.getDocuments() || [];
    const jsonItems = document.querySelectorAll("#jsonUploadList .upload-item");
    const pdfItems = document.querySelectorAll("#pdfUploadList .upload-item");

    let rows = docs.map(
      (d) => `<tr>
        <td>${escapeHtml(d.name)}</td><td>${d.type}</td><td>${d.size}</td>
        <td>—</td><td>${d.chunks}</td>
        <td><span class="status-badge status-mock">${d.status}</span></td>
      </tr>`
    );

    jsonItems.forEach((item) => {
      const name = item.querySelector("strong")?.textContent || "upload.json";
      rows.push(`<tr><td>${escapeHtml(name)}</td><td>JSON</td><td>—</td><td>—</td><td>—</td><td><span class="status-badge status-pending">Pending</span></td></tr>`);
    });
    pdfItems.forEach((item) => {
      const name = item.querySelector("strong")?.textContent || "upload.pdf";
      const pages = item.querySelector("span")?.textContent.match(/(\d+) pages/)?.[1] || "—";
      rows.push(`<tr><td>${escapeHtml(name)}</td><td>PDF</td><td>—</td><td>${pages}</td><td>—</td><td><span class="status-badge status-pending">Pending</span></td></tr>`);
    });

    tbody.innerHTML = rows.join("") || `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:2rem;">No documents uploaded yet.</td></tr>`;
  }

  /* ── RAG Pipeline (Week 3 preserved) ── */
  function initRAGPipeline() {
    const btnAnimate = document.getElementById("btnAnimatePipeline");
    const btnRun = document.getElementById("btnRunLiveQuery");
    const input = document.getElementById("liveQueryInput");
    const outputGrid = document.getElementById("liveOutputGrid");
    const ragOut = document.getElementById("liveRagOutput");
    const baseOut = document.getElementById("liveBaseOutput");
    const ragBadge = document.getElementById("liveRagStatusBadge");
    const baseBadge = document.getElementById("liveBaseStatusBadge");
    const backendBadge = document.getElementById("ragBackendBadge");

    if (btnAnimate) {
      btnAnimate.addEventListener("click", () => {
        const pipeSteps = [1, 2, 3, 4, 5, 6, 7].map((i) => document.getElementById("pipeStep" + i));
        const flowNodes = [1, 2, 3, 4, 5, 6, 7, 8].map((i) => document.getElementById("flowNode" + i));
        [...pipeSteps, ...flowNodes].forEach((n) => n && n.classList.remove("pipe-active"));

        [...pipeSteps, ...flowNodes].forEach((n, idx) => {
          setTimeout(() => n && n.classList.add("pipe-active"), idx * 350);
          setTimeout(() => n && n.classList.remove("pipe-active"), idx * 350 + 600);
        });
      });
    }

    if (btnRun) {
      btnRun.addEventListener("click", async () => {
        const query = input?.value.trim();
        if (!query) return;

        if (outputGrid) outputGrid.style.display = "grid";
        if (ragOut) ragOut.textContent = "Querying RAG backend...";
        if (baseOut) baseOut.textContent = "Querying baseline...";
        if (ragBadge) ragBadge.textContent = "Loading";
        if (baseBadge) baseBadge.textContent = "Loading";

        const result = await SVC.ragService.compare(query, 3);

        if (result.source === "backend") {
          if (backendBadge) {
            backendBadge.textContent = "Backend Connected";
            backendBadge.className = "pill-badge pill-emerald";
          }
          if (ragOut) ragOut.textContent = result.data.with_rag?.answer || "No response";
          if (baseOut) baseOut.textContent = result.data.without_rag?.answer || "No response";
          if (ragBadge) ragBadge.textContent = "Live";
          if (baseBadge) baseBadge.textContent = "Live";
        } else {
          if (ragOut) ragOut.textContent = result.data.with_rag?.answer || "[Backend Pending]";
          if (baseOut) baseOut.textContent = result.data.without_rag?.answer || "[Backend Pending]";
          if (ragBadge) ragBadge.textContent = "Pending";
          if (baseBadge) baseBadge.textContent = "Pending";
        }
      });
    }
  }

  /* ── Model Router ── */
  function initModelRouter() {
    const btn = document.getElementById("btnClassifyQuery");
    const input = document.getElementById("routerQueryInput");
    const resultEl = document.getElementById("routerResult");
    const examplesEl = document.getElementById("routerExamples");
    const examples = SVC.modelRouterService?.getExamples() || [];

    if (examplesEl) {
      examplesEl.innerHTML = examples
        .map(
          (ex) => `
        <div class="router-example-card">
          <p class="router-ex-query">"${escapeHtml(ex.query)}"</p>
          <div class="router-ex-details">
            <span><strong>Difficulty:</strong> ${ex.difficulty}</span>
            <span><strong>Model:</strong> ${ex.model}</span>
            <span><strong>Reason:</strong> ${escapeHtml(ex.reason)}</span>
            <span><strong>Fallback:</strong> ${ex.fallback}</span>
          </div>
        </div>`
        )
        .join("");
    }

    function showResult(ex) {
      if (!resultEl) return;
      resultEl.innerHTML = `
        <div class="router-result-grid">
          <div class="router-field"><span>Query</span><strong>${escapeHtml(ex.query)}</strong></div>
          <div class="router-field"><span>Detected Difficulty</span><strong class="highlight-cyan">${ex.difficulty}</strong></div>
          <div class="router-field"><span>Selected Model</span><strong>${ex.model}</strong></div>
          <div class="router-field"><span>Reason for Selection</span><strong>${escapeHtml(ex.reason)}</strong></div>
          <div class="router-field"><span>Fallback Model</span><strong>${ex.fallback}</strong></div>
          <div class="router-field"><span>Status</span><span class="status-badge status-pending">Backend Pending</span></div>
        </div>`;
    }

    btn?.addEventListener("click", () => {
      const q = input?.value.trim() || "What is hypertension?";
      const ex = SVC.modelRouterService?.classifyQuery(q);
      if (ex) showResult({ ...ex, query: q });
    });

    if (examples[0]) showResult(examples[0]);
  }

  /* ── Guardrails ── */
  function renderGuardrails() {
    const el = document.getElementById("guardrailCards");
    const checks = SVC.guardrailsService?.getChecks() || [];
    if (!el) return;

    el.innerHTML = checks
      .map(
        (c) => `
      <article class="guardrail-card">
        <h4>${escapeHtml(c.name)}</h4>
        <p>${escapeHtml(c.description)}</p>
        <span class="status-badge status-pending">${c.status}</span>
      </article>`
      )
      .join("");
  }

  /* ── Model Cards (W4 Task 1) ── */
  function renderModelCards() {
    const grid = document.getElementById("modelCardsGrid");
    const models = SVC.modelService?.getModels() || [];
    if (!grid) return;

    grid.innerHTML = models
      .map(
        (m) => `
      <article class="model-card">
        <div class="model-header">
          <div class="model-icon">${m.id === "qwen" ? "⚡" : m.id === "gemma" ? "💎" : "🔬"}</div>
          <div>
            <h3 class="model-title">${escapeHtml(m.name)}</h3>
            <p class="model-tagline">${escapeHtml(m.developer)} · ${escapeHtml(m.difficulty)}</p>
          </div>
        </div>
        <div class="model-specs">
          <div class="spec-item"><span>Model Size</span><strong>${escapeHtml(m.size)}</strong></div>
          <div class="spec-item"><span>Status</span><strong>${escapeHtml(m.status)}</strong></div>
          <div class="spec-item"><span>Latency</span><strong>${m.latency}</strong></div>
          <div class="spec-item"><span>Accuracy</span><strong>${m.accuracy}</strong></div>
        </div>
        <span class="status-badge status-pending">Waiting for API</span>
      </article>`
      )
      .join("");
  }

  /* ── Evaluation Dataset (W4 Task 2) ── */
  let evalFilter = "all";
  let evalSearch = "";

  function renderEvalDataset() {
    const categories = SVC.evaluationService?.getCategories() || [];
    const pillsEl = document.getElementById("evalFilterPills");
    const countEl = document.getElementById("evalQuestionCount");

    if (pillsEl) {
      pillsEl.innerHTML =
        `<button class="filter-pill active" data-cat="all">All</button>` +
        categories.map((c) => `<button class="filter-pill" data-cat="${escapeHtml(c)}">${escapeHtml(c)}</button>`).join("");

      pillsEl.querySelectorAll(".filter-pill").forEach((p) => {
        p.addEventListener("click", () => {
          pillsEl.querySelectorAll(".filter-pill").forEach((x) => x.classList.remove("active"));
          p.classList.add("active");
          evalFilter = p.getAttribute("data-cat");
          renderEvalTable();
        });
      });
    }

    document.getElementById("evalSearchInput")?.addEventListener("input", (e) => {
      evalSearch = e.target.value.toLowerCase();
      renderEvalTable();
    });

    renderEvalTable();
    if (countEl) {
      const qs = SVC.evaluationService?.getQuestions() || [];
      countEl.textContent = qs.length + " Questions";
    }
  }

  function renderEvalTable() {
    const tbody = document.getElementById("evalTableBody");
    if (!tbody) return;

    let questions = SVC.evaluationService?.getQuestions() || [];

    if (evalFilter !== "all") {
      questions = questions.filter((q) => q.category === evalFilter);
    }
    if (evalSearch) {
      questions = questions.filter(
        (q) => q.id.toLowerCase().includes(evalSearch) || q.question.toLowerCase().includes(evalSearch) || q.category.toLowerCase().includes(evalSearch)
      );
    }

    tbody.innerHTML =
      questions.length === 0
        ? `<tr><td colspan="4" style="text-align:center;padding:2rem;color:var(--text-muted);">No matching questions.</td></tr>`
        : questions
            .map(
              (q) => `
          <tr>
            <td><strong style="color:var(--cyan);font-family:var(--font-mono);">${q.id}</strong></td>
            <td><span class="tag-badge tag-medical">${escapeHtml(q.category)}</span></td>
            <td>${escapeHtml(q.question)}</td>
            <td><span class="status-badge status-mock">Mock</span></td>
          </tr>`
            )
            .join("");
  }

  /* ── Metrics (W4 Task 3) ── */
  function renderMetrics() {
    const metrics = SVC.evaluationService?.getDemoMetrics() || {};
    const qEl = document.getElementById("qualityMetricCards");
    const pEl = document.getElementById("perfMetricCards");
    const tbody = document.getElementById("metricsCompareBody");

    const qualityLabels = {
      accuracy: "Accuracy",
      relevance: "Relevance",
      retrievalQuality: "Retrieval Quality",
      hallucinationRate: "Hallucination Rate",
      testPassRate: "Test Pass Rate",
    };
    const perfLabels = {
      latency: "Response Latency (s)",
      tokenUsage: "Token Usage",
      cpuUsage: "CPU Usage (%)",
      gpuUsage: "GPU Usage (%)",
      memoryUsage: "Memory (MB)",
    };

    if (qEl && metrics.quality) {
      qEl.innerHTML = Object.entries(metrics.quality)
        .map(([k, v]) => {
          const avg = Math.round((v.qwen + v.gemma + v.smollm) / 3);
          return `<div class="metric-card"><span class="metric-label">${qualityLabels[k] || k}</span><span class="metric-val">${avg}%</span><span class="metric-demo-tag">Demo</span></div>`;
        })
        .join("");
    }

    if (pEl && metrics.performance) {
      pEl.innerHTML = Object.entries(metrics.performance)
        .map(([k, v]) => {
          const avg = ((v.qwen + v.gemma + v.smollm) / 3).toFixed(k === "latency" ? 1 : 0);
          return `<div class="metric-card"><span class="metric-label">${perfLabels[k] || k}</span><span class="metric-val">${avg}</span><span class="metric-demo-tag">Demo</span></div>`;
        })
        .join("");
    }

    if (tbody && metrics.quality && metrics.performance) {
      const rows = [];
      Object.entries({ ...metrics.quality, ...metrics.performance }).forEach(([k, v]) => {
        const label = qualityLabels[k] || perfLabels[k] || k;
        const suffix = k.includes("Rate") || k.includes("Usage") && k !== "latency" && k !== "tokenUsage" && k !== "memoryUsage" ? "%" : k === "latency" ? "s" : k === "memoryUsage" ? " MB" : k === "tokenUsage" ? "" : k.includes("accuracy") || k.includes("relevance") || k.includes("Quality") || k.includes("Pass") ? "%" : "";
        rows.push(`<tr><td>${label}</td><td>${v.qwen}${suffix}</td><td>${v.gemma}${suffix}</td><td>${v.smollm}${suffix}</td></tr>`);
      });
      tbody.innerHTML = rows.join("");
    }
  }

  /* ── Charts ── */
  function initCharts() {
    Chart.defaults.color = "#94a3b8";
    Chart.defaults.font.family = '"Outfit", "Inter", sans-serif';

    initQualityChart();
    initLatencyChart();
    initRadarChart();
    initResourceChart();
    initTradeoffChart();
  }

  function initQualityChart() {
    const ctx = document.getElementById("chartQualityBars");
    if (!ctx) return;
    charts.quality = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Accuracy", "Relevance", "Retrieval Quality", "Hallucination ↓", "Test Pass Rate"],
        datasets: [
          { label: "Qwen2.5 0.5B", data: [72, 68, 74, 18, 65], backgroundColor: "rgba(0,240,255,0.7)", borderRadius: 6 },
          { label: "Gemma 3 1B", data: [81, 79, 82, 12, 78], backgroundColor: "rgba(16,185,129,0.8)", borderRadius: 6 },
          { label: "SmolLM2 1.7B", data: [88, 85, 87, 8, 84], backgroundColor: "rgba(168,85,247,0.8)", borderRadius: 6 },
        ],
      },
      options: chartOpts("%"),
    });
  }

  function initLatencyChart() {
    const ctx = document.getElementById("chartLatencyBars");
    if (!ctx) return;
    charts.latency = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Qwen2.5 0.5B", "Gemma 3 1B", "SmolLM2 1.7B"],
        datasets: [{ label: "Latency (s)", data: [1.2, 2.8, 4.5], backgroundColor: ["rgba(0,240,255,0.7)", "rgba(16,185,129,0.7)", "rgba(168,85,247,0.8)"], borderRadius: 6 }],
      },
      options: chartOpts("s"),
    });
  }

  function initRadarChart() {
    const ctx = document.getElementById("chartRadarProfile");
    if (!ctx) return;
    charts.radar = new Chart(ctx, {
      type: "radar",
      data: {
        labels: ["Accuracy", "Speed", "Low Memory", "Retrieval", "Low Hallucination"],
        datasets: [
          { label: "Qwen2.5 0.5B", data: [72, 95, 90, 74, 70], borderColor: "#00f0ff", backgroundColor: "rgba(0,240,255,0.15)", borderWidth: 2 },
          { label: "Gemma 3 1B", data: [81, 70, 75, 82, 80], borderColor: "#10b981", backgroundColor: "rgba(16,185,129,0.15)", borderWidth: 2 },
          { label: "SmolLM2 1.7B", data: [88, 45, 50, 87, 92], borderColor: "#a855f7", backgroundColor: "rgba(168,85,247,0.15)", borderWidth: 2 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: { r: { angleLines: { color: "rgba(255,255,255,0.1)" }, grid: { color: "rgba(255,255,255,0.08)" }, pointLabels: { font: { size: 11 } }, ticks: { display: false, max: 100 } } },
        plugins: { legend: { position: "bottom", labels: { boxWidth: 12 } } },
      },
    });
  }

  function initResourceChart() {
    const ctx = document.getElementById("chartResourceBars");
    if (!ctx) return;
    charts.resource = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Qwen2.5 0.5B", "Gemma 3 1B", "SmolLM2 1.7B"],
        datasets: [
          { label: "Memory (MB)", data: [890, 1450, 2100], backgroundColor: "rgba(0,240,255,0.75)", borderRadius: 6 },
          { label: "CPU (%)", data: [22, 38, 55], backgroundColor: "rgba(168,85,247,0.75)", borderRadius: 6 },
        ],
      },
      options: chartOpts(""),
    });
  }

  function initTradeoffChart() {
    const ctx = document.getElementById("chartTradeoff");
    if (!ctx) return;
    charts.tradeoff = new Chart(ctx, {
      type: "scatter",
      data: {
        datasets: [
          { label: "Qwen2.5 0.5B", data: [{ x: 1.2, y: 72 }], backgroundColor: "#00f0ff", pointRadius: 10 },
          { label: "Gemma 3 1B", data: [{ x: 2.8, y: 81 }], backgroundColor: "#10b981", pointRadius: 10 },
          { label: "SmolLM2 1.7B", data: [{ x: 4.5, y: 88 }], backgroundColor: "#a855f7", pointRadius: 10 },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "Latency (s)" }, grid: { color: "rgba(255,255,255,0.06)" } },
          y: { title: { display: true, text: "Accuracy (%)" }, min: 60, max: 95, grid: { color: "rgba(255,255,255,0.06)" } },
        },
        plugins: { legend: { position: "bottom" } },
      },
    });
  }

  function chartOpts(suffix) {
    return {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, grid: { color: "rgba(255,255,255,0.06)" }, ticks: { callback: (v) => v + (suffix || "") } },
        x: { grid: { display: false } },
      },
      plugins: { legend: { position: "bottom", labels: { boxWidth: 12, padding: 8 } } },
    };
  }

  /* ── Trade-offs (W4 Task 4) ── */
  function renderTradeOffs() {
    const grid = document.getElementById("tradeoffGrid");
    const items = SVC.evaluationService?.getTradeOffs() || [];
    if (!grid) return;

    grid.innerHTML = items
      .map(
        (t) => `
      <article class="tradeoff-card">
        <span class="tradeoff-icon">${t.icon}</span>
        <h4>${escapeHtml(t.label)}</h4>
        <p class="tradeoff-val">${escapeHtml(t.value)}</p>
        <p class="tradeoff-detail">${escapeHtml(t.detail)} <span class="metric-demo-tag">Demo</span></p>
      </article>`
      )
      .join("");
  }

  /* ── RAG Analysis (W4 Task 5) ── */
  const TRACE_CASES = [
    {
      id: "MED-010",
      category: "Relevant Context",
      question: "I smashed my shin about 2 weeks ago and now feel a dent in the bone area. Was hoping just a deep bone bruise.",
      chunks: ["[C1] (Doc #10) The dent is in soft tissues, not bone. Soreness increases in evening due to swelling. Cold compresses and crepe bandage recommended. Recovery 3-6 weeks."],
      withRag: "Based on verified context, the dent is in soft tissues covering the shin bone, not the bone itself. Evening soreness is from accumulating swelling. Cold compresses and crepe bandage are recommended.",
      withoutRag: "A dent after trauma could indicate a depressed fracture. Obtain an X-ray urgently.",
      grounding: "Correct Answer",
      contextQuality: "Relevant Context",
      retrievalQuality: "High",
    },
    {
      id: "MED-022",
      category: "Irrelevant Context",
      question: "What is the recommended paediatric dose of amoxicillin for acute otitis media?",
      chunks: ["[C1] Adult amoxicillin 500mg TDS...", "[C2] Pediatric paracetamol syrup...", "[C3] Otitis externa ear drops..."],
      withRag: "Amoxicillin is commonly given at 40-45 mg/kg/day in divided doses.",
      withoutRag: "Pediatric dose is usually 80-90 mg/kg/day divided every 12 hours.",
      grounding: "Potential Hallucination",
      contextQuality: "Irrelevant Context",
      retrievalQuality: "Low",
    },
    {
      id: "MED-025",
      category: "Missing Information",
      question: "What is the interaction between apixaban and experimental drug ZX-417?",
      chunks: ["[C1] Apixaban anticoagulant indications. No data on ZX-417."],
      withRag: "The knowledge base does not contain information about ZX-417 interactions. Consult a clinical pharmacist.",
      withoutRag: "ZX-417 is unverified. Check medical literature.",
      grounding: "Correct Answer",
      contextQuality: "Missing Information",
      retrievalQuality: "Medium",
    },
  ];

  function initRagAnalysis() {
    const bar = document.getElementById("ragAnalysisSelector");
    if (!bar) return;

    bar.innerHTML = TRACE_CASES.map(
      (t, i) => `<button class="trace-btn ${i === 0 ? "active" : ""}" data-idx="${i}">${t.id}</button>`
    ).join("");

    bar.querySelectorAll(".trace-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        bar.querySelectorAll(".trace-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        displayRagAnalysis(TRACE_CASES[parseInt(btn.getAttribute("data-idx"), 10)]);
      });
    });

    displayRagAnalysis(TRACE_CASES[0]);
  }

  function displayRagAnalysis(trace) {
    const panel = document.getElementById("ragAnalysisPanel");
    if (!panel) return;

    const badgeClass = {
      "Correct Answer": "tag-medical",
      "Potential Hallucination": "tag-abstain",
      "Relevant Context": "tag-medical",
      "Irrelevant Context": "tag-abstain",
      "Missing Information": "tag-repo",
    };

    panel.innerHTML = `
      <div class="trace-meta-row">
        <span class="tag-badge tag-medical">${trace.id}</span>
        <span class="tag-badge ${badgeClass[trace.grounding] || "tag-medical"}">${trace.grounding}</span>
        <span class="tag-badge ${badgeClass[trace.contextQuality] || "tag-repo"}">${trace.contextQuality}</span>
      </div>
      <div class="trace-question-box"><span style="color:var(--text-muted);font-size:0.8rem;">QUESTION</span><br/>${escapeHtml(trace.question)}</div>
      <div class="pipeline-arrow" style="text-align:center;">↓</div>
      <div class="trace-chunks-box"><h5>RETRIEVED CONTEXT</h5>${trace.chunks.map((c) => `<div class="chunk-snippet">${escapeHtml(c)}</div>`).join("")}</div>
      <div class="pipeline-arrow" style="text-align:center;">↓</div>
      <div class="trace-comparison-grid">
        <div class="trace-col with-rag"><div class="trace-col-title">LLM Response (With-RAG)</div><div class="trace-text">${escapeHtml(trace.withRag)}</div></div>
        <div class="trace-col without-rag"><div class="trace-col-title">Baseline (Without-RAG)</div><div class="trace-text">${escapeHtml(trace.withoutRag)}</div></div>
      </div>
      <div class="pipeline-arrow" style="text-align:center;">↓</div>
      <div class="trace-conclusion"><strong>GROUNDING RESULT:</strong> ${trace.grounding} · Retrieval: ${trace.retrievalQuality} · Context: ${trace.contextQuality}</div>`;
  }

  /* ── Repository (W4 Task 6) ── */
  function renderRepoExplorer() {
    const el = document.getElementById("repoExplorer");
    const data = SVC.repositoryService?.getAnalysis() || {};
    if (!el) return;

    el.innerHTML = `
      <div class="repo-explorer-grid">
        <div class="repo-explorer-card"><h4>Files Involved</h4><ul>${(data.files || []).map((f) => `<li><code>${escapeHtml(f)}</code></li>`).join("")}</ul></div>
        <div class="repo-explorer-card"><h4>Components</h4><ul>${(data.components || []).map((c) => `<li>${escapeHtml(c)}</li>`).join("")}</ul></div>
        <div class="repo-explorer-card"><h4>Dependencies</h4><ul>${(data.dependencies || []).map((d) => `<li>${escapeHtml(d)}</li>`).join("")}</ul></div>
        <div class="repo-explorer-card"><h4>Function Calls</h4><ul>${(data.functionCalls || []).map((f) => `<li><code>${escapeHtml(f)}</code></li>`).join("")}</ul></div>
        <div class="repo-explorer-card"><h4>Impact Analysis</h4><ul>${(data.impactAnalysis || []).map((i) => `<li>${escapeHtml(i)}</li>`).join("")}</ul></div>
        <div class="repo-explorer-card"><h4>Related Tests</h4><ul>${(data.relatedTests || []).map((t) => `<li><code>${escapeHtml(t)}</code></li>`).join("")}</ul></div>
      </div>
      <h4 class="subsection-title">Sample Repository Questions</h4>
      <div class="repo-questions">${(data.sampleQuestions || []).map((q) => `<div class="repo-q-item">"${escapeHtml(q)}"</div>`).join("")}</div>`;
  }

  const REPO_PROBES = [
    { id: "REP-001", title: "End-to-End RAG Architecture", question: "Which files implement the medical RAG path from ingestion to generation?", files: ["src/rag_ingest.py", "src/rag_server.py", "src/ollama_client.py"], keyInsight: "Ingest → FAISS index → Server endpoints → LLM client." },
    { id: "REP-002", title: "RAG Compare Request Flow", question: "What happens after POST /api/v1/rag/compare?", files: ["src/rag_server.py", "dashboard/app.js"], keyInsight: "Embed → FAISS → Baseline + RAG generation → JSON response." },
    { id: "REP-003", title: "Configuration Dependencies", question: "Where are model and embedding settings configured?", files: ["config/week4_models.json", "src/ollama_client.py"], keyInsight: "Config files + environment variable overrides." },
    { id: "REP-004", title: "Chunking Change Impact", question: "If chunking strategy changes, what is affected?", files: ["src/rag_ingest.py", "outputs/rag_index/"], keyInsight: "Full re-index required; retrieval precision changes." },
    { id: "REP-005", title: "Docker Service Network", question: "Which services does Docker Compose start?", files: ["docker-compose.yml"], keyInsight: "Ollama, RAG API, Dashboard orchestration." },
    { id: "REP-006", title: "Frontend-Backend Contract", question: "Which frontend files call RAG endpoints?", files: ["dashboard/app.js", "dashboard/api/services.js"], keyInsight: "ragService.compare() → /api/v1/rag/compare." },
  ];

  function renderRepoProbes() {
    const grid = document.getElementById("repoProbesGrid");
    if (!grid) return;

    grid.innerHTML = REPO_PROBES.map(
      (p) => `
      <article class="repo-card">
        <div class="repo-card-head"><span class="tag-badge tag-repo">${p.id}</span></div>
        <h4 class="repo-q-title">${escapeHtml(p.title)}</h4>
        <p style="font-size:0.82rem;color:var(--text-muted);">${escapeHtml(p.question)}</p>
        <div class="repo-files-list">${p.files.map((f) => `<span class="file-chip">${escapeHtml(f)}</span>`).join("")}</div>
        <p style="font-size:0.8rem;color:#34d399;margin-top:0.5rem;"><strong>Insight:</strong> ${escapeHtml(p.keyInsight)}</p>
      </article>`
    ).join("");
  }

  /* ── W4 Summaries (Tasks 7 & 8) ── */
  function renderW4Summaries() {
    const routerEl = document.getElementById("w4RouterSummary");
    const guardEl = document.getElementById("w4GuardrailsSummary");

    if (routerEl) {
      routerEl.innerHTML = `
        <p class="section-desc">Evaluates difficulty-based routing across Easy (Qwen), Medium (Gemma), and Complex (SmolLM2) tiers.</p>
        <div class="router-models-row">${["Easy → Qwen2.5 0.5B", "Medium → Gemma 3 1B", "Complex → SmolLM2 1.7B"].map((t) => `<div class="router-model-chip">${t}</div>`).join("")}</div>
        <p class="placeholder-note">Routing evaluation results will populate when backend is connected.</p>`;
    }

    if (guardEl) {
      const checks = SVC.guardrailsService?.getChecks() || [];
      guardEl.innerHTML = `
        <p class="section-desc">Evaluates each guardrail stage against test queries for injection, safety, and grounding.</p>
        <div class="guardrail-cards">${checks.map((c) => `<article class="guardrail-card"><h4>${escapeHtml(c.name)}</h4><span class="status-badge status-pending">${c.status}</span></article>`).join("")}</div>
        <p class="placeholder-note">Guardrail evaluation metrics will populate when backend is connected.</p>`;
    }
  }

  /* ── Modal ── */
  function initModal() {
    const modal = document.getElementById("taskDetailModal");
    document.getElementById("modalCloseBtn")?.addEventListener("click", () => modal?.classList.remove("open"));
    modal?.addEventListener("click", (e) => { if (e.target === modal) modal.classList.remove("open"); });
  }

  /* ── Utils ── */
  function escapeHtml(str) {
    if (!str) return "";
    return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / 1048576).toFixed(1) + " MB";
  }
})();
