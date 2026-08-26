"use strict";

const API = window.AIDEVOP_API_URL ||
  (((location.hostname === "localhost" || location.hostname === "127.0.0.1") && location.port === "8000")
    ? "http://localhost:8001"
    : "");

const $ = (selector) => document.querySelector(selector);

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[char]);
}

function percent(value, digits = 1) {
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function seconds(milliseconds) {
  return `${(Number(milliseconds) / 1000).toFixed(2)} s`;
}

function megabytes(value) {
  return `${Math.round(Number(value)).toLocaleString()} MB`;
}

async function api(path, options = {}) {
  const response = await fetch(`${API}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
  }
  return response.json();
}

async function loadStatus() {
  const pill = $("#servicePill");
  try {
    const status = await api("/status");
    pill.classList.add("online");
    pill.querySelector("b").textContent = "Service ready";
    $("#heroChunks").textContent = Number(status.chunks).toLocaleString();
    $("#heroBackend").textContent = status.services.retrieval_rag_service.backend || "semantic";
    $("#heroFallback").textContent = status.ready ? "Ready" : "Unavailable";
  } catch (error) {
    pill.classList.add("offline");
    pill.querySelector("b").textContent = "Service unavailable";
    $("#heroChunks").textContent = "Unavailable";
    $("#heroBackend").textContent = "—";
    $("#heroFallback").textContent = "Offline";
  }
}

function renderModels(data) {
  const cards = $("#modelCards");
  const rows = $("#metricsRows");
  const bestCorrectness = Math.max(...data.models.map((model) => model.correctness));
  const lowestLatency = Math.min(...data.models.map((model) => model.latency_ms_mean));
  const lowestHallucination = Math.min(...data.models.map((model) => model.hallucination_rate));

  cards.innerHTML = data.models.map((model) => {
    const leader = model.correctness === bestCorrectness;
    const label = leader ? "Quality leader" : model.latency_ms_mean === lowestLatency ? "Latency leader" : "Repository leader";
    return `<article class="model-card ${leader ? "leader" : ""}">
      <div class="model-top"><h3>${escapeHTML(model.name)}</h3><span>${label}</span></div>
      <div class="score-block">
        <div><span>Medical correctness</span><strong>${percent(model.correctness)}</strong></div>
        <div class="score-track"><i style="width:${model.correctness * 100}%"></i></div>
      </div>
      <div class="model-mini">
        <div><span>Mean latency</span><strong>${seconds(model.latency_ms_mean)}</strong></div>
        <div><span>Hallucination</span><strong>${percent(model.hallucination_rate)}</strong></div>
        <div><span>Relevance</span><strong>${percent(model.relevance)}</strong></div>
        <div><span>Recall@3</span><strong>${Number(model.retrieval_recall_at_3).toFixed(3)}</strong></div>
      </div>
    </article>`;
  }).join("");

  rows.innerHTML = data.models.map((model) => `<tr>
    <td><b>${escapeHTML(model.name)}</b></td>
    <td class="${model.correctness === bestCorrectness ? "metric-good" : ""}">${percent(model.correctness)}</td>
    <td>${percent(model.relevance)}</td>
    <td class="${model.hallucination_rate === lowestHallucination ? "metric-good" : ""}">${percent(model.hallucination_rate)}</td>
    <td class="${model.latency_ms_mean === lowestLatency ? "metric-good" : ""}">${seconds(model.latency_ms_mean)}</td>
    <td>${Number(model.total_tokens).toLocaleString()}</td>
    <td>${megabytes(model.vram_mb_mean)}</td>
  </tr>`).join("");

  $("#testPassNote").textContent = data.test_pass_rate_note;
  const validity = data.validity;
  $("#validityStrip").innerHTML = `
    <div><strong>${validity.actual_records} / ${validity.expected_records}</strong><span>medical runs complete</span></div>
    <div><strong>${validity.models.length}</strong><span>models compared</span></div>
    <div><strong>${validity.tasks.length}</strong><span>shared questions</span></div>
    <div><strong>${validity.missing_pairs.length}</strong><span>missing model-task pairs</span></div>`;
}

async function loadWeek4Results() {
  try {
    renderModels(await api("/api/v1/week4/results"));
  } catch (error) {
    $("#modelCards").innerHTML = `<div class="loading-card">Evaluation evidence could not be loaded from the service.</div>`;
    $("#metricsRows").innerHTML = `<tr><td colspan="7">Evidence endpoint unavailable.</td></tr>`;
  }
}

function renderObservability(data) {
  $("#heroVersion").textContent = data.deployment_version || "local";
  $("#obsRequests").textContent = Number(data.requests).toLocaleString();
  $("#obsLatency").textContent = `${Number(data.avg_latency_ms).toLocaleString()} ms`;
  $("#obsFallback").textContent = percent(data.fallback_rate, 0);
  $("#obsSafety").textContent = percent(data.safety_flag_rate, 0);

  const routeEntries = Object.entries(data.routes || {});
  const total = Math.max(1, Number(data.requests));
  $("#routeBars").innerHTML = routeEntries.length ? routeEntries.map(([route, count]) => `<div class="route-bar">
    <div><span>${escapeHTML(route.replaceAll("_", " "))}</span><b>${count}</b></div>
    <div class="route-bar-track"><i style="width:${(count / total) * 100}%"></i></div>
  </div>`).join("") : "<p>No live requests yet. Use the safety router above.</p>";

  $("#eventList").innerHTML = data.recent?.length ? data.recent.map((event) => `<div class="event-row">
    <span>${escapeHTML(event.risk_level)}</span>
    <b>${escapeHTML(event.route.replaceAll("_", " "))}</b>
    <small>${Number(event.latency_ms).toLocaleString()} ms</small>
  </div>`).join("") : "<p>No live requests yet. Use the safety router above.</p>";
}

async function loadObservability() {
  try {
    renderObservability(await api("/api/v1/observability"));
  } catch (error) {
    $("#heroVersion").textContent = "Unavailable";
  }
}

function renderResponse(data) {
  $("#responseEmpty").hidden = true;
  $("#responseContent").hidden = false;
  const level = data.risk.level;
  const riskBadge = $("#riskBadge");
  riskBadge.className = `risk-${level}`;
  riskBadge.textContent = `${level} risk`;
  $("#routeBadge").textContent = data.routing.selected_route.replaceAll("_", " ");
  $("#latencyBadge").textContent = `${Number(data.latency_ms).toLocaleString()} ms`;
  $("#answerText").textContent = data.answer;

  const sourceList = $("#sourceList");
  sourceList.innerHTML = data.retrieved_chunks.length ? data.retrieved_chunks.map((source, index) => `<div class="source-item">
    <b>Evidence ${index + 1} · score ${Number(source.score).toFixed(4)}</b><br />${escapeHTML(source.text.slice(0, 340))}${source.text.length > 340 ? "…" : ""}
  </div>`).join("") : `<div class="source-item"><b>No model evidence used.</b><br />The emergency safety policy blocked generation and returned deterministic urgent-care guidance.</div>`;
}

function initPlayground() {
  document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => {
      $("#question").value = button.dataset.question;
      $("#question").focus();
    });
  });

  $("#sourceToggle").addEventListener("click", () => {
    const list = $("#sourceList");
    const expanded = list.hidden;
    list.hidden = !expanded;
    $("#sourceToggle").setAttribute("aria-expanded", String(expanded));
    $("#sourceToggle span").textContent = expanded ? "−" : "+";
  });

  $("#askForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const query = $("#question").value.trim();
    if (!query) return;
    const button = $("#askButton");
    button.textContent = "Evaluating risk and evidence…";
    button.classList.add("is-loading");
    try {
      const data = await api("/api/v1/pulsemirror/ask", {
        method: "POST",
        body: JSON.stringify({ query, top_k: 3 }),
      });
      renderResponse(data);
      await loadObservability();
    } catch (error) {
      $("#responseEmpty").hidden = false;
      $("#responseEmpty h3").textContent = "Request could not be completed";
      $("#responseEmpty p").textContent = "The service may be waking up. Please try again in a moment.";
      $("#responseContent").hidden = true;
    } finally {
      button.textContent = "Run risk-aware pipeline";
      button.classList.remove("is-loading");
    }
  });

  $("#refreshOps").addEventListener("click", loadObservability);
}

async function init() {
  initPlayground();
  await Promise.all([loadStatus(), loadWeek4Results(), loadObservability()]);
}

init();
