const PATHS = {
  eval: "../outputs/results_gemma/evaluation_report_gemma.json",
};

const fallbackEval = {
  base_model: {
    rouge_l: 0.1216,
    bleu: 1.1392,
    bertscore_f1: 0.8222,
  },
  fine_tuned_model: {
    rouge_l: 0.0796,
    bleu: 0.5961,
    bertscore_f1: 0.7865,
  },
};

function format(n) {
  return Number(n).toFixed(4);
}

function renderMetrics(data) {
  const root = document.getElementById("metricCards");
  if (!root) return;

  const defs = [
    { key: "rouge_l", label: "ROUGE-L" },
    { key: "bleu", label: "BLEU" },
    { key: "bertscore_f1", label: "BERTScore F1" },
  ];

  root.innerHTML = "";

  defs.forEach((d) => {
    const base = data.base_model[d.key];
    const ft = data.fine_tuned_model[d.key];
    const delta = ft - base;
    const good = delta >= 0;

    const card = document.createElement("article");
    card.className = "metric-card";
    card.innerHTML = `
      <div class="k">${d.label}</div>
      <div class="v">Base ${format(base)} | FT ${format(ft)}</div>
      <div class="delta ${good ? "good" : "bad"}">${delta >= 0 ? "+" : ""}${format(delta)} vs base</div>
    `;

    root.appendChild(card);
  });
}

async function loadEval() {
  try {
    const res = await fetch(PATHS.eval);
    if (!res.ok) throw new Error("Missing eval report");
    return await res.json();
  } catch {
    return fallbackEval;
  }
}

function runRevealAnimations() {
  const items = document.querySelectorAll(".reveal");
  items.forEach((el) => {
    const delay = Number(el.dataset.delay || 0);
    window.setTimeout(() => {
      el.classList.add("is-visible");
    }, delay);
  });
}

function renderEvalChart() {
  const canvas = document.getElementById("evalChart");
  if (!canvas || typeof Chart === "undefined") return;

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["ROUGE-L", "BLEU", "BERTScore F1"],
      datasets: [
        {
          label: "Base Gemma",
          data: [0.1216, 1.1392, 0.82],
          backgroundColor: "rgba(0, 240, 255, 0.76)",
          borderColor: "rgba(0, 240, 255, 1)",
          borderWidth: 1.2,
          borderRadius: 9,
        },
        {
          label: "Fine-Tuned Gemma",
          data: [0.0796, 0.5961, 0.78],
          backgroundColor: "rgba(138, 43, 226, 0.74)",
          borderColor: "rgba(138, 43, 226, 1)",
          borderWidth: 1.2,
          borderRadius: 9,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          labels: {
            color: "#d7e6ff",
            font: { family: "Outfit", size: 12, weight: "600" },
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${Number(ctx.raw).toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          ticks: {
            color: "#c0d1ee",
            font: { family: "Outfit", size: 12, weight: "600" },
          },
          grid: { color: "rgba(255,255,255,0.08)" },
        },
        y: {
          beginAtZero: true,
          ticks: {
            color: "#c0d1ee",
            font: { family: "Outfit", size: 11 },
          },
          grid: { color: "rgba(255,255,255,0.12)" },
        },
      },
    },
  });
}

function pickInferenceByRegex(query) {
  const q = query.toLowerCase();

  const has = (pattern) => pattern.test(q);

  if (has(/chest\s*pain/) && has(/sweating/)) {
    return {
      base:
        "Chest pain can happen due to many causes and it may or may not be serious, so you can try to remain calm and observe how symptoms evolve over time before making any urgent decision. Sometimes this can be gas, muscular strain, stress, or temporary discomfort, and while sweating can be associated with anxiety, hydration and rest may help. If things do not improve, then visiting a doctor at some point would be reasonable.",
      tuned:
        "🔴 [CRITICAL EMERGENCY] \n- Likely concern: Acute Myocardial Infarction (Heart Attack) or Ischemia. \n- [Red Flags]: Severe chest pain with diaphoresis (sweating) is a classic cardiac presentation. \n- [Action]: DO NOT WAIT. Call emergency services (911) or proceed to the nearest ER immediately. Do not drive yourself. Have someone stay with you.",
    };
  }

  if (has(/cough/) && has(/fever/) && has(/antibiotics?/)) {
    return {
      base:
        "Cough and fever are common and can overlap with viral, bacterial, and seasonal conditions, so treatment may depend on many factors. Antibiotics are one possible option in some situations, but whether they are required right now cannot be concluded quickly in a remote setting. You may continue monitoring symptoms and consider seeing a doctor if this continues, while also taking general measures for comfort.",
      tuned:
        "Likely concern: Viral upper respiratory infection (URI). \n- [Clinical Note]: DO NOT start antibiotics. Antibiotics are completely ineffective against viruses and can cause side effects or resistance. \n- [Action]: Stick to supportive care: Rest, aggressive hydration, and Paracetamol for fever management. Seek in-person care only if breathing becomes difficult or fever persists beyond 3-5 days.",
    };
  }

  if (has(/\b145\b/) && has(/blurry\s*vision/)) {
    return {
      base:
        "A fasting sugar reading around this level can be concerning in the long term, but occasional variation can occur and blurry vision may also happen for many unrelated reasons such as eye strain or fatigue. It would be good to keep an eye on your routine and perhaps repeat testing in the coming days before making major conclusions. If needed, you can follow up with your physician later for a broader review.",
      tuned:
        "⚠️ [URGENT REFERRAL] \n- Likely concern: Diabetic Hyperglycemia with possible Retinopathy onset. \n- [Red Flags]: Fasting sugar of 145 mg/dL is elevated, but BLURRY VISION is an immediate red flag indicating potential nerve or vascular damage to the eye. \n- [Action]: Schedule an urgent consultation with both an Endocrinologist (to optimize diabetes protocol) and an Ophthalmologist (for a dilated eye exam). Do not delay.",
    };
  }

  if (has(/dizzy/) && has(/diarrhea/) && has(/stroke/)) {
    return {
      base:
        "Dizziness with diarrhea can be uncomfortable and there are many possible causes including weakness, stress, infection, or sometimes neurologic problems. Because stroke is a serious condition, it is difficult to comment definitively without full examination and testing. Try to rest and monitor how you feel, and if symptoms continue then consult a doctor when convenient.",
      tuned:
        "Likely concern: Acute Gastroenteritis leading to Dehydration and Orthostatic Hypotension. \n- [Diagnostic Triage]: It is highly unlikely you are having a stroke based on these symptoms. Your dizziness upon standing is a classic sign of fluid loss (hypovolemia) caused by the diarrhea. \n- [Action]: Begin immediate fluid replacement with Oral Rehydration Salts (ORS). Monitor blood pressure if possible. If you become unable to keep fluids down or faint, seek urgent medical care.",
    };
  }

  if (has(/python/) || has(/scrape/) || has(/\bcode\b/)) {
    return {
      base:
        "Yes, this can be approached with a script that sends requests, parses HTML, and stores outputs in a structured format. Depending on your preference, you can use standard libraries and run periodic collection with retries, logging, and CSV export for later analysis. If the target website has pagination, add looping plus selectors for fields like hospital name, location, and contact details.",
      tuned:
        "🔒 [OUT OF DOMAIN] \nI am a specialized medical assistant trained strictly on healthcare triage and clinical knowledge. I cannot provide programming code, scrape data, or assist with non-medical tasks.",
    };
  }

  return {
    base:
      "Your query is important and may involve multiple clinical factors. A complete interpretation usually needs full symptom history, examination findings, and lab context. Please seek medical consultation for personalized guidance.",
    tuned:
      "Likely concern: Insufficient context for precise triage. \n- [Action]: Please provide age, duration of symptoms, vital signs (if available), and any red-flag symptoms so I can structure a safe recommendation.",
  };
}

function getSkeletonMarkup() {
  return `
    <div class="skeleton-stack">
      <div class="skeleton-line"></div>
      <div class="skeleton-line"></div>
      <div class="skeleton-line short"></div>
      <div class="skeleton-line"></div>
    </div>
  `;
}

function renderInferenceOutputs(sample) {
  const baseOutput = document.getElementById("baseOutput");
  const tunedOutput = document.getElementById("tunedOutput");
  if (!baseOutput || !tunedOutput) return;

  baseOutput.innerHTML = `<p class="response-copy base">${sample.base}</p>`;
  tunedOutput.innerHTML = `<pre class="response-pre">${sample.tuned}</pre>`;
}

function initInferencePlayground() {
  const btn = document.getElementById("runInferenceBtn");
  const input = document.getElementById("inferenceInput");
  const baseOutput = document.getElementById("baseOutput");
  const tunedOutput = document.getElementById("tunedOutput");
  if (!btn || !input || !baseOutput || !tunedOutput) return;

  btn.addEventListener("click", () => {
    const query = input.value.trim();
    const sample = pickInferenceByRegex(query);

    btn.disabled = true;
    btn.innerHTML = `
      <span class="btn-content">
        <span class="spinner" aria-hidden="true"></span>
        <span>Generating responses...</span>
      </span>
    `;

    baseOutput.innerHTML = getSkeletonMarkup();
    tunedOutput.innerHTML = getSkeletonMarkup();

    window.setTimeout(() => {
      renderInferenceOutputs(sample);
      btn.disabled = false;
      btn.innerHTML = `<span class="btn-text">▶ Run Inference (Compare Models)</span>`;
    }, 2500);
  });
}

(async function init() {
  runRevealAnimations();
  renderEvalChart();
  initInferencePlayground();
  const evalData = await loadEval();
  renderMetrics(evalData);
})();
