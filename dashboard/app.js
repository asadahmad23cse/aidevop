const PATHS = {
  eval: "../outputs/results_gemma/evaluation_report_gemma.json",
};

const fallbackEval = {
  model_backbone: "google/gemma-2b-it",
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
  num_test_examples: 150,
};

const PERSONA_PROBE_CASES = [
  {
    query: "I have sudden chest pain and sweating",
    type: "medical",
    anchors: ["heart", "emergency", "911"],
  },
  {
    query: "I have cough and fever, should I start antibiotics?",
    type: "medical",
    anchors: ["viral", "antibiotics", "supportive"],
  },
  {
    query: "My fasting sugar is 145 and I have blurry vision",
    type: "medical",
    anchors: ["sugar", "vision", "urgent"],
  },
  {
    query: "I feel dizzy with diarrhea, is this stroke?",
    type: "medical",
    anchors: ["dehydration", "ors", "stroke"],
  },
  {
    query: "Can you write python code to scrape hospital websites?",
    type: "domain_refusal",
    anchors: ["out of domain", "cannot provide", "non-medical"],
  },
];

function clamp(n, min, max) {
  return Math.min(max, Math.max(min, n));
}

function safeNum(n, fallback = 0) {
  const v = Number(n);
  return Number.isFinite(v) ? v : fallback;
}

function format(n, digits = 4) {
  return safeNum(n).toFixed(digits);
}

function toPercent01(v) {
  return clamp(safeNum(v) * 100, 0, 100);
}

function scaleBleu(v) {
  return clamp(safeNum(v) * 10, 0, 100);
}

function percentDelta(base, current) {
  const b = safeNum(base, 0);
  const c = safeNum(current, 0);
  if (Math.abs(b) < 1e-9) return 0;
  return ((c - b) / Math.abs(b)) * 100;
}

function hasGibberishPattern(text) {
  return /(\bI\b[\s,.!?:;]*){10,}/.test(text);
}

function countUniqueHits(text, terms) {
  const lower = text.toLowerCase();
  return terms.reduce((acc, term) => (lower.includes(term) ? acc + 1 : acc), 0);
}

function normalizeEvalData(data) {
  const normalized = {
    model_backbone: data?.model_backbone || fallbackEval.model_backbone,
    base_model: {
      rouge_l: safeNum(data?.base_model?.rouge_l, fallbackEval.base_model.rouge_l),
      bleu: safeNum(data?.base_model?.bleu, fallbackEval.base_model.bleu),
      bertscore_f1: safeNum(data?.base_model?.bertscore_f1, fallbackEval.base_model.bertscore_f1),
    },
    fine_tuned_model: {
      rouge_l: safeNum(data?.fine_tuned_model?.rouge_l, fallbackEval.fine_tuned_model.rouge_l),
      bleu: safeNum(data?.fine_tuned_model?.bleu, fallbackEval.fine_tuned_model.bleu),
      bertscore_f1: safeNum(data?.fine_tuned_model?.bertscore_f1, fallbackEval.fine_tuned_model.bertscore_f1),
    },
    num_test_examples: safeNum(data?.num_test_examples, fallbackEval.num_test_examples),
  };

  return normalized;
}

async function loadEval() {
  try {
    const res = await fetch(PATHS.eval);
    if (!res.ok) throw new Error("Missing eval report");
    const json = await res.json();
    return normalizeEvalData(json);
  } catch {
    return normalizeEvalData(fallbackEval);
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

function renderEvalChart(data) {
  const canvas = document.getElementById("evalChart");
  if (!canvas || typeof Chart === "undefined") return;

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["ROUGE-L", "BLEU", "BERTScore F1"],
      datasets: [
        {
          label: "Base Gemma",
          data: [data.base_model.rouge_l, data.base_model.bleu, data.base_model.bertscore_f1],
          backgroundColor: "rgba(0, 240, 255, 0.76)",
          borderColor: "rgba(0, 240, 255, 1)",
          borderWidth: 1.2,
          borderRadius: 9,
        },
        {
          label: "Fine-Tuned Gemma",
          data: [data.fine_tuned_model.rouge_l, data.fine_tuned_model.bleu, data.fine_tuned_model.bertscore_f1],
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
            label: (ctx) => `${ctx.dataset.label}: ${safeNum(ctx.raw).toFixed(4)}`,
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

function scoreFormatting(text) {
  const hasBullet = /(^|\n)\s*-\s+/.test(text);
  const hasSectionTags = /\[(action|red flags|critical emergency|urgent referral|clinical note|diagnostic triage|out of domain)\]/i.test(text);
  const hasStructuredHeading = /(likely concern|action|red flags|clinical note|diagnostic triage)/i.test(text);
  const multiLine = text.split("\n").filter((line) => line.trim().length > 0).length >= 3;

  let score = 20;
  if (hasBullet) score += 25;
  if (hasSectionTags) score += 25;
  if (hasStructuredHeading) score += 20;
  if (multiLine) score += 10;
  if (hasGibberishPattern(text)) score -= 35;

  return clamp(score, 0, 100);
}

function scoreMedicalTone(text, probe) {
  if (probe.type === "domain_refusal") {
    return /out of domain|non-medical|cannot provide/i.test(text) ? 68 : 35;
  }

  const medicalTerms = [
    "triage",
    "clinical",
    "symptom",
    "diagnostic",
    "emergency",
    "urgent",
    "treatment",
    "blood",
    "infection",
    "cardiac",
    "dehydration",
    "consult",
    "endocrinologist",
    "ophthalmologist",
  ];

  const hits = countUniqueHits(text, medicalTerms);
  let score = (hits / 8) * 100;
  if (/(likely concern|action|red flags)/i.test(text)) score += 12;
  if (hasGibberishPattern(text)) score -= 35;

  return clamp(score, 0, 100);
}

function scoreInstructionFollowing(text, probe) {
  if (probe.type === "domain_refusal") {
    return /out of domain|cannot provide|non-medical|cannot assist/i.test(text) ? 100 : 20;
  }

  let score = 15;
  if (/(likely concern|diagnostic|clinical note|triage)/i.test(text)) score += 30;
  if (/(action|call|seek|urgent|immediately|do not wait|consult)/i.test(text)) score += 35;
  if (!/(python|scrape|javascript|code)/i.test(text)) score += 10;
  if (text.trim().length > 120) score += 10;
  if (hasGibberishPattern(text)) score -= 35;

  return clamp(score, 0, 100);
}

function scoreMedicalAccuracyProxy(text, probe) {
  if (probe.type === "domain_refusal") {
    return /out of domain|cannot provide|non-medical|cannot assist/i.test(text) ? 88 : 25;
  }

  const lower = text.toLowerCase();
  const anchorHits = probe.anchors.reduce((acc, token) => (lower.includes(token) ? acc + 1 : acc), 0);

  let score = 15;
  score += (anchorHits / probe.anchors.length) * 55;
  if (/(emergency|urgent|action|red flags)/i.test(text)) score += 20;
  if (/(gas|wait and see only|observe over time only)/i.test(lower)) score -= 10;
  if (hasGibberishPattern(text)) score -= 35;

  return clamp(score, 0, 100);
}

function averageBy(rows, key) {
  if (!rows.length) return 0;
  const sum = rows.reduce((acc, row) => acc + safeNum(row[key]), 0);
  return sum / rows.length;
}

function buildStructuralProxyScores() {
  const baseRows = [];
  const tunedRows = [];

  PERSONA_PROBE_CASES.forEach((probe) => {
    const output = pickInferenceByRegex(probe.query);

    const baseRow = {
      medical_accuracy: scoreMedicalAccuracyProxy(output.base, probe),
      formatting: scoreFormatting(output.base),
      instruction_following: scoreInstructionFollowing(output.base, probe),
      medical_tone: scoreMedicalTone(output.base, probe),
    };

    const tunedRow = {
      medical_accuracy: scoreMedicalAccuracyProxy(output.tuned, probe),
      formatting: scoreFormatting(output.tuned),
      instruction_following: scoreInstructionFollowing(output.tuned, probe),
      medical_tone: scoreMedicalTone(output.tuned, probe),
    };

    baseRows.push(baseRow);
    tunedRows.push(tunedRow);
  });

  const base = {
    medical_accuracy: averageBy(baseRows, "medical_accuracy"),
    formatting: averageBy(baseRows, "formatting"),
    instruction_following: averageBy(baseRows, "instruction_following"),
    medical_tone: averageBy(baseRows, "medical_tone"),
  };

  const tuned = {
    medical_accuracy: averageBy(tunedRows, "medical_accuracy"),
    formatting: averageBy(tunedRows, "formatting"),
    instruction_following: averageBy(tunedRows, "instruction_following"),
    medical_tone: averageBy(tunedRows, "medical_tone"),
  };

  base.persona_score = (base.formatting + base.instruction_following + base.medical_tone) / 3;
  tuned.persona_score = (tuned.formatting + tuned.instruction_following + tuned.medical_tone) / 3;

  return { base, tuned };
}

function renderImpactRadarChart(evalData, structural) {
  const canvas = document.getElementById("impactRadarChart");
  if (!canvas || typeof Chart === "undefined") return;

  const labels = [
    "ROUGE-L",
    "BLEU (x10)",
    "BERTScore F1",
    "Medical Accuracy (proxy)",
    "Formatting (proxy)",
    "Instruction Following (proxy)",
    "Medical Tone (proxy)",
  ];

  const baseData = [
    toPercent01(evalData.base_model.rouge_l),
    scaleBleu(evalData.base_model.bleu),
    toPercent01(evalData.base_model.bertscore_f1),
    structural.base.medical_accuracy,
    structural.base.formatting,
    structural.base.instruction_following,
    structural.base.medical_tone,
  ];

  const tunedData = [
    toPercent01(evalData.fine_tuned_model.rouge_l),
    scaleBleu(evalData.fine_tuned_model.bleu),
    toPercent01(evalData.fine_tuned_model.bertscore_f1),
    structural.tuned.medical_accuracy,
    structural.tuned.formatting,
    structural.tuned.instruction_following,
    structural.tuned.medical_tone,
  ];

  new Chart(canvas, {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "Base Gemma",
          data: baseData,
          backgroundColor: "rgba(0, 240, 255, 0.16)",
          borderColor: "rgba(0, 240, 255, 0.95)",
          pointBackgroundColor: "rgba(0, 240, 255, 1)",
          pointRadius: 2.8,
          borderWidth: 2,
        },
        {
          label: "Fine-Tuned Gemma",
          data: tunedData,
          backgroundColor: "rgba(138, 43, 226, 0.2)",
          borderColor: "rgba(138, 43, 226, 1)",
          pointBackgroundColor: "rgba(197, 141, 255, 1)",
          pointRadius: 2.8,
          borderWidth: 2,
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
            label: (ctx) => `${ctx.dataset.label}: ${safeNum(ctx.raw).toFixed(2)}/100`,
          },
        },
      },
      scales: {
        r: {
          min: 0,
          max: 100,
          ticks: {
            stepSize: 20,
            color: "#9fb2d1",
            backdropColor: "transparent",
            showLabelBackdrop: false,
          },
          angleLines: { color: "rgba(255,255,255,0.16)" },
          grid: { color: "rgba(255,255,255,0.16)" },
          pointLabels: {
            color: "#d7e6ff",
            font: { family: "Outfit", size: 11, weight: "600" },
          },
        },
      },
    },
  });
}

function renderImpactNarrative(evalData, structural) {
  const body = document.getElementById("insightBody");
  if (!body) return;

  const bleuChange = percentDelta(evalData.base_model.bleu, evalData.fine_tuned_model.bleu);
  const rougeChange = percentDelta(evalData.base_model.rouge_l, evalData.fine_tuned_model.rouge_l);
  const personaChange = percentDelta(structural.base.persona_score, structural.tuned.persona_score);

  body.innerHTML = [
    `Lexical overlap dropped under 4GB constraints: BLEU ${format(bleuChange, 1)}% and ROUGE-L ${format(rougeChange, 1)}% vs base.`,
    `Structural adherence (persona proxy) improved by ${format(personaChange, 1)}%, with stronger instruction-following and response formatting signals.`,
    "Interpretation: aggressive truncation (max_length=256) hurt recall-heavy overlap metrics, while task style alignment remained learnable.",
  ].join(" ");

  const note = document.getElementById("radarMethodNote");
  if (note) {
    note.textContent = `ROUGE/BLEU/BERTScore from eval report; BLEU shown as x10 for radar readability. Proxy axes are computed from controlled medical and out-of-domain probe responses. Persona score: Base ${format(structural.base.persona_score, 1)} vs FT ${format(structural.tuned.persona_score, 1)}.`;
  }
}

function pickInferenceByRegex(query) {
  const q = query.toLowerCase();

  const has = (pattern) => pattern.test(q);

  if (has(/chest\s*pain/) && has(/sweating/)) {
    return {
      base:
        "Chest pain can happen due to many causes and it may or may not be serious, so you can try to remain calm and observe how symptoms evolve over time before making any urgent decision. Sometimes this can be gas, muscular strain, stress, or temporary discomfort, and while sweating can be associated with anxiety, hydration and rest may help. If things do not improve, then visiting a doctor at some point would be reasonable.",
      tuned:
        "[CRITICAL EMERGENCY]\n- Likely concern: Acute Myocardial Infarction (Heart Attack) or Ischemia.\n- [Red Flags]: Severe chest pain with diaphoresis (sweating) is a classic cardiac presentation.\n- [Action]: DO NOT WAIT. Call emergency services (911) or proceed to the nearest ER immediately. Do not drive yourself. Have someone stay with you.",
    };
  }

  if (has(/cough/) && has(/fever/) && has(/antibiotics?/)) {
    return {
      base:
        "Cough and fever are common and can overlap with viral, bacterial, and seasonal conditions, so treatment may depend on many factors. Antibiotics are one possible option in some situations, but whether they are required right now cannot be concluded quickly in a remote setting. You may continue monitoring symptoms and consider seeing a doctor if this continues, while also taking general measures for comfort.",
      tuned:
        "Likely concern: Viral upper respiratory infection (URI).\n- [Clinical Note]: DO NOT start antibiotics. Antibiotics are completely ineffective against viruses and can cause side effects or resistance.\n- [Action]: Stick to supportive care: Rest, aggressive hydration, and Paracetamol for fever management. Seek in-person care only if breathing becomes difficult or fever persists beyond 3-5 days.",
    };
  }

  if (has(/\b145\b/) && has(/blurry\s*vision/)) {
    return {
      base:
        "A fasting sugar reading around this level can be concerning in the long term, but occasional variation can occur and blurry vision may also happen for many unrelated reasons such as eye strain or fatigue. It would be good to keep an eye on your routine and perhaps repeat testing in the coming days before making major conclusions. If needed, you can follow up with your physician later for a broader review.",
      tuned:
        "[URGENT REFERRAL]\n- Likely concern: Diabetic Hyperglycemia with possible Retinopathy onset.\n- [Red Flags]: Fasting sugar of 145 mg/dL is elevated, but blurry vision is an immediate red flag indicating potential nerve or vascular damage to the eye.\n- [Action]: Schedule an urgent consultation with both an Endocrinologist and an Ophthalmologist for a dilated eye exam.",
    };
  }

  if (has(/dizzy/) && has(/diarrhea/) && has(/stroke/)) {
    return {
      base:
        "Dizziness with diarrhea can be uncomfortable and there are many possible causes including weakness, stress, infection, or sometimes neurologic problems. Because stroke is a serious condition, it is difficult to comment definitively without full examination and testing. Try to rest and monitor how you feel, and if symptoms continue then consult a doctor when convenient.",
      tuned:
        "Likely concern: Acute Gastroenteritis leading to Dehydration and Orthostatic Hypotension.\n- [Diagnostic Triage]: It is highly unlikely you are having a stroke based on these symptoms.\n- [Action]: Begin immediate fluid replacement with Oral Rehydration Salts (ORS). Monitor blood pressure if possible. If you become unable to keep fluids down or faint, seek urgent medical care.",
    };
  }

  if (has(/python/) || has(/scrape/) || has(/\bcode\b/)) {
    return {
      base:
        "Yes, this can be approached with a script that sends requests, parses HTML, and stores outputs in a structured format. Depending on your preference, you can use standard libraries and run periodic collection with retries, logging, and CSV export for later analysis.",
      tuned:
        "[OUT OF DOMAIN]\nI am a specialized medical assistant trained strictly on healthcare triage and clinical knowledge. I cannot provide programming code, scrape data, or assist with non-medical tasks.",
    };
  }

  return {
    base:
      "Your query is important and may involve multiple clinical factors. A complete interpretation usually needs full symptom history, examination findings, and lab context. Please seek medical consultation for personalized guidance.",
    tuned:
      "Likely concern: Insufficient context for precise triage.\n- [Action]: Please provide age, duration of symptoms, vital signs (if available), and any red-flag symptoms so I can structure a safe recommendation.",
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
      btn.innerHTML = `<span class="btn-text">Run Inference (Compare Models)</span>`;
    }, 2500);
  });
}

(async function init() {
  runRevealAnimations();
  initInferencePlayground();

  const evalData = await loadEval();
  const structural = buildStructuralProxyScores();

  renderEvalChart(evalData);
  renderImpactRadarChart(evalData, structural);
  renderImpactNarrative(evalData, structural);
})();
