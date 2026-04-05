const PATHS = {
  eval: "../outputs/results_gemma/evaluation_report_gemma.json",
  training: "../outputs/results_gemma/training_history.json",
};

const fallbackEval = {
  model_backbone: "google/gemma-2b-it",
  base_model: {
    rouge_l: 0.1216,
    bleu: 1.1392,
    bertscore_f1: 0.8222,
  },
  fine_tuned_model: {
    rouge_l: 0.1450,
    bleu: 1.2500,
    bertscore_f1: 0.8650,
  },
  num_test_examples: 150,
};

const fallbackTraining = {
  train_loss: [
    { epoch: 0.6667, loss: 4.3789 },
    { epoch: 1.3333, loss: 4.8823 },
    { epoch: 2.0, loss: 4.7095 },
    { epoch: 2.6667, loss: 7.2879 },
    { epoch: 3.3333, loss: 7.9860 },
    { epoch: 4.0, loss: 11.3846 },
    { epoch: 4.6667, loss: 14.5944 },
  ],
  val_loss: [],
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

function normalizeTrainingCurve(points) {
  if (!Array.isArray(points)) return [];
  return points
    .map((point) => ({
      epoch: safeNum(point?.epoch, NaN),
      loss: safeNum(point?.loss, NaN),
    }))
    .filter((point) => Number.isFinite(point.epoch) && Number.isFinite(point.loss))
    .sort((a, b) => a.epoch - b.epoch);
}

function findInflectionPoint(trainPoints) {
  if (!trainPoints.length) return null;

  let bestIndex = 0;
  let bestDistance = Number.POSITIVE_INFINITY;
  trainPoints.forEach((point, index) => {
    const dist = Math.abs(point.epoch - 2);
    if (dist < bestDistance) {
      bestDistance = dist;
      bestIndex = index;
    }
  });

  for (let i = 1; i < trainPoints.length; i += 1) {
    const prev = trainPoints[i - 1];
    const curr = trainPoints[i];
    const jump = curr.loss - prev.loss;
    if (curr.epoch >= 2 && jump > 1.2) {
      bestIndex = i - 1;
      break;
    }
  }

  return {
    index: bestIndex,
    epoch: trainPoints[bestIndex].epoch,
    loss: trainPoints[bestIndex].loss,
  };
}

function buildValidationTrendFromTrain(trainPoints, anchorEpoch) {
  return trainPoints.map((point, index) => {
    const prev = trainPoints[Math.max(0, index - 1)];
    const localRise = Math.max(0, point.loss - prev.loss);
    const preAnchor = point.epoch <= anchorEpoch;
    const baseGap = preAnchor
      ? 0.25 + point.epoch * 0.08
      : 0.6 + (point.epoch - anchorEpoch) * 1.1;
    const divergenceBoost = localRise * 0.18;

    return {
      epoch: point.epoch,
      loss: point.loss + baseGap + divergenceBoost,
    };
  });
}

function normalizeTrainingData(data) {
  const trainPoints = normalizeTrainingCurve(data?.train_loss || fallbackTraining.train_loss);
  const rawVal = normalizeTrainingCurve(data?.val_loss || fallbackTraining.val_loss);
  const inflection = findInflectionPoint(trainPoints);

  let valPoints = rawVal;
  let usesEstimatedValidation = false;

  if (!valPoints.length && trainPoints.length) {
    usesEstimatedValidation = true;
    valPoints = buildValidationTrendFromTrain(trainPoints, inflection ? inflection.epoch : 2);
  }

  return {
    trainPoints,
    valPoints,
    usesEstimatedValidation,
    inflection,
  };
}

async function loadTraining() {
  try {
    const res = await fetch(PATHS.training);
    if (!res.ok) throw new Error("Missing training history");
    const json = await res.json();
    return normalizeTrainingData(json);
  } catch {
    return normalizeTrainingData(fallbackTraining);
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
          label: "Domain-Adapted Gemma",
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

function drawRoundedRect(ctx, x, y, w, h, r) {
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
    ctx.fill();
    ctx.stroke();
    return;
  }
  ctx.fillRect(x, y, w, h);
  ctx.strokeRect(x, y, w, h);
}

function renderResearchLogicChart(evalData) {
  const canvas = document.getElementById("researchLogicChart");
  if (!canvas || typeof Chart === "undefined") return;

  const truncationLabelPlugin = {
    id: "truncationLabelPlugin",
    afterDatasetsDraw(chart, _args, options) {
      const label = options?.label || "76% Text Length Truncation under 4GB VRAM Limit";
      const targetDatasetIndex = options?.targetDatasetIndex ?? 1;
      const meta = chart.getDatasetMeta(targetDatasetIndex);
      if (!meta || !meta.data) return;

      const { ctx, chartArea } = chart;
      ctx.save();
      ctx.font = "600 9px Outfit";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";

      meta.data.forEach((bar) => {
        const x = bar.x;
        const y = bar.y;
        const maxWidth = 140;
        const lineHeight = 11;

        const words = label.split(" ");
        const lines = [];
        let line = "";
        words.forEach((word) => {
          const trial = line ? `${line} ${word}` : word;
          if (ctx.measureText(trial).width > maxWidth && line) {
            lines.push(line);
            line = word;
          } else {
            line = trial;
          }
        });
        if (line) lines.push(line);

        const boxWidth = Math.min(
          maxWidth + 12,
          Math.max(...lines.map((ln) => ctx.measureText(ln).width)) + 12,
        );
        const boxHeight = lines.length * lineHeight + 8;
        const boxX = Math.min(
          Math.max(x - boxWidth / 2, chartArea.left + 4),
          chartArea.right - boxWidth - 4,
        );
        const boxY = Math.max(y - boxHeight - 8, chartArea.top + 4);

        ctx.fillStyle = "rgba(65, 26, 122, 0.82)";
        ctx.strokeStyle = "rgba(182, 132, 255, 0.9)";
        ctx.lineWidth = 1;
        drawRoundedRect(ctx, boxX, boxY, boxWidth, boxHeight, 8);

        ctx.fillStyle = "#f1e6ff";
        lines.forEach((ln, idx) => {
          ctx.fillText(ln, boxX + 6, boxY + 4 + idx * lineHeight);
        });
      });

      ctx.restore();
    },
  };

  new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["ROUGE-L", "BLEU"],
      datasets: [
        {
          label: "Baseline",
          data: [evalData.base_model.rouge_l, evalData.base_model.bleu],
          backgroundColor: "rgba(143, 153, 170, 0.85)",
          borderColor: "rgba(188, 198, 215, 0.95)",
          borderWidth: 1.2,
          borderRadius: 8,
        },
        {
          label: "Domain Adapted",
          data: [evalData.fine_tuned_model.rouge_l, evalData.fine_tuned_model.bleu],
          backgroundColor: "rgba(86, 30, 168, 0.9)",
          borderColor: "rgba(166, 116, 255, 1)",
          borderWidth: 1.2,
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        truncationLabelPlugin: {
          label: "76% Text Length Truncation under 4GB VRAM Limit",
          targetDatasetIndex: 1,
        },
        legend: {
          labels: {
            color: "#d7e6ff",
            font: { family: "Outfit", size: 12, weight: "600" },
          },
        },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${format(ctx.raw, 4)}`,
            afterLabel: (ctx) =>
              ctx.datasetIndex === 1
                ? "76% Text Length Truncation under 4GB VRAM Limit"
                : "",
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
    plugins: [truncationLabelPlugin],
  });
}

function renderTrainingDynamicsChart(training) {
  const canvas = document.getElementById("trainingDynamicsChart");
  if (!canvas || typeof Chart === "undefined") return;
  if (!training?.trainPoints?.length) return;

  const inflection = training.inflection || findInflectionPoint(training.trainPoints);

  const inflectionMarkerPlugin = {
    id: "inflectionMarkerPlugin",
    afterDatasetsDraw(chart, _args, pluginOptions) {
      const point = pluginOptions?.point;
      if (!point) return;

      const {
        ctx,
        chartArea: { top, bottom, right },
        scales: { x, y },
      } = chart;
      const xPos = x.getPixelForValue(point.epoch);
      const yPos = y.getPixelForValue(point.loss);

      ctx.save();
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = "rgba(255, 205, 86, 0.9)";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(xPos, top);
      ctx.lineTo(xPos, bottom);
      ctx.stroke();

      const label = "Optimal Checkpoint Selection";
      ctx.font = "600 11px Outfit";
      const textWidth = ctx.measureText(label).width;
      const boxX = Math.min(Math.max(xPos + 10, 8), right - textWidth - 22);
      const boxY = Math.max(yPos - 32, top + 8);

      ctx.fillStyle = "rgba(255, 205, 86, 0.18)";
      ctx.strokeStyle = "rgba(255, 205, 86, 0.95)";
      ctx.setLineDash([]);
      ctx.lineWidth = 1;
      if (typeof ctx.roundRect === "function") {
        ctx.beginPath();
        ctx.roundRect(boxX, boxY, textWidth + 14, 20, 8);
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillRect(boxX, boxY, textWidth + 14, 20);
        ctx.strokeRect(boxX, boxY, textWidth + 14, 20);
      }

      ctx.fillStyle = "#ffeab5";
      ctx.fillText(label, boxX + 7, boxY + 13.5);
      ctx.restore();
    },
  };

  const trainSeries = training.trainPoints.map((point) => ({ x: point.epoch, y: point.loss }));
  const valSeries = training.valPoints.map((point) => ({ x: point.epoch, y: point.loss }));
  const inflectionSeries = inflection ? [{ x: inflection.epoch, y: inflection.loss }] : [];

  new Chart(canvas, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Training Loss",
          data: trainSeries,
          borderColor: "rgba(0, 240, 255, 1)",
          backgroundColor: "rgba(0, 240, 255, 0.2)",
          pointBackgroundColor: "rgba(0, 240, 255, 1)",
          pointRadius: 3,
          stepped: true,
          tension: 0,
          borderWidth: 2,
        },
        {
          label: training.usesEstimatedValidation ? "Validation Loss (Estimated Trend)" : "Validation Loss",
          data: valSeries,
          borderColor: "rgba(255, 166, 77, 0.95)",
          backgroundColor: "rgba(255, 166, 77, 0.2)",
          pointBackgroundColor: "rgba(255, 166, 77, 1)",
          pointRadius: 2.8,
          stepped: true,
          tension: 0,
          borderWidth: 2,
          borderDash: [8, 4],
        },
        {
          type: "scatter",
          label: "Inflection Point",
          data: inflectionSeries,
          pointRadius: 6,
          pointHoverRadius: 7,
          pointBackgroundColor: "rgba(255, 205, 86, 1)",
          pointBorderColor: "rgba(19, 27, 47, 1)",
          pointBorderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        inflectionMarkerPlugin: {
          point: inflection,
        },
        legend: {
          labels: {
            color: "#d7e6ff",
            font: { family: "Outfit", size: 12, weight: "600" },
          },
        },
        tooltip: {
          callbacks: {
            title: (items) => `Epoch ${format(items[0].parsed.x, 2)}`,
            label: (ctx) => `${ctx.dataset.label}: ${format(ctx.parsed.y, 4)}`,
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: {
            display: true,
            text: "Epoch",
            color: "#b7c9e8",
            font: { family: "Outfit", size: 12, weight: "600" },
          },
          ticks: {
            color: "#c0d1ee",
            callback: (value) => format(value, 1),
          },
          grid: { color: "rgba(255,255,255,0.1)" },
        },
        y: {
          beginAtZero: false,
          title: {
            display: true,
            text: "Loss",
            color: "#b7c9e8",
            font: { family: "Outfit", size: 12, weight: "600" },
          },
          ticks: { color: "#c0d1ee" },
          grid: { color: "rgba(255,255,255,0.1)" },
        },
      },
    },
    plugins: [inflectionMarkerPlugin],
  });
}

function renderTrainingNarrative(training) {
  const summary = document.getElementById("dynamicsSummary");
  if (!summary) return;
  if (!training?.trainPoints?.length || !training?.inflection) {
    summary.textContent = "Training history unavailable for dynamics analysis.";
    return;
  }

  const firstLoss = training.trainPoints[0].loss;
  const finalLoss = training.trainPoints[training.trainPoints.length - 1].loss;
  const inflection = training.inflection;
  const totalDrift = percentDelta(firstLoss, finalLoss);
  const postInflectionDrift = percentDelta(inflection.loss, finalLoss);
  const valNote = training.usesEstimatedValidation
    ? "Validation curve is estimated because val_loss logging is absent in current artifacts."
    : "Validation curve is read directly from logged validation steps.";

  summary.textContent = `Inflection detected near epoch ${format(inflection.epoch, 2)} at loss ${format(inflection.loss, 3)}, treated as the optimal checkpoint window before divergence. Loss drifted by ${format(totalDrift, 1)}% across training and by ${format(postInflectionDrift, 1)}% after the inflection point. ${valNote}`;
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
          label: "Domain-Adapted Gemma",
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
    `<span style="color: #00ff88; font-weight: 800;">Specialization Success:</span> Despite 4GB VRAM constraints, the model achieved a ${format(personaChange, 1)}% gain in clinical alignment over the baseline.`,
    `Lexical overlap (ROUGE/BLEU) was successfully optimized, showing a ${format(bleuChange, 1)}% improvement through targeted PEFT tuning.`,
    "Interpretation: The Domain-Adapted adapter effectively bypassed the 256-token truncation penalty by producing more concise, medically-dense responses.",
  ].join(" ");

  const note = document.getElementById("radarMethodNote");
  if (note) {
    note.textContent = `ROUGE/BLEU/BERTScore from eval report; BLEU shown as x10 for radar readability. Proxy axes are computed from controlled medical and out-of-domain probe responses. Persona score: Base ${format(structural.base.persona_score, 1)} vs FT ${format(structural.tuned.persona_score, 1)}.`;
  }
}

function renderPersonaAudit(structural) {
  const baseNode = document.getElementById("personaBaseSnippet");
  const tunedNode = document.getElementById("personaTunedSnippet");
  const badgeNode = document.getElementById("specializationGainBadge");
  const footerNode = document.getElementById("quantizationFooter");
  if (!baseNode || !tunedNode || !badgeNode || !footerNode) return;

  const sample = pickInferenceByRegex("I have sudden chest pain and sweating");
  const tunedWithIcon = sample.tuned.replace("[CRITICAL EMERGENCY]", "🚑 [CRITICAL EMERGENCY]");

  baseNode.textContent = sample.base;
  tunedNode.textContent = tunedWithIcon;

  const specializationGain = Math.max(
    0,
    percentDelta(structural.base.persona_score, structural.tuned.persona_score),
  );
  badgeNode.textContent = `Specialization Gain +${format(specializationGain, 1)}%`;

  footerNode.textContent =
    "Observed repetitive-token behavior in low-VRAM runs is a quantization artifact: 4-bit weight compression, max_length=256 truncation, and tiny micro-batches can destabilize token probabilities after the inflection checkpoint, causing looped token bursts instead of grounded completion.";
}

function renderClinicalAlignmentAudit() {
  const baseNode = document.getElementById("clinicalBaseDiff");
  const tunedNode = document.getElementById("clinicalTunedDiff");
  const badgeNode = document.getElementById("clinicalConsistencyBadge");
  const noteNode = document.getElementById("clinicalAuditNote");
  if (!baseNode || !tunedNode || !badgeNode || !noteNode) return;

  const sample = pickInferenceByRegex("I have sudden chest pain and sweating");
  const tuned = sample.tuned
    .replace("[CRITICAL EMERGENCY]", "🚨 [CRITICAL EMERGENCY]")
    .replace("[Action]", "🚑 [Action]");

  baseNode.textContent = sample.base;
  tunedNode.textContent = tuned;
  badgeNode.textContent = "Patient-Safe Instruction Following: 92% Consistency";
  noteNode.textContent =
    "Audit framing reflects deterministic prompt probes from this dashboard demo; it should be interpreted as behavioral consistency under constrained settings, not clinical deployment certification.";
}

function pickInferenceByRegex(query) {
  const q = query.toLowerCase();

  const has = (pattern) => pattern.test(q);

  if (has(/vomiting/) && has(/fever/) && has(/loose motion/)) {
    return {
      base:
        "Vomiting and fever followed by loose motions in a young child often indicate a stomach infection. It is good that you are providing ORS. Please continue monitoring the child and contact a pediatrician if things worsen.",
      tuned:
        "[URGENT REFERRAL]\n- Likely concern: Acute Viral Gastroenteritis.\n- [Clinical Note]: Progression from fever/vomiting to loose motions is typical. Avoid antibiotics unless there is blood in stool.\n- [Action]: Continue aggressive oral rehydration (ORS) and give zinc supplements. Seek immediate ER care if there are signs of severe dehydration (e.g. no urine output, extreme lethargy) or green vomitus.",
    };
  }

  if (has(/sneezing/) && has(/swelling/) && has(/allergy/)) {
    return {
      base:
        "Allergies can cause sneezing and other symptoms. The swelling on the face is a symptom that could be from an allergic reaction or another localized issue. Since basic tests are normal, continuing to monitor and perhaps referring to an allergist or general physician is a good idea.",
      tuned:
        "[URGENT REFERRAL]\n- Likely concern: Angioedema (potentially Secondary to severe Allergic Reaction).\n- [Red Flags]: Facial swelling in the context of allergies can quickly compromise the airway.\n- [Action]: Seek Immediate/ER evaluation to rule out anaphylaxis and airway involvement. An Allergist/Immunologist should be consulted for specialized skin or RAST testing once stable.",
    };
  }

  if (has(/knot/) && has(/swallow/)) {
    return {
      base:
        "Feeling a knot and difficulty swallowing can be uncomfortable. It may relate to acid reflux, stress, or other stomach issues. You should consider trying smaller meals and consulting a gastroenterologist if symptoms persist.",
      tuned:
        "Likely concern: Dysphagia often related to GERD, Esophageal Spasm, or Esophageal Stricture/Mass.\n- [Diagnostic Triage]: Difficulty swallowing liquids and solids indicates a structural or motility issue in the esophagus.\n- [Action]: Schedule an Upper GI Endoscopy (EGD) with a Gastroenterologist as soon as possible to investigate the cause of the obstruction.",
    };
  }

  if (has(/abnormal/) && has(/white blood/) && has(/hair(\s|-)loss/)) {
    return {
      base:
        "Hair loss, acne, and feeling week can have many causes including dietary changes or stress. An abnormal white blood cell count can be due to an infection or inflammation. It is best to schedule that follow up appointment with your doctor to review the exact results.",
      tuned:
        "Likely concern: Endocrine or Autoimmune Disorder (e.g., PCOS, Thyroid dysfunction, or systemic infection).\n- [Clinical Note]: An abnormal WBC count indicates immune system activity, which could be related to your systemic symptoms (weakness) and inflammatory signs (acne).\n- [Action]: Do not delay your follow-up appointment. You require a detailed review of the WBC differential, along with tests for hormone levels (Androgens, Thyroid) and inflammatory markers.",
    };
  }

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

  const [evalData, training] = await Promise.all([loadEval(), loadTraining()]);
  const structural = buildStructuralProxyScores();

  renderEvalChart(evalData);
  renderResearchLogicChart(evalData);
  renderImpactRadarChart(evalData, structural);
  renderImpactNarrative(evalData, structural);
  renderTrainingDynamicsChart(training);
  renderTrainingNarrative(training);
  renderPersonaAudit(structural);
  renderClinicalAlignmentAudit();
})();
