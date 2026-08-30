"""
guardrails.py - HealthRAG AI safety checks (PRD sections 7, 8, 11)
=================================================================
Each guardrail is a small, isolated, deterministic function so it can later be
replaced by a stronger classifier. None of these call an LLM.

Input guardrails (run BEFORE any Ollama call):
  - check_prompt_injection(query)
  - check_domain(query)

Output guardrails (run AFTER the Ollama response):
  - check_grounding(answer, context_chunks)
  - check_medical_safety(answer, query)

Every function returns a dict: {"passed": bool, "status": str, "reason": str, ...}
These are basic first-layer checks and are labelled as such - they do not provide
complete security or a medical-safety guarantee.
"""

from __future__ import annotations

import re
from typing import Dict, List, Sequence

try:  # reuse the Week 4 proxy metrics when available
    from week4_metrics import abstention_detected, unsupported_claim_proxy, content_tokens
except Exception:  # pragma: no cover
    try:
        from src.week4_metrics import abstention_detected, unsupported_claim_proxy, content_tokens
    except Exception:  # fallback keeps guardrails self-contained
        _NO_WEEK4 = True
    else:
        _NO_WEEK4 = False
else:
    _NO_WEEK4 = False

if _NO_WEEK4:  # pragma: no cover - minimal self-contained fallbacks
    def abstention_detected(answer: str) -> bool:
        lowered = (answer or "").lower()
        return any(p in lowered for p in (
            "does not contain", "not enough information", "insufficient information",
            "cannot answer", "no information", "knowledge base does not",
        ))

    def content_tokens(text: str):
        return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}

    def unsupported_claim_proxy(answer: str, retrieved_contexts: Sequence[str]) -> Dict:
        return {"unsupported_sentence_rate": 0.0, "unsupported_sentences": []}


# ── Input guardrail 1: prompt injection ──────────────────────────────────────
_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above|earlier) (instructions|prompts?|messages?)",
    r"ignore the (system|developer) (prompt|message|instructions?)",
    r"disregard (all |any )?(previous|prior|the) (instructions|rules|prompt)",
    r"(reveal|show|print|repeat|leak|expose) (me )?(your |the )?(system|developer|initial) (prompt|message|instructions?)",
    r"what (is|are) your (system|initial) (prompt|instructions?)",
    r"bypass (your |the )?(instructions|rules|guardrails?|filters?|restrictions?)",
    r"you are now (a|an|in) ",
    r"(enter|activate) (developer|debug|god|dan) mode",
    r"pretend (you are|to be) (an? )?(unfiltered|uncensored|different)",
    r"do anything now",
]
_INJECTION_RE = [re.compile(p, re.I) for p in _INJECTION_PATTERNS]


def check_prompt_injection(query: str) -> Dict[str, object]:
    q = query or ""
    for rx in _INJECTION_RE:
        m = rx.search(q)
        if m:
            return {
                "passed": False,
                "status": "Blocked by Prompt Injection Guardrail",
                "reason": "The query appears to try to override or reveal the system instructions.",
                "matched": m.group(0)[:120],
            }
    return {"passed": True, "status": "Passed", "reason": "No obvious prompt-injection pattern detected."}


# ── Input guardrail 2: domain relevance ──────────────────────────────────────
_OUT_OF_SCOPE_PATTERNS = [
    r"\b(weather|temperature outside|forecast|will it rain)\b",
    r"what('?s| is) (today'?s|the) (date|day|time)\b",
    r"\b(cricket|football|soccer|nba|match|tournament|world cup)\b.*\b(score|won|winner|result)\b",
    r"\b(who won|score of)\b",
    r"\b(write|create|generate|debug|fix)\b.{0,20}\b(python|java(script)?|program|code|function|script|algorithm|regex)\b",
    r"c\+\+",
    r"\btell me a (joke|story|poem)\b",
    r"\b(stock price|share price|bitcoin|crypto|forex)\b",
    r"\b(recipe for|how to cook|directions to|book a flight|nearest restaurant)\b",
    r"\btranslate .* (to|into) (french|spanish|german|hindi|urdu)\b",
]
_OUT_OF_SCOPE_RE = [re.compile(p, re.I) for p in _OUT_OF_SCOPE_PATTERNS]

# Broad health / medical / knowledge-base vocabulary. A query is treated as
# in-domain only if it touches one of these (or the knowledge base strongly
# matches it). Everything else is rejected as out-of-domain.
_HEALTH_TERMS = (
    # general
    "health", "healthy", "medical", "medicine", "medicin", "clinic", "clinical",
    "hospital", "doctor", "physician", "nurse", "patient", "pharmac", "prescri",
    "wellness", "wellbeing", "first aid", "emergency", "ambulance", "icu",
    "knowledge base", "document", "guideline", "protocol", "report", "leaflet",
    # actions / framings
    "diagnos", "treat", "treatment", "therap", "cure", "heal", "recover", "manage",
    "symptom", "sign of", "cause of", "risk factor", "prevent", "screening",
    "side effect", "adverse", "contraindicat", "interaction", "overdose",
    "dose", "dosage", "dosing", "mg", "ml", "tablet", "capsule", "injection",
    "is it safe to", "should i take", "should i see", "how long does it take to heal",
    "what should i do if", "home remedy", "when to worry",
    # body / systems
    "body", "blood", "bone", "muscle", "joint", "nerve", "skin", "heart", "cardiac",
    "lung", "respirat", "kidney", "renal", "liver", "hepatic", "brain", "neuro",
    "stomach", "gut", "intestin", "bowel", "bladder", "throat", "sinus", "ear",
    "eye", "vision", "tooth", "teeth", "gum", "spine", "back", "shoulder", "knee",
    "hip", "ankle", "wrist", "chest", "abdomen", "pelvi", "thyroid", "gland",
    "hormone", "immune", "lymph", "artery", "vein", "vessel",
    # symptoms
    "pain", "ache", "hurt", "sore", "swelling", "swollen", "inflammation", "inflamed",
    "fever", "temperature", "chills", "cough", "cold", "flu", "sneeze", "congestion",
    "rash", "itch", "bruis", "bleed", "wound", "cut", "burn", "blister", "lump",
    "nausea", "vomit", "diarrhea", "diarrhoea", "constipat", "cramp", "bloat",
    "dizzy", "dizziness", "faint", "vertigo", "headache", "migraine", "fatigue",
    "tired", "weak", "numb", "tingl", "short of breath", "breathless", "palpitat",
    "seizure", "tremor", "spasm", "stiff", "discharge", "ulcer", "infection",
    # conditions
    "disease", "disorder", "syndrome", "condition", "chronic", "acute", "cancer",
    "tumor", "tumour", "diabet", "hypertension", "hypotension", "blood pressure",
    "cholesterol", "asthma", "copd", "pneumonia", "bronchitis", "tuberculosis",
    "arthritis", "osteoporosis", "anemia", "anaemia", "allerg", "eczema", "psoriasis",
    "stroke", "heart attack", "angina", "arrhythmia", "kidney disease", "hepatitis",
    "cirrhosis", "gastritis", "reflux", "ibs", "crohn", "colitis", "thyroid",
    "depression", "anxiety", "insomnia", "adhd", "epilepsy", "parkinson", "alzheimer",
    "dementia", "obesity", "malnutrition", "sepsis", "covid", "hiv", "malaria",
    "dengue", "typhoid", "measles", "chickenpox", "shingles",
    # drugs
    "antibiotic", "amoxicillin", "penicillin", "azithromycin", "paracetamol",
    "acetaminophen", "ibuprofen", "aspirin", "naproxen", "metformin", "insulin",
    "statin", "amlodipine", "lisinopril", "losartan", "omeprazole", "steroid",
    "antihistamine", "anticoagulant", "warfarin", "apixaban", "vaccine", "vaccinat",
    "antidepressant", "painkiller", "analgesic", "chemotherapy", "antiviral",
    # life stages / reproductive
    "pregnan", "prenatal", "postnatal", "breastfeed", "menstru", "period", "menopause",
    "fertility", "contracepti", "paediatric", "pediatric", "infant", "toddler",
    "newborn", "elderly", "geriatric", "child", "baby",
    # nutrition / lifestyle (health context)
    "nutrition", "diet", "vitamin", "mineral", "supplement", "hydration", "dehydrat",
    "exercise", "physiotherapy", "rehabilitation", "sleep apnea", "smoking cessation",
    "blood sugar", "glucose", "bmi", "calorie",
    # care
    "surgery", "operation", "biopsy", "scan", "x-ray", "mri", "ct scan", "ultrasound",
    "blood test", "lab result", "vaccination schedule", "dressing", "bandage",
    "stitches", "cast", "crutches",
)


def check_domain(query: str, retrieval_scores=None, score_threshold: float = 0.5) -> Dict[str, object]:
    """Decide whether a query belongs to the HealthRAG (health / medical /
    knowledge-base) domain. Anything that is not clearly health-related is
    rejected as out-of-domain and must NOT be sent to the LLM (PRD section 7).

    `retrieval_scores` (optional): *semantic* similarity scores (cosine, 0-1) of
    the top retrieved chunks. A strong match also counts as in-domain, catching
    health questions phrased with uncommon wording. The caller should only pass
    these when using the semantic retrieval backend - lexical word-overlap
    scores are not a reliable topic signal.
    """
    q = (query or "").lower().strip()
    if not q:
        return {"passed": False, "status": "Out of Scope",
                "reason": "Empty query.", "matched": ""}

    # 1. Explicit out-of-scope topics (weather, sport, coding, jokes, ...).
    for rx in _OUT_OF_SCOPE_RE:
        m = rx.search(q)
        if m:
            return {
                "passed": False,
                "status": "Out of Scope",
                "reason": "This question is outside the scope of HealthRAG AI. Please ask a health or knowledge-base related question.",
                "matched": m.group(0)[:120],
            }

    # 2. Health / medical / knowledge-base vocabulary.
    hit = next((t for t in _HEALTH_TERMS if t in q), None)
    if hit:
        return {"passed": True, "status": "In Scope",
                "reason": f"Query references a health or knowledge-base topic ('{hit}')."}

    # 3. Knowledge-base match signal (uncommon phrasing of a real health question).
    try:
        top = max((float(s) for s in (retrieval_scores or [])), default=0.0)
    except (TypeError, ValueError):
        top = 0.0
    if top >= score_threshold:
        return {"passed": True, "status": "In Scope",
                "reason": f"Knowledge base strongly matches the query (score {round(top, 2)})."}

    # 4. Default: reject as out-of-domain.
    return {
        "passed": False,
        "status": "Out of Domain",
        "reason": "This question does not appear to be about health, medicine, or the knowledge base. HealthRAG AI only answers health and knowledge-base questions.",
        "matched": "",
    }


# ── Output guardrail 1: grounding ────────────────────────────────────────────
def check_grounding(answer: str, context_chunks: Sequence[dict]) -> Dict[str, object]:
    answer = answer or ""
    contexts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in (context_chunks or [])]

    if not answer.strip():
        return {"passed": False, "status": "Backend Validation Pending",
                "reason": "No answer text to validate."}

    if abstention_detected(answer):
        return {"passed": True, "status": "Grounded (abstained)",
                "reason": "The model stated the knowledge base does not contain enough information."}

    if not contexts:
        return {"passed": False, "status": "Ungrounded",
                "reason": "No retrieved context was available to support the answer."}

    proxy = unsupported_claim_proxy(answer, contexts)
    rate = float(proxy.get("unsupported_sentence_rate", 0.0) or 0.0)
    if rate >= 0.5:
        return {
            "passed": False,
            "status": "Weak Grounding",
            "reason": f"About {round(rate * 100)}% of factual sentences have low overlap with the retrieved context (lexical proxy).",
            "unsupported_rate": round(rate, 2),
        }
    return {
        "passed": True,
        "status": "Grounded",
        "reason": "Most factual sentences overlap the retrieved context (lexical proxy).",
        "unsupported_rate": round(rate, 2),
    }


# ── Output guardrail 2: medical safety ───────────────────────────────────────
_ABSOLUTE_CLAIM_RE = re.compile(
    r"\b(you (should|must)|the (correct|right) (dose|dosage) is|take \d+\s?(mg|ml|g)\b|"
    r"definitely have|you have (cancer|diabetes|a heart attack)|stop taking|is safe to)\b",
    re.I,
)
_DISCLAIMER_MARKERS = ("professional", "doctor", "clinician", "pharmacist", "not a substitute", "seek medical")


def check_medical_safety(answer: str, query: str = "") -> Dict[str, object]:
    answer = answer or ""
    if not answer.strip():
        return {"passed": False, "status": "Backend Validation Pending",
                "reason": "No answer text to validate.", "annotated_answer": answer}

    flags: List[str] = []
    m = _ABSOLUTE_CLAIM_RE.search(answer)
    if m and not abstention_detected(answer):
        flags.append(f"absolute clinical statement: \"{m.group(0)}\"")

    has_disclaimer = any(k in answer.lower() for k in _DISCLAIMER_MARKERS)
    annotated = answer
    treatment_context = any(t in (query + " " + answer).lower()
                            for t in ("treat", "dose", "medication", "should i take", "diagnos"))
    if treatment_context and not has_disclaimer and not abstention_detected(answer):
        annotated = answer.rstrip() + "\n\nNote: This is general information from the knowledge base, not personalised medical advice - consult a qualified professional."

    if flags:
        return {
            "passed": False,
            "status": "Safety Review",
            "reason": "Answer contains an absolute clinical claim; treat with caution.",
            "flags": flags,
            "annotated_answer": annotated,
        }
    return {
        "passed": True,
        "status": "OK",
        "reason": "No absolute clinical claims detected (basic heuristic check).",
        "flags": [],
        "annotated_answer": annotated,
    }
