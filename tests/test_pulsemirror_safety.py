import os

os.environ["RETRIEVAL_BACKEND"] = "lexical"

from src.rag_server import (
    PulseMirrorRequest,
    apply_medical_safety_gate,
    classify_medical_risk,
    pulsemirror_risk_aware_ask,
)
import src.rag_server as rag_server


def test_general_wellness_query_uses_low_risk_policy():
    result = classify_medical_risk("What habits support healthy sleep?")
    assert result["level"] == "low"
    assert result["score"] == 0.2


def test_medication_change_query_uses_high_risk_policy():
    result = classify_medical_risk("Should I stop taking my prescription medication?")
    assert result["level"] == "high"
    answer, flags, blocked = apply_medical_safety_gate("Evidence-based response.", result)
    assert not blocked
    assert "clinical_review_recommended" in flags
    assert "qualified clinician" in answer


def test_emergency_request_blocks_model_generation():
    result = pulsemirror_risk_aware_ask(
        PulseMirrorRequest(query="I have chest pain and difficulty breathing")
    )
    assert result["risk"]["level"] == "emergency"
    assert result["routing"]["selected_route"] == "emergency_safety_pipeline"
    assert result["routing"]["output_blocked"] is True
    assert result["retrieved_chunks"] == []
    assert "emergency service" in result["answer"].lower()


def test_high_risk_request_uses_strict_route_and_fallback(monkeypatch):
    chunk = {"doc_id": 7, "text": "Medication changes require clinical review.", "instruction": ""}
    monkeypatch.setattr(rag_server, "CHUNK_META", [chunk])
    monkeypatch.setattr(
        rag_server,
        "retrieve_chunks",
        lambda query, top_k: ([chunk], [0.91], 0, 0.0, 0.001),
    )
    monkeypatch.setattr(
        rag_server,
        "generate_rag_answer",
        lambda request: {"answer": "Use the retrieved medication evidence.", "generation_source": "extractive_fallback"},
    )

    result = rag_server.pulsemirror_risk_aware_ask(
        PulseMirrorRequest(query="Should I stop taking my prescription medication?")
    )
    assert result["risk"]["level"] == "high"
    assert result["routing"]["selected_route"] == "strict_grounded_pipeline"
    assert result["routing"]["fallback_used"] is True
    assert "clinical_review_recommended" in result["safety_flags"]
