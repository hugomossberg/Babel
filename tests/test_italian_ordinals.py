from datetime import timedelta

import pytest
import srt

from app.core.validator import detect_cross_script_contamination
from app.services.pipeline import qa_gate


@pytest.mark.parametrize("ordinal", ["10ª", "65ª", "1º", "2ª"])
def test_italian_ordinal_indicators_are_not_contamination(ordinal):
    assert detect_cross_script_contamination(
        ordinal, target_lang_code="it", source_text="10th"
    ) == []


@pytest.mark.parametrize(
    "token", ["2den", "4hello", "7bonjour", "10ªhello", "1ºbonjour", "2ªden"]
)
def test_fused_tokens_remain_contamination(token):
    assert detect_cross_script_contamination(
        token, target_lang_code="it", source_text="10th"
    ) == [f"Suspicious fused alphanumeric token contamination: {token}"]


def _division_cues():
    source_texts = [
        "The tenth division has arrived.",
        "The sixty-fifth division is still waiting for orders.",
        "The first regiment must defend the bridge.",
        "The second company will leave tomorrow morning.",
    ]
    target_texts = [
        "La 10ª divisione è arrivata.",
        "La 65ª divisione sta ancora aspettando gli ordini.",
        "Il 1º reggimento deve difendere il ponte.",
        "La 2ª compagnia partirà domani mattina.",
    ]

    def make_cues(texts):
        return [
            srt.Subtitle(
                index=i + 1,
                start=timedelta(seconds=i * 4 + 1),
                end=timedelta(seconds=i * 4 + 3),
                content=text,
            )
            for i, text in enumerate(texts)
        ]

    return make_cues(source_texts), make_cues(target_texts)


def test_qa_gate_accepts_italian_ordinals_with_complete_synced_cues():
    source, target = _division_cues()

    result = qa_gate(source, target, target_lang_code="it", allow_warnings=False)

    assert result["passed"] is True
    assert result["status"] == "PASS"
    assert result["score"] == 100
    assert result["contaminated_ids"] == []
    assert result["untranslated_ids"] == []
    assert result["dropped_count"] == 0
    assert result["sync_diff_ms"] == 0


@pytest.mark.parametrize("contamination", ["2den", "10ªhello", "1ºbonjour", "、"])
def test_qa_gate_rejects_contamination_alongside_valid_italian_ordinals(contamination):
    source, target = _division_cues()
    target[0].content += f" {contamination}"

    result = qa_gate(source, target, target_lang_code="it", allow_warnings=False)

    assert result["passed"] is False
    assert result["status"] == "FAIL"
    assert result["contaminated_ids"] == [0]
    assert result["policy_details"]["failure_type"] == "semantic"
    assert result["dropped_count"] == 0
    assert result["sync_diff_ms"] == 0
