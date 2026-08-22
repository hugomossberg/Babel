import pytest
import os
import srt
import secrets
from datetime import timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import Request, HTTPException

from app.services.translator import (
    is_deterministically_safe_keep,
    normalize_for_compare,
    validate_classifier_output,
    ProviderConfigurationError,
    ProviderUnavailableError
)
from app.services.pipeline import SubtitlePipeline
from app.api.webhooks import validate_webhook_token
from app.core.extractor import extract_embedded_srt

def test_deterministic_safe_keep_rules():
    # Dialogue words must never be kept as proper nouns or symbols
    assert not is_deterministically_safe_keep("Hello", "proper_noun")
    assert not is_deterministically_safe_keep("Come here.", "proper_noun")
    assert not is_deterministically_safe_keep("Get out!", "proper_noun")
    assert not is_deterministically_safe_keep("What happened?", "proper_noun")
    assert not is_deterministically_safe_keep("Thank you", "non_verbal")

    # Proper nouns fail-closed to False (must default to TRANSLATE)
    assert not is_deterministically_safe_keep("John Smith", "proper_noun")
    assert not is_deterministically_safe_keep("New York", "proper_noun")
    assert not is_deterministically_safe_keep("Red Alert", "proper_noun")
    assert not is_deterministically_safe_keep("Green Light", "proper_noun")
    assert not is_deterministically_safe_keep("Dead Body", "proper_noun")
    assert not is_deterministically_safe_keep("Black Car", "proper_noun")
    assert not is_deterministically_safe_keep("White House", "proper_noun")
    assert not is_deterministically_safe_keep("Good Morning", "proper_noun")
    assert not is_deterministically_safe_keep("Happy Birthday", "proper_noun")
    assert not is_deterministically_safe_keep("Big Problem", "proper_noun")
    assert not is_deterministically_safe_keep("Last Chance", "proper_noun")
    assert not is_deterministically_safe_keep("First Time", "proper_noun")
    assert not is_deterministically_safe_keep("New Plan", "proper_noun")
    assert not is_deterministically_safe_keep("Bad Idea", "proper_noun")
    assert not is_deterministically_safe_keep("Blue Moon", "proper_noun")

    # Brand is fail-closed
    assert not is_deterministically_safe_keep("Microsoft", "brand")
    assert not is_deterministically_safe_keep("Apple", "brand")

    # Valid acronyms, tech brands and numbers from explicit allowlist
    assert is_deterministically_safe_keep("NASA", "acronym")
    assert is_deterministically_safe_keep("FBI", "acronym")
    assert is_deterministically_safe_keep("CIA", "acronym")
    assert is_deterministically_safe_keep("GPS", "acronym")
    assert is_deterministically_safe_keep("USB", "acronym")
    assert is_deterministically_safe_keep("DNA", "acronym")
    assert is_deterministically_safe_keep("WiFi!", "brand")
    assert is_deterministically_safe_keep("WiFi!", "acronym")
    assert is_deterministically_safe_keep("Wi-Fi", "acronym")
    assert is_deterministically_safe_keep("Bluetooth", "brand")
    assert is_deterministically_safe_keep("911", "number")
    assert is_deterministically_safe_keep("2026", "number")

    # Non-verbal with descriptive words must fail-closed to False
    assert not is_deterministically_safe_keep("[sighs]", "non_verbal")
    assert not is_deterministically_safe_keep("(door closes)", "non_verbal")
    assert not is_deterministically_safe_keep("[I love you]", "non_verbal")
    assert not is_deterministically_safe_keep("♪ music playing ♪", "non_verbal")
    # Pure symbols/music notes without letters are True
    assert is_deterministically_safe_keep("♪ ♪", "non_verbal")
    assert is_deterministically_safe_keep("♬", "non_verbal")
    # Valid vocalizations and sounds
    assert is_deterministically_safe_keep("Mmm.", "non_verbal")
    assert is_deterministically_safe_keep("Mmm!", "non_verbal")
    assert is_deterministically_safe_keep("Ha!", "non_verbal")
    assert is_deterministically_safe_keep("Hmm...", "non_verbal")
    assert is_deterministically_safe_keep("Ugh!", "non_verbal")

    # Proper nouns in show_title or single words must NOT be auto-kept
    assert not is_deterministically_safe_keep("Millennials!", "proper_noun", show_title="Mostly 4 Millennials")
    assert not is_deterministically_safe_keep("Seinfeld", "proper_noun", show_title="Seinfeld")
    assert not is_deterministically_safe_keep("Bear!", "proper_noun", show_title="The Bear")
    assert not is_deterministically_safe_keep("Office!", "proper_noun", show_title="The Office")
    assert not is_deterministically_safe_keep("Friends!", "proper_noun", show_title="Friends")
    assert not is_deterministically_safe_keep("Lost!", "proper_noun", show_title="Lost")
    assert not is_deterministically_safe_keep("House!", "proper_noun", show_title="House")
    assert not is_deterministically_safe_keep("You!", "proper_noun", show_title="You")
    assert not is_deterministically_safe_keep("May!", "proper_noun", show_title="May")
    assert not is_deterministically_safe_keep("From!", "proper_noun", show_title="From")

    # Single words / names without multi-token proof are rejected
    assert not is_deterministically_safe_keep("May!", "proper_noun")
    assert not is_deterministically_safe_keep("Will!", "proper_noun")
    assert not is_deterministically_safe_keep("Rose!", "proper_noun")
    assert not is_deterministically_safe_keep("Falcon?", "proper_noun")
    assert not is_deterministically_safe_keep("Falcon?", "acronym")
    assert not is_deterministically_safe_keep("Falcon?", "brand")
    assert not is_deterministically_safe_keep("SadStar.", "proper_noun")
    assert not is_deterministically_safe_keep("SadStar.", "brand")
    assert not is_deterministically_safe_keep("BlueMoon", "brand")
    assert not is_deterministically_safe_keep("GreenLight", "brand")

    # Real dialogue words must never be kept as non-verbal
    for word in ["No!", "Yes!", "Why?", "Hey!", "Stop!", "Help!", "Right!", "Okay!", "What?", "Go!", "Run!"]:
        assert not is_deterministically_safe_keep(word, "non_verbal")

def test_adversarial_keep_cases():
    # 1. Title Case multi-word phrases must NOT be kept as proper nouns
    for phrase in [
        "Red Alert", "Green Light", "Dead Body", "Black Car", "White House",
        "Good Morning", "Happy Birthday", "Big Problem", "Last Chance",
        "First Time", "New Plan", "Bad Idea", "Blue Moon", "John Smith", "New York"
    ]:
        assert not is_deterministically_safe_keep(phrase, "proper_noun")

    # 2. Generic all-caps / consonant heuristics must NOT be kept as acronyms
    for pseudo_acronym in [
        "BRR", "PSST", "SHH", "GRR", "HMM", "WTF", "OMG", "LOL",
        "RUN", "STOP", "HELP", "FALCON", "ATTACK"
    ]:
        assert not is_deterministically_safe_keep(pseudo_acronym, "acronym")
        assert not is_deterministically_safe_keep(pseudo_acronym, "brand")

    # 3. show_title="Mostly 4 Millennials", cue="Millennials!" -> NOT KEEP on title-match
    assert not is_deterministically_safe_keep("Millennials!", "proper_noun", show_title="Mostly 4 Millennials")

    # 4. show_title matches for other series -> NOT KEEP
    assert not is_deterministically_safe_keep("Bear!", "proper_noun", show_title="The Bear")
    assert not is_deterministically_safe_keep("Office!", "proper_noun", show_title="The Office")
    assert not is_deterministically_safe_keep("Friends!", "proper_noun", show_title="Friends")

    # 5. Ambiguous single names / words -> NOT KEEP
    assert not is_deterministically_safe_keep("May!", "proper_noun")
    assert not is_deterministically_safe_keep("Will!", "proper_noun")
    assert not is_deterministically_safe_keep("Rose!", "proper_noun")

    # 6. Strict allowlist for tech & brand & acronyms -> KEEP
    for token in ["WiFi", "Wi-Fi", "Bluetooth", "GPS", "USB", "DNA", "FBI", "NASA", "CIA"]:
        assert is_deterministically_safe_keep(token, "acronym") or is_deterministically_safe_keep(token, "brand")

def test_normalize_for_compare():
    assert normalize_for_compare("Hello, World!") == normalize_for_compare("hello world")
    assert normalize_for_compare("<i>Hello</i>...") == "hello"
    assert normalize_for_compare("123-456") == "123456"

def test_fail_closed_qa_gate():
    pipeline = SubtitlePipeline()

    # 1. Exact match passes
    source = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hello")]
    trans_good = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hej")]
    res = pipeline.qa_gate(source, trans_good, "sv")
    assert res["passed"] is True

    # 2. Untranslated line fails closed
    trans_untrans = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "Hello")]
    res = pipeline.qa_gate(source, trans_untrans, "sv")
    assert res["passed"] is False
    assert 0 in res["real_untranslated_ids"]

    # 3. Dropped line fails closed
    trans_dropped = [srt.Subtitle(1, timedelta(seconds=1), timedelta(seconds=2), "<i></i>")]
    res = pipeline.qa_gate(source, trans_dropped, "sv")
    assert res["passed"] is False
    assert res["dropped_count"] == 1

    # 4. Sync drift fails closed
    trans_drift = [srt.Subtitle(1, timedelta(seconds=2), timedelta(seconds=3), "Hej")]
    res = pipeline.qa_gate(source, trans_drift, "sv")
    assert res["passed"] is False
    assert res["sync_diff_ms"] > 0

def test_webhook_token_constant_time_and_headers():
    with patch("app.core.db.get_setting", return_value="secret123"):
        # Valid query param
        req_query = MagicMock()
        req_query.query_params = {"secret": "secret123"}
        req_query.headers = {}
        validate_webhook_token(req_query)  # Should not raise

        # Valid X-Webhook-Secret header
        req_hdr = MagicMock()
        req_hdr.query_params = {}
        req_hdr.headers = {"X-Webhook-Secret": "secret123"}
        validate_webhook_token(req_hdr)  # Should not raise

        # Valid Authorization Bearer header
        req_auth = MagicMock()
        req_auth.query_params = {}
        req_auth.headers = {"Authorization": "Bearer secret123"}
        validate_webhook_token(req_auth)  # Should not raise

        # Invalid secret raises 401
        req_bad = MagicMock()
        req_bad.query_params = {"secret": "wrong"}
        req_bad.headers = {}
        with pytest.raises(HTTPException) as exc:
            validate_webhook_token(req_bad)
        assert exc.value.status_code == 401

def test_extractor_fail_closed_on_corrupt(tmp_path):
    # If mkv extraction results in corrupt/unparseable file, extract_embedded_srt must return False
    output_srt = str(tmp_path / "corrupt.en.srt")
    with open(output_srt, "w") as f:
        f.write("NOT A VALID SRT CONTENT")

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        with patch("app.core.extractor.inspect_mkv_tracks", return_value={"subtitles": [{"id": 0, "codec": "SubRip/SRT", "language": "eng"}], "duration": 3000}):
            # Ensure corrupted file is rejected
            res = extract_embedded_srt("fake.mkv", "en", output_srt)
            assert res is False
