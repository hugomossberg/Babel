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

    # Valid proper nouns
    assert is_deterministically_safe_keep("John Smith", "proper_noun")
    assert is_deterministically_safe_keep("New York", "proper_noun")
    # Brand is fail-closed
    assert not is_deterministically_safe_keep("Microsoft", "brand")

    # Valid acronyms, tech brands and numbers
    assert is_deterministically_safe_keep("NASA", "acronym")
    assert is_deterministically_safe_keep("FBI", "acronym")
    assert is_deterministically_safe_keep("WiFi!", "brand")
    assert is_deterministically_safe_keep("WiFi!", "acronym")
    assert is_deterministically_safe_keep("Wi-Fi", "acronym")
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
