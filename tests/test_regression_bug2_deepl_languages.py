import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.services.translator import SubtitleTranslator
from app.core.languages import get_deepl_source_code, get_deepl_target_code


def test_deepl_language_code_mappings():
    """Verify that get_deepl_source_code and get_deepl_target_code map languages correctly without truncation bugs."""
    # German
    assert get_deepl_source_code("German") == "DE"
    assert get_deepl_source_code("de") == "DE"
    assert get_deepl_target_code("German") == "DE"

    # Spanish
    assert get_deepl_source_code("Spanish") == "ES"
    assert get_deepl_source_code("es") == "ES"
    assert get_deepl_target_code("Spanish") == "ES"

    # Dutch
    assert get_deepl_source_code("Dutch") == "NL"
    assert get_deepl_source_code("nl") == "NL"
    assert get_deepl_target_code("Dutch") == "NL"

    # Swedish
    assert get_deepl_source_code("Swedish") == "SV"
    assert get_deepl_source_code("sv") == "SV"
    assert get_deepl_target_code("Swedish") == "SV"

    # French
    assert get_deepl_source_code("French") == "FR"
    assert get_deepl_source_code("fr") == "FR"
    assert get_deepl_target_code("French") == "FR"

    # Portuguese
    assert get_deepl_source_code("Portuguese") == "PT"
    assert get_deepl_source_code("pt") == "PT"
    assert get_deepl_target_code("Portuguese") == "PT-PT"
    assert get_deepl_target_code("pt-br") == "PT-BR"

    # English
    assert get_deepl_source_code("English") == "EN"
    assert get_deepl_target_code("English") == "EN-US"
    assert get_deepl_target_code("en-gb") == "EN-GB"


@pytest.mark.asyncio
async def test_deepl_translate_batch_non_english_source():
    """Verify translate_batch_deepl generates correct source_lang and target_lang for non-English sources."""
    translator = SubtitleTranslator()

    with patch("app.services.translator.get_setting") as mock_gs, \
         patch("httpx.AsyncClient.post") as mock_post:

        mock_gs.side_effect = lambda k, d="": "deepl" if k == "ai_provider" else ("test-key" if k == "deepl_api_key" else d)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"translations": [{"text": "God morgon"}]}
        mock_post.return_value = mock_resp

        # German to Swedish
        items = [{"id": 1, "text": "Guten Morgen"}]
        res = await translator.translate_batch_deepl(items, target_language="Swedish", source_language="German")
        assert len(res) == 1
        assert res[0]["text"] == "God morgon"

        sent_json = mock_post.call_args.kwargs["json"]
        assert sent_json["source_lang"] == "DE"
        assert sent_json["target_lang"] == "SV"

        # Spanish to Dutch
        items_es = [{"id": 2, "text": "Buenos días"}]
        mock_resp.json.return_value = {"translations": [{"text": "Goedemorgen"}]}
        await translator.translate_batch_deepl(items_es, target_language="Dutch", source_language="Spanish")
        sent_json_es = mock_post.call_args.kwargs["json"]
        assert sent_json_es["source_lang"] == "ES"
        assert sent_json_es["target_lang"] == "NL"


@pytest.mark.asyncio
async def test_deepl_escalation_single_line_non_english_source():
    """Verify _execute_single_escalation_call does not hardcode EN and uses source_language."""
    translator = SubtitleTranslator()

    with patch("app.services.translator.get_setting") as mock_gs, \
         patch("httpx.AsyncClient.post") as mock_post:

        mock_gs.side_effect = lambda k, d="": "deepl" if k == "ai_provider" else ("test-key" if k == "deepl_api_key" else d)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"translations": [{"text": "Hallo wereld"}]}
        mock_post.return_value = mock_resp

        # Call with French source -> Dutch target
        res = await translator._execute_single_escalation_call(
            provider="deepl",
            model_name="deepl",
            system_prompt="",
            prompt="",
            schema={},
            target_language="Dutch",
            target_text="Bonjour le monde",
            source_language="French",
        )
        assert res is not None
        sent_json = mock_post.call_args.kwargs["json"]
        assert sent_json["source_lang"] == "FR"
        assert sent_json["target_lang"] == "NL"


@pytest.mark.asyncio
async def test_deepl_first_pass_micro_repair_batch():
    """Verify first_pass_micro_repair_batch forwards correct source_lang to DeepL API."""
    from app.core.ai_providers import ProviderContext
    translator = SubtitleTranslator()

    with patch("app.services.translator.get_setting") as mock_gs, \
         patch("httpx.AsyncClient.post") as mock_post:

        # After hardening, provider is resolved via provider_ctx, not get_setting("ai_provider").
        # get_setting is still called for deepl_api_key.
        mock_gs.side_effect = lambda k, d="": ("test-key" if k == "deepl_api_key" else d)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"translations": [{"text": "Reparerad text"}]}
        mock_post.return_value = mock_resp

        repair_items = [{"id": 1, "target": "Tekst te repareren"}]
        res = await translator.first_pass_micro_repair_batch(
            repair_items=repair_items,
            target_language="Swedish",
            source_language="Dutch",
            provider_ctx=ProviderContext(provider="deepl", model="prefer_quality_optimized"),
        )
        assert len(res) == 1
        sent_json = mock_post.call_args.kwargs["json"]
        assert sent_json["source_lang"] == "NL"
        assert sent_json["target_lang"] == "SV"


@pytest.mark.asyncio
async def test_deepl_bulk_contextual_and_strict_recovery():
    """Verify bulk_contextual_recovery and bulk_strict_recovery pass correct non-English source codes."""
    from app.core.ai_providers import ProviderContext
    translator = SubtitleTranslator()

    with patch("app.services.translator.get_setting") as mock_gs, \
         patch("httpx.AsyncClient.post") as mock_post:

        # After hardening, provider is resolved via provider_ctx.
        mock_gs.side_effect = lambda k, d="": ("test-key" if k == "deepl_api_key" else d)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"translations": [{"text": "Återställd rad"}]}
        mock_post.return_value = mock_resp

        _deepl_ctx = ProviderContext(provider="deepl", model="prefer_quality_optimized")

        # Contextual recovery with German source -> Swedish target
        recovery_items = [{"id": 1, "target": "Hallo Welt"}]
        await translator.bulk_contextual_recovery(
            recovery_items=recovery_items,
            target_language="Swedish",
            source_language="German",
            provider_ctx=_deepl_ctx,
        )
        sent_ctx = mock_post.call_args.kwargs["json"]
        assert sent_ctx["source_lang"] == "DE"
        assert sent_ctx["target_lang"] == "SV"

        # Strict recovery with Portuguese source -> Swedish target
        await translator.bulk_strict_recovery(
            recovery_items=recovery_items,
            target_language="Swedish",
            source_language="Portuguese",
            provider_ctx=_deepl_ctx,
        )
        sent_strict = mock_post.call_args.kwargs["json"]
        assert sent_strict["source_lang"] == "PT"
        assert sent_strict["target_lang"] == "SV"
