import pytest
from unittest.mock import patch, ANY
from app.api.dashboard import api_save_ai_settings, AISettingsRequest
from app.core.db import DeferStage
from app.services.pipeline import SubtitlePipeline

@pytest.mark.asyncio
@patch("app.api.dashboard.set_setting")
@patch("app.api.dashboard.get_setting")
async def test_a_partial_settings_save(mock_get, mock_set):
    settings_db = {
        "ai_provider": "gemini",
        "gemini_api_key": "old_gemini",
        "openai_api_key": "old_openai",
        "batch_size": "150",
        "escalate_to_pro": "true"
    }
    
    def mock_set_fn(key, val):
        settings_db[key] = val
        
    def mock_get_fn(key, default=""):
        return settings_db.get(key, default)
        
    mock_set.side_effect = mock_set_fn
    mock_get.side_effect = mock_get_fn
    
    req = AISettingsRequest(openai_api_key="new_openai")
    await api_save_ai_settings(req)
    
    assert settings_db["openai_api_key"] == "new_openai"
    assert settings_db["gemini_api_key"] == "old_gemini"
    assert settings_db["ai_provider"] == "gemini"
    assert settings_db["batch_size"] == "150"
    
    req2 = AISettingsRequest(openai_api_key="")
    await api_save_ai_settings(req2)
    
    assert settings_db["openai_api_key"] == ""
    assert settings_db["gemini_api_key"] == "old_gemini"
    
    # Test ignored empty strings for models, urls, providers
    req3 = AISettingsRequest(ai_provider="", gemini_model="", ollama_url="", custom_openai_url="")
    await api_save_ai_settings(req3)
    assert settings_db["ai_provider"] == "gemini"  # Should NOT be overwritten with ""
    assert settings_db["gemini_model"] == "gemini-3.5-flash-lite" if "gemini_model" in settings_db else True

    # Test explicit false
    req4 = AISettingsRequest(escalate_to_pro=False)
    await api_save_ai_settings(req4)
    assert settings_db["escalate_to_pro"] == "false"

@pytest.mark.asyncio
@patch("app.services.pipeline.os.path.exists")
@patch("app.services.pipeline.SubtitlePipeline.get_configured_languages")
@patch("app.services.pipeline.find_external_subtitle")
@patch("app.services.source_resolver.SourceResolver.resolve")
@patch("app.core.db.update_deferred_metadata")
@patch("app.core.db.update_job")
@patch("app.core.db.get_job_by_id")
@patch("app.core.ai_providers.context_from_settings")
async def test_b_primary_budget_progress(mock_ctx, mock_gjbi, mock_upd_job, mock_upd_meta, mock_resolve, mock_find, mock_langs, mock_exists):
    mock_gjbi.return_value = {"id": 1, "processed_lines": 50, "defer_stage": "primary", "primary_provider": "gemini"}
    mock_exists.return_value = True
    mock_langs.return_value = [{"name": "Swedish", "code": "sv"}]
    mock_find.return_value = None
    from unittest.mock import MagicMock
    m = MagicMock()
    m.path = "test.en.srt"
    m.language = "eng"
    m.origin = MagicMock()
    m.origin.value = "EXTERNAL"
    m.cues = [1]
    mock_resolve.return_value = m

    def raise_quota(*args, **kwargs):
        from app.core.quota import RequestBudgetExhaustedError
        raise RequestBudgetExhaustedError("gemini", 100, 100)

    mock_ctx.side_effect = raise_quota

    pipeline = SubtitlePipeline()
    res = await pipeline._run_pipeline_logic_impl(1, "test.mkv")

    assert res["status"] == "deferred"
    mock_upd_meta.assert_called_with(
        1,
        defer_reason=ANY,
        waiting_provider="gemini",
        waiting_model=None,
        defer_stage=DeferStage.PRIMARY
    )

@pytest.mark.asyncio
@patch("app.services.pipeline.get_setting")
@patch("app.services.pipeline.os.path.exists")
@patch("app.services.pipeline.SubtitlePipeline.get_configured_languages")
@patch("app.services.pipeline.find_external_subtitle")
@patch("app.services.source_resolver.SourceResolver.resolve")
@patch("app.core.db.update_deferred_metadata")
@patch("app.core.db.update_job")
@patch("app.core.db.get_job_by_id")
@patch("app.core.ai_providers.context_from_settings")
@patch("app.core.ai_providers.resolve_job_provider_context")
@patch("app.services.pipeline.SubtitlePipeline.check_semantic_cue_alignment")
@patch("app.services.pipeline.qa_gate")
@patch("app.services.translator.SubtitleTranslator.bulk_strict_recovery")
@patch("app.services.translator.SubtitleTranslator.bulk_contextual_recovery")
@patch("app.services.translator.SubtitleTranslator.escalate_single_line")
@patch("app.services.translator.SubtitleTranslator.translate_srt_content")
async def test_c_escalation_budget_progress(
    mock_trans_srt,
    mock_escalate,
    mock_bulk_rec,
    mock_bulk_strict,
    mock_qa,
    mock_alignment,
    mock_resolve_ctx,
    mock_ctx_settings,
    mock_gjbi,
    mock_upd_job,
    mock_upd_meta,
    mock_resolve_src,
    mock_find_ext,
    mock_langs,
    mock_exists,
    mock_get_setting
):
    import datetime
    import srt
    from unittest.mock import MagicMock
    from app.core.ai_providers import ProviderContext
    from app.core.quota import RequestBudgetExhaustedError

    mock_exists.return_value = True
    mock_langs.return_value = [{"name": "Swedish", "code": "sv"}]
    mock_find_ext.return_value = None
    mock_gjbi.return_value = {"id": 1, "processed_lines": 0, "defer_stage": "primary", "primary_provider": "gemini"}

    def fake_get_setting(key, default=""):
        if key == "escalate_to_pro":
            return "true"
        if key == "clean_sdh":
            return "false"
        if key == "enable_bazarr_check":
            return "false"
        return default

    mock_get_setting.side_effect = fake_get_setting

    srt_content = "1\n00:00:01,000 --> 00:00:02,000\nHello\n\n"
    cue = srt.Subtitle(index=1, start=datetime.timedelta(seconds=1), end=datetime.timedelta(seconds=2), content="Hello")
    
    src = MagicMock()
    src.path = "test.en.srt"
    src.language = "eng"
    src.language_name = "English"
    src.origin = MagicMock()
    src.origin.value = "EXTERNAL"
    src.content = srt_content
    src.cues = [cue]
    mock_resolve_src.return_value = src

    def get_ctx(escalation=False, **kwargs):
        if escalation:
            return ProviderContext(provider="openai", model="gpt-4o")
        return ProviderContext(provider="gemini", model="gemini-1.5-flash")

    mock_ctx_settings.side_effect = get_ctx
    mock_resolve_ctx.side_effect = lambda jid, escalation=False: get_ctx(escalation=escalation)

    mock_trans_srt.return_value = [cue]
    mock_alignment.return_value = {"incidents": [], "affected_indices": []}
    mock_bulk_rec.return_value = []
    mock_bulk_strict.return_value = []
    mock_qa.return_value = {
        "passed": False,
        "score": 50,
        "untranslated_ids": [],
        "real_untranslated_ids": [0],
        "wrong_language_ids": [0],
        "dropped_details": []
    }

    mock_escalate.side_effect = RequestBudgetExhaustedError("openai", 100, 100)

    pipeline = SubtitlePipeline()
    res = await pipeline._run_pipeline_logic_impl(1, "test.mkv")

    assert res["status"] == "deferred"
    mock_upd_meta.assert_called_with(
        1,
        defer_reason=ANY,
        waiting_provider="openai",
        waiting_model=None,
        defer_stage=DeferStage.ESCALATION
    )

