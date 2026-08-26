import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.core.ai_providers import context_from_settings, resolve_job_provider_context, ProviderContext
from app.core.db import create_job, pin_job_provider, set_setting, get_job_by_id, init_db
from app.services.translator import SubtitleTranslator
from app.services.pipeline import SubtitlePipeline


@pytest.fixture(autouse=True)
def setup_clean_db():
    init_db()
    set_setting("ai_provider", "gemini")
    set_setting("gemini_model", "gemini-3.5-flash-lite")
    set_setting("gemini_api_key", "test_key")
    set_setting("openai_model", "gpt-4o-mini")
    set_setting("openai_api_key", "test_key")
    set_setting("escalate_to_pro", "false")
    set_setting("escalation_provider", "none")
    set_setting("escalation_model", "")
    yield
    set_setting("ai_provider", "gemini")
    set_setting("gemini_model", "gemini-3.5-flash-lite")
    set_setting("escalate_to_pro", "false")
    set_setting("escalation_provider", "none")
    set_setting("escalation_model", "")


@pytest.mark.asyncio
async def test_scenario_1_pinned_model_preserved_when_same_provider_settings_change():
    """
    TEST 1:
    Job starts with provider A (gemini) and model X (gemini-3.5-flash-lite).
    Settings change to provider A (gemini) and model Y (gemini-3.7-flash).
    Verify all recovery / QA / repair paths use model X, not model Y.
    """
    job_id = create_job("test_s1.mkv")
    pin_job_provider(job_id, primary_provider="gemini", primary_model="gemini-3.5-flash-lite")

    # Change settings to new model for gemini
    set_setting("gemini_model", "gemini-3.7-flash")

    translator = SubtitleTranslator()
    dispatches = []

    async def mock_dispatch(provider, model_name, system_prompt, user_prompt, **kwargs):
        dispatches.append({"provider": provider, "model": model_name, "func": kwargs.get("job_id")})
        # Return generic valid JSON
        if "results" in system_prompt or "results" in user_prompt:
            return '{"results": [{"id": 1, "action": "keep", "reason": "proper_noun", "text": "Test", "classification": "DIALOGUE", "invariant_in_target": true, "explanation": "ok"}]}'
        if "translations" in system_prompt or "translations" in user_prompt:
            return '{"translations": [{"id": 1, "text": "Test"}]}'
        return '{"results": [{"id": 1, "text": "Test"}]}'

    with patch.object(translator, "_dispatch_llm_completion", side_effect=mock_dispatch):
        # 1. classify_and_recover_identical
        dispatches.clear()
        await translator.classify_and_recover_identical(
            [{"id": 1, "text": "Hello"}],
            target_language="Swedish",
            show_title="TestShow",
            job_id=job_id
        )
        assert len(dispatches) == 1
        assert dispatches[0]["provider"] == "gemini"
        assert dispatches[0]["model"] == "gemini-3.5-flash-lite", f"Expected pinned model, got {dispatches[0]['model']}"

        # 2. verify_alphabetic_invariants_batch
        dispatches.clear()
        await translator.verify_alphabetic_invariants_batch(
            [{"id": 1, "text": "Hello", "target": "Hello"}],
            target_language="Swedish",
            show_title="TestShow",
            job_id=job_id
        )
        assert len(dispatches) == 1
        assert dispatches[0]["provider"] == "gemini"
        assert dispatches[0]["model"] == "gemini-3.5-flash-lite"

        # 3. classify_sdh_segments
        dispatches.clear()
        await translator.classify_sdh_segments(
            [{"id": 1, "text": "[sighs]"}],
            source_language="English",
            job_id=job_id
        )
        assert len(dispatches) == 1
        assert dispatches[0]["provider"] == "gemini"
        assert dispatches[0]["model"] == "gemini-3.5-flash-lite"

        # 4. first_pass_micro_repair_batch
        dispatches.clear()
        await translator.first_pass_micro_repair_batch(
            [{"id": 1, "target": "Hello"}],
            target_language="Swedish",
            show_title="TestShow",
            job_id=job_id
        )
        assert len(dispatches) == 1
        assert dispatches[0]["provider"] == "gemini"
        assert dispatches[0]["model"] == "gemini-3.5-flash-lite"

        # 5. bulk_contextual_recovery
        dispatches.clear()
        await translator.bulk_contextual_recovery(
            [{"id": 1, "target": "Hello"}],
            target_language="Swedish",
            show_title="TestShow",
            job_id=job_id
        )
        assert len(dispatches) == 1
        assert dispatches[0]["provider"] == "gemini"
        assert dispatches[0]["model"] == "gemini-3.5-flash-lite"

        # 6. bulk_strict_recovery
        dispatches.clear()
        await translator.bulk_strict_recovery(
            [{"id": 1, "target": "Hello"}],
            target_language="Swedish",
            show_title="TestShow",
            job_id=job_id
        )
        assert len(dispatches) == 1
        assert dispatches[0]["provider"] == "gemini"
        assert dispatches[0]["model"] == "gemini-3.5-flash-lite"

        # 7. repair_alignment_region
        dispatches.clear()
        await translator.repair_alignment_region(
            repair_cue_ids=[1],
            source_context_items=[{"id": 1, "text": "Hello"}],
            target_context_items=[{"id": 1, "text": "Hej"}],
            target_language="Swedish",
            show_title="TestShow",
            job_id=job_id
        )
        assert len(dispatches) == 1
        assert dispatches[0]["provider"] == "gemini"
        assert dispatches[0]["model"] == "gemini-3.5-flash-lite"


@pytest.mark.asyncio
async def test_scenario_2_pinned_provider_and_model_preserved_when_both_settings_change():
    """
    TEST 2:
    Job starts with provider A (gemini) and model X (gemini-3.5-flash-lite).
    Settings change completely to provider B (openai) and model Z (gpt-4o).
    Ongoing job must STILL use provider A (gemini) and model X (gemini-3.5-flash-lite).
    """
    job_id = create_job("test_s2.mkv")
    pin_job_provider(job_id, primary_provider="gemini", primary_model="gemini-3.5-flash-lite")

    # Global settings changed to OpenAI gpt-4o
    set_setting("ai_provider", "openai")
    set_setting("openai_model", "gpt-4o")

    translator = SubtitleTranslator()
    dispatches = []

    async def mock_dispatch(provider, model_name, system_prompt, user_prompt, **kwargs):
        dispatches.append({"provider": provider, "model": model_name})
        return '{"results": [{"id": 1, "text": "Hej"}]}'

    with patch.object(translator, "_dispatch_llm_completion", side_effect=mock_dispatch):
        await translator.bulk_contextual_recovery(
            [{"id": 1, "target": "Hello"}],
            target_language="Swedish",
            show_title="TestShow",
            job_id=job_id
        )
        assert len(dispatches) == 1
        assert dispatches[0]["provider"] == "gemini"
        assert dispatches[0]["model"] == "gemini-3.5-flash-lite"


@pytest.mark.asyncio
async def test_scenario_3_new_jobs_use_new_settings():
    """
    TEST 3:
    Pinning job 1 does not affect new jobs.
    Job 2 created after settings change uses the new provider & model.
    """
    # Job 1 pinned to gemini / gemini-3.5-flash-lite
    job_1 = create_job("test_job_1.mkv")
    pin_job_provider(job_1, primary_provider="gemini", primary_model="gemini-3.5-flash-lite")

    # Settings changed to Anthropic claude-3-5-sonnet
    set_setting("ai_provider", "anthropic")
    set_setting("anthropic_model", "claude-3-5-sonnet-20241022")

    # Job 2 created and pinned with new settings
    job_2 = create_job("test_job_2.mkv")
    ctx_2 = context_from_settings()
    pin_job_provider(job_2, primary_provider=ctx_2.provider, primary_model=ctx_2.model)

    resolved_1 = resolve_job_provider_context(job_1)
    resolved_2 = resolve_job_provider_context(job_2)

    assert resolved_1.provider == "gemini"
    assert resolved_1.model == "gemini-3.5-flash-lite"

    assert resolved_2.provider == "anthropic"
    assert resolved_2.model == "claude-3-5-sonnet-20241022"


@pytest.mark.asyncio
async def test_scenario_4_pinning_happens_before_first_ai_call():
    """
    TEST 4:
    Verify that job pinning in pipeline logic occurs BEFORE any AI call,
    including SDH classification.
    """
    pipeline_service = SubtitlePipeline()
    job_id = create_job("test_s4.mkv")

    # When SDH classifier runs, assert that the job is already pinned in DB
    sdh_called_with_pinned_job = False

    async def mock_classify_sdh(items, source_language="unknown", job_id=None, provider_ctx=None):
        nonlocal sdh_called_with_pinned_job
        assert job_id is not None
        job_db = get_job_by_id(job_id)
        assert job_db is not None
        assert job_db["primary_provider"] == "gemini"
        assert job_db["primary_model"] == "gemini-3.5-flash-lite"
        sdh_called_with_pinned_job = True
        return []

    # Verify translator.classify_sdh_segments resolves pinned context
    with patch.object(pipeline_service.translator, "classify_sdh_segments", side_effect=mock_classify_sdh):
        # Mock source subtitle
        from app.services.source_resolver import SubtitleSource, SourceOrigin
        import srt
        from datetime import timedelta
        mock_cues = [srt.Subtitle(index=1, start=timedelta(seconds=1), end=timedelta(seconds=2), content="(sighs)")]
        mock_source = SubtitleSource(
            path="/tmp/mock_src.srt",
            language="en",
            origin=SourceOrigin.EMBEDDED,
            cues=mock_cues,
            content=srt.compose(mock_cues)
        )

        with patch("app.services.pipeline.inspect_mkv_tracks", return_value={"subtitles": [], "audio": []}), \
             patch("app.services.pipeline.find_external_subtitle", return_value=None), \
             patch("app.services.pipeline.SourceResolver.resolve", return_value=mock_source), \
             patch.object(pipeline_service.translator, "translate_srt_content", return_value=mock_cues), \
             patch("app.services.pipeline._publish_subtitle_atomic", return_value={"published": True}), \
             patch("os.path.exists", return_value=True):

            await pipeline_service._run_pipeline_logic_impl(
                job_id=job_id,
                video_path="test_s4.mkv"
            )

    assert sdh_called_with_pinned_job, "SDH classification must be invoked with job already pinned in DB"


@pytest.mark.asyncio
async def test_scenario_5_escalation_uses_pinned_escalation_model():
    """
    TEST 5:
    Job starts with primary (gemini / gemini-3.5-flash-lite) and escalation (openai / gpt-4o).
    After pinning, escalation settings change to (openai / gpt-4o-mini).
    Escalation must still use the pinned escalation model (gpt-4o).
    """
    job_id = create_job("test_s5.mkv")
    pin_job_provider(
        job_id,
        primary_provider="gemini",
        primary_model="gemini-3.5-flash-lite",
        escalation_enabled=True,
        escalation_provider="openai",
        escalation_model="gpt-4o"
    )

    # Change settings
    set_setting("escalate_to_pro", "true")
    set_setting("escalation_provider", "openai")
    set_setting("escalation_model", "gpt-4o-mini")

    esc_ctx = resolve_job_provider_context(job_id, escalation=True)
    assert esc_ctx.provider == "openai"
    assert esc_ctx.model == "gpt-4o"

    translator = SubtitleTranslator()
    escalation_dispatches = []

    async def mock_exec_escalation(provider, model_name, system_prompt, prompt, schema, target_language, target_text, source_language="English", job_id=None):
        escalation_dispatches.append({"provider": provider, "model": model_name, "job_id": job_id})
        return '{"translation": "Hej"}'

    with patch.object(translator, "_execute_single_escalation_call", side_effect=mock_exec_escalation):
        res = await translator.escalate_single_line(
            target_idx=0,
            target_text="Hello",
            prev_text="",
            next_text="",
            target_language="Swedish",
            show_title="TestShow",
            job_id=job_id
        )
        assert res == "Hej"
        assert len(escalation_dispatches) >= 1
        assert escalation_dispatches[0]["provider"] == "openai"
        assert escalation_dispatches[0]["model"] == "gpt-4o", f"Expected pinned escalation model gpt-4o, got {escalation_dispatches[0]['model']}"


@pytest.mark.asyncio
async def test_non_job_context_uses_current_settings():
    """
    Verify that when no job_id or provider_ctx is passed (e.g. settings test / standalone invocation),
    the current settings model is used.
    """
    set_setting("ai_provider", "gemini")
    set_setting("gemini_model", "gemini-2.5-pro")

    translator = SubtitleTranslator()
    dispatches = []

    async def mock_dispatch(provider, model_name, system_prompt, user_prompt, **kwargs):
        dispatches.append({"provider": provider, "model": model_name})
        return '{"results": [{"id": 1, "text": "Hej"}]}'

    with patch.object(translator, "_dispatch_llm_completion", side_effect=mock_dispatch):
        await translator.bulk_contextual_recovery(
            [{"id": 1, "target": "Hello"}],
            target_language="Swedish",
            show_title="TestShow",
            job_id=None
        )
        assert len(dispatches) == 1
        assert dispatches[0]["provider"] == "gemini"
        assert dispatches[0]["model"] == "gemini-2.5-pro"
