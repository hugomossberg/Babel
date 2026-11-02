"""Issue #18: slow inference must be configurable and timeout storms must stop."""
import copy
import json
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import srt
from fastapi.testclient import TestClient

from app.core import db
from app.main import app
from app.services import translator as tr
from app.services.pipeline import SubtitlePipeline


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    # Keep settings mutations and job retries isolated from the rest of the suite.
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'settings.db'))
    db.init_db()
    db.set_setting('llm_request_timeout_seconds', '0')
    db.set_setting('ai_provider', 'ollama')
    db.set_setting('ollama_model', 'llama3')
    db.set_setting('batch_concurrency', '1')
    db.set_setting('enable_bazarr_check', 'false')
    db.set_setting('clean_sdh', 'false')
    db.set_setting('languages', '[{"name":"Swedish","code":"sv","enabled":true}]')


def test_timeout_settings_roundtrip_and_partial_update():
    with TestClient(app) as client:
        response = client.post('/api/settings/ai', json={'llm_request_timeout_seconds': 900})
        assert response.status_code == 200
        assert client.get('/api/settings/all').json()['ai']['llm_request_timeout_seconds'] == 900
        client.post('/api/settings/ai', json={'batch_size': 150})
        assert db.get_setting('llm_request_timeout_seconds') == '900'
        assert client.post('/api/settings/ai', json={'llm_request_timeout_seconds': 0}).status_code == 200


@pytest.mark.parametrize('value', [-1, 3601, 1.5, 'NaN'])
def test_invalid_timeout_is_rejected(value):
    with TestClient(app) as client:
        assert client.post('/api/settings/ai', json={'llm_request_timeout_seconds': value}).status_code == 422


@pytest.mark.parametrize('value', ['', 'junk', '-1', '3601', 'NaN', 'inf', '1.5'])
def test_corrupt_legacy_timeout_keeps_provider_default(value):
    db.set_setting('llm_request_timeout_seconds', value)
    assert tr.get_llm_request_timeout() is None


@pytest.mark.asyncio
@pytest.mark.parametrize('entrypoint', ['primary', 'qa', 'audit', 'escalation'])
async def test_ollama_timeout_applies_to_primary_and_all_recovery_dispatch(entrypoint):
    db.set_setting('llm_request_timeout_seconds', '900')
    client = AsyncMock()
    client.post.return_value = httpx.Response(200, json={'response': '{"results":[{"id":0,"text":"Hej"}]}'} , request=httpx.Request('POST', 'http://ollama/api/generate'))
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=client)
    manager.__aexit__ = AsyncMock(return_value=False)
    translator = tr.SubtitleTranslator()
    with patch('httpx.AsyncClient', return_value=manager) as factory:
        if entrypoint == 'primary':
            await translator.translate_batch_ollama.__wrapped__(translator, [{'id': 0, 'text': 'Hello'}], 'Swedish', 'llama3')
        elif entrypoint == 'qa':
            await translator.classify_and_recover_identical.__wrapped__(translator, [{'id': 0, 'text': 'Hello'}], 'Swedish', '')
        elif entrypoint == 'audit':
            await translator.audit_cue_alignment_window.__wrapped__(translator, [{'id':0,'text':'Hello'}], [{'id':0,'text':'Hej'}])
        else:
            await translator._execute_single_escalation_call.__wrapped__(translator, 'ollama', 'llama3', '', '', {}, 'Swedish', 'Hello')
        assert factory.call_args.kwargs['timeout'] == 900
        assert client.post.await_count == 1


@pytest.mark.parametrize('provider', ['gemini', 'openai'])
def test_cached_sdk_clients_refresh_when_timeout_changes(provider):
    db.set_setting(f'{provider}_api_key', 'test-key')
    translator = tr.SubtitleTranslator()
    getter = getattr(translator, f'get_{provider}_client')
    target = 'google.genai.Client' if provider == 'gemini' else 'openai.OpenAI'
    with patch(target, side_effect=[MagicMock(), MagicMock()]) as factory:
        original = getter()
        assert 'timeout' not in factory.call_args.kwargs
        assert 'http_options' not in factory.call_args.kwargs
        db.set_setting('llm_request_timeout_seconds', '900')
        changed = getter()
        assert changed is not original
        assert getter() is changed
        kwargs = factory.call_args.kwargs
        if provider == 'gemini':
            assert kwargs['http_options'].timeout == 900000
        else:
            assert kwargs['timeout'] == 900
        assert factory.call_count == 2


@pytest.mark.asyncio
async def test_empty_timeout_is_typed_and_does_not_retry_inference():
    translator = tr.SubtitleTranslator()
    client = AsyncMock()
    client.post.side_effect = httpx.ReadTimeout('')
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=client)
    manager.__aexit__ = AsyncMock(return_value=False)
    with patch('httpx.AsyncClient', return_value=manager), patch('asyncio.sleep', new_callable=AsyncMock) as sleep:
        with pytest.raises(tr.ProviderRequestTimeout, match='ollama.*timeout'):
            await translator.classify_and_recover_identical([{'id':0,'text':'Hello'}], 'Swedish', '')
        assert client.post.await_count == 1
        sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_primary_batches_resume_after_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'test.db'))
    db.init_db()
    db.set_setting('batch_concurrency', '1')
    translator = tr.SubtitleTranslator()
    subs = [srt.Subtitle(i + 1, timedelta(seconds=i), timedelta(seconds=i + 1), text)
            for i, text in enumerate(['Hello there', 'Where are you?'])]
    job_id = db.create_job(str(tmp_path / 'video.mkv'))
    with patch.object(translator, 'translate_batch', side_effect=[
            [{'id':0,'text':'Hej där'}], tr.ProviderRequestTimeout('ollama request timeout')]):
        with pytest.raises(tr.ProviderRequestTimeout):
            await translator.translate_srt_content(subs, 'Swedish', batch_size=1, job_id=job_id)
    partial = json.loads((tmp_path / f'job_{job_id}_sv_partial.json').read_text())
    assert partial['lines'] == {'0': 'Hej där'}
    with patch.object(translator, 'translate_batch', new_callable=AsyncMock, return_value=[{'id':1,'text':'Var är du?'}]) as translate:
        result = await translator.translate_srt_content(subs, 'Swedish', batch_size=1, job_id=job_id)
    assert translate.await_count == 1
    assert translate.call_args.args[0] == [{'id':1,'text':'Where are you?'}]
    assert [cue.content for cue in result] == ['Hej där', 'Var är du?']
    assert [(cue.start, cue.end) for cue in result] == [(cue.start, cue.end) for cue in subs]


@pytest.mark.asyncio
@pytest.mark.parametrize('retry_count, expected_status', [(0, 'waiting_provider'), (3, 'failed')])
async def test_qa_timeout_stops_remaining_chunks_and_eventually_stops_automatic_retries(tmp_path, retry_count, expected_status):
    pipeline = SubtitlePipeline()
    video = tmp_path / 'timeout.mkv'
    video.touch()
    subs = [srt.Subtitle(i+1, timedelta(seconds=i*2), timedelta(seconds=i*2+1), f'We should leave this place quickly {i}.') for i in range(45)]
    video.with_suffix('.en.srt').write_text(srt.compose(subs))
    job_id = db.create_job(str(video))
    db.update_job(job_id, retry_count=retry_count)
    # Exercise the real classifier and httpx timeout through pipeline's chunk loop.
    client = AsyncMock()
    client.post.side_effect = httpx.ReadTimeout('')
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=client)
    manager.__aexit__ = AsyncMock(return_value=False)
    with patch.object(pipeline.translator, 'translate_srt_content', new_callable=AsyncMock, return_value=copy.deepcopy(subs)), \
         patch.object(pipeline, 'check_semantic_cue_alignment', new_callable=AsyncMock, return_value={'affected_indices':[]}), \
         patch('httpx.AsyncClient', return_value=manager):
        result = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)
    assert result['status'] == expected_status, result
    assert client.post.await_count == 1
    job = db.get_job_by_id(job_id)
    assert 'timeout' in job['error_message'].lower()
    assert not video.with_suffix('.sv.srt').exists()
    if expected_status == 'failed':
        assert job['next_retry_at'] is None


@pytest.mark.asyncio
async def test_timeout_waits_for_running_success_and_skips_queued_batches(tmp_path, monkeypatch):
    import asyncio
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'test.db'))
    db.init_db()
    db.set_setting('batch_concurrency', '2')
    translator = tr.SubtitleTranslator()
    subs = [srt.Subtitle(i+1, timedelta(seconds=i), timedelta(seconds=i+1), 'Hello there!') for i in range(3)]
    job_id = db.create_job(str(tmp_path / 'video.mkv'))
    second_started = asyncio.Event()
    called = []

    async def translate(payload, **kwargs):
        idx = payload[0]['id']
        called.append(idx)
        if idx == 0:
            await second_started.wait()
            raise tr.ProviderRequestTimeout('ollama request timeout')
        second_started.set()
        await asyncio.sleep(0)
        return [{'id':idx, 'text':'Hej där!'}]

    with patch.object(translator, 'translate_batch', side_effect=translate):
        with pytest.raises(tr.ProviderRequestTimeout):
            await asyncio.wait_for(translator.translate_srt_content(subs, 'Swedish', batch_size=1, job_id=job_id), timeout=2)
    assert called == [0, 1]
    partial = json.loads((tmp_path / f'job_{job_id}_sv_partial.json').read_text())
    assert partial['lines'] == {'1': 'Hej där!'}


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['gemini', 'openai', 'openrouter', 'deepseek', 'custom', 'anthropic'])
async def test_configured_timeout_reaches_other_provider_dispatch(provider):
    db.set_setting('llm_request_timeout_seconds', '900')
    translator = tr.SubtitleTranslator()
    if provider == 'gemini':
        with patch('google.genai.Client') as factory:
            await translator._dispatch_llm_completion(provider, '', '', '')
            assert factory.call_args.kwargs['http_options'].timeout == 900000
    elif provider == 'anthropic':
        client = AsyncMock()
        client.post.return_value = httpx.Response(200, json={'content':[]}, request=httpx.Request('POST', 'http://test'))
        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=client)
        manager.__aexit__ = AsyncMock(return_value=False)
        with patch('httpx.AsyncClient', return_value=manager) as factory:
            await translator._dispatch_llm_completion(provider, '', '', '')
            assert factory.call_args.kwargs['timeout'] == 900
    else:
        with patch('openai.OpenAI') as factory:
            await translator._dispatch_llm_completion(provider, '', '', '')
            assert factory.call_args.kwargs['timeout'] == 900


@pytest.mark.asyncio
async def test_escalation_preserves_timeout_type():
    translator = tr.SubtitleTranslator()
    with patch.object(translator, '_execute_single_escalation_call', side_effect=tr.ProviderRequestTimeout('ollama request timeout')) as call:
        with pytest.raises(tr.ProviderRequestTimeout):
            await translator.escalate_single_line(0, 'Hello there', '', '', 'Swedish', '')
        assert call.await_count == 1


@pytest.mark.asyncio
async def test_audit_does_not_swallow_empty_timeout():
    translator = tr.SubtitleTranslator()
    with patch.object(translator, '_dispatch_llm_completion', side_effect=tr.ProviderRequestTimeout('ollama request timeout')):
        with pytest.raises(tr.ProviderRequestTimeout):
            await translator.audit_cue_alignment_window([{'id':0,'text':'Hello'}], [{'id':0,'text':'Hej'}])


@pytest.mark.asyncio
@pytest.mark.parametrize('override, expected', [('0', 60), ('3600', 3600)])
async def test_ollama_default_and_maximum_timeout(override, expected):
    db.set_setting('llm_request_timeout_seconds', override)
    client = AsyncMock()
    client.post.return_value = httpx.Response(200, json={'response':'{}'}, request=httpx.Request('POST', 'http://test'))
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=client)
    manager.__aexit__ = AsyncMock(return_value=False)
    with patch('httpx.AsyncClient', return_value=manager) as factory:
        await tr.SubtitleTranslator()._dispatch_llm_completion('ollama', '', '', '')
        assert factory.call_args.kwargs['timeout'] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('override, expected', [('0', 30), ('900', 900)])
async def test_deepl_keeps_default_and_honors_override(override, expected):
    db.set_setting('llm_request_timeout_seconds', override)
    client = AsyncMock()
    client.post.return_value = httpx.Response(200, json={'translations':[{'text':'Hej'}]}, request=httpx.Request('POST', 'http://test'))
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=client)
    manager.__aexit__ = AsyncMock(return_value=False)
    with patch('httpx.AsyncClient', return_value=manager) as factory, patch.object(tr, 'get_setting', side_effect=lambda key, default='': 'test-key' if key == 'deepl_api_key' else db.get_setting(key, default)):
        await tr.SubtitleTranslator().translate_batch_deepl.__wrapped__(tr.SubtitleTranslator(), [{'id':0,'text':'Hello'}], 'Swedish')
        assert factory.call_args.kwargs['timeout'] == expected


@pytest.mark.parametrize('provider', ['gemini', 'openai'])
def test_failed_client_refresh_does_not_poison_cache(provider):
    db.set_setting(f'{provider}_api_key', 'test-key')
    translator = tr.SubtitleTranslator()
    getter = getattr(translator, f'get_{provider}_client')
    target = 'google.genai.Client' if provider == 'gemini' else 'openai.OpenAI'
    with patch(target, side_effect=[MagicMock(), RuntimeError('client setup failed'), MagicMock()]) as factory:
        original = getter()
        db.set_setting('llm_request_timeout_seconds', '900')
        with pytest.raises(RuntimeError):
            getter()
        assert getter() is not original
        assert factory.call_count == 3


@pytest.mark.asyncio
async def test_successful_qa_chunks_are_checkpointed_before_later_timeout(tmp_path):
    pipeline = SubtitlePipeline()
    video = tmp_path / 'qa-progress.mkv'
    video.touch()
    subs = [srt.Subtitle(i+1, timedelta(seconds=i*2), timedelta(seconds=i*2+1), f'We should leave this place quickly {i}.') for i in range(45)]
    video.with_suffix('.en.srt').write_text(srt.compose(subs))
    job_id = db.create_job(str(video))
    first_chunk = [{'id':i, 'action':'translate', 'reason':'none', 'text':f'Vi måste lämna den här platsen snabbt {i}.'} for i in range(20)]
    client = AsyncMock()
    client.post.side_effect = [
        httpx.Response(200, json={'response':json.dumps({'results':first_chunk})}, request=httpx.Request('POST', 'http://test')),
        httpx.ReadTimeout(''),
    ]
    manager = MagicMock()
    manager.__aenter__ = AsyncMock(return_value=client)
    manager.__aexit__ = AsyncMock(return_value=False)
    with patch.object(pipeline.translator, 'translate_srt_content', new_callable=AsyncMock, return_value=copy.deepcopy(subs)), \
         patch.object(pipeline, 'check_semantic_cue_alignment', new_callable=AsyncMock, return_value={'affected_indices':[]}), \
         patch('httpx.AsyncClient', return_value=manager):
        result = await pipeline._run_pipeline_logic(job_id, str(video), wait_seconds=0)
    assert result['status'] == 'waiting_provider', result
    assert client.post.await_count == 2
    assert not video.with_suffix('.sv.srt').exists()
    partial_path = tmp_path / f'job_{job_id}_sv_partial.json'
    partial = json.loads(partial_path.read_text())
    assert partial['lines']['0'] == first_chunk[0]['text']
    assert partial['lines']['19'] == first_chunk[19]['text']
    assert partial['lines']['20'] == subs[20].content
    # Check that normal resume sees the same candidates without changing cue timing.
    with patch.object(pipeline.translator, 'translate_batch', new_callable=AsyncMock) as primary:
        resumed = await pipeline.translator.translate_srt_content(subs, 'Swedish', job_id=job_id)
        primary.assert_not_awaited()
    assert len(resumed) == len(subs)
    assert resumed[0].content == first_chunk[0]['text']
    assert resumed[20].content == subs[20].content
    assert [(cue.start, cue.end) for cue in resumed] == [(cue.start, cue.end) for cue in subs]


@pytest.mark.asyncio
async def test_semantic_audit_timeout_stops_queued_chunks():
    pipeline = SubtitlePipeline()
    subs = [srt.Subtitle(i+1, timedelta(seconds=i), timedelta(seconds=i+1), f'Hello there {i}.') for i in range(100)]
    with patch.object(pipeline.translator, 'audit_batch_semantic_integrity', side_effect=tr.ProviderRequestTimeout('ollama request timeout')) as audit:
        with pytest.raises(tr.ProviderRequestTimeout):
            await pipeline.check_semantic_cue_alignment(subs, subs, 'Swedish', 'English', batch_size=2)
        assert audit.await_count == 1


def test_failed_qa_checkpoint_keeps_prior_progress(tmp_path, monkeypatch):
    translator = tr.SubtitleTranslator()
    source = [srt.Subtitle(1, timedelta(), timedelta(seconds=1), 'Hello there')]
    translated = [srt.Subtitle(1, timedelta(), timedelta(seconds=1), 'Hej där')]
    # SQLite may independently remove its WAL/SHM sidecars on connection GC.
    # Isolate checkpoint I/O so the cleanup assertion concerns only our writer.
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    monkeypatch.setattr(db, 'DB_PATH', str(checkpoint_dir / 'state.db'))
    checkpoint = checkpoint_dir / 'job_123_sv_partial.json'
    checkpoint.write_text('previous valid checkpoint')
    original_files = {p.name for p in checkpoint_dir.iterdir()}
    with patch('os.replace', side_effect=OSError('disk full')):
        assert not translator.save_partial_progress(source, translated, 'Swedish', 123)
    assert checkpoint.read_text() == 'previous valid checkpoint'
    assert {p.name for p in checkpoint_dir.iterdir()} == original_files


@pytest.mark.asyncio
@pytest.mark.parametrize('error_class', [tr.ProviderRequestTimeout, tr.ProviderConfigurationError, tr.ProviderUnavailableError])
async def test_structural_retry_provider_error_stops_partitioning(error_class):
    translator = tr.SubtitleTranslator()
    subs = [srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1),
                         text) for i, text in enumerate(['Where are you going?', 'I must find my sister.'])]
    with patch.object(translator, 'translate_batch', new_callable=AsyncMock, side_effect=[
        [], error_class('request failed'),
        [{'id': 0, 'text': 'Vart är du på väg?'}],
        [{'id': 1, 'text': 'Jag måste hitta min syster.'}],
    ]) as translate:
        with pytest.raises(error_class):
            await translator.translate_srt_content(subs, 'Swedish', batch_size=2)
    assert translate.await_count == 2


@pytest.mark.asyncio
async def test_successful_micro_repair_chunks_survive_later_timeout(tmp_path):
    translator = tr.SubtitleTranslator()
    subs = [srt.Subtitle(i + 1, timedelta(seconds=i * 2), timedelta(seconds=i * 2 + 1),
                         f'We should leave this place quickly {i}.') for i in range(51)]
    job_id = db.create_job(str(tmp_path / 'video.mkv'))
    recovered = [{'id': i, 'text': f'Vi måste lämna den här platsen snabbt {i}.'} for i in range(50)]
    with patch.object(translator, 'translate_batch', new_callable=AsyncMock,
                      return_value=[{'id': i, 'text': cue.content} for i, cue in enumerate(subs)]), \
         patch.object(translator, 'first_pass_micro_repair_batch', new_callable=AsyncMock,
                      side_effect=[recovered, tr.ProviderRequestTimeout('request timed out')]) as micro:
        with pytest.raises(tr.ProviderRequestTimeout):
            await translator.translate_srt_content(subs, 'Swedish', batch_size=51, job_id=job_id)
    assert micro.await_count == 2
    partial = json.loads((tmp_path / f'job_{job_id}_sv_partial.json').read_text())
    assert partial['lines']['0'] == recovered[0]['text']
    assert partial['lines']['49'] == recovered[49]['text']
    assert partial['lines']['50'] == subs[50].content
