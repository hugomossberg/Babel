"""Persistent Bazarr grace periods use virtual UTC time, never real waiting."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import sqlite3
from unittest.mock import AsyncMock, patch

from fastapi import BackgroundTasks
from fastapi.testclient import TestClient
import pytest
import srt

from app.core import db
from app.main import app
from app.services.pipeline import SubtitlePipeline
from app.services.source_resolver import BazarrResult, BazarrResultCode
from app.services.bazarr_coordinator import (bazarr_coordinator, BazarrJobsPollResult,
                                              BazarrJobPollStatus, PublicationOwnershipResult)


class Clock(datetime):
    current = datetime(2026, 11, 2, 12, tzinfo=timezone.utc)

    @classmethod
    def now(cls, tz=None):
        return cls.current if tz else cls.current.replace(tzinfo=None)


@pytest.fixture(autouse=True)
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'grace.db'))
    monkeypatch.setattr(db, 'datetime', Clock)
    monkeypatch.setattr(Clock, 'current', datetime(2026, 11, 2, 12, tzinfo=timezone.utc))
    db.init_db()
    monkeypatch.setattr(bazarr_coordinator, 'poll_system_jobs', AsyncMock(
        return_value=BazarrJobsPollResult(status=BazarrJobPollStatus.KNOWN_IDLE)))
    monkeypatch.setattr(bazarr_coordinator, 'acquire_publication_ownership', AsyncMock(
        return_value=PublicationOwnershipResult(granted=True, reason='controlled_idle')))
    for key, value in {
        'enable_bazarr_check': 'true', 'bazarr_api_key': 'fixture-bazarr',
        'grace_period_hours': '24', 'extract_target_embedded': 'false',
        'extract_source_embedded': 'false', 'clean_sdh': 'false',
        'hybrid_bazarr_max_wait_sec': '0', 'source_search_deadline_seconds': '0',
        'notify_jellyfin': 'false', 'notify_plex': 'false',
    }.items():
        db.set_setting(key, value)


def media(tmp_path, name='Episode'):
    video = tmp_path / f'{name}.mkv'
    video.touch()
    cues = [srt.Subtitle(i + 1, timedelta(seconds=i * 3), timedelta(seconds=i * 3 + 2),
                         f'We must discuss what happened here last night {i}.') for i in range(60)]
    video.with_suffix('.en.srt').write_text(srt.compose(cues))
    return video, cues


def swedish(cues):
    return [srt.Subtitle(cue.index, cue.start, cue.end,
                         f'Vi måste prata om det som har hänt ikväll {i}.') for i, cue in enumerate(cues)]


def engine():
    pipeline = SubtitlePipeline()
    pipeline.translator.translate_srt_content = AsyncMock(side_effect=lambda subs, *args, **kwargs: swedish(subs))
    pipeline.trigger_bazarr_search = AsyncMock(return_value=BazarrResult(
        code=BazarrResultCode.TEMPORARY_ERROR, language='sv', detail='Controlled failure'))
    return pipeline


@pytest.mark.parametrize('hours', [0, 1, 25, 48, 8760])
def test_duration_uses_original_start_and_survives_restart(hours):
    job = db.create_job('/test.mkv', 'SONARR')
    result = db.apply_grace_period(job, 'SONARR', hours)
    assert result['waiting'] == (hours > 0)
    if not hours:
        return
    original = db.get_job_by_id(job)
    assert datetime.fromisoformat(original['grace_until']) == Clock.current + timedelta(hours=hours)
    Clock.current += timedelta(minutes=30)
    db.init_db()
    again = db.apply_grace_period(job, 'RETRY', hours)
    assert again['waiting']
    assert db.get_job_by_id(job)['grace_started_at'] == original['grace_started_at']
    assert db.get_job_by_id(job)['grace_until'] == original['grace_until']


@pytest.mark.parametrize('hours, waiting', [(0, False), (1, False), (12, True), (48, True)])
def test_setting_changes_recalculate_from_original_start(hours, waiting):
    job = db.create_job('/test.mkv', 'RADARR')
    db.apply_grace_period(job, 'RADARR', 24)
    start = db.get_job_by_id(job)['grace_started_at']
    Clock.current += timedelta(hours=2)
    result = db.apply_grace_period(job, 'RETRY', hours)
    assert result['waiting'] == waiting
    assert db.get_job_by_id(job)['grace_started_at'] == start
    if not waiting:
        assert db.get_job_by_id(job)['grace_completed'] == 1
        assert not db.apply_grace_period(job, 'RETRY', 48)['waiting']


@pytest.mark.asyncio
async def test_new_automatic_item_waits_without_source_or_ai_or_search_storm(tmp_path):
    video, _ = media(tmp_path)
    pipeline = engine()
    with patch('app.services.pipeline.SourceResolver.resolve', new_callable=AsyncMock) as source:
        result = await pipeline.process_video_file(str(video), event_source='SONARR')
    assert result['status'] == 'waiting_grace'
    job = db.get_job_by_id(result['job_id'])
    assert job['status'] == 'WAITING_GRACE'
    assert job['primary_provider'] is None
    source.assert_not_awaited()
    pipeline.translator.translate_srt_content.assert_not_awaited()
    pipeline.trigger_bazarr_search.assert_not_awaited()
    for _ in range(5):
        assert db.create_job_if_no_active(str(video), 'SONARR')['job_id'] == job['id']
    assert db.get_active_job_for_video(str(video))['id'] == job['id']


@pytest.mark.asyncio
@pytest.mark.parametrize('source', ['MANUAL', 'SONARR'])
async def test_default_zero_and_manual_translation_keep_existing_behavior(tmp_path, source):
    if source == 'SONARR':
        db.set_setting('grace_period_hours', '0')
    video, _ = media(tmp_path)
    pipeline = engine()
    result = await pipeline.process_video_file(str(video), event_source=source)
    assert result['status'] in ('translated', 'success'), result
    assert pipeline.translator.translate_srt_content.await_count == 1
    assert db.get_job_by_id(result['job_id'])['status'] == 'TRANSLATED'


@pytest.mark.asyncio
async def test_valid_target_arrives_during_grace_and_avoids_ai(tmp_path):
    video, cues = media(tmp_path)
    pipeline = engine()
    first = await pipeline.process_video_file(str(video), event_source='RADARR')
    assert first['status'] == 'waiting_grace'
    video.with_suffix('.sv.srt').write_text(srt.compose(swedish(cues)))
    Clock.current += timedelta(minutes=15)
    assert db.claim_job_for_retry(first['job_id'])
    result = await pipeline.process_video_file(str(video), event_source='RETRY', job_id=first['job_id'])
    assert result['status'] == 'skipped', result
    pipeline.translator.translate_srt_content.assert_not_awaited()
    pipeline.trigger_bazarr_search.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('bazarr_finds', [False, True])
async def test_expiry_awaits_new_bazarr_opportunity_then_reuses_existing_fallback(tmp_path, bazarr_finds):
    video, cues = media(tmp_path)
    pipeline = engine()
    first = await pipeline.process_video_file(str(video), event_source='SONARR')
    assert first['status'] == 'waiting_grace'
    Clock.current += timedelta(hours=24)
    assert db.claim_job_for_retry(first['job_id'])

    async def search(*args, **kwargs):
        await asyncio.sleep(0)  # The response must finish before source resolution can cancel it.
        if bazarr_finds:
            video.with_suffix('.sv.srt').write_text(srt.compose(swedish(cues)))
        return BazarrResult(code=BazarrResultCode.TEMPORARY_ERROR, language='sv', detail='Controlled failure')

    pipeline.trigger_bazarr_search.side_effect = search
    result = await pipeline.process_video_file(str(video), event_source='RETRY', job_id=first['job_id'])
    pipeline.trigger_bazarr_search.assert_awaited_once()
    assert pipeline.translator.translate_srt_content.await_count == (0 if bazarr_finds else 1)
    assert result['status'] in ('translated', 'skipped', 'success'), result
    assert db.get_job_by_id(first['job_id'])['grace_completed'] == 1


@pytest.mark.asyncio
async def test_invalid_target_does_not_bypass_validation(tmp_path):
    video, _ = media(tmp_path)
    video.with_suffix('.sv.srt').write_text('not a valid subtitle' * 30)
    pipeline = engine()
    result = await pipeline.process_video_file(str(video), event_source='RADARR')
    assert result['status'] == 'waiting_grace'
    pipeline.translator.translate_srt_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_manual_override_reuses_waiting_job_and_preserves_force_intent(tmp_path):
    from app.api.webhooks import ManualProcessRequest, manual_process
    video, _ = media(tmp_path)
    pipeline = engine()
    first = await pipeline.process_video_file(str(video), event_source='SONARR')
    assert first['status'] == 'waiting_grace'
    tasks = BackgroundTasks()
    with patch('app.api.webhooks.validate_path', side_effect=lambda value: value), \
         patch('app.api.webhooks.pipeline.process_video_file', new=pipeline.process_video_file):
        response = await manual_process(ManualProcessRequest(video_path=str(video), force_retranslate=True), tasks)
        assert response['status'] == 'queued'
        assert response['job_id'] == first['job_id']
        assert db.get_job_by_id(first['job_id'])['grace_completed'] == 1
        await tasks()
    assert db.get_job_by_id(first['job_id'])['force_retranslate'] == 1
    assert pipeline.translator.translate_srt_content.await_count == 1
    with sqlite3.connect(db.DB_PATH) as conn:
        assert conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1


def test_manual_and_scheduler_claim_race_has_one_owner():
    job = db.create_job('/test.mkv', 'RADARR')
    db.apply_grace_period(job, 'RADARR', 24)
    with ThreadPoolExecutor(max_workers=2) as executor:
        claimed, manual = list(executor.map(lambda action: db.claim_job_for_retry(job) if action == 'claim'
                                           else db.create_job_if_no_active('/test.mkv', 'MANUAL'), ['claim', 'manual']))
    assert manual['job_id'] == job
    assert int(claimed) + int(manual.get('resumed', False)) == 1
    assert db.get_job_by_id(job)['grace_completed'] == 1


@pytest.mark.asyncio
async def test_scheduler_throttles_waiting_items_and_reacts_to_duration_changes(tmp_path, monkeypatch):
    import app.main as main
    monkeypatch.setattr(main, 'datetime', Clock)
    ids = []
    for i in range(3):
        job = db.create_job(str(tmp_path / f'{i}.mkv'), 'SONARR')
        db.apply_grace_period(job, 'SONARR', 24)
        ids.append(job)
    worker = AsyncMock(return_value={})
    with patch.object(main.pipeline, 'process_video_file', worker):
        assert [task async for task in main.process_one_retry_pass()] == []
        Clock.current += timedelta(minutes=15)
        tasks = [task async for task in main.process_one_retry_pass()]
        assert len(tasks) == 3
        await asyncio.gather(*tasks)
        assert [task async for task in main.process_one_retry_pass()] == []
        for job in ids:
            db.apply_grace_period(job, 'RETRY', 24)
        db.set_setting('grace_period_hours', '0')
        tasks = [task async for task in main.process_one_retry_pass()]
        assert len(tasks) == 3
        await asyncio.gather(*tasks)


@pytest.mark.parametrize('value, status', [(0, 200), (48, 200), (8760, 200), (-1, 422), (8761, 422), (1.5, 422)])
def test_api_duration_validation_and_roundtrip(value, status):
    with TestClient(app) as client:
        settings = client.get('/api/settings/all').json()['integrations']
        settings['grace_period_hours'] = value
        assert client.post('/api/settings/integrations', json=settings).status_code == status
        if status == 200:
            assert client.get('/api/settings/all').json()['integrations']['grace_period_hours'] == value


def test_old_client_omitting_duration_does_not_reset_it():
    with TestClient(app) as client:
        settings = client.get('/api/settings/all').json()['integrations']
        settings.pop('grace_period_hours', None)
        assert client.post('/api/settings/integrations', json=settings).status_code == 200
    assert db.get_setting('grace_period_hours') == '24'


def test_existing_database_migrates_without_changing_existing_job():
    job = db.create_job('/legacy.mkv', 'SONARR')
    with sqlite3.connect(db.DB_PATH) as conn:
        for column in ('grace_started_at', 'grace_until', 'grace_completed'):
            conn.execute(f'ALTER TABLE jobs DROP COLUMN {column}')
        conn.execute("DELETE FROM settings WHERE key='grace_period_hours'")
    db.init_db()
    assert db.get_setting('grace_period_hours') == '0'
    legacy = db.get_job_by_id(job)
    assert legacy['status'] == 'QUEUED'
    assert legacy['grace_started_at'] is None
    assert legacy['grace_completed'] == 0


@pytest.mark.asyncio
async def test_library_scans_are_read_only_and_show_persisted_deadline(tmp_path, monkeypatch):
    from app.api import dashboard
    video, _ = media(tmp_path)
    pipeline = engine()
    result = await pipeline.process_video_file(str(video), event_source='SONARR')
    assert result['status'] == 'waiting_grace'
    before = db.get_job_by_id(result['job_id'])
    monkeypatch.setattr(dashboard, 'scan_library_folders', lambda path, category:
                        [{'path': str(video), 'filename': video.name}] if category == 'movies' else [])
    monkeypatch.setattr(dashboard, '_scan_lock', asyncio.Lock())
    monkeypatch.setattr(dashboard, '_media_cache', {})
    for _ in range(3):
        response = await dashboard.api_get_media_files(force=True)
        assert response['movies'][0]['active_job']['status'] == 'WAITING_GRACE'
        assert response['movies'][0]['active_job']['grace_until'] == before['grace_until']
    active = await dashboard.api_active_jobs()
    assert active[str(video)]['grace_until'] == before['grace_until']
    assert db.get_job_by_id(result['job_id'])['grace_started_at'] == before['grace_started_at']
    pipeline.trigger_bazarr_search.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_and_retry_jobs_do_not_start_a_new_grace_period(tmp_path):
    video, _ = media(tmp_path)
    job = db.create_job(str(video), 'SONARR')
    db.update_job(job, status='RETRY_PENDING', retry_count=1)
    pipeline = engine()
    result = await pipeline.process_video_file(str(video), job_id=job, event_source='RETRY')
    assert result['status'] == 'translated', result
    assert db.get_job_by_id(job)['grace_started_at'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize('disabled', ['enable_bazarr_check', 'bazarr_api_key'])
async def test_disabling_bazarr_releases_waiting_items(tmp_path, disabled):
    video, _ = media(tmp_path)
    pipeline = engine()
    result = await pipeline.process_video_file(str(video), event_source='SONARR')
    assert result['status'] == 'waiting_grace'
    db.set_setting(disabled, 'false' if disabled == 'enable_bazarr_check' else '')
    assert db.claim_job_for_retry(result['job_id'])
    resumed = await pipeline.process_video_file(str(video), event_source='RETRY', job_id=result['job_id'])
    assert resumed['status'] == 'translated', resumed
    assert db.get_job_by_id(result['job_id'])['grace_completed'] == 1


@pytest.mark.asyncio
async def test_grace_preserves_configured_language_eligibility(tmp_path):
    video, cues = media(tmp_path)
    video.with_suffix('.sv.srt').write_text(srt.compose(swedish(cues)))
    db.set_setting('languages', '[{"name":"Swedish","code":"sv","enabled":true},{"name":"French","code":"fr","enabled":true}]')
    pipeline = engine()
    result = await pipeline.process_video_file(str(video), event_source='RADARR')
    assert result['status'] == 'waiting_grace'
    pipeline.translator.translate_srt_content.assert_not_awaited()

    db.set_setting('languages', '[{"name":"Swedish","code":"sv","enabled":true},{"name":"French","code":"fr","enabled":false}]')
    assert db.claim_job_for_retry(result['job_id'])
    resumed = await pipeline.process_video_file(str(video), event_source='RETRY', job_id=result['job_id'])
    assert resumed['status'] == 'skipped'
    pipeline.translator.translate_srt_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_first_ai_after_grace_still_runs_minimum_budget_admission(tmp_path):
    video, _ = media(tmp_path)
    pipeline = engine()
    result = await pipeline.process_video_file(str(video), event_source='RADARR')
    assert result['status'] == 'waiting_grace'
    Clock.current += timedelta(hours=24)
    assert db.claim_job_for_retry(result['job_id'])
    with patch('app.core.quota.check_minimum_budget_admission', return_value={
        'admitted': False, 'estimated_minimum': 1, 'available': 0,
    }) as admission:
        resumed = await pipeline.process_video_file(str(video), event_source='RETRY', job_id=result['job_id'])
    assert resumed['status'] == 'deferred', resumed
    admission.assert_called_once()
    pipeline.translator.translate_srt_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_post_grace_bazarr_timeout_is_bounded_and_does_not_restart_wait(tmp_path):
    video, _ = media(tmp_path)
    pipeline = engine()
    result = await pipeline.process_video_file(str(video), event_source='SONARR')
    assert result['status'] == 'waiting_grace'
    Clock.current += timedelta(hours=24)
    assert db.claim_job_for_retry(result['job_id'])
    real_wait_for = asyncio.wait_for

    async def limited(awaitable, timeout):
        return await real_wait_for(awaitable, timeout=0.001 if timeout == 20.0 else timeout)

    async def stalled(*args, **kwargs):
        await asyncio.Event().wait()

    pipeline.trigger_bazarr_search.side_effect = stalled
    with patch('asyncio.wait_for', side_effect=limited):
        resumed = await pipeline.process_video_file(str(video), event_source='RETRY', job_id=result['job_id'])
    assert resumed['status'] == 'translated', resumed
    assert pipeline.trigger_bazarr_search.await_count == 1
    assert db.get_job_by_id(result['job_id'])['grace_completed'] == 1
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith('bazarr_tgt_')]


@pytest.mark.asyncio
async def test_simultaneous_items_wait_and_translate_once_with_original_cue_timing(tmp_path):
    items = [media(tmp_path, f'Episode{i}') for i in range(3)]
    pipeline = engine()
    waiting = await asyncio.gather(*(pipeline.process_video_file(str(video), event_source='SONARR')
                                     for video, _ in items))
    assert {result['status'] for result in waiting} == {'waiting_grace'}
    assert len({result['job_id'] for result in waiting}) == 3
    pipeline.translator.translate_srt_content.assert_not_awaited()
    Clock.current += timedelta(hours=24)
    for result in waiting:
        assert db.claim_job_for_retry(result['job_id'])
    completed = await asyncio.gather(*(pipeline.process_video_file(str(video), event_source='RETRY', job_id=result['job_id'])
                                       for (video, _), result in zip(items, waiting)))
    assert {result['status'] for result in completed} == {'translated'}
    assert pipeline.translator.translate_srt_content.await_count == 3
    for video, source in items:
        target = list(srt.parse(video.with_suffix('.sv.srt').read_text()))
        assert len(target) == len(source)
        assert [(cue.start, cue.end) for cue in target] == [(cue.start, cue.end) for cue in source]
