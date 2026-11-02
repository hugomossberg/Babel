"""Source preparation must not masquerade as translation or reset checkpoints."""
import json

import pytest

from app.core import db
from app.services import pipeline as pipeline_module
from app.services.source_resolver import SourceResolver
from app.api.dashboard import api_active_jobs, _attach_active_jobs_to_media


@pytest.mark.asyncio
async def test_probe_and_resolver_report_preparation_without_resetting_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'activity.db'))
    db.init_db()
    for key, value in {
        'enable_bazarr_check': 'false', 'extract_target_embedded': 'false',
        'extract_source_embedded': 'true',
        'languages': json.dumps([{'code': 'sv', 'name': 'Swedish', 'enabled': True}]),
    }.items():
        db.set_setting(key, value)
    video = tmp_path / 'episode.mkv'
    video.write_bytes(b'owned media')
    job = db.create_job(str(video))
    db.update_job(job, total_lines=827, processed_lines=100)
    phases = []

    def record(expected):
        row = db.get_job_by_id(job)
        assert row['current_batch'].startswith('Preparing source:'), row['current_batch']
        assert expected in row['current_batch'].lower()
        assert row['total_lines'] == 827 and row['processed_lines'] == 100
        phases.append(expected)

    def probe(path):
        record('inspect')
        return {'audio': [{'language': 'eng', 'default': True}], 'subtitles': [], 'duration': 20}

    async def resolve(self):
        record('resolv')
        raise RuntimeError('Controlled stop before AI dispatch')

    monkeypatch.setattr(pipeline_module, 'inspect_mkv_tracks', probe)
    monkeypatch.setattr(SourceResolver, 'resolve', resolve)
    result = await pipeline_module.SubtitlePipeline()._run_pipeline_logic(job, str(video), force_retranslate=True)
    assert result['status'] == 'error'
    assert phases == ['inspect', 'resolv']
    assert db.get_job_by_id(job)['status'] == 'FAILED'


@pytest.mark.asyncio
async def test_preparation_phase_survives_both_library_api_views(tmp_path, monkeypatch):
    monkeypatch.setattr(db, 'DB_PATH', str(tmp_path / 'views.db'))
    db.init_db()
    paths = [str(tmp_path / 'episode.mkv'), str(tmp_path / 'movie.mkv')]
    phase = 'Preparing source: resolving subtitles'
    for path in paths:
        job = db.create_job(path)
        db.update_job(job, status='TRANSLATING', current_batch=phase, total_lines=827, processed_lines=100)
    active = await api_active_jobs()
    assert all(active[path].get('current_batch') == phase for path in paths)
    series = [{'episodes': [{'path': paths[0]}]}]
    movies = [{'path': paths[1]}]
    await _attach_active_jobs_to_media(series, movies, paths)
    assert series[0]['episodes'][0]['active_job']['current_batch'] == phase
    assert movies[0]['active_job']['current_batch'] == phase
    assert movies[0]['active_job']['processed_lines'] == 100
