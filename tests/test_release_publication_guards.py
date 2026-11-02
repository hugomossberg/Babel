"""Release preflight is offline and fails before any registry mutation."""
import importlib.util
from pathlib import Path
import subprocess

import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]
SHA='a'*40


@pytest.fixture
def guard():
    spec=importlib.util.spec_from_file_location('release_preflight',ROOT/'.github/scripts/release_preflight.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    (tmp_path/'app').mkdir()
    (tmp_path/'app/config.py').write_text('import os\nVERSION = os.getenv("BABEL_VERSION") or "2.5.6-beta"\n')
    (tmp_path/'docs/releases').mkdir(parents=True)
    (tmp_path/'docs/releases/v2.5.6-beta.md').write_text('# Release\nReviewed notes\n')
    return tmp_path


def test_approved_public_tag_is_valid(guard,source):
    assert guard.validate_release('hugomossberg/Babel','v2.5.6-beta',SHA,source,lambda ref:SHA)=='2.5.6-beta'


@pytest.mark.parametrize('repository',['hugomossberg/babel-dev','someone/Babel',''])
def test_private_or_other_repository_is_rejected(guard,source,repository):
    with pytest.raises(ValueError,match='repository'):
        guard.validate_release(repository,'v2.5.6-beta',SHA,source,lambda ref:SHA)


@pytest.mark.parametrize('tag',['vfoo','v2.5.6-rc1','v2.5.6-beta-extra','v02.5.6-beta','2.5.6-beta','v2.5.6;echo x','v../../main'])
def test_invalid_tags_are_rejected(guard,source,tag):
    with pytest.raises(ValueError,match='tag'):
        guard.validate_release('hugomossberg/Babel',tag,SHA,source,lambda ref:SHA)


def test_version_mismatch_stops_release(guard,source):
    (source/'app/config.py').write_text('VERSION="2.5.5-beta"')
    with pytest.raises(ValueError,match='version'):
        guard.validate_release('hugomossberg/Babel','v2.5.6-beta',SHA,source,lambda ref:SHA)


@pytest.mark.parametrize('body',[None,'','   \n'])
def test_missing_or_empty_notes_stop_release(guard,source,body):
    path=source/'docs/releases/v2.5.6-beta.md'
    if body is None:path.unlink()
    else:path.write_text(body)
    with pytest.raises(ValueError,match='notes'):
        guard.validate_release('hugomossberg/Babel','v2.5.6-beta',SHA,source,lambda ref:SHA)


def test_tag_must_be_current_public_main(guard,source):
    with pytest.raises(ValueError,match='main'):
        guard.validate_release('hugomossberg/Babel','v2.5.6-beta',SHA,source,lambda ref:'b'*40 if 'origin/main' in ref else SHA)


def test_stable_release_remains_supported(guard,source):
    (source/'app/config.py').write_text('VERSION="2.5.6"')
    (source/'docs/releases/v2.5.6.md').write_text('Reviewed stable notes')
    assert guard.validate_release('hugomossberg/Babel','v2.5.6',SHA,source,lambda ref:SHA)=='2.5.6'


def test_workflow_can_only_publish_reviewed_public_tags():
    workflow=yaml.load((ROOT/'.github/workflows/docker-publish.yml').read_text(),Loader=yaml.BaseLoader)
    assert workflow['on']=={'push':{'tags':['v*']}}
    assert workflow['concurrency']['cancel-in-progress']=='false'
    for name in ['test','publish','release']:
        assert "github.repository == 'hugomossberg/Babel'" in workflow['jobs'][name]['if']
    steps=workflow['jobs']['test']['steps']
    checkout=next(s for s in steps if s.get('uses','').startswith('actions/checkout@'))
    assert checkout['with']['fetch-depth']=='0'
    guard_index=next(i for i,s in enumerate(steps) if '.github/scripts/release_preflight.py' in s.get('run',''))
    tests_index=next(i for i,s in enumerate(steps) if 'pytest' in s.get('run','') and 'pip install' not in s.get('run',''))
    assert guard_index<tests_index
    assert workflow['jobs']['publish']['needs']=='test'
    assert workflow['jobs']['release']['needs']=='publish'
    release=next(s for s in workflow['jobs']['release']['steps'] if s.get('uses','').startswith('softprops/action-gh-release@'))
    assert release['with']['body_path']=='docs/releases/${{ github.ref_name }}.md'
    assert release['with']['generate_release_notes']=='false'
    tags=next(s for s in workflow['jobs']['publish']['steps'] if s.get('uses','').startswith('docker/metadata-action@'))['with']['tags']
    assert "value=beta,enable=${{ contains(github.ref_name, '-beta') }}" in tags
    assert "value=latest,enable=${{ !contains(github.ref_name, '-') }}" in tags
