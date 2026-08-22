import os
import tempfile
import pytest
import srt
from datetime import timedelta
from unittest.mock import patch

from app.services.pipeline import _publish_subtitle_atomic
from app.core.cleaner import subs_to_srt_string


def create_sample_subs(texts):
    subs = []
    for i, t in enumerate(texts, 1):
        subs.append(srt.Subtitle(
            index=i,
            start=timedelta(seconds=i * 2),
            end=timedelta(seconds=i * 2 + 1),
            content=t
        ))
    return subs


HEALTHY_SWEDISH_TEXTS = [
    "Det här är en bra svensk text.",
    "Vi ska gå och titta på filmen ikväll.",
    "Det var verkligen en fantastisk upplevelse.",
    "Kan du hjälpa mig att förstå detta?",
    "Jag tror inte att det kommer att regna idag.",
    "Välkommen till den stora staden och vårt hem.",
    "Vi har mycket att prata om inför morgondagen.",
    "Allt kommer att ordna sig till slut."
]


@pytest.mark.asyncio
async def test_1_regression_target_backup_rollback_bug(tmp_path):
    """
    Test 1 (Mandatory Regression):
    Target does not exist initially -> Babel has valid temp ->
    os.link raises FileExistsError because another process created target ->
    target is unhealthy -> backup succeeds -> final publish operation fails ->
    original target MUST be restored via rollback.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"

    subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    translated_srt = subs_to_srt_string(subs)

    unhealthy_content = "1\n00:00:01,000 --> 00:00:02,000\nUnhealthy content created by race\n\n"

    original_link = os.link
    link_call_count = 0

    def mock_link(src, dst):
        nonlocal link_call_count
        link_call_count += 1
        if dst == str(target_srt):
            if link_call_count == 1:
                # External process created unhealthy target right before first link
                target_srt.write_text(unhealthy_content, encoding="utf-8")
                raise FileExistsError(f"File exists: {dst}")
            elif link_call_count == 2:
                # Second link fails (e.g. disk failure / I/O error) after backup succeeded
                raise OSError("Disk write failure during final publish")
        return original_link(src, dst)

    with patch("os.link", side_effect=mock_link):
        with pytest.raises(OSError) as exc_info:
            _publish_subtitle_atomic(
                video_path=video_path,
                target_output_path=str(target_srt),
                lang_code="sv",
                translated_srt_text=translated_srt,
                expected_cue_count=len(subs),
            )
        assert "Disk write failure" in str(exc_info.value)

    # VERIFY:
    # 1. target_srt MUST exist and be restored with the unhealthy content that was backed up
    assert target_srt.exists(), "Target file must be restored after publish crash"
    assert "Unhealthy content created by race" in target_srt.read_text(encoding="utf-8")

    # 2. Babel's translated content was not published
    assert "Det här är en bra svensk text" not in target_srt.read_text(encoding="utf-8")

    # 3. No leftover temp files or backup files
    remaining_files = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining_files)
    assert not any(".babel-replaced." in f for f in remaining_files)


@pytest.mark.asyncio
async def test_2_publish_backup_failure_stops_publish(tmp_path):
    """
    Test 2:
    Existing unhealthy target -> mock backup failure (os.replace raises OSError) ->
    Babel raises RuntimeError, original target untouched, Babel output not published, temp cleaned.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"
    original_unhealthy = "1\n00:00:01,000 --> 00:00:02,000\nOld Unhealthy Dialogue\n\n"
    target_srt.write_text(original_unhealthy, encoding="utf-8")

    subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    translated_srt = subs_to_srt_string(subs)

    original_replace = os.replace
    def mock_replace(src, dst):
        if ".babel-replaced." in dst:
            raise OSError("Permission denied on backup")
        return original_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace):
        with pytest.raises(RuntimeError) as exc_info:
            _publish_subtitle_atomic(
                video_path=video_path,
                target_output_path=str(target_srt),
                lang_code="sv",
                translated_srt_text=translated_srt,
                expected_cue_count=len(subs),
            )
        assert "Cannot safely back up" in str(exc_info.value)

    assert target_srt.exists()
    assert target_srt.read_text(encoding="utf-8") == original_unhealthy
    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)


@pytest.mark.asyncio
async def test_3_publish_temp_validation_failure(tmp_path):
    """
    Test 3:
    Invalid temp SRT or cue count mismatch -> validation fails ->
    raises RuntimeError, original target untouched, temp cleaned.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"
    target_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nOriginal Intact\n\n", encoding="utf-8")

    # 1. Corrupt SRT text
    with pytest.raises(RuntimeError) as exc_info:
        _publish_subtitle_atomic(
            video_path=video_path,
            target_output_path=str(target_srt),
            lang_code="sv",
            translated_srt_text="NOT VALID SRT CONTENT",
            expected_cue_count=5,
        )
    assert "validation failed" in str(exc_info.value).lower()
    assert target_srt.read_text(encoding="utf-8") == "1\n00:00:01,000 --> 00:00:02,000\nOriginal Intact\n\n"

    # 2. Cue count mismatch
    subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS[:4])  # 4 cues
    with pytest.raises(RuntimeError) as exc_info2:
        _publish_subtitle_atomic(
            video_path=video_path,
            target_output_path=str(target_srt),
            lang_code="sv",
            translated_srt_text=subs_to_srt_string(subs),
            expected_cue_count=8,  # Expected 8, got 4
        )
    assert "cue count mismatch" in str(exc_info2.value).lower()
    assert target_srt.read_text(encoding="utf-8") == "1\n00:00:01,000 --> 00:00:02,000\nOriginal Intact\n\n"

    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)


@pytest.mark.asyncio
async def test_4_publish_failure_after_backup_restores_original(tmp_path):
    """
    Test 4:
    Unhealthy target on disk -> backup succeeds -> simulated link failure ->
    production rollback restores original target.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"
    original_text = "1\n00:00:01,000 --> 00:00:02,000\nUnhealthy Source Dialogue\n\n"
    target_srt.write_text(original_text, encoding="utf-8")

    subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    translated_srt = subs_to_srt_string(subs)

    with patch("os.link", side_effect=OSError("Disk write error on publish")):
        with pytest.raises(OSError) as exc_info:
            _publish_subtitle_atomic(
                video_path=video_path,
                target_output_path=str(target_srt),
                lang_code="sv",
                translated_srt_text=translated_srt,
                expected_cue_count=len(subs),
            )
        assert "Disk write error" in str(exc_info.value)

    assert target_srt.exists()
    assert target_srt.read_text(encoding="utf-8") == original_text
    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)
    assert not any(".babel-replaced." in f for f in remaining)


@pytest.mark.asyncio
async def test_5_external_healthy_file_replaces_unhealthy_file_before_backup(tmp_path):
    """
    Test 5 (TOCTOU Race Before Backup):
    Target appears unhealthy initially -> external process replaces it with a healthy file ->
    Babel's atomic backup moves the healthy file -> Babel inspects backup health ->
    finds GREEN -> restores healthy target and skips publishing Babel's translation.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"

    # Initially target is unhealthy
    target_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nUnhealthy Initial Text\n\n", encoding="utf-8")

    healthy_subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    healthy_external_text = subs_to_srt_string(healthy_subs)

    babel_subs = create_sample_subs(["Babel översättning rad 1", "Babel översättning rad 2"])
    babel_text = subs_to_srt_string(babel_subs)

    original_replace = os.replace
    def mock_replace_simulating_race(src, dst):
        if ".babel-replaced." in dst:
            # Right as os.replace is called, external process has written healthy content to src
            with open(src, "w", encoding="utf-8") as f:
                f.write(healthy_external_text)
        return original_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace_simulating_race):
        res = _publish_subtitle_atomic(
            video_path=video_path,
            target_output_path=str(target_srt),
            lang_code="sv",
            translated_srt_text=babel_text,
            expected_cue_count=len(babel_subs),
        )

    assert res["skipped"] is True
    assert res["published"] is False
    assert target_srt.exists()
    content = target_srt.read_text(encoding="utf-8")
    assert "Det här är en bra svensk text" in content
    assert "Babel översättning" not in content
    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)


@pytest.mark.asyncio
async def test_6_external_healthy_target_appears_after_backup(tmp_path):
    """
    Test 6:
    Unhealthy target is backed up -> before Babel links temp, external process creates healthy target ->
    Babel gets FileExistsError -> inspects target -> finds GREEN -> skips publish, cleans temp.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"
    target_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nOld Unhealthy\n\n", encoding="utf-8")

    healthy_subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    healthy_external_text = subs_to_srt_string(healthy_subs)

    babel_subs = create_sample_subs(["Babel översättning 1", "Babel översättning 2"])
    babel_text = subs_to_srt_string(babel_subs)

    original_link = os.link
    def mock_link_with_race(src, dst):
        if dst == str(target_srt):
            # External process wrote healthy target after backup
            target_srt.write_text(healthy_external_text, encoding="utf-8")
            raise FileExistsError(f"File exists: {dst}")
        return original_link(src, dst)

    with patch("os.link", side_effect=mock_link_with_race):
        res = _publish_subtitle_atomic(
            video_path=video_path,
            target_output_path=str(target_srt),
            lang_code="sv",
            translated_srt_text=babel_text,
            expected_cue_count=len(babel_subs),
        )

    assert res["skipped"] is True
    assert res["published"] is False
    assert target_srt.exists()
    content = target_srt.read_text(encoding="utf-8")
    assert "Det här är en bra svensk text" in content
    assert "Babel översättning" not in content
    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)


@pytest.mark.asyncio
async def test_7_clean_no_target_publish(tmp_path):
    """
    Test 7:
    No existing target -> valid temp -> os.link succeeds -> target complete -> temp deleted.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"

    subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    translated_srt = subs_to_srt_string(subs)

    res = _publish_subtitle_atomic(
        video_path=video_path,
        target_output_path=str(target_srt),
        lang_code="sv",
        translated_srt_text=translated_srt,
        expected_cue_count=len(subs),
    )

    assert res["published"] is True
    assert res["skipped"] is False
    assert target_srt.exists()
    content = target_srt.read_text(encoding="utf-8")
    assert "Det här är en bra svensk text" in content
    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)
    assert not any(".babel-replaced." in f for f in remaining)


@pytest.mark.asyncio
async def test_8_concurrent_publishers(tmp_path):
    """
    Test 8 (Deterministic Concurrent Publishers):
    Two concurrent publisher tasks attempt to atomically publish to the same target simultaneously.
    One wins and publishes; the other encounters FileExistsError / GREEN target and safely skips.
    Exactly one valid target survives with zero corruption and all temp files cleaned up.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"

    subs_a = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    text_a = subs_to_srt_string(subs_a)

    subs_b = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    text_b = subs_to_srt_string(subs_b)

    import asyncio

    # Launch publisher A and publisher B concurrently in separate threads
    task_a = asyncio.to_thread(
        _publish_subtitle_atomic,
        video_path=video_path,
        target_output_path=str(target_srt),
        lang_code="sv",
        translated_srt_text=text_a,
        expected_cue_count=len(subs_a),
    )
    task_b = asyncio.to_thread(
        _publish_subtitle_atomic,
        video_path=video_path,
        target_output_path=str(target_srt),
        lang_code="sv",
        translated_srt_text=text_b,
        expected_cue_count=len(subs_b),
    )

    results = await asyncio.gather(task_a, task_b)
    published_count = sum(1 for r in results if r["published"] is True)
    skipped_count = sum(1 for r in results if r["skipped"] is True)

    assert published_count == 1, f"Expected exactly 1 publisher to succeed, got {published_count}"
    assert skipped_count == 1, f"Expected exactly 1 publisher to skip, got {skipped_count}"

    assert target_srt.exists()
    parsed = list(srt.parse(target_srt.read_text(encoding="utf-8")))
    assert len(parsed) == len(HEALTHY_SWEDISH_TEXTS)

    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)
    assert not any(".babel-replaced." in f for f in remaining)


@pytest.mark.asyncio
async def test_9_moved_healthy_backup_target_exists_green_vs_unhealthy_race(tmp_path):
    """
    Test 9 (Regression for Punkt 1):
    Target is moved to backup (backup is GREEN). Right as it is moved, an external process creates a file at target_path.
    - Case A: New file at target_path is GREEN -> keep current target, safely retain backup, return skipped=True.
    - Case B: New file at target_path is UNHEALTHY -> fail-closed (RuntimeError), retain both files so nothing is lost.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"

    healthy_subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    healthy_text = subs_to_srt_string(healthy_subs)
    babel_subs = create_sample_subs(["Babel translation cue 1", "Babel translation cue 2"])
    babel_text = subs_to_srt_string(babel_subs)

    unhealthy_text = "1\n00:00:01,000 --> 00:00:02,000\nCorrupt/English text\n\n"

    # --- CASE A: New file at target_path is GREEN ---
    target_srt.write_text(unhealthy_text, encoding="utf-8")
    original_replace = os.replace
    def mock_replace_case_a(src, dst):
        if ".babel-replaced." in dst:
            # External process wrote healthy text to src right before backup move
            with open(src, "w", encoding="utf-8") as f:
                f.write(healthy_text)
            res = original_replace(src, dst)
            # Immediately another process creates a SECOND healthy Swedish subtitle at target_srt
            target_srt.write_text(healthy_text, encoding="utf-8")
            return res
        return original_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace_case_a):
        res_a = _publish_subtitle_atomic(
            video_path=video_path,
            target_output_path=str(target_srt),
            lang_code="sv",
            translated_srt_text=babel_text,
            expected_cue_count=len(babel_subs),
        )

    assert res_a["skipped"] is True
    assert res_a["published"] is False
    assert target_srt.exists()
    assert "Det här är en bra svensk text" in target_srt.read_text(encoding="utf-8")
    # Retained backup must still exist on disk so no data was lost
    backup_files = [f for f in tmp_path.iterdir() if ".babel-replaced." in f.name]
    assert len(backup_files) >= 1
    # Clean up for Case B
    for f in list(tmp_path.iterdir()):
        f.unlink()

    # --- CASE B: New file at target_path is UNHEALTHY ---
    target_srt.write_text(unhealthy_text, encoding="utf-8")
    def mock_replace_case_b(src, dst):
        if ".babel-replaced." in dst:
            # External process wrote healthy text to src right before backup move
            with open(src, "w", encoding="utf-8") as f:
                f.write(healthy_text)
            res = original_replace(src, dst)
            # Immediately another process creates an UNHEALTHY file at target_srt
            target_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nUnhealthy Second Text\n\n", encoding="utf-8")
            return res
        return original_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace_case_b):
        with pytest.raises(RuntimeError) as exc_info:
            _publish_subtitle_atomic(
                video_path=video_path,
                target_output_path=str(target_srt),
                lang_code="sv",
                translated_srt_text=babel_text,
                expected_cue_count=len(babel_subs),
            )
        assert "Target conflict" in str(exc_info.value)

    # Fail-closed: both target file and backup file MUST exist on disk intact!
    assert target_srt.exists()
    assert "Unhealthy Second Text" in target_srt.read_text(encoding="utf-8")
    backup_files_b = [f for f in tmp_path.iterdir() if ".babel-replaced." in f.name]
    assert len(backup_files_b) == 1
    assert "Det här är en bra svensk text" in backup_files_b[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_10_fsync_failure_aborts_publish_leaves_target_untouched(tmp_path):
    """
    Test 10 (Punkt 2):
    Simulate fsync failure (os.fsync raises OSError) ->
    Publish aborts immediately BEFORE touching target or creating backup,
    existing target remains 100% untouched, temp file is cleaned up.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"
    original_content = "1\n00:00:01,000 --> 00:00:02,000\nExisting Target Content\n\n"
    target_srt.write_text(original_content, encoding="utf-8")

    subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    translated_srt = subs_to_srt_string(subs)

    with patch("os.fsync", side_effect=OSError("Disk I/O failure during fsync")):
        with pytest.raises(OSError) as exc_info:
            _publish_subtitle_atomic(
                video_path=video_path,
                target_output_path=str(target_srt),
                lang_code="sv",
                translated_srt_text=translated_srt,
                expected_cue_count=len(subs),
            )
        assert "Disk I/O failure" in str(exc_info.value)

    # Existing target untouched
    assert target_srt.exists()
    assert target_srt.read_text(encoding="utf-8") == original_content
    # No temp files and no backups
    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)
    assert not any(".babel-replaced." in f for f in remaining)


@pytest.mark.asyncio
async def test_11_partial_checkpoint_isolation_from_repo_data(tmp_path, monkeypatch):
    """
    Test 11 (Punkt 3):
    Verify that tests redirect checkpoint/recovery state (*_partial.json) to tmp_path
    and never pollute repository ./data directory.
    """
    import app.core.db
    from app.services.translator import SubtitleTranslator, ProviderUnavailableError

    test_db = tmp_path / "isolation_test.db"
    monkeypatch.setattr(app.core.db, "DB_PATH", str(test_db))
    app.core.db.init_db()

    repo_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    repo_files_before = set(os.listdir(repo_data_dir)) if os.path.exists(repo_data_dir) else set()

    job_id = 99999
    translator = SubtitleTranslator()
    subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)

    call_count = {"count": 0}
    async def mock_translate_batch(payload, **kwargs):
        call_count["count"] += 1
        if call_count["count"] > 1:
            raise ProviderUnavailableError("Rate limit")
        return [{"id": p["id"], "text": f"Svenska {p['id']}"} for p in payload]

    with patch.object(translator, "translate_batch", side_effect=mock_translate_batch):
        try:
            await translator.translate_srt_content(subs, job_id=job_id, target_language="sv", batch_size=2)
        except ProviderUnavailableError:
            pass

    # Partial file must be created in tmp_path (test db directory), NOT in repo data/
    expected_partial = tmp_path / f"job_{job_id}_sv_partial.json"
    assert expected_partial.exists(), "Partial checkpoint must be created in test directory"

    # Repository data/ must have NO new files
    repo_files_after = set(os.listdir(repo_data_dir)) if os.path.exists(repo_data_dir) else set()
    new_repo_files = repo_files_after - repo_files_before
    assert len(new_repo_files) == 0, f"Repository data directory polluted with: {new_repo_files}"


@pytest.mark.asyncio
async def test_12_moved_healthy_backup_target_absent_restore_failure_fail_closed(tmp_path):
    """
    Test 12 (Regression for Item 1 - moved_health):
    GREEN backup + target absent + restore os.replace raises OSError ->
    raises RuntimeError (fail-closed), retains backup intact, no false skipped/published, no data destroyed.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"

    healthy_subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    healthy_text = subs_to_srt_string(healthy_subs)
    babel_subs = create_sample_subs(["Babel translation cue 1", "Babel translation cue 2"])
    babel_text = subs_to_srt_string(babel_subs)

    unhealthy_text = "1\n00:00:01,000 --> 00:00:02,000\nCorrupt/English text\n\n"
    target_srt.write_text(unhealthy_text, encoding="utf-8")

    original_replace = os.replace
    def mock_replace(src, dst):
        if ".babel-replaced." in dst:
            # First move: target -> backup. Write healthy text to simulate race
            with open(src, "w", encoding="utf-8") as f:
                f.write(healthy_text)
            return original_replace(src, dst)
        elif ".babel-replaced." in src:
            # Second move: restore backup -> target. Inject OSError!
            raise OSError("Disk I/O error during backup restore")
        return original_replace(src, dst)

    with patch("os.replace", side_effect=mock_replace):
        with pytest.raises(RuntimeError) as exc_info:
            _publish_subtitle_atomic(
                video_path=video_path,
                target_output_path=str(target_srt),
                lang_code="sv",
                translated_srt_text=babel_text,
                expected_cue_count=len(babel_subs),
            )
        assert "Failed to restore captured healthy subtitle" in str(exc_info.value)

    # Backup MUST still exist on disk intact with healthy Swedish content
    backup_files = [f for f in tmp_path.iterdir() if ".babel-replaced." in f.name]
    assert len(backup_files) == 1
    assert "Det här är en bra svensk text" in backup_files[0].read_text(encoding="utf-8")
    # Temp file must be cleaned up
    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)


@pytest.mark.asyncio
async def test_13_moved_c_healthy_backup_target_absent_restore_failure_fail_closed(tmp_path):
    """
    Test 13 (Regression for Item 1 - moved_c_health):
    GREEN concurrent backup + target absent + restore os.replace raises OSError ->
    raises RuntimeError (fail-closed), retains concurrent backup intact, no false skipped/published, no data destroyed.
    """
    video_path = str(tmp_path / "Movie.mkv")
    target_srt = tmp_path / "Movie.sv.srt"

    healthy_subs = create_sample_subs(HEALTHY_SWEDISH_TEXTS)
    healthy_text = subs_to_srt_string(healthy_subs)
    babel_subs = create_sample_subs(["Babel translation cue 1", "Babel translation cue 2"])
    babel_text = subs_to_srt_string(babel_subs)

    # No initial target. During _link_temp_no_clobber, target appears.
    # When health checked initially on target, it returns UNHEALTHY, so it moves it to concurrent_backup.
    # But when moved, it is healthy Swedish text.
    original_replace = os.replace
    first_replace_done = False

    def mock_link(src, dst):
        if dst == str(target_srt):
            # Create concurrent target
            target_srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nCorrupt cue\n\n", encoding="utf-8")
            return False
        return False

    def mock_replace(src, dst):
        nonlocal first_replace_done
        if ".babel-replaced." in dst and not first_replace_done:
            first_replace_done = True
            # Write healthy text into src before moving to concurrent_backup
            with open(src, "w", encoding="utf-8") as f:
                f.write(healthy_text)
            return original_replace(src, dst)
        elif ".babel-replaced." in src:
            # Restore move: concurrent_backup -> target_output_path. Inject OSError!
            raise OSError("Disk write failure during concurrent restore")
        return original_replace(src, dst)

    with patch("app.services.pipeline._link_temp_no_clobber", side_effect=mock_link), \
         patch("os.replace", side_effect=mock_replace):
        with pytest.raises(RuntimeError) as exc_info:
            _publish_subtitle_atomic(
                video_path=video_path,
                target_output_path=str(target_srt),
                lang_code="sv",
                translated_srt_text=babel_text,
                expected_cue_count=len(babel_subs),
            )
        assert "Failed to restore captured healthy concurrent subtitle" in str(exc_info.value)

    # Concurrent backup MUST still exist on disk intact with healthy Swedish content
    backup_files = [f for f in tmp_path.iterdir() if ".babel-replaced." in f.name]
    assert len(backup_files) == 1
    assert "Det här är en bra svensk text" in backup_files[0].read_text(encoding="utf-8")
    # Temp file must be cleaned up
    remaining = [f.name for f in tmp_path.iterdir()]
    assert not any(f.startswith(".Movie.sv.srt.tmp") for f in remaining)
