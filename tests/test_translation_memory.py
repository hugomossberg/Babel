import pytest
import sqlite3
from app.core.db import DB_PATH, init_db, save_translation_memory_bulk, get_translation_memory
from app.services.pipeline import SubtitlePipeline

@pytest.fixture(autouse=True)
def setup_tmp_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_tm.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    init_db()
    yield test_db

def test_translation_memory_crud():
    show = "Test Show"
    items = [
        {"original": "Hello world", "translated": "Hej världen"},
        {"original": "Good morning", "translated": "God morgon"},
    ]

    # Save TM
    save_translation_memory_bulk(show, items)

    # Lookup TM
    tm = get_translation_memory(show)
    originals = [entry["original"] for entry in tm]
    translateds = [entry["translated"] for entry in tm]
    assert "Hello world" in originals
    assert "Hej världen" in translateds
    assert "Good morning" in originals
    assert "God morgon" in translateds

def test_translation_memory_case_insensitive():
    show = "Test Show 2"
    items = [
        {"original": "Good night", "translated": "God natt"},
    ]
    save_translation_memory_bulk(show, items)

    tm = get_translation_memory(show)
    assert len(tm) == 1
    assert tm[0]["original"] == "Good night"
    assert tm[0]["translated"] == "God natt"
