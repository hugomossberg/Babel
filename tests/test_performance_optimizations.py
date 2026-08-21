import pytest
import sqlite3
import json
from unittest.mock import patch
from app.core.db import init_db, DB_PATH, save_translation_memory_bulk, append_job_log, create_job, get_job_by_id
from app.services.translator import SubtitleTranslator
import os

@pytest.fixture
def test_db():
    # Setup test DB
    import tempfile
    temp_dir = tempfile.TemporaryDirectory()
    original_db_path = os.environ.get("BABEL_DB_PATH", "/app/data/babel.db")
    test_path = os.path.join(temp_dir.name, "test_babel.db")
    
    with patch("app.core.db.DB_PATH", test_path):
        init_db()
        yield test_path
        
    temp_dir.cleanup()

def test_save_translation_memory_bulk(test_db):
    items = [
        {"original": "Hello", "translated": "Hej"},
        {"original": "World", "translated": "Värld"},
        {"original": "Too long sentence to be saved in translation memory ideally", "translated": "För lång mening"},
        {"original": "", "translated": "Tom"}
    ]
    
    save_translation_memory_bulk("Test Series", items)
    
    with sqlite3.connect(test_db) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        rows = cursor.execute("SELECT original_text, translated_text FROM translation_memory WHERE series_title = 'Test Series'").fetchall()
        
    assert len(rows) == 2
    memories = {r["original_text"]: r["translated_text"] for r in rows}
    assert "Hello" in memories
    assert memories["Hello"] == "Hej"
    assert "World" in memories
    assert memories["World"] == "Värld"
    assert "Too long sentence to be saved in translation memory ideally" not in memories

def test_append_job_log_json_insert(test_db):
    job_id = create_job("test_video.mkv")
    
    append_job_log(job_id, "Log entry 1")
    append_job_log(job_id, "Log entry 2")
    
    job = get_job_by_id(job_id)
    assert len(job["logs"]) == 3 # 1 from create_job, 2 from appends
    assert "Log entry 1" in job["logs"][1]
    assert "Log entry 2" in job["logs"][2]

def test_client_caching():
    translator = SubtitleTranslator()
    
    with patch("app.services.translator.get_setting", return_value="fake_key_1"):
        client1 = translator.get_gemini_client()
        client2 = translator.get_gemini_client()
        assert client1 is client2 # Should be cached
        
    with patch("app.services.translator.get_setting", return_value="fake_key_2"):
        client3 = translator.get_gemini_client()
        assert client1 is not client3 # Key changed, should return new client
        
    with patch("app.services.translator.get_setting", return_value="fake_key_openai"):
        client_o1 = translator.get_openai_client()
        client_o2 = translator.get_openai_client()
        assert client_o1 is client_o2
