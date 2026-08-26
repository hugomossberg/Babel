import os
import sqlite3
import pytest
from app.core import db
from app.core.db import parse_legacy_tm_series_key


@pytest.fixture(autouse=True)
def setup_tmp_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "test_tm.db")
    monkeypatch.setattr("app.core.db.DB_PATH", test_db)
    db.init_db()
    yield test_db


def test_parse_legacy_tm_series_key_helper():
    """Unit tests for parse_legacy_tm_series_key helper."""
    # Valid legacy keys
    title, src, tgt = parse_legacy_tm_series_key("Breaking Bad::en::sv")
    assert title == "Breaking Bad" and src == "en" and tgt == "sv"

    title, src, tgt = parse_legacy_tm_series_key("Show::Special::Edition::en::sv")
    assert title == "Show::Special::Edition" and src == "en" and tgt == "sv"

    # Legitimate non-language colons
    title, src, tgt = parse_legacy_tm_series_key("Something::Else")
    assert title == "Something::Else" and src is None and tgt is None

    title, src, tgt = parse_legacy_tm_series_key("Show::Special::Edition")
    assert title == "Show::Special::Edition" and src is None and tgt is None

    title, src, tgt = parse_legacy_tm_series_key("Show::foo::bar")
    assert title == "Show::foo::bar" and src is None and tgt is None

    title, src, tgt = parse_legacy_tm_series_key("Normal Show Title")
    assert title == "Normal Show Title" and src is None and tgt is None


def test_tm_save_retrieve_legitimate_colons_something_else():
    """Requirement A: Legitimate '::' in series name ('Something::Else') must be preserved as opaque text."""
    series = "Something::Else"
    db.save_translation_memory(series, "Original line.", "Översatt rad.", source_language="en", target_language="sv")

    # Exact retrieve with same opaque name
    res = db.get_translation_memory(series, source_language="en", target_language="sv")
    assert len(res) == 1
    assert res[0]["original"] == "Original line."
    assert res[0]["translated"] == "Översatt rad."

    # Must NOT have been saved under "Something"
    res_truncated = db.get_translation_memory("Something", source_language="en", target_language="sv")
    assert len(res_truncated) == 0


def test_tm_save_retrieve_multiple_colons_special_edition():
    """Requirement B: Multiple '::' in series name ('Show::Special::Edition') must be preserved as opaque text."""
    series = "Show::Special::Edition"
    db.save_translation_memory(series, "Key scene.", "Viktig scen.", source_language="en", target_language="sv")

    res = db.get_translation_memory(series, source_language="en", target_language="sv")
    assert len(res) == 1
    assert res[0]["translated"] == "Viktig scen."

    # Must NOT have been saved under "Show"
    res_truncated = db.get_translation_memory("Show", source_language="en", target_language="sv")
    assert len(res_truncated) == 0


def test_tm_migration_from_legacy_compound_keys(tmp_path, monkeypatch):
    """Requirement C: Legacy compound key 'Breaking Bad::en::sv' must migrate to title='Breaking Bad', src='en', tgt='sv'."""
    db_file = str(tmp_path / "test_legacy.db")

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE translation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_title TEXT NOT NULL,
            original_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(series_title, original_text)
        )
    """)
    cur.execute("INSERT INTO translation_memory (series_title, original_text, translated_text) VALUES (?, ?, ?)",
                ("Breaking Bad::en::sv", "I am the one who knocks.", "Det är jag som knackar."))
    conn.commit()
    conn.close()

    monkeypatch.setattr("app.core.db.DB_PATH", db_file)
    db.init_db()

    # Query with clean series title and lang pair
    res = db.get_translation_memory("Breaking Bad", source_language="en", target_language="sv")
    assert len(res) == 1
    assert res[0]["original"] == "I am the one who knocks."
    assert res[0]["translated"] == "Det är jag som knackar."


def test_tm_migration_preserves_false_legacy_titles(tmp_path, monkeypatch):
    """Requirement D: False legacy titles ('Show::foo::bar') must NOT be truncated during migration."""
    db_file = str(tmp_path / "test_false_legacy.db")

    conn = sqlite3.connect(db_file)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE translation_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            series_title TEXT NOT NULL,
            original_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(series_title, original_text)
        )
    """)
    cur.execute("INSERT INTO translation_memory (series_title, original_text, translated_text) VALUES (?, ?, ?)",
                ("Show::foo::bar", "Some line.", "Någon rad."))
    conn.commit()
    conn.close()

    monkeypatch.setattr("app.core.db.DB_PATH", db_file)
    db.init_db()

    # Must retain exact title 'Show::foo::bar'
    res_exact = db.get_translation_memory("Show::foo::bar", source_language="en", target_language="sv")
    assert len(res_exact) == 1
    assert res_exact[0]["translated"] == "Någon rad."

    # Must NOT have been truncated to 'Show'
    res_truncated = db.get_translation_memory("Show", source_language="en", target_language="sv")
    assert len(res_truncated) == 0


def test_tm_language_pair_and_series_strict_isolation():
    """Requirement E: Verify strict matrix: same/diff series, same/diff source, same/diff target."""
    # Seed data
    db.save_translation_memory("Show Alpha", "Target word", "Målord Alpha SV", source_language="en", target_language="sv")
    db.save_translation_memory("Show Alpha", "Target word", "Mot cible Alpha FR", source_language="en", target_language="fr")
    db.save_translation_memory("Show Alpha", "Zielwort", "Målord Alpha DE-SV", source_language="de", target_language="sv")
    db.save_translation_memory("Show Beta", "Target word", "Målord Beta SV", source_language="en", target_language="sv")

    # Same series + same source + same target = HIT
    res = db.get_translation_memory("Show Alpha", source_language="en", target_language="sv")
    assert len(res) == 1 and res[0]["translated"] == "Målord Alpha SV"

    # Different target = MISS / isolated
    res_fr = db.get_translation_memory("Show Alpha", source_language="en", target_language="fr")
    assert len(res_fr) == 1 and res_fr[0]["translated"] == "Mot cible Alpha FR"

    # Different source = MISS / isolated
    res_de = db.get_translation_memory("Show Alpha", source_language="de", target_language="sv")
    assert len(res_de) == 1 and res_de[0]["translated"] == "Målord Alpha DE-SV"

    # Different series = MISS / isolated
    res_beta = db.get_translation_memory("Show Beta", source_language="en", target_language="sv")
    assert len(res_beta) == 1 and res_beta[0]["translated"] == "Målord Beta SV"

    # Unknown language pair = MISS
    res_none = db.get_translation_memory("Show Alpha", source_language="en", target_language="es")
    assert len(res_none) == 0
