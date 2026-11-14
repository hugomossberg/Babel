import os
import sqlite3
import tempfile

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="babel_test_data_")
os.environ["BABEL_DB_PATH"] = os.path.join(_TEST_DATA_DIR, "test.db")

# Test databases are throwaway: skip the fsync on every commit, which
# otherwise dominates DB-heavy tests on a real disk.
_real_sqlite_connect = sqlite3.connect


def _connect_without_fsync(*args, **kwargs):
    conn = _real_sqlite_connect(*args, **kwargs)
    conn.execute("PRAGMA synchronous=OFF")
    return conn


sqlite3.connect = _connect_without_fsync

from app.core.db import init_db
init_db()
