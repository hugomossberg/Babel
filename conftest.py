import os
import tempfile

_TEST_DATA_DIR = tempfile.mkdtemp(prefix="babel_test_data_")
os.environ["BABEL_DB_PATH"] = os.path.join(_TEST_DATA_DIR, "test.db")

from app.core.db import init_db
init_db()
