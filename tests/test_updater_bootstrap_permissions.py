import os
import pytest
from unittest.mock import patch
from app.updater.main import bootstrap_updater_secret
import stat

def test_updater_bootstrap_file_permissions(tmp_path):
    auth_file = tmp_path / "updater_secret"

    with patch("os.getenv", return_value=""), patch("app.updater.main.AUTH_FILE_PATH", str(auth_file)):
        secret = bootstrap_updater_secret()
        assert len(secret) == 64
        assert auth_file.exists()

        # Check permissions
        st = os.stat(auth_file)
        perms = stat.S_IMODE(st.st_mode)
        # 0o600 means exactly owner read and owner write
        assert perms == 0o600, f"Permissions were {oct(perms)}, expected 0o600"
