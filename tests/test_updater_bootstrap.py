import os
import pytest
from unittest.mock import patch
from app.updater.main import bootstrap_updater_secret
from app.services.updates_controller import UpdatesController

def test_updater_bootstrap_generates_secret(tmp_path):
    auth_file = tmp_path / "updater_secret"

    with patch("os.getenv", return_value=""), patch("app.updater.main.AUTH_FILE_PATH", str(auth_file)):
        secret1 = bootstrap_updater_secret()
        assert len(secret1) == 64  # 32 bytes hex
        assert auth_file.exists()
        assert auth_file.read_text().strip() == secret1

        # Test it persists
        secret2 = bootstrap_updater_secret()
        assert secret2 == secret1

def test_updater_bootstrap_env_override():
    with patch("os.getenv", return_value="override_secret"):
        secret = bootstrap_updater_secret()
        assert secret == "override_secret"

def test_updates_controller_dynamic_secret(tmp_path):
    auth_file = tmp_path / "updater_secret"
    auth_file.write_text("file_secret")

    controller = UpdatesController()
    controller.AUTH_FILE_PATH = str(auth_file)

    with patch("os.getenv", return_value=""):
        assert controller._get_updater_secret() == "file_secret"

    with patch("os.getenv", return_value="env_secret"):
        assert controller._get_updater_secret() == "env_secret"

def test_updates_controller_no_secret():
    controller = UpdatesController()
    controller.AUTH_FILE_PATH = "/nonexistent/path/for/test"
    with patch("os.getenv", return_value=""):
        assert controller._get_updater_secret() == ""
