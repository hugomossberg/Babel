import os

# Bug #23: This file now only provides environment defaults and server-level config.
# All application logic should read from SQLite via get_setting().

APP_NAME = "Babel"
VERSION = "2.3.29-beta"
PORT = int(os.getenv("PORT", "8765"))

# Bug #39: Authentication config
# Set BABEL_AUTH_USERNAME and BABEL_AUTH_PASSWORD in environment to enable HTTP Basic Auth
AUTH_USERNAME = os.getenv("BABEL_AUTH_USERNAME", "")
AUTH_PASSWORD = os.getenv("BABEL_AUTH_PASSWORD", "")

# Updater configuration & internal authentication
UPDATER_SECRET = os.getenv("BABEL_UPDATER_SECRET", os.getenv("UPDATER_SECRET", "")).strip()
UPDATER_URL = os.getenv("BABEL_UPDATER_URL", "http://babel-updater:8767").rstrip("/")
GITHUB_REPO = "hugomossberg/Babel"
DOCKER_REPO = "ghcr.io/hugomossberg/babel"

