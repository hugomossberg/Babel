"""Opt-in real-browser smoke against a disposable, loopback-only Babel server.

Install Playwright, Chromium and Alpine in a temporary npm directory; supply
NODE_PATH and PLAYWRIGHT_BROWSERS_PATH. No production DB, socket or media is used.
Run with the project's Python: python tests/run_browser_smoke.py --output /tmp/babel-smoke
"""
import argparse
import base64
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVER = '''
import sys
from app.core.db import init_db
init_db()
from app.services.docker_controller import docker_controller
docker_controller.socket_path = sys.argv[2]
from app.main import app
import uvicorn
uvicorn.run(app, fd=int(sys.argv[1]), log_level="warning")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='babel-browser-') as temporary:
        data = Path(temporary)
        movies, series = data / 'movies', data / 'series'
        movies.mkdir()
        series.mkdir()
        (movies / 'Smoke Movie.mkv').touch()
        (movies / 'Smoke Movie.sv.srt').write_text('1\n00:00:01,000 --> 00:00:02,000\nHej världen!\n')
        season = series / 'Smoke Show' / 'Season 01'
        season.mkdir(parents=True)
        (season / 'Smoke Show - S01E01.mkv').touch()
        (season / 'Smoke Show - S01E01.sv.srt').write_text('1\n00:00:01,000 --> 00:00:02,000\nHej världen!\n')
        database = data / 'babel.db'
        # Existing settings survive normal startup/migration and a second boot.
        with sqlite3.connect(database) as conn:
            conn.execute('CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT)')
            conn.executemany('INSERT INTO settings VALUES (?, ?)', [
                ('media_movies_path', str(movies)), ('media_series_path', str(series)),
                ('enable_bazarr_check', 'false'), ('legacy_custom_setting', 'preserved'),
            ])
        env = dict(os.environ, BABEL_DB_PATH=str(database), BABEL_AUTH_USERNAME='smoke',
                   BABEL_AUTH_PASSWORD='disposable-test-password', BABEL_WEBHOOK_SECRET='disposable-webhook',
                   BABEL_UPDATER_SECRET='disposable-updater', BABEL_UPDATER_URL='http://127.0.0.1:9',
                   BABEL_VERSION='stabilization-smoke')
        auth = 'Basic ' + base64.b64encode(b'smoke:disposable-test-password').decode()
        for boot in range(2):
            with socket.socket() as listener, (args.output / f'server-{boot}.log').open('w') as log:
                listener.bind(('127.0.0.1', 0))
                listener.listen(128)
                base = f'http://127.0.0.1:{listener.getsockname()[1]}'
                process = subprocess.Popen([sys.executable, '-c', SERVER, str(listener.fileno()),
                                            str(data / 'nonexistent-docker.sock')], cwd=root, env=env,
                                           pass_fds=(listener.fileno(),), stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 30
                    while True:
                        try:
                            with urlopen(base + '/health', timeout=1) as response:
                                assert json.load(response)['status'] == 'healthy'
                            break
                        except (URLError, TimeoutError):
                            if process.poll() is not None or time.monotonic() > deadline:
                                raise RuntimeError('Disposable Babel server failed to start')
                            time.sleep(0.1)
                    try:
                        urlopen(base + '/api/settings/all', timeout=3)
                        raise AssertionError('Unauthenticated settings request succeeded')
                    except HTTPError as exc:
                        assert exc.code == 401
                    if boot == 0:
                        subprocess.run(['node', str(root / 'tests/browser_smoke.cjs'), base,
                                        str(args.output)], env=env, check=True, timeout=90)
                        request = Request(base + '/api/settings/ai', data=b'{"llm_request_timeout_seconds":300}',
                                          headers={'Authorization': auth, 'Content-Type': 'application/json'})
                        with urlopen(request, timeout=3) as response:
                            assert response.status == 200
                    else:
                        with urlopen(Request(base + '/api/settings/all', headers={'Authorization': auth}), timeout=3) as response:
                            settings = json.load(response)
                            assert settings['ai']['llm_request_timeout_seconds'] == 300
                            assert settings['integrations']['grace_period_hours'] == 51
                        with sqlite3.connect(database) as conn:
                            assert conn.execute("SELECT value FROM settings WHERE key='legacy_custom_setting'").fetchone()[0] == 'preserved'
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
        print('PASS: isolated startup, health, Basic Auth, settings migration/persistence and restart')


if __name__ == '__main__':
    main()
