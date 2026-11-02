"""Execute the real UI component with fake browser timers and deferred fetches."""
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_alpine_initializes_component_once():
    html = (ROOT / "app/templates/index.html").read_text()
    body = html.split("<body", 1)[1].split(">", 1)[0]
    assert 'x-data="babelApp()"' in body
    assert 'x-init="init()"' not in body  # Alpine 3 calls the init method itself.


@pytest.mark.parametrize("scenario", [
    "model_reactive_identity", "model_saved_selection", "model_concurrent_refresh", "model_changed_endpoint",
    "model_destroy_cleanup", "model_error_status", "model_endpoint_secrets",
    "duplicate_init", "startup_library", "overlapping_dashboard", "library_refresh",
    "destroy_timers", "destroy_init", "destroy_embedded", "destroy_updates",
    "destroy_health", "health_overlap", "health_terminal", "health_timeout", "health_deadline",
    "job_completion", "embedded_completion", "queued_refresh", "null_job_bindings", "grace_settings",
])
def test_ui_runtime_lifecycle(scenario):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required to execute browser lifecycle regression tests")
    result = subprocess.run(
        [node, str(ROOT / "tests/ui_polling_lifecycle.cjs"), str(ROOT / "app/templates/index.html"), scenario],
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
