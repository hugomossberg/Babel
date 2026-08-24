import os
import pytest
import yaml

def test_docker_compose_security_and_structure():
    compose_path = os.path.join(os.path.dirname(__file__), "..", "docker-compose.yml")
    assert os.path.exists(compose_path), "docker-compose.yml must exist"

    with open(compose_path, "r") as f:
        compose_data = yaml.safe_load(f)

    services = compose_data.get("services", {})
    assert "babel" in services, "babel service is missing"
    assert "babel-updater" in services, "babel-updater service is missing"

    babel = services["babel"]
    updater = services["babel-updater"]

    babel_volumes = babel.get("volumes", [])
    updater_volumes = updater.get("volumes", [])

    # 1. Babel auth-volume is read-only
    assert any("babel_updater_auth" in str(v) and str(v).endswith(":ro") for v in babel_volumes), "Babel auth volume must be read-only (:ro)"

    # 2. Updater auth-volume is read/write
    assert any("babel_updater_auth" in str(v) and not str(v).endswith(":ro") for v in updater_volumes), "Updater auth volume must be read-write"

    # 3. Docker socket exists only on updater
    assert not any("docker.sock" in str(v) for v in babel_volumes), "Babel container must NOT mount docker.sock"
    assert any("docker.sock" in str(v) for v in updater_volumes), "Updater container must mount docker.sock"

    # 4. Updater exposes no host ports
    assert "ports" not in updater, "babel-updater must not expose ports"

    # 5. Updater image restriction
    allowed_image = None
    env = updater.get("environment", [])
    if isinstance(env, dict):
        allowed_image = env.get("ALLOWED_IMAGE")
    elif isinstance(env, list):
        for item in env:
            assert not isinstance(item, dict), "Environment entry parsed as dict instead of string. Missing quotes in compose file?"
            if isinstance(item, str) and item.startswith("ALLOWED_IMAGE="):
                allowed_image = item.split("=", 1)[1]

    assert allowed_image is not None, "babel-updater is missing ALLOWED_IMAGE"
    assert "ghcr.io/hugomossberg/babel" in allowed_image, "ALLOWED_IMAGE must be ghcr.io/hugomossberg/babel"
