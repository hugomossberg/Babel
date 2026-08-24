import pytest
from app.services.updates_controller import updates_controller

@pytest.fixture(autouse=True)
def reset_updates_controller_state():
    updates_controller.is_maintenance_locked = False
    updates_controller.update_status = "idle"
    yield
    updates_controller.is_maintenance_locked = False
    updates_controller.update_status = "idle"
