import pytest
from unittest.mock import patch
from app.services.updates_controller import UpdatesController

@pytest.mark.asyncio
async def test_update_checker_logic():
    controller = UpdatesController()
    
    # Test version parsing
    assert controller._parse_version("v2.3.24-beta") == (2, 3, 24)
    assert controller._parse_version("2.3.24-beta") == (2, 3, 24)
    assert controller._parse_version("v2.3.9-beta") < controller._parse_version("v2.3.10-beta")
    
    # Test malformed
    assert controller._parse_version("junk") == (0, 0, 0)

