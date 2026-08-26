import pytest
from unittest.mock import patch, AsyncMock
from app.services.updates_controller import updates_controller

@pytest.fixture(autouse=True)
def reset_updates_controller_state():
    updates_controller.is_maintenance_locked = False
    updates_controller.update_status = "idle"
    yield
    updates_controller.is_maintenance_locked = False
    updates_controller.update_status = "idle"


@pytest.fixture(autouse=True)
def hermetic_semantic_audit(request):
    """
    Dependency seam: ensures that audit_batch_semantic_integrity never makes
    live API calls during the test suite.

    For tests that do NOT explicitly mock audit_batch_semantic_integrity on their
    SubtitleTranslator instance, this autouse fixture provides a safe ALIGNED
    default response (class-level patch on SubtitleTranslator).

    Tests that set pipeline.translator.audit_batch_semantic_integrity = AsyncMock(...)
    (i.e. instance-level override) will use their own mock and this default will be
    shadowed, so the patch is fully transparent to test-specific mocks.

    Tests decorated with @pytest.mark.skip_hermetic_audit receive the REAL
    audit_batch_semantic_integrity implementation so they can test contract logic
    directly via _dispatch_llm_completion mocks.

    This guarantees:
    - 0 live paid API calls from pytest.
    - No fail-closed SUSPECT cascades in hermetic tests that don't care about semantics.
    - Explicit semantic audit tests control their own mock and are unaffected.
    - Contract hardening tests can test the real code via the marker.
    """
    if request.node.get_closest_marker("skip_hermetic_audit"):
        # Let the real implementation run — the test is responsible for
        # mocking _dispatch_llm_completion to prevent live API calls.
        yield
        return

    async def _default_audit(self, batch_payloads, **kwargs):
        return {
            bp["batch_id"]: {
                "batch_id": bp["batch_id"],
                "verdict": "ALIGNED",
                "confidence": "HIGH",
                "details": "hermetic_semantic_audit fixture default"
            }
            for bp in batch_payloads
        }

    with patch(
        "app.services.translator.SubtitleTranslator.audit_batch_semantic_integrity",
        new=_default_audit,
    ):
        yield
