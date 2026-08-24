import pytest
import re

def test_poll_health_state_awareness():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    poll_health_match = re.search(r"pollHealth\(expectedVersion\) \{.*?(?=formatDate)", html, re.DOTALL)
    assert poll_health_match, "pollHealth function not found"
    poll_func = poll_health_match.group(0)

    # A) old version + updater_status=pulling => continues polling
    # B) old version + updater_status=verifying => continues polling
    # To verify this, check that clearInterval is NOT called unconditionally after fetch
    
    # Assert unconditional clearInterval is GONE
    assert "                                    const upData = await upRes.json();\n                                    clearInterval(iv);" not in poll_func
    
    # Check that failed explicitly leads to terminal
    assert "st === 'failed'" in poll_func
    
    # Check that rolled_back explicitly leads to terminal
    assert "st === 'rolled_back'" in poll_func
    
    # Check that success requires both version match and terminal state
    assert "isExpected" in poll_func
    assert "(st === 'success' || st === 'idle')" in poll_func
    
