import pytest
import re

def test_poll_health_state_awareness():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    poll_health_match = re.search(r"pollHealth\(expectedVersion\) \{.*?(?=formatDate)", html, re.DOTALL)
    assert poll_health_match, "pollHealth function not found"
    poll_func = poll_health_match.group(0)

    # 1. Terminal failed state cleanly handled
    assert "st === 'failed'" in poll_func
    assert "this.updateInProgress = false;" in poll_func
    assert "this.updateFailed = true;" in poll_func
    assert "this.stopHealthPoll();" in poll_func

    # 2. Terminal rolled_back state cleanly handled (10)
    assert "st === 'rolled_back'" in poll_func

    # 3. Success state requires expected version match and terminal success/idle (9, 13)
    assert "isExpected" in poll_func
    assert "(st === 'success' || st === 'idle')" in poll_func
    assert "sessionStorage.setItem('babelUpdateSuccess'" in poll_func
    assert "window.location.replace(newUrl)" in poll_func

    # 4. Timeout stops polling and flags failure (14)
    assert "attempts >= maxAttempts" in poll_func
    assert "Update timed out" in poll_func
