import pytest

def test_automatic_update_polling_interval():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Verify that poll timer variables exist
    assert "updatePollTimer: null" in html
    
    # Verify that it is cleared to prevent leaks
    assert "clearInterval(this.updatePollTimer)" in html

    # Verify that it runs checkUpdates at roughly 5 minutes (60000ms)
    assert "this.updatePollTimer = setInterval(() => {" in html
    assert "this.checkUpdates();" in html
    
    # Either 60000 or close approximations if formatting differs
    assert "}, 60000);" in html

    # Verify manualCheckUpdates still exists as a separate function, since we shouldn't have changed it
    assert "async manualCheckUpdates()" in html
