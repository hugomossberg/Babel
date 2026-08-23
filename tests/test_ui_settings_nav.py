import pytest
import re

def test_settings_navigation_integrity():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # The actual settings buttons should be exactly 7
    tabs = ["'ai'", "'modules'", "'languages'", "'folders'", "'integrations'", "'webhooks'", "'system'"]

    for tab in tabs:
        # Check that there is exactly one button with @click="settingsTab = <tab>"
        counts = html.count(f"@click=\"settingsTab = {tab}\"")
        assert counts == 1, f"Expected exactly 1 button for settingsTab {tab}, found {counts}"

        # Check that the corresponding panel exists!
        panel_counts = html.count(f"x-show=\"settingsTab === {tab}\"")
        assert panel_counts == 1, f"Expected exactly 1 panel for settingsTab {tab}, found {panel_counts}"

    assert html.count("Webhooks & Automation") == 1, "Webhooks text should appear exactly once in nav"
    assert " :class=\"settingsTab === 'webhooks'" not in html.replace(" <button @click=\"settingsTab = 'webhooks'\" :class=\"settingsTab === 'webhooks'", ""), "Dangling Alpine bindings found!"

def test_system_panel_content():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "System & Updates" in html
    assert "Current Version" in html
    assert "Check for updates" in html
    assert "checkingUpdates" in html
    assert "updateCheckError" in html

    # Test it's not nested inside webhooks
    assert "SUB-TAB 6: WEBHOOKS & ARRS" in html
    assert "SUB-TAB 7: SYSTEM" in html
    assert html.find("SUB-TAB 6: WEBHOOKS & ARRS") < html.find("SUB-TAB 7: SYSTEM"), "System panel must come after Webhooks panel"

def test_concurrent_batches_ui():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    assert "settingsData.ai.batch_concurrency" in html, "Concurrent Batches input is missing"
    assert "Concurrent Batches" in html

def test_header_update_pill_and_single_popover():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Header pill exists with discrete conditional visibility
    assert 'id="header-update-badge"' in html
    assert 'x-show="updateData.update_available' in html

    # 2. Popover toggle and click outside handling
    assert 'popoverOpen' in html
    assert '@click.outside="if (updateData.updater_status !== \'updating\') popoverOpen = false"' in html

    # 3. What's new and preview bounds
    assert "What's new" in html
    assert "getPreviewLines(updateData.release_notes)" in html

    # 4. View details & Update now in popover
    assert "View details" in html
    assert 'id="btn-popover-update"' in html
    assert "Update now" in html

    # 5. No redundant second confirmation modal
    assert 'showUpdateConfirm' not in html

    # 6. Active jobs waiting state
    assert "Waiting for active jobs to finish" in html
    assert "stats.active_jobs === 0" in html

    # 7. Updating state inside popover
    assert "Updating Babel..." in html
    assert "Babel will restart briefly" in html

    # 8. Success and failure states in popover
    assert "Updated to" in html
    assert "Update failed" in html
    assert "Babel is still running the previous version" in html

def test_toast_and_js_update_methods():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        html = f.read()

    # Toast styles
    assert "toastType === 'info'" in html
    assert "toastType === 'error'" in html
    assert "text-emerald-400" in html

    # JS Methods
    assert "getPreviewLines(notes)" in html
    assert "manualCheckUpdates()" in html
    assert "startUpdate()" in html
    assert "pollHealth(expectedVersion)" in html
