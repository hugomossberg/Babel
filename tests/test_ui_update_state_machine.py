import re
import pytest

def get_index_html():
    with open("app/templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

def test_scenario_a_local_state_update_in_progress():
    """A) Local state handles update-in-progress even if backend returns 'idle' initially."""
    html = get_index_html()
    assert "updateInProgress: false" in html
    assert "updateTargetVersion: ''" in html
    assert "updateJustSucceeded: false" in html
    assert "this.updateInProgress = true;" in html
    assert "isUpdating()" in html

def test_scenario_b_header_and_settings_updating_state_coordination():
    """B) Header and Settings show 'Updating...' simultaneously without conflicts."""
    html = get_index_html()
    assert "isUpdating() ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse'" in html
    assert "<template x-if=\"isUpdating()\">" in html
    assert 'x-show="isUpdating()"' in html
    assert "Updating Babel..." in html

def test_scenario_c_up_to_date_cannot_show_while_updating():
    """C) 'Up to date' cannot be shown while update is in progress."""
    html = get_index_html()
    assert 'x-show="!isUpdating() && !updateJustSucceeded && !isUpdateFailed() && !checkingUpdates && !updateCheckError && !updateData.update_available"' in html

def test_scenario_d_update_failed_cannot_show_while_updating():
    """D) 'Update failed' cannot be shown while update is still in progress."""
    html = get_index_html()
    assert '<template x-if="!isUpdating() && !updateJustSucceeded && isUpdateFailed()">' in html
    assert 'x-show="!isUpdating() && !updateJustSucceeded && isUpdateFailed()"' in html

def test_scenario_e_full_reload_triggered_on_verified_success():
    """E) Cache-busted reload is triggered on verified success."""
    html = get_index_html()
    assert "window.location.replace(" in html
    assert "hasTriggeredReload = true;" in html
    assert "updated=" in html

def test_scenario_f_session_storage_transient_success_state():
    """F) After reload, 'Updated to vX.Y.Z' is verified and displayed transiently via sessionStorage."""
    html = get_index_html()
    assert "sessionStorage.setItem('babelUpdateSuccess'" in html
    assert "sessionStorage.getItem('babelUpdateSuccess')" in html
    assert "sessionStorage.removeItem('babelUpdateSuccess')" in html
    assert "normCurrent === normMarker" in html
    assert "this.updateJustSucceeded = true;" in html
    assert "this.updateSuccessVersion = this.updateData.current_version;" in html
    assert "this.updateSuccessTimer = setTimeout(" in html
    assert "10000" in html

def test_scenario_g_header_version_binds_to_current_version():
    """G) Header version binds to current_version (same source as Settings)."""
    html = get_index_html()
    assert 'x-text="updateData.current_version || \'v{{VERSION}}\'"' in html

def test_scenario_h_check_updates_runs_during_update_in_progress():
    """H) 7. checkUpdates is NOT blocked by updateInProgress and continues reading live status."""
    html = get_index_html()
    assert "if (this.checkingUpdates) return;" in html
    assert "if (this.checkingUpdates || this.updateInProgress) return;" not in html
    assert "['inspecting', 'pulling', 'replacing', 'verifying', 'updating', 'rolling_back'].includes" in html

def test_scenario_i_network_disconnect_tolerated():
    """I) Polling 502/504 / network reset does NOT trigger immediate failure."""
    html = get_index_html()
    assert "// Connection errors during restart are expected" in html
    assert "maxAttempts = 60;" in html

def test_scenario_j_rolled_back_sets_correct_failure_rollback_state():
    """J) 10. Rolled back sets correct failure/rollback state and stops updating."""
    html = get_index_html()
    assert "st === 'rolled_back'" in html
    assert "this.updateData.updater_status = 'rolled_back';" in html
    assert "this.updateInProgress = false;" in html
    assert "this.updateFailed = true;" in html

def test_scenario_k_dismiss_clears_failure_state():
    """K) Dismiss clears popover and failure banner."""
    html = get_index_html()
    assert "@click=\"updateFailed = false; updateData.updater_status = 'idle'; popoverOpen = false\"" in html
    assert "@click=\"updateFailed = false; updateData.updater_status = 'idle'\"" in html

def test_scenario_l_start_update_buttons_disabled_during_update():
    """L) Start buttons are disabled during active update."""
    html = get_index_html()
    assert "if (this.isUpdating()) return;" in html
    assert ':disabled="updateInProgress"' in html

def test_scenario_m_mutual_exclusivity_of_all_ui_states():
    """M) 12. All UI states (idle, updating, success, failed, rollback) are mutually exclusive."""
    html = get_index_html()
    assert 'x-show="isUpdating()"' in html
    assert 'x-show="!isUpdating() && updateJustSucceeded"' in html
    assert 'x-show="!isUpdating() && !updateJustSucceeded && isUpdateFailed()"' in html
    assert 'x-show="!isUpdating() && !updateJustSucceeded && !isUpdateFailed() && updateData.update_available"' in html

def test_init_fresh_check_and_background_polling():
    """8. Page load initiates check and starts monitoring if status is active (e.g. verifying)."""
    html = get_index_html()
    assert "await this.checkUpdates();" in html
    assert "if (!this.healthPollActive) {" in html
    assert "this.pollHealth(this.updateTargetVersion || data.latest_version);" in html
    assert "this.updatePollTimer = setInterval(() => {\n            this.checkUpdates();\n          }, 60000);" in html

def test_update_already_in_progress_message_handled_as_ongoing():
    """11. 'already in progress' is treated as ongoing update, not terminal failure."""
    html = get_index_html()
    assert "if (res.message && res.message.toLowerCase().includes('already in progress'))" in html
