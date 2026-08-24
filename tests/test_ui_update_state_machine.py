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
    # startUpdate sets local updateInProgress = true immediately
    assert "this.updateInProgress = true;" in html
    assert "isUpdating()" in html

def test_scenario_b_header_and_settings_updating_state_coordination():
    """B) Header and Settings show 'Updating...' simultaneously without conflicts."""
    html = get_index_html()
    # Header badge uses isUpdating()
    assert "isUpdating() ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40 animate-pulse'" in html
    assert "<template x-if=\"isUpdating()\">" in html
    # Settings panel has mutually exclusive updating indicator
    assert 'x-show="isUpdating()"' in html
    assert "Updating Babel..." in html

def test_scenario_c_up_to_date_cannot_show_while_updating():
    """C) 'Up to date' cannot be shown while update is in progress."""
    html = get_index_html()
    # Check that the Up to date block in settings requires !isUpdating()
    assert 'x-show="!isUpdating() && !updateJustSucceeded && !isUpdateFailed() && !checkingUpdates && !updateCheckError && !updateData.update_available"' in html

def test_scenario_d_update_failed_cannot_show_while_updating():
    """D) 'Update failed' cannot be shown while update is still in progress."""
    html = get_index_html()
    # In header template
    assert '<template x-if="!isUpdating() && !updateJustSucceeded && isUpdateFailed()">' in html
    # In settings
    assert 'x-show="!isUpdating() && !updateJustSucceeded && isUpdateFailed()"' in html

def test_scenario_e_full_reload_triggered_on_verified_success():
    """E) Full reload is triggered on verified success."""
    html = get_index_html()
    assert "window.location.reload();" in html
    assert "hasTriggeredReload = true;" in html

def test_scenario_f_session_storage_transient_success_state():
    """F) After reload, 'Updated to vX.Y.Z' is displayed transiently via sessionStorage."""
    html = get_index_html()
    # In pollHealth before reload
    assert "sessionStorage.setItem('babelUpdateSuccess'" in html
    # In init() after reload
    assert "sessionStorage.getItem('babelUpdateSuccess')" in html
    assert "sessionStorage.removeItem('babelUpdateSuccess')" in html
    assert "this.updateJustSucceeded = true;" in html
    assert "this.updateSuccessVersion = marker.version;" in html
    # Transient timeout exists
    assert "this.updateSuccessTimer = setTimeout(" in html

def test_scenario_g_header_version_binds_to_current_version():
    """G) Header version binds to current_version (not just static v{{VERSION}})."""
    html = get_index_html()
    assert 'x-text="updateData.current_version || \'v{{VERSION}}\'"' in html

def test_scenario_h_polling_cannot_reset_update_in_progress_prematurely():
    """H) checkUpdates polling cannot toggle updateInProgress to false until terminal state."""
    html = get_index_html()
    # In checkUpdates
    assert "if (this.checkingUpdates || this.updateInProgress) return;" in html
    # In isUpdating
    assert "['inspecting', 'pulling', 'replacing', 'verifying', 'updating', 'rolling_back'].includes" in html

def test_scenario_i_network_disconnect_tolerated():
    """I) Polling 502/504 / network reset does NOT trigger immediate failure."""
    html = get_index_html()
    # Connection errors in fetch do not clearInterval or fail immediately
    assert "// Connection errors during restart are expected" in html
    assert "maxAttempts = 40;" in html

def test_scenario_j_rolled_back_sets_correct_failure_rollback_state():
    """J) Rolled back sets correct failure/rollback state."""
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
    """L) Start-knappar är disablade under pågående update."""
    html = get_index_html()
    # startUpdate guard
    assert "if (this.updateInProgress) return;" in html
    # Popover button
    assert ':disabled="updateInProgress"' in html
    # Settings button
    assert ':disabled="updateInProgress"' in html
    # Release notes modal button
    assert ':disabled="updateInProgress"' in html

def test_scenario_m_mutual_exclusivity_of_all_ui_states():
    """M) All UI states (idle, updating, success, failed, rollback) are mutually exclusive."""
    html = get_index_html()
    
    # Priority order in Popover:
    # 1. isUpdating()
    assert 'x-show="isUpdating()"' in html
    # 2. !isUpdating() && updateJustSucceeded
    assert 'x-show="!isUpdating() && updateJustSucceeded"' in html
    # 3. !isUpdating() && !updateJustSucceeded && isUpdateFailed()
    assert 'x-show="!isUpdating() && !updateJustSucceeded && isUpdateFailed()"' in html
    # 4. !isUpdating() && !updateJustSucceeded && !isUpdateFailed() && updateData.update_available
    assert 'x-show="!isUpdating() && !updateJustSucceeded && !isUpdateFailed() && updateData.update_available"' in html
