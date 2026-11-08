// Opt-in smoke: invoked only by run_browser_smoke.py with a disposable server.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const base = process.argv[2];
assert.equal(new URL(base).hostname, '127.0.0.1', 'Only disposable loopback servers are supported');
const output = process.argv[3];
const alpine = path.join(path.dirname(require.resolve('alpinejs/package.json')), 'dist/cdn.min.js');

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    const context = await browser.newContext({ httpCredentials: { username: 'smoke', password: 'disposable-test-password' } });
    const page = await context.newPage();
    const errors = [], counts = new Map();
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      const key = new URL(request.url()).pathname;
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    await page.addInitScript(() => {
      window.__intervals = new Map();
      const start = window.setInterval, stop = window.clearInterval;
      window.setInterval = (callback, delay, ...args) => {
        const id = start(callback, delay, ...args);
        window.__intervals.set(id, delay);
        return id;
      };
      window.clearInterval = id => { window.__intervals.delete(id); stop(id); };
    });
    let releaseUpdates, releaseModels;
    const updatesGate = new Promise(resolve => { releaseUpdates = resolve; });
    const modelsGate = new Promise(resolve => { releaseModels = resolve; });
    let slowStats = false, releaseStats, statsPending = 0, maximumStats = 0;
    let mediaMode = 'actual', manualOverride;
    let modelReply = { models: [], source: 'live', error: null };
    await page.route('**/*', async route => {
      const url = new URL(route.request().url());
      if (url.href.includes('alpinejs')) return route.fulfill({ path: alpine, contentType: 'application/javascript' });
      if (url.href === 'https://cdn.tailwindcss.com/') return route.fulfill({ body: 'window.tailwind={};', contentType: 'application/javascript' });
      if (url.origin !== base) return route.abort();
      if (url.pathname === '/webhook/process') {
        manualOverride = route.request().postDataJSON();
        return route.fulfill({ json: { status: 'queued', job_id: 999 } });
      }
      if (url.pathname === '/api/updates') {
        await updatesGate;
        return route.fulfill({ json: { current_version: 'stabilization-smoke', updater_status: 'idle', update_available: false, update_check_ok: true } });
      }
      if (url.pathname === '/api/settings/models') {
        await modelsGate;
        return route.fulfill({ json: modelReply });
      }
      if (url.pathname === '/api/stats' && slowStats) {
        statsPending++; maximumStats = Math.max(maximumStats, statsPending);
        await new Promise(resolve => { releaseStats = resolve; });
        statsPending--;
      }
      if (url.pathname === '/api/media-files' && url.searchParams.has('force')) {
        if (mediaMode === 'failed') return route.fulfill({ status: 500, json: { detail: 'Controlled failed scan' } });
        if (mediaMode === 'empty') return route.fulfill({ json: { series: [], movies: [], scanning_embedded: false } });
      }
      return route.continue();
    });
    await page.goto(base, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.Alpine && Alpine.$data(document.body)._initialized);
    assert.equal(counts.get('/api/media-files') || 0, 0, 'Dashboard must not load the library');
    assert.equal(counts.get('/api/stats'), 1);
    await page.getByRole('button', { name: 'Library', exact: true }).click();
    await page.waitForFunction(() => Alpine.$data(document.body).mediaLoaded);
    assert.equal(counts.get('/api/media-files'), 1, 'Opening Library loads it once');
    assert.equal(await page.evaluate(() => Alpine.$data(document.body).mediaData.series.length), 1);
    await page.evaluate(() => { const app = Alpine.$data(document.body); app.toggleShow(app.mediaData.series[0].title); });
    await page.getByText('Smoke Show - S01E01.mkv', { exact: true }).waitFor({ state: 'visible' });
    await page.getByRole('button', { name: /^Movies \(/ }).click();
    await page.getByText('Smoke Movie.mkv', { exact: true }).waitFor({ state: 'visible' });
    // Cached library is visible while update/model discovery are still blocked.
    releaseUpdates(); releaseModels();
    await page.waitForFunction(() => [...window.__intervals.values()].filter(ms => ms === 2000).length === 1);
    assert.equal(await page.evaluate(() => [...window.__intervals.values()].filter(ms => ms === 60000).length), 1);
    await page.evaluate(() => { Alpine.$data(document.body).currentTab = 'dashboard'; });
    slowStats = true;
    await page.waitForFunction(() => true);
    const deadline = Date.now() + 6000;
    while (!releaseStats && Date.now() < deadline) await page.waitForTimeout(100);
    assert(releaseStats, 'Expected a dashboard poll');
    await page.waitForTimeout(2400);
    assert.equal(maximumStats, 1, 'Slow polling requests must not overlap');
    slowStats = false; releaseStats();
    await page.evaluate(() => { Alpine.$data(document.body).currentTab = 'media'; });
    mediaMode = 'failed';
    await page.evaluate(() => Alpine.$data(document.body).loadMediaFiles(true));
    assert.equal(await page.evaluate(() => Alpine.$data(document.body).scanningFolders), false);
    assert.equal(await page.evaluate(() => Alpine.$data(document.body).mediaData.movies.length), 1);
    mediaMode = 'empty';
    await page.evaluate(() => Alpine.$data(document.body).loadMediaFiles(true));
    assert.equal(await page.evaluate(() => Alpine.$data(document.body).mediaData.movies.length), 0);
    mediaMode = 'actual';
    await page.evaluate(() => Alpine.$data(document.body).loadMediaFiles(true));
    await page.getByText('Smoke Movie.mkv', { exact: true }).waitFor({ state: 'visible' });
    await page.evaluate(() => { Alpine.$data(document.body).mediaData.movies[0].active_job = { status: 'TRANSLATING', processed_lines: 5, total_lines: 10 }; });
    await page.getByText('Translating 50%', { exact: true }).filter({ visible: true }).first().waitFor({ state: 'visible' });
    await page.evaluate(() => { Alpine.$data(document.body).mediaData.movies[0].active_job = null; });
    // Exercise the shipped selects and Alpine DOM bindings, with only discovery HTTP mocked.
    await page.evaluate(() => {
      const app = Alpine.$data(document.body);
      app.currentTab = 'settings'; app.settingsTab = 'ai';
      app.settingsData.ai.ai_provider = 'openai';
      app.settingsData.ai.openai_model = 'ft:Exact/Saved-Model';
      app.settingsData.ai.escalation_provider = 'openai';
      app.settingsData.ai.escalation_model = 'Saved-Fallback';
    });
    modelReply = { models: [{ id: 'gpt-4o-mini', name: 'GPT-4o mini' }], source: 'live', error: null };
    await page.evaluate(() => Alpine.$data(document.body).fetchProviderModels('openai', true));
    const primaryModel = page.locator('select[x-model="settingsData.ai[currentModelKey(settingsData.ai.ai_provider)]"]');
    const fallbackModel = page.locator('select[x-model="settingsData.ai.escalation_model"]');
    await page.waitForFunction(() => document.querySelector('select[x-model="settingsData.ai[currentModelKey(settingsData.ai.ai_provider)]"]').value === 'ft:Exact/Saved-Model');
    assert.equal(await primaryModel.inputValue(), 'ft:Exact/Saved-Model');
    assert.equal(await fallbackModel.inputValue(), 'Saved-Fallback');
    assert.equal(await primaryModel.locator('option').count(), 2);
    assert.match(await primaryModel.locator('option:checked').innerText(), /not in current list/);
    modelReply = { models: [{ id: 'gpt-4o-mini' }], source: 'fallback', error: 'authentication', listing_verified: false };
    await page.evaluate(() => Alpine.$data(document.body).fetchProviderModels('openai', true));
    await page.getByText(/Offline suggestions — availability unverified/).first().waitFor({ state: 'visible' });
    assert.equal(await primaryModel.inputValue(), 'ft:Exact/Saved-Model');
    await page.evaluate(() => {
      const app = Alpine.$data(document.body);
      app.settingsData.ai.ai_provider = 'custom';
      app.settingsData.ai.custom_openai_model = 'Namespace/Model:Exact';
    });
    modelReply = { models: [], source: 'live', error: null };
    await page.evaluate(() => Alpine.$data(document.body).fetchProviderModels('custom', true));
    assert.equal(await primaryModel.inputValue(), 'Namespace/Model:Exact');
    assert.equal(await primaryModel.locator('option').count(), 1);
    await page.evaluate(() => {
      const app = Alpine.$data(document.body);
      app.currentTab = 'settings'; app.settingsTab = 'integrations';
      app.settingsData.integrations.enable_bazarr_check = true;
    });
    await page.getByLabel('Grace Period days').fill('2');
    await page.getByLabel('Grace Period hours').fill('3');
    await page.evaluate(() => Alpine.$data(document.body).saveIntegrationsSettings());
    assert.equal(await page.getByLabel('Grace Period days').inputValue(), '2');
    assert.equal(await page.getByLabel('Grace Period hours').inputValue(), '3');
    const saved = await (await page.request.get(base + '/api/settings/all')).json();
    assert.equal(saved.integrations.grace_period_hours, 51);
    await page.evaluate(() => {
      const app = Alpine.$data(document.body);
      app.currentTab = 'media';
      app.mediaData.movies[0].active_job = { status: 'WAITING_GRACE', grace_until: '2026-11-04T12:00:00Z' };
    });
    await page.getByText(/Grace Period until/).filter({ visible: true }).first().waitFor({ state: 'visible' });
    await page.getByRole('button', { name: 'Translate now', exact: true }).click();
    const manualDeadline = Date.now() + 5000;
    while (!manualOverride && Date.now() < manualDeadline) await page.waitForTimeout(50);
    assert(manualOverride, 'Grace Period must offer a manual bypass');
    assert.equal(manualOverride.force_retranslate, false);
    assert.match(manualOverride.video_path, /Smoke Movie\.mkv$/);
    await page.waitForTimeout(150);
    await page.screenshot({ path: path.join(output, 'browser-library.png'), fullPage: true });
    await page.evaluate(() => { window.__smokeApp = Alpine.$data(document.body); document.body.remove(); });
    await page.waitForFunction(() => window.__smokeApp._destroyed && window.__intervals.size === 0);
    const before = [...counts.entries()].filter(([url]) => url.startsWith('/api/')).reduce((sum, [, n]) => sum + n, 0);
    await page.waitForTimeout(2300);
    const after = [...counts.entries()].filter(([url]) => url.startsWith('/api/')).reduce((sum, [, n]) => sum + n, 0);
    assert.equal(after, before, 'Unmounted Alpine component must stop polling');
    fs.writeFileSync(path.join(output, 'browser-results.json'), JSON.stringify({ browser: browser.version(), counts: Object.fromEntries(counts), errors }, null, 2));
    assert.deepEqual(errors, [], 'No browser JavaScript errors are allowed');
    console.log('PASS: real Chromium/Alpine startup, library, polling, scans, model discovery/selection, Grace Period settings/status/manual action and cleanup');
    await context.close();
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
