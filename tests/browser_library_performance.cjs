// Opt-in real-browser regression; dependencies stay outside the production app:
// npm install --prefix /tmp/babel-browser-deps playwright alpinejs
// /tmp/babel-browser-deps/node_modules/.bin/playwright install chromium
// NODE_PATH=/tmp/babel-browser-deps/node_modules node tests/browser_library_performance.cjs
// Optional arguments: <template.html> --baseline measures an unpatched template.
// BABEL_BROWSER_EXECUTABLE may select a locally installed Chromium/Chrome binary.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('playwright');
const template = process.argv[2] || path.join(__dirname, '../app/templates/index.html');
const baseline = process.argv.includes('--baseline');
const alpine = process.env.BABEL_ALPINE_PATH || path.join(path.dirname(require.resolve('alpinejs/package.json')), 'dist/cdn.min.js');

(async () => {
  const browser = await chromium.launch({ headless: true, executablePath: process.env.BABEL_BROWSER_EXECUTABLE });
  try {
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    const fixture = `<script>
      const originalApp = babelApp;
      babelApp = () => {
        const app = originalApp();
        for (const method of ['loadStats', 'loadJobs', 'loadAllSettings', 'loadQuotaStatus',
          'loadTodayUsage', 'loadUsageStats', 'loadDeferredQueue', 'checkUpdates', 'refreshLibraryJobState']) {
          app[method] = async () => {};
        }
        app.loadMediaFiles = async function() {
          window.mediaLoads = (window.mediaLoads || 0) + 1;
          if (this.mediaLoaded) return;
          const item = n => ({path: '/media/' + n + '.mkv', filename: n + '.mkv',
            season: 'Season 01', subtitles: [], has_target_sub: false, embedded_status_known: true, active_job: null});
          this.mediaData = {
            series: Array.from({length: 60}, (_, s) => ({title: 'Show ' + String(s).padStart(3, '0'),
              episodes: Array.from({length: 120}, (_, e) => item('S' + s + 'E' + e))})),
            movies: Array.from({length: 240}, (_, m) => item('Movie ' + String(m).padStart(3, '0')))
          };
          this.mediaLoaded = true;
        };
        return app;
      };
    </script>`;
    const html = fs.readFileSync(template, 'utf8').replace('</body>', fixture + '</body>');
    await page.route('**/*', route => {
      const url = route.request().url();
      if (url.includes('cdn.tailwindcss.com')) return route.fulfill(process.env.BABEL_TAILWIND_PATH ? {path: process.env.BABEL_TAILWIND_PATH, contentType: 'application/javascript'} : {body: 'window.tailwind = {};', contentType: 'application/javascript'});
      if (url.includes('alpinejs')) return route.fulfill({path: alpine, contentType: 'application/javascript'});
      if (url === 'http://127.0.0.1:8765/') return route.fulfill({body: html, contentType: 'text/html'});
      return route.fulfill({body: '', contentType: 'text/plain'});
    });
    const session = await page.context().newCDPSession(page);
    await session.send('Performance.enable');
    const started = Date.now();
    await page.goto('http://127.0.0.1:8765/', {waitUntil: 'load', timeout: 120000});
    await page.waitForFunction(() => window.Alpine && Alpine.$data(document.body).pollTimer, null, {timeout: 120000});
    const startupMs = Date.now() - started;
    const nodes = () => page.locator('*').count();
    const before = await nodes();
    const metrics = async () => Object.fromEntries((await session.send('Performance.getMetrics')).metrics.map(m => [m.name, m.value]));
    const first = await metrics();
    await page.waitForTimeout(2000);
    const second = await metrics();
    console.log(JSON.stringify({template, startupMs, dashboardNodes: before,
      idleTaskSeconds: second.TaskDuration - first.TaskDuration}));
    if (baseline) { assert.deepEqual(errors, []); return; }
    assert.equal(await page.evaluate(() => window.mediaLoads || 0), 0, 'Dashboard must not load the library');
    assert(before < 5000, 'Dashboard DOM must not grow with library size');
    await page.getByRole('button', {name: 'Library', exact: true}).click();
    await page.waitForFunction(() => Alpine.$data(document.body).mediaLoaded);
    const shows = page.locator('h3').filter({hasText: /^Show /});
    await shows.first().waitFor();
    assert.equal(await shows.count(), 25);
    assert.equal(await page.getByText('S0E0.mkv', {exact: true}).count(), 0, 'Collapsed episodes must not exist in DOM');
    await page.getByText('Show 000', {exact: true}).click();
    await page.getByText('S0E0.mkv', {exact: true}).waitFor();
    assert.equal(await page.locator('[data-episode-row]').count(), 50);
    await page.getByRole('button', {name: 'Next episodes', exact: true}).click();
    await page.getByText('S0E50.mkv', {exact: true}).waitFor();
    await page.evaluate(() => Alpine.$data(document.body).setSelectedEpisodeFilter('Show 000', 'complete'));
    await page.waitForFunction(() => !document.querySelector('[data-episode-row]'));
    await page.evaluate(() => Alpine.$data(document.body).setSelectedEpisodeFilter('Show 000', 'all'));
    await page.getByText('S0E0.mkv', {exact: true}).waitFor();
    await page.getByRole('button', {name: 'Next episodes', exact: true}).click();
    await page.getByText('S0E50.mkv', {exact: true}).waitFor();
    assert.equal(await page.getByText('S0E0.mkv', {exact: true}).count(), 0);
    await page.getByText('Show 000', {exact: true}).click();
    await page.waitForFunction(() => !document.querySelector('[data-episode-row]'));
    await page.getByRole('button', {name: 'Next library page', exact: true}).click();
    await page.getByText('Show 025', {exact: true}).waitFor();
    await page.locator('input[x-model="mediaSearchQuery"]').fill('Show 059');
    await page.getByText('Show 059', {exact: true}).waitFor();
    assert.equal(await shows.count(), 1, 'Filtering resets pagination and searches the full library');
    await page.locator('input[x-model="mediaSearchQuery"]').fill('');
    await page.getByRole('button', {name: /^Movies \(/}).click();
    await page.getByText('Movie 000.mkv', {exact: true}).waitFor();
    assert.equal(await page.locator('[data-movie-row]').count(), 25);
    assert.equal(await shows.count(), 0, 'Inactive category must not exist in DOM');
    await page.getByRole('button', {name: 'Next library page', exact: true}).click();
    await page.getByText('Movie 025.mkv', {exact: true}).waitFor();
    await page.evaluate(() => { Alpine.$data(document.body).mediaData.movies = Alpine.$data(document.body).mediaData.movies.slice(0, 1); });
    await page.getByText('Movie 000.mkv', {exact: true}).waitFor();
    assert.equal(await page.locator('[data-movie-row]').count(), 1, 'A shrinking refresh clamps the current page');
    await page.getByRole('button', {name: 'Activity', exact: true}).click();
    await page.waitForFunction(() => !document.querySelector('[data-movie-row]'));
    assert.deepEqual(errors, []);
    console.log('PASS: lazy library load, bounded series/movie/episode DOM, pagination, search reset and unmount');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
