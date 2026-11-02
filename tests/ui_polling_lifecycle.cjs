// No DOM or dependency shim: execute the shipped component with controlled I/O.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(process.argv[2], 'utf8');
const source = html.slice(html.indexOf('function babelApp()'), html.lastIndexOf('</script>'));
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };
const flush = async () => { for (let i = 0; i < 40; i++) await Promise.resolve(); };
const response = data => ({ ok: true, json: async () => data });
const library = { series: [], movies: [{ path: '/movie.mkv', title: 'Cached movie', active_job: null }], scanning_embedded: false };

function harness() {
  let nextTimer = 0;
  const timers = new Map(), calls = [], replacements = [], storage = new Map();
  const context = {
    console, URLSearchParams, Date, Set, AbortController,
    navigator: { clipboard: { writeText: () => {} } },
    setInterval: (callback, ms) => { const id = ++nextTimer; timers.set(id, { callback, ms, interval: true }); return id; },
    setTimeout: (callback, ms) => { const id = ++nextTimer; timers.set(id, { callback, ms, interval: false }); return id; },
    clearInterval: id => timers.delete(id), clearTimeout: id => timers.delete(id),
    sessionStorage: { getItem: k => storage.get(k), setItem: (k, v) => storage.set(k, v), removeItem: k => storage.delete(k) },
    window: { isSecureContext: true, location: { pathname: '/', search: '', replace: url => replacements.push(url) } },
    document: { querySelector: () => null },
  };
  context.fetch = async url => {
    calls.push(url);
    if (url === '/api/settings/all') return response({ ai: { ai_provider: 'gemini' }, integrations: {}, languages: [] });
    if (url.startsWith('/api/media-files/status')) return response({ scanning_embedded: false });
    if (url.startsWith('/api/media-files')) return response(structuredClone(library));
    if (url.startsWith('/api/jobs') || url.startsWith('/api/queue/deferred')) return response([]);
    if (url.startsWith('/api/updates')) return response({ updater_status: 'idle', current_version: 'v1', latest_version: 'v2' });
    return response({});
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  const app = context.babelApp();
  app.$nextTick = callback => callback();
  const fire = ms => {
    const pending = [];
    for (const [id, timer] of [...timers]) {
      if (timer.ms !== ms || !timers.has(id)) continue;
      if (!timer.interval) timers.delete(id);
      pending.push(timer.callback());
    }
    return pending;
  };
  return { app, timers, calls, context, replacements, fire, storage };
}

const scenarios = {
  async source_preparation_labels() {
    const h = harness();
    const preparing = { status: 'TRANSLATING', total_lines: 827, processed_lines: 100,
      current_batch: 'Preparing source: resolving subtitles' };
    assert.equal(h.app.isPreparingSource(preparing), true);
    assert.equal(h.app.getActiveJobLabel(preparing), preparing.current_batch);
    assert.equal(h.app.isPreparingSource(null), false);
    assert.equal(h.app.getActiveJobLabel(null), '');
    assert.equal(h.app.isPreparingSource({ ...preparing, status: 'CANCELLED' }), false);
    assert.equal(h.app.getActiveJobLabel({ ...preparing, status: 'CANCELLED' }), 'CANCELLED');
    const translating = { ...preparing, current_batch: 'Translating lines 1-150' };
    assert.equal(h.app.isPreparingSource(translating), false);
    assert.equal(h.app.getActiveJobLabel(translating), 'Translating 12%');
    assert.equal(h.app.getActiveJobLabel({ status: 'TRANSLATING', total_lines: 100, processed_lines: 0 }), 'Translating 0%');
    assert.match(html, /getActiveJobLabel\(ep\.active_job\)/);
    assert.match(html, /getActiveJobLabel\(movie\.active_job\)/);
    assert.match(html, /getActiveJobLabel\(job\)/);
  },
  async model_reactive_identity() {
    const h = harness();
    h.app._modelRequests = new Proxy({}, { get(target, key) {
      const value = target[key];
      return value && typeof value === 'object' ? new Proxy(value, {}) : value;
    } });
    h.context.fetch = async () => response({ models: [{ id: 'exact-model' }], source: 'live' });
    await h.app.fetchProviderModels('openai');
    assert.equal(h.app.fetchedModels.openai[0].id, 'exact-model');
    assert.equal(h.app.fetchingModels.openai, false);
    assert.equal(Object.keys(h.app._modelRequests).length, 0);
  },
  async model_saved_selection() {
    const h = harness();
    h.app.settingsData.ai.openai_model = 'ft:Exact/Legacy';
    h.app.settingsData.ai.escalation_model = 'Legacy-Fallback';
    h.app.settingsData.ai.escalation_provider = 'openai';
    h.context.fetch = async () => response({ models: [{ id: 'gpt-4o-mini' }, { id: 'gpt-4o-mini' }], source: 'live' });
    await h.app.fetchProviderModels('openai', true);
    await h.app.onEscalationProviderChange();
    assert.equal(h.app.settingsData.ai.openai_model, 'ft:Exact/Legacy');
    assert.equal(h.app.settingsData.ai.escalation_model, 'Legacy-Fallback');
    assert.equal(h.app.fetchedModels.openai.length, 1);
    assert.equal(h.app.modelOptions('openai', 'ft:Exact/Legacy')[0].id, 'ft:Exact/Legacy');
    assert.match(h.app.modelStatus('openai'), /Live/);
  },
  async model_concurrent_refresh() {
    const h = harness(), gate = deferred();
    let calls = 0;
    h.context.fetch = async () => { calls++; await gate.promise; return response({ models: [], source: 'live' }); };
    const first = h.app.fetchProviderModels('ollama', true);
    const second = h.app.fetchProviderModels('ollama', true);
    assert.equal(calls, 1);
    assert.equal(h.app.fetchingModels.ollama, true);
    gate.resolve(); await Promise.all([first, second]);
    assert.equal(h.app.fetchingModels.ollama, false);
    assert.match(h.app.modelStatus('ollama'), /no compatible models/);
  },
  async model_changed_endpoint() {
    const h = harness(), gate = deferred();
    let firstSignal;
    h.context.fetch = async (url, options) => {
      if (url.includes('first')) { firstSignal = options.signal; await gate.promise; return response({ models: [{ id: 'old' }], source: 'live' }); }
      return response({ models: [{ id: 'new' }], source: 'live' });
    };
    h.app.settingsData.ai.ollama_url = 'http://first:11434';
    const first = h.app.fetchProviderModels('ollama');
    h.app.settingsData.ai.ollama_url = 'http://second:11434';
    await h.app.fetchProviderModels('ollama');
    assert.equal(firstSignal.aborted, true);
    gate.resolve(); await first;
    assert.equal(h.app.fetchedModels.ollama[0].id, 'new');
  },
  async model_destroy_cleanup() {
    const h = harness(), gate = deferred(); let signal;
    h.context.fetch = async (url, options) => { signal = options.signal; await gate.promise; return response({ models: [{ id: 'late' }], source: 'live' }); };
    const pending = h.app.fetchProviderModels('openai');
    const before = h.app.fetchedModels.openai;
    h.app.destroy(); assert.equal(signal.aborted, true);
    gate.resolve(); await pending;
    assert.equal(h.app.fetchedModels.openai, before);
    assert.equal(h.app.fetchedModels.openai.length, 0);
    assert.equal(Object.keys(h.app._modelRequests).length, 0);
  },
  async model_error_status() {
    const h = harness();
    h.app.fetchedModels.openai = [{ id: 'selected' }];
    h.app.modelDiscovery.openai = { source: 'live', endpoint: '' };
    h.context.fetch = async () => { throw new Error('secret must not appear'); };
    await h.app.fetchProviderModels('openai', true);
    assert.equal(h.app.fetchedModels.openai[0].id, 'selected');
    assert.match(h.app.modelStatus('openai'), /Cached.*stale/);
    assert(!h.app.modelStatus('openai').includes('secret'));
    h.context.fetch = async () => response({ models: [{ id: 'suggestion' }], source: 'fallback', error: 'authentication' });
    await h.app.fetchProviderModels('openai', true);
    assert.match(h.app.modelStatus('openai'), /Offline suggestions.*unverified/);
    assert.match(h.app.modelStatus('openai'), /saved API key/);
  },
  async model_endpoint_secrets() {
    const h = harness(); let calls = 0;
    h.context.fetch = async () => { calls++; return response({}); };
    for (const url of ['http://user:secret@localhost', 'http://localhost?key=secret']) {
      h.app.settingsData.ai.custom_openai_url = url;
      await h.app.fetchProviderModels('custom');
      assert.equal(h.app.modelDiscovery.custom.error, 'invalid_endpoint');
    }
    assert.equal(calls, 0);
  },
  async grace_settings() {
    const h = harness();
    await h.app.init();
    assert.equal(h.app.gracePeriodDays, 0);
    assert.equal(h.app.gracePeriodHours, 0);
    const original = h.context.fetch;
    let posted;
    h.context.fetch = async (url, options) => {
      if (url === '/api/settings/integrations') {
        posted = JSON.parse(options.body);
        return response({ status: 'saved' });
      }
      if (url === '/api/settings/all') return response({ ai: {}, integrations: { grace_period_hours: 51 } });
      return original(url);
    };
    h.app.gracePeriodDays = 2;
    h.app.gracePeriodHours = 3;
    await h.app.saveIntegrationsSettings();
    assert.equal(posted.grace_period_hours, 51);
    assert.equal(h.app.gracePeriodDays, 2);
    assert.equal(h.app.gracePeriodHours, 3);
    posted = null;
    h.app.gracePeriodHours = 24;
    await h.app.saveIntegrationsSettings();
    assert.equal(posted, null, 'invalid duration must not make a request');
    assert.match(h.app.getActiveJobLabel({ status: 'WAITING_GRACE', grace_until: '2026-11-04T12:00:00Z' }), /Grace Period/);
    h.app.destroy();
    assert.equal(h.timers.size, 0);
  },
  async null_job_bindings() {
    // Alpine evaluates x-text even inside an x-show=false badge.
    const h = harness();
    const bindings = [...html.matchAll(/x-text="([^"]*active_job[^"]*)"/g)]
      .map(match => match[1]);
    assert.equal(bindings.length, 6);
    for (const expression of bindings) {
      const evaluate = new Function('ep', 'movie', 'getActiveJobLabel', `return (${expression})`);
      const label = h.app.getActiveJobLabel.bind(h.app);
      assert.equal(evaluate({ active_job: null }, { active_job: null }, label), '');
      const active = { active_job: { status: 'TRANSLATING', processed_lines: 5, total_lines: 10 } };
      assert.equal(typeof evaluate(active, active, label), 'string');
    }
  },
  async duplicate_init() {
    const h = harness();
    await Promise.all([h.app.init(), h.app.init()]);
    assert.equal(h.calls.filter(u => u === '/api/stats').length, 1);
    assert.equal([...h.timers.values()].filter(t => t.interval && t.ms === 2000).length, 1);
    assert.equal([...h.timers.values()].filter(t => t.interval && t.ms === 60000).length, 1);
  },
  async startup_library() {
    const h = harness(), update = deferred(), models = deferred();
    const original = h.context.fetch;
    h.context.fetch = url => url === '/api/updates' ? update.promise : url.startsWith('/api/settings/models') ? models.promise : original(url);
    const init = h.app.init();
    await flush();
    assert.equal(h.app.mediaLoaded, true, 'library must load independently of updater/model discovery');
    assert.equal(h.app.mediaData.movies[0].title, 'Cached movie');
    update.resolve(response({ updater_status: 'idle' })); models.resolve(response({ models: [] }));
    await init;
  },
  async overlapping_dashboard() {
    const h = harness(); await h.app.init();
    const blocked = deferred(); let loads = 0;
    h.app.loadStats = async () => { loads++; await blocked.promise; };
    h.fire(2000); h.fire(2000); await flush();
    assert.equal(loads, 1, 'slow polling must not overlap');
    blocked.resolve(); await flush(); h.fire(2000); await flush();
    assert.equal(loads, 2);
  },
  async library_refresh() {
    const h = harness(); await h.app.init(); h.calls.length = 0; h.app.currentTab = 'media';
    for (let i = 0; i < 5; i++) { await Promise.all(h.fire(2000)); await flush(); }
    assert.equal(h.calls.filter(u => u.startsWith('/api/media-files')).length, 0, 'no forced scan every ten seconds');
    for (let i = 0; i < 25; i++) { await Promise.all(h.fire(2000)); await flush(); }
    assert.deepEqual(h.calls.filter(u => u.startsWith('/api/media-files')), ['/api/media-files']);
    await h.app.loadMediaFiles(true);
    assert.equal(h.calls.at(-1), '/api/media-files?force=true', 'manual scan remains explicit');
  },
  async destroy_timers() {
    const h = harness();
    h.storage.set('babelUpdateSuccess', JSON.stringify({ version: 'v1', timestamp: Date.now() }));
    await h.app.init(); h.app.showToast('hello'); h.app.scheduleEmbeddedStatusPoll(); h.app.pollHealth('v2');
    h.app.copyWebhook('process');
    await h.app.manualRefreshActivity();
    assert(h.timers.size >= 6);
    h.app.destroy(); assert.equal(h.timers.size, 0);
    const count = h.calls.length;
    h.app.showToast('late'); h.app.scheduleEmbeddedStatusPoll(); h.app.pollHealth('v2');
    await h.app.loadMediaFiles(true); await h.app.init();
    assert.equal(h.timers.size, 0); assert.equal(h.calls.length, count);
  },
  async destroy_init() {
    const h = harness(), pending = deferred();
    h.app.loadStats = () => pending.promise;
    const init = h.app.init(); await flush(); h.app.destroy(); pending.resolve(); await init;
    assert.equal(h.timers.size, 0, 'late init must not recreate polling timers');
  },
  async destroy_embedded() {
    for (const statusPoll of [false, true]) {
      const h = harness(), pending = deferred();
      h.context.fetch = () => pending.promise;
      const request = statusPoll ? h.app._pollEmbeddedStatus() : h.app.loadMediaFiles();
      h.app.destroy(); pending.resolve(response({ scanning_embedded: true, series: [], movies: [] }));
      await request; assert.equal(h.timers.size, 0, 'late embedded response must not rearm');
    }
  },
  async destroy_updates() {
    const h = harness(), pending = deferred(); h.context.fetch = () => pending.promise;
    const update = h.app.checkUpdates(); h.app.destroy();
    pending.resolve(response({ updater_status: 'verifying', latest_version: 'v2' }));
    await update; assert.equal(h.timers.size, 0);
  },
  async destroy_health() {
    const h = harness(), pending = deferred(); let signal;
    h.context.fetch = (url, options) => { signal = options.signal; return pending.promise; };
    h.app.pollHealth('v2'); const running = h.fire(2500); h.app.destroy();
    assert.equal(signal.aborted, true, 'destroy cancels in-flight health network I/O');
    pending.resolve(response({ current_version: 'v2', updater_status: 'success' }));
    await Promise.all(running); assert.equal(h.timers.size, 0); assert.equal(h.replacements.length, 0);
  },
  async health_overlap() {
    const h = harness(), pending = deferred(); let requests = 0;
    h.context.fetch = () => { requests++; return pending.promise; };
    h.app.pollHealth('v2'); h.fire(2500); h.fire(2500); await flush();
    assert.equal(requests, 1, 'health requests must not overlap');
    h.app.stopHealthPoll(); pending.resolve(response({ current_version: 'v2', updater_status: 'success' })); await flush();
    assert.equal(h.replacements.length, 0, 'stopped health request must not reload');
  },
  async health_terminal() {
    for (const status of ['failed', 'rolled_back', 'success']) {
      const h = harness();
      h.context.fetch = async () => response({ current_version: 'v2', updater_status: status });
      h.app.pollHealth('v2'); await Promise.all(h.fire(2500)); await flush();
      assert.equal(h.app.healthPollActive, false); assert.equal(h.timers.size, 0);
      assert.equal(h.replacements.length, status === 'success' ? 1 : 0);
      assert.equal(h.app.updateFailed, status !== 'success');
    }
  },
  async health_timeout() {
    const h = harness(); h.context.fetch = async () => { throw new Error('restarting'); };
    h.app.pollHealth('v2');
    for (let i = 0; i < 60; i++) { await Promise.all(h.fire(2500)); await flush(); }
    assert.equal(h.app.healthPollActive, false); assert.equal(h.app.updateFailed, true);
    assert.equal(h.app.updateData.updater_status, 'failed');
    assert.equal([...h.timers.values()].filter(t => t.interval).length, 0);
  },
  async health_deadline() {
    const h = harness(), pending = deferred(); h.context.fetch = () => pending.promise;
    h.app.pollHealth('v2'); h.fire(2500); await flush();
    h.fire(150000); await flush();
    assert.equal(h.app.healthPollActive, false, 'hung health request must not extend the update deadline');
    assert.equal(h.app.updateFailed, true);
    pending.resolve(response({ current_version: 'v2', updater_status: 'success' })); await flush();
    assert.equal(h.replacements.length, 0);
  },
  async job_completion() {
    const h = harness(); h.app.mediaData = structuredClone(library);
    h.app.mediaData.movies[0].active_job = { id: 1, status: 'TRANSLATING' };
    await h.app.refreshLibraryJobState();
    assert(h.calls.includes('/api/media-files?force=true'));
    assert.equal(h.app.mediaData.movies[0].active_job, null);
  },
  async queued_refresh() {
    const h = harness(), pending = deferred(); let loads = 0;
    h.context.fetch = url => {
      h.calls.push(url); loads++;
      return loads === 1 ? pending.promise : Promise.resolve(response(structuredClone(library)));
    };
    const initial = h.app.loadMediaFiles();
    await h.app.loadMediaFiles(true, { silent: true });
    pending.resolve(response(structuredClone(library))); await initial; await flush();
    assert.deepEqual(h.calls, ['/api/media-files', '/api/media-files?force=true'], 'one completion refresh survives an in-flight library load');
  },
  async embedded_completion() {
    const h = harness(); h.app.mediaData = { ...structuredClone(library), scanning_embedded: true };
    await h.app._pollEmbeddedStatus();
    assert.deepEqual(h.calls, ['/api/media-files/status', '/api/media-files?force=true']);
    assert.equal(h.timers.size, 0);
  },
};

(async () => {
  const scenario = scenarios[process.argv[3]];
  if (!scenario) throw new Error('Unknown scenario');
  await scenario();
})().catch(error => { console.error(error); process.exitCode = 1; });
