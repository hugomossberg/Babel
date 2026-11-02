# Model discovery and compatibility

Reviewed 2026-11-02. Model listing is not an inference/access guarantee. Babel
uses Chat Completions for OpenAI-compatible services, generateContent for Gemini,
Messages for Anthropic, Ollama chat and DeepL translate. Discovery never performs
inference or changes saved settings or pinned jobs.

## Official contracts

- [Gemini model listing](https://aistudio.google.com/api-reference/models/list-models):
  `/v1beta/models`, `nextPageToken` / `pageToken`, and explicit `generateContent`
  support. Strip only the resource's leading `models/`, preserving the remaining ID.
  [Lifecycle](https://ai.google.dev/gemini-api/docs/deprecations/) and
  [release notes](https://ai.google.dev/gemini-api/docs/changelog) distinguish the
  Developer API from Vertex/Agent Platform. Do not transfer shutdown dates between
  platforms. Omit deprecated 2.5 recommendations and the 3.5 Flash alias, which
  Google routes to 3.6 Flash. [Gemini 3.7 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash)
  is also a deprecated routing alias; replace that suggestion with documented
  [3.8 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash).
  Label these aliases if the live API lists them; never rewrite their IDs.
  [3.6 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash)
  remains supported. Keep the existing 3.5 Flash-Lite default.
- [OpenAI models](https://developers.openai.com/api/reference/resources/models/methods/list):
  `/v1/models` lists account-visible IDs, without per-model endpoint capabilities.
  Filter recognizable non-chat products and Responses-only Pro/Codex/research
  families; retain a standalone snapshot when its alias is absent. This heuristic
  cannot prove compatibility for every future model. Saved/fine-tuned IDs are
  still accepted by settings and dispatched unchanged.
  [Deprecations](https://developers.openai.com/api/docs/deprecations): remove
  retired `o1-mini` from suggestions. Keep `gpt-4o-mini` as default to avoid an
  automatic cost/behavior change; its documented snapshot retirement on
  2027-01-14 requires another review before that date.
  [GPT-6 request compatibility](https://developers.openai.com/api/docs/guides/latest-model):
  the documented families support text Chat Completions and structured outputs;
  custom temperature requires reasoning `none`, which Astra/6.1 Sol do not support.
  Unknown families keep conservative capabilities.
- [Anthropic models](https://platform.claude.com/docs/en/api/models/list):
  `/v1/models`, `has_more`, `last_id` / `after_id`. Keep the existing API version
  header; do not require the newly introduced lifecycle query parameter. Honor
  lifecycle metadata when present and known retired 3.5/3.7/early-4 families when
  older responses omit it. [Lifecycle](https://platform.claude.com/docs/en/about-claude/model-deprecations):
  remove retired 3.5 aliases and deprecated Sonnet 4.5 suggestions; retain active
  Sonnet 5 as default, add Sonnet/Haiku 5.5 suggestions. Listing is the authority
  for account-specific IDs. [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)
  documents native `output_config.format`; include known modern 4.x and Haiku 5.x
  families without assuming future unknown models have native format support.
- [OpenRouter models](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties):
  public `/api/v1/models`, exact `provider/model` IDs, architecture input/output
  modalities and supported parameters. Exclude batch and non-text products;
  respect endpoint metadata when supplied. The listing normally lacks endpoint
  compatibility metadata, so unknown compatibility remains unverified. Retired
  `anthropic/claude-3.5-sonnet` is absent from the public listing and removed from
  suggestions, along with absent `deepseek/deepseek-r1` and
  `mistralai/mistral-large-2411`. Retain the existing OpenRouter Gemini 3.7 Flash
  default, which remains listed under OpenRouter's own model ID; do not infer its
  availability from Google's separate lifecycle. A public, unauthenticated
  listing was checked during this review; no inference was performed.
- [DeepSeek models](https://api-docs.deepseek.com/api/list-models/):
  authenticated `https://api.deepseek.com/models`; honor text modalities.
  [Current models and alias policy](https://api-docs.deepseek.com/quick_start/pricing/):
  use `deepseek-flash` for new configurations instead of the retired-model alias
  `deepseek-v4-flash`, which the provider still accepts and bills as Flash. Preserve
  saved aliases verbatim. [Thinking mode](https://api-docs.deepseek.com/guides/thinking_mode/)
  documents `extra_body.thinking.type=disabled`; apply the existing non-thinking
  JSON-output behavior to the current alias as well.
- [Ollama tags](https://docs.ollama.com/api/tags): only installed models returned by
  the configured `/api/tags`. Preserve namespaces/tags and arbitrary local IDs;
  exclude recognizable embedding-only families. Offline examples are never
  returned as installed models. No installed-model capability is inferred from
  an example catalog.
- [DeepL translation](https://developers.deepl.com/api-reference/translate/request-translation):
  `prefer_quality_optimized`, `quality_optimized`, `latency_optimized` are translation
  strategies, returned as `configuration`, not discovered LLMs.
- Custom OpenAI-compatible services use the configured `/models`; no hardcoded
  family/name allowlist. Honor explicit text/endpoint metadata if supplied.
  Keep the existing `Bearer dummy` header when no custom key is configured.
  Unlisted models can be entered manually. The same applies to custom Ollama IDs.

## Cache and UI semantics

API responses retain `provider`, `default_model`, `models` and `cached`, adding:

- `source`: `live`, `cache`, `fallback` or `configuration`.
- `listing_verified`: whether an endpoint actually supplied the list. It never
  certifies inference access, payment status or every request parameter.
- `fetched_at`: original successful listing time, or null for offline suggestions.
- `stale` and sanitized `error`: failed refreshes preserve the previous listing
  and timestamp while explicitly marking it stale.

Cache scopes include provider, effective endpoint and credential fingerprint;
secrets and endpoint hostnames are not stored in cache keys. Successful empty
lists stay empty. Local/custom successes expire after 60 seconds, hosted after
one hour; failure results expire after 15 seconds. Refresh bypasses TTL; concurrent
requests share a bounded fetch. Limit the cache to 128 scopes, discovery to 20
pages/20,000 records/20 seconds, without inference retries. Locks are weakly held.

The UI preserves primary and escalation selections, including absent/legacy IDs,
deduplicates options and labels unavailable listings separately from offline
suggestions. An absent ID is *not listed*, not necessarily inaccessible. Refresh
has loading/error feedback, coalesces requests and aborts obsolete endpoint
requests and requests on component destruction. Request IDs work with Alpine's
reactive proxies. Custom/local model IDs remain editable.

Only new/unset settings receive defaults. Database initialization uses existing
`INSERT OR IGNORE` semantics. Database defaults now use the same central catalog
as the API: OpenRouter's previously inconsistent new-database Sonnet preset now
matches its existing UI/catalog Gemini default. Existing rows are never migrated.

## Security and validation limits

Keys use headers, never query strings. Errors omit provider bodies, exception text,
keys and URLs. Local/custom discovery requires HTTP(S), no userinfo, queries,
fragments, backslashes or whitespace. Reject link-local metadata IP literals and
known metadata hostnames; keep loopback, LAN addresses and Docker DNS names.
Redirects are disabled. Configured custom endpoints are administrator-trusted;
this is not a general-purpose network sandbox or DNS-rebinding defense.

Authenticated provider discovery was validated using deterministic mocked HTTP
contracts, not production credentials. Actual account entitlements and paid
translation quality are unverified. Google recommends its default sampling for
3.x; Babel continues omitting a custom temperature rather than claiming that the
SDK rejects it. No QA threshold, recovery path, subtitle rule or publishing gate
is changed by this maintenance task. The OpenAI connection probe uses documented
`max_completion_tokens` on known reasoning families, avoiding `max_tokens`
rejection there while retaining the legacy parameter for older non-reasoning models.

Connection-probe failures are also sanitized instead of returning SDK exception
strings or provider response bodies. Local/custom probes use the same endpoint
validation as discovery. No raw error detail is rendered in model refresh feedback.

[OpenAI's structured-output contract](https://developers.openai.com/api/docs/guides/structured-outputs)
distinguishes JSON mode from strict schemas. GPT-3.5 Turbo, GPT-4 Turbo and
`gpt-4o-2024-05-13` must use JSON mode instead of unsupported `json_schema`.
The base GPT-4 alias gets no optional JSON response format. Later GPT-4o snapshots,
GPT-4o mini and GPT-4.1 retain strict schemas. This changes provider parameters,
not Babel's parsing, cue/timestamp validation, QA or publication requirements.
