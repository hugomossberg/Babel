# Changelog

## v2.5.7-beta

### Fixed

- Prevent duplicate physical extraction scans under concurrent requests by rechecking subtitle cache after I/O semaphore admission.
- Clarify active job status during media inspection, subtitle resolving, and SDH cleaning, replacing inaccurate `TRANSLATING 0%` displays with descriptive preparation stages.
- Propagate `current_batch` across library and dashboard active job views with null-safe Alpine template evaluation.

### Added

- Regression test suites covering concurrent cache handoff, semaphore release on cancellation, and source preparation activity preservation.

### Upgrade Notes

No database migration is required. Existing settings, API keys, and job histories are preserved.

Linux amd64 images: `ghcr.io/hugomossberg/babel:v2.5.7-beta` and `:beta`; `latest` is unchanged. Storage-level MKV extraction performance variance and typography case handling for specific media sources remain under investigation.

[Release notes](docs/releases/v2.5.7-beta.md) · [Full Changelog](https://github.com/hugomossberg/Babel/compare/v2.5.6-beta...v2.5.7-beta)

## v2.5.6-beta

### Added

- Optional Bazarr Grace Period in days and hours, with persistent deadlines, manual bypass and duplicate-job prevention. Default: 0 hours.
- Configurable AI request timeout for translation, QA and recovery.

### Fixed

- Library startup, Alpine initialization, polling overlap and lifecycle cleanup.
- Subtitle discovery across language metadata, sidecars and alternate embedded tracks.
- Oversized font sizes from ASS/SSA-to-SRT conversion, preserving native SRT formatting.
- Recovery progress after timeouts, bounded retries and stale timing verdicts.
- Jellyfin authentication-header compatibility.
- Italian ordinal QA false positives without weakening contamination checks.

### Improved

- Provider model discovery, pagination, filtering and live/cache/fallback feedback while preserving saved and custom model IDs.
- Ollama installed-model listing, DeepSeek live discovery and OpenAI request compatibility.
- Timing-offset evidence across consolidated dialogue cues.

### Security

- Move Plex authentication tokens from URLs to headers and sanitize notifier errors.
- Update certifi to 2026.10.19, pyasn1 to 0.6.5 and urllib3 to 2.8.0.
- Guard tag-triggered publication with public-repository, version, reviewed-notes and main-source checks.

### Upgrade Notes

Back up the database before upgrading; Grace Period fields are added automatically. Existing settings and model selections are preserved. Older versions require a matching database backup for rollback. Custom discovery URLs must not contain credentials, query strings or fragments.

Linux amd64 images: `ghcr.io/hugomossberg/babel:v2.5.6-beta` and `:beta`; `latest` is unchanged. Reporter-specific integration checks and PT-BR dialect-quality verification remain outstanding; multiple Gemini keys are not included.

[Release notes](docs/releases/v2.5.6-beta.md) · [Full Changelog](https://github.com/hugomossberg/Babel/compare/v2.5.5-beta...v2.5.6-beta)
