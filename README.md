# Babel — Self-Hosted Subtitle Automation

**Babel** is a self-hosted subtitle automation engine built for Sonarr, Radarr, Bazarr and modern media servers.

It automatically finds the best available subtitle source, translates missing subtitles with AI, validates the result, repairs recoverable failures and publishes only subtitles that pass strict Quality Assurance.

> **Install it. Connect your media stack. Forget about subtitles.**

---

## Why Babel?

AI translation is powerful, but unreliable.

Models can silently:

- Skip subtitle lines
- Leave dialogue untranslated
- Return malformed output
- Break subtitle structure
- Produce incomplete translations
- Fail because of rate limits or temporary provider outages

Babel treats the AI provider as an **untrusted component**.

Every generated subtitle must pass a strict QA pipeline before it can be published.

> **No valid subtitle → no publish.**

---

## Features

### Subtitle Resolution

- Detect existing target-language subtitles
- Prefer human subtitles when available
- Bazarr integration with configurable grace period
- Extract embedded text subtitles directly from media
- Prefer embedded source tracks for accurate synchronization
- External subtitle fallback
- Automatic source selection

### AI Translation

Supported providers include:

- Google Gemini
- OpenAI
- DeepL
- Ollama / local models

Translation features include:

- Batch translation
- Context-aware translation
- Show-aware glossary support
- Multi-language target support
- Configurable providers and models
- Provider fallback and retry handling

### Strict Quality Assurance & QA Policy

Every generated subtitle is validated before publication. Babel evaluates subtitles against a three-tiered QA policy:

- **`PASS`**: All structural, synchronicity, and linguistic checks pass with zero unresolved cues (100% translated/verified). Published with full confidence and saved to Translation Memory.
- **`PASS_WITH_WARNINGS`**: All structural and timing checks pass, and all recoverable dialogue is translated, but a small bounded number of stubborn cues ($\le 1\%$ or max 5 cues) were safely preserved from source to prevent deadlock. Published cleanly; source-preserved cues are excluded from Translation Memory.
- **`FAIL`**: Structural defects, line count mismatches, timestamp corruption, significant untranslated dialogue, or detected wrong language immediately block publication and trigger automatic recovery or retry.

Babel checks:

- Subtitle structure & formatting
- Line count & cue ordering
- Dropped lines & missing blocks
- Timestamp synchronization (millisecond precision)
- Stratified target language distribution (beginning, middle, and end)
- Untranslated dialogue & normalized echo detection
- Invalid or incomplete AI output

### Automatic Recovery

When something goes wrong, Babel attempts to repair it automatically.

Recovery features include:

- Retry dropped lines
- Detect legitimate unchanged text
- Re-translate unresolved dialogue
- Contextual single-line recovery
- Optional stronger-model escalation
- Provider retry and exponential backoff
- Persistent retry states

Recoverable failures do not immediately become terminal failures.

### SDH & Noise Cleaning

Babel can conservatively remove unnecessary subtitle noise before translation.

Examples include:

```text
[door closes]
[music playing]
[phone ringing]
```

The cleaner is designed to preserve real dialogue and subtitle timing.

### Media Automation

- Sonarr webhooks
- Radarr webhooks
- Bazarr integration
- Media-server library notification
- Automatic processing after downloads and upgrades
- Manual processing from the web UI
- Background retry for recoverable failures

### Safety Features

- Original Language Guard
- Atomic subtitle publishing
- Existing subtitle protection
- Strict source synchronization
- Embedded subtitle validation
- Wrong-language detection
- Persistent recovery states
- Fail-safe QA blocking

---

## Core Philosophy

Babel follows two rules:

> **Never publish bad subtitles.**

> **Never give up on recoverable failures.**

Temporary provider errors, dropped AI output and difficult individual lines should be recovered automatically whenever possible.

If Babel cannot produce a subtitle that satisfies its hard QA requirements, the subtitle is not published.

---

## How It Works

```text
Media arrives
      │
      ▼
Target subtitle already available?
      │
      ├── Yes ──► Validate / Keep
      │
      ▼
Give Bazarr a chance to find a human subtitle
      │
      ├── Found ─► Use subtitle
      │
      ▼
Resolve best source subtitle
      │
      ├── Embedded text track
      ├── External subtitle
      └── Bazarr source fallback
      │
      ▼
Clean SDH / subtitle noise
      │
      ▼
AI translation
      │
      ▼
Smart Recovery
      │
      ▼
KEEP / TRANSLATE classification
      │
      ▼
Contextual recovery / escalation
      │
      ▼
Strict QA Gate
      │
      ├── FAIL ──► Recover / Retry / Block
      │
      ▼
     PASS
      │
      ▼
Atomic subtitle publish
      │
      ▼
Notify media server
```

---

## Pipeline Modes

### Hybrid Mode

```text
Bazarr → Human subtitle → AI fallback
```

Babel gives Bazarr a configurable amount of time to find a human subtitle before AI translation begins.

This mode is useful when you prefer professionally created or community subtitles whenever they are available.

### Pure AI Mode

```text
Source subtitle → AI translation → QA
```

Babel operates without Bazarr and immediately resolves and translates the best available source subtitle.

---

## Installation

### Docker Compose

Clone the repository:

```bash
git clone https://github.com/hugomossberg/babel-subtitles.git
cd babel-subtitles
```

Configure your media mounts in `docker-compose.yml`:

```yaml
volumes:
  - /path/to/tv:/tv
  - /path/to/movies:/movies
```

Start Babel:

```bash
docker compose up -d
```

Open the web interface:

```text
http://your-server:8765
```

Then configure:

1. AI provider
2. AI model
3. Target languages
4. Media folders
5. Optional integrations
6. Pipeline preferences

After that, Babel can run automatically.

---

## Media Paths

Babel uses paths **inside the container**.

Recommended container paths:

```text
TV Series: /tv
Movies:    /movies
```

Example Docker mounts:

```yaml
volumes:
  - /your/tv/library:/tv
  - /your/movie/library:/movies
```

Your host paths can be located anywhere.

Babel only needs the correct container paths.

---

## Sonarr / Radarr Webhooks

Babel can automatically process newly downloaded or upgraded media.

### Sonarr

Go to:

```text
Settings → Connect → Add → Webhook
```

Webhook URL:

```text
http://babel:8765/webhook/sonarr
```

Recommended triggers:

- Download
- Upgrade

### Radarr

Webhook URL:

```text
http://babel:8765/webhook/radarr
```

Recommended triggers:

- Download
- Upgrade

If Sonarr/Radarr and Babel use different container paths, configure Babel's Remote Path Mapping.

Example:

```text
Remote Path Prefix: /media/tv
Local Path Prefix:  /tv
```

---

## Bazarr Integration

Babel can work alongside Bazarr instead of replacing it.

In Hybrid Mode:

```text
Media arrives
      │
      ▼
Babel checks for target subtitle
      │
      ▼
Bazarr gets a chance to find one
      │
      ├── Found → use human subtitle
      │
      └── Missing → Babel AI fallback
```

The Bazarr grace period is configurable through the web UI.

---

## Media Server Integration

When a subtitle is successfully published, Babel can notify the configured media server so the new subtitle is detected without waiting for a scheduled library scan.

Media-server integrations are optional.

---

## Configuration

Most configuration is managed through Babel's web interface.

### AI Engine

| Setting | Description |
|---|---|
| AI Provider | Translation provider |
| AI Model | Model used for translation |
| Batch Size | Number of subtitle lines per request |
| Retry Policy | Controls temporary provider retries |
| Escalation Model | Optional stronger model for difficult lines |

### Target Languages

| Setting | Description |
|---|---|
| Target Languages | Languages Babel should maintain |
| Original Language Guard | Avoid unnecessary translation when audio already matches the target |
| Language Validation | Verify generated subtitle language before publishing |

### Media Folders

| Setting | Description |
|---|---|
| TV Series Path | Container path to series |
| Movies Path | Container path to movies |
| Remote Path Mapping | Translate external container paths into Babel paths |

### Pipeline

| Setting | Description |
|---|---|
| Hybrid Mode | Bazarr-first with AI fallback |
| Pure AI Mode | Immediate AI processing |
| Bazarr Grace Delay | Time Babel waits for a human subtitle |
| Embedded Target Extraction | Use suitable embedded target subtitles |
| Embedded Source Extraction | Prefer embedded source tracks |
| SDH Cleaner | Remove unnecessary subtitle annotations |
| Auto Repair | Detect and recover unhealthy subtitles |
| Strict Source Sync | Preserve source timestamps |
| Escalation | Retry difficult lines using contextual recovery |

---

## QA Pipeline

A translation is not automatically considered successful just because the AI provider returned a response.

Babel verifies the completed subtitle before publishing.

Hard requirements include:

```text
Valid subtitle structure
Correct line count
No dropped subtitle blocks
No unresolved dialogue
Valid timestamps
Valid synchronization
Correct target language
```

If any hard requirement fails:

```text
DO NOT PUBLISH
```

Babel will attempt recovery when possible.

---

## Smart Recovery

AI providers occasionally skip or mishandle individual subtitle lines.

Instead of translating the entire subtitle again, Babel can isolate the affected lines and retry only those.

Example:

```text
558 subtitle blocks
       │
       ▼
AI translation
       │
       ▼
2 missing lines detected
       │
       ▼
Retry those 2 lines
       │
       ▼
Re-run QA
```

This reduces cost and unnecessary API usage.

---

## KEEP / TRANSLATE Classification

Not every unchanged subtitle line is actually untranslated.

Examples:

```text
NASA
Microsoft
LeBron James
911
```

These may legitimately remain unchanged.

Babel separates safe unchanged content from genuinely untranslated dialogue before the final QA decision.

Suspicious dialogue is sent through recovery instead of being silently accepted.

---

## Contextual Recovery

Difficult isolated lines can be retried using surrounding dialogue.

Example:

```text
Previous: Are you coming with us?

TARGET: Get out!

Next: I don't want to see you again.
```

The surrounding context helps the model understand the intended meaning without retranslating the entire subtitle.

---

## Provider Failure Handling

Temporary provider problems should not permanently kill a subtitle job.

Examples include:

```text
HTTP 429
HTTP 500
HTTP 502
HTTP 503
HTTP 504
Timeouts
Temporary network failures
```

Babel can retry these failures using backoff and persistent job states.

Example internal flow:

```text
TRANSLATING
     │
     ▼
Provider unavailable
     │
     ▼
RETRY_PENDING
     │
     ▼
WAITING_PROVIDER
     │
     ▼
Retry
```

---

## Atomic Publishing

Babel does not intentionally expose partially generated subtitle files to the media server.

Temporary output is generated and validated first.

Only after validation succeeds is the final subtitle published.

```text
Generate temporary subtitle
        │
        ▼
Validate
        │
        ├── FAIL → discard / recover
        │
        ▼
       PASS
        │
        ▼
Atomic publish
```

---

## Project Structure

```text
app/
├── api/
│   └── REST endpoints and webhooks
│
├── core/
│   ├── subtitle validation
│   ├── extraction
│   ├── cleaner
│   ├── language handling
│   └── database
│
├── services/
│   ├── translation pipeline
│   ├── AI providers
│   ├── Bazarr integration
│   └── media-server integrations
│
└── templates/
    └── Web interface

tests/
└── Automated unit and regression tests

data/
└── Runtime application data and SQLite database
```

---

## Development Status

Babel is under active development.

The current focus is reliability, automatic recovery and real-world testing across different media libraries, subtitle formats and AI providers.

Expect bugs and edge cases while the project is still evolving.

Bug reports and testing feedback are welcome.

---

## Testing

Run the test suite with:

```bash
pytest -v
```

Babel includes automated tests covering core validation, pipeline behavior and regression scenarios.

Real-world media testing is also an important part of development because subtitle files can vary significantly between releases and providers.

---

## Security

Babel is designed primarily for self-hosted environments.

Do not expose Babel directly to the public internet without appropriate authentication, reverse-proxy configuration and network security.

API keys and credentials should never be committed to Git.

---

## Contributing

Contributions, bug reports and feature suggestions are welcome.

If you find a reproducible problem, please include:

- Babel version / commit
- Media type
- Subtitle source
- AI provider and model
- Relevant job log
- Expected behavior
- Actual behavior

Please remove API keys, tokens, private hostnames and personal paths before posting logs publicly.

---

## Support Babel

Babel is free and open source.

If Babel saves you time and you want to support development, you can buy me a beer 🍺.

---

## Author

**Hugo Mossberg**
