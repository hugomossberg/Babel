# Babel — Automated AI Subtitle Translation & QA Engine

**Babel** is a self-hosted subtitle translation engine built for the Sonarr/Radarr/Bazarr/Jellyfin ecosystem. It automatically finds, translates, validates and publishes subtitle files for your media library using AI.

Babel treats AI as an **unreliable component** — every translation passes through a strict Quality Assurance gate before it's published. If the AI drops lines, leaves text untranslated, or breaks timing sync, Babel catches it and either repairs it automatically or blocks the file.

## Features

- **AI Translation** — Gemini, OpenAI, DeepL, Ollama (local LLMs)
- **Smart Recovery** — If AI drops lines, Babel retries only the missing ones
- **QA Gate** — Every file is validated before publishing (sync, completeness, language detection)
- **SDH Cleaning** — Strips hearing-impaired tags before translation to save tokens and improve quality
- **Bazarr Integration** — Checks for human subtitles first, falls back to AI only when needed
- **Sonarr/Radarr Webhooks** — Fully automated pipeline triggered on download/upgrade
- **Glossary & Context** — Consistent translations across episodes with show-aware glossaries
- **Multi-Language** — Translate to any language, not just Swedish
- **Original Language Guard** — Won't translate to a language the audio is already in
- **Docker** — One command to install, runs alongside your existing media stack

## Quick Start

```bash
git clone https://github.com/hugomossberg/babel-subtitles.git
cd babel-subtitles

# Set your media paths
export TV_PATH=/path/to/your/tv/shows
export MOVIES_PATH=/path/to/your/movies

docker compose up -d
```

Open **http://your-server:8765** → Enter your AI API key → Configure languages → Done.

## How It Works

```
Video arrives (Sonarr/Radarr webhook)
        │
        ▼
  ┌─────────────┐
  │ Target Check │ ← Already have Swedish subtitle? Skip.
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Bazarr Check │ ← Human subtitle available? Use it.
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Source Find  │ ← Embedded English → External → Bazarr fallback
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ SDH Cleaner  │ ← Strip [door closes], ♪ music ♪, etc.
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ AI Translate │ ← Batch translation with context & glossary
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │   QA Gate    │ ← Line count ✓ Language ✓ Sync ✓ No drops ✓
  └──────┬──────┘
         ▼
    PASS → Publish .sv.srt
    FAIL → Retry / Block
```

## Sonarr/Radarr Webhook Setup

1. In Sonarr/Radarr, go to **Settings → Connect → Add → Webhook**
2. Set URL to: `http://babel:8765/webhook/sonarr` (or `/webhook/radarr`)
3. Trigger on: **Download**, **Upgrade**

## Configuration

All settings are managed through the web UI at `http://your-server:8765`.

| Setting | Description |
|---------|-------------|
| AI Provider | Gemini, OpenAI, DeepL, or Ollama |
| Batch Size | Lines per AI request (50 recommended) |
| Target Languages | Any combination (Swedish, German, French, etc.) |
| Bazarr Integration | URL + API key for human subtitle fallback |
| Glossary | Custom term translations for consistency |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TV_PATH` | `/data/media/tv` | Path to TV show library |
| `MOVIES_PATH` | `/data/media/movies` | Path to movie library |
| `BABEL_PORT` | `8765` | Web UI port |
| `TZ` | `Europe/Stockholm` | Timezone |

## Project Structure

- `app/` — Main application (FastAPI)
  - `api/` — REST endpoints & webhooks
  - `core/` — Cleaner, validator, extractor, database
  - `services/` — Pipeline, translator, Bazarr/Jellyfin integration
- `tests/` — Test suite
- `data/` — SQLite database (auto-created, git-ignored)

## Author

- Hugo Mossberg
