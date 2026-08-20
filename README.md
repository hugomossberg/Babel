# Babel

Babel is a local web service/app for managing and processing media files (TV shows and movies).

## Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.x (if running locally without Docker)

### Installation (Docker)

The easiest way to run Babel is using Docker Compose. It is configured to run on port `18765` and mounts your media directories.

```bash
docker compose up -d
```

To view the logs:
```bash
docker compose logs -f
```

### Local Development

If you want to run the project locally without Docker:

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application (typically with Uvicorn/FastAPI):
   ```bash
   # Make sure you are in the correct directory
   # uvicorn app.main:app --reload --port 8765
   ```

## Project Structure
- `app/` - Main application code (API, Core logic, Services)
- `data/` - SQLite database storage (ignored by git)
- `tests/` - Pytest test suite
- `docker-compose.yml` - Docker configuration

## Author
- Hugo Mossberg
