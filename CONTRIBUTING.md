# Contributing to Babel

Thank you for your interest in improving Babel!

## Guidelines

1. **Fork and Branch:** Create a fork and work on a focused branch.
2. **Run Tests:** Ensure all tests pass (`pytest -v`) before submitting a pull request.
3. **Focused PRs:** Keep PRs small and focused on a single issue or feature.
4. **Testing:** Coding changes should include or update tests when applicable.
5. **No Secrets:** Never commit or upload log dumps containing API keys or tokens.

## Development Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest pytest-asyncio
pytest -v
```
