import pytest
from app.services.pipeline import SubtitlePipeline
from app.core.db import get_translation_memory
from unittest.mock import patch, MagicMock

# This test requires mocking QA gate and DB, so maybe it's complex. Let's see if we can just test that translator.py doesn't call it, and pipeline does.
