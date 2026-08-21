import pytest
from app.services.translator import SubtitleTranslator
import app.services.translator
import google.genai
import asyncio

@pytest.mark.asyncio
async def test_semantic_retry_success(monkeypatch):
    translator = SubtitleTranslator()
    
    # Mock settings to force gemini
    monkeypatch.setattr(app.services.translator, "get_setting", lambda k, d="": "gemini" if k == "ai_provider" else "false")
    
    call_count = 0
    class MockResponse:
        def __init__(self, text):
            self.text = text
            
    def mock_generate_content(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return MockResponse('{"translation": " "}')
        elif call_count == 2:
            return MockResponse('{"translation": "Hello"}')
        else:
            return MockResponse('{"translation": "Hej"}')

    class MockModels:
        def generate_content(self, *args, **kwargs):
            return mock_generate_content(*args, **kwargs)
            
    class MockClient:
        def __init__(self, *args, **kwargs):
            self.models = MockModels()

    monkeypatch.setattr(google.genai, "Client", MockClient)

    result = await translator.escalate_single_line(
        target_idx=100,
        target_text="Hello",
        prev_text="Hi",
        next_text="Hey",
        target_language="Swedish",
        show_title="Test",
        is_real_untranslated=True,
        job_id=1
    )
    
    assert call_count == 3
    assert result == "Hej"

@pytest.mark.asyncio
async def test_semantic_retry_exhausted(monkeypatch):
    translator = SubtitleTranslator()
    
    monkeypatch.setattr(app.services.translator, "get_setting", lambda k, d="": "gemini" if k == "ai_provider" else "false")
    
    call_count = 0
    class MockResponse:
        def __init__(self, text):
            self.text = text
            
    def mock_generate_content(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return MockResponse('{"translation": "Hello"}') # Always identical

    class MockModels:
        def generate_content(self, *args, **kwargs):
            return mock_generate_content(*args, **kwargs)
            
    class MockClient:
        def __init__(self, *args, **kwargs):
            self.models = MockModels()

    monkeypatch.setattr(google.genai, "Client", MockClient)

    result = await translator.escalate_single_line(
        target_idx=100,
        target_text="Hello",
        prev_text="Hi",
        next_text="Hey",
        target_language="Swedish",
        show_title="Test",
        is_real_untranslated=True,
        job_id=1
    )
    
    assert call_count == 3
    assert result is None
