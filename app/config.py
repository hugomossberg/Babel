import os
from pydantic import BaseModel

class Settings(BaseModel):
    app_name: str = "Babel"
    version: str = "0.1.0"
    port: int = int(os.getenv("PORT", "8765"))
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    batch_size: int = int(os.getenv("BATCH_SIZE", "50"))
    wait_time_seconds: int = int(os.getenv("WAIT_TIME_SECONDS", "15"))
    jellyfin_url: str = os.getenv("JELLYFIN_URL", "http://dev-jellyfin:8096")
    jellyfin_api_key: str = os.getenv("JELLYFIN_API_KEY", "devtestkey1234567890abcdef")

settings = Settings()
