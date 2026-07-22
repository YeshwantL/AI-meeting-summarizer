from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "AI Meeting Summarizer"
    database_url: str = f"sqlite:///{BASE_DIR / 'meetings.db'}"
    upload_dir: Path = BASE_DIR / "uploads"
    max_upload_mb: int = 1000
    whisper_model: str = "base"
    summarizer_model: str = "facebook/bart-large-cnn"
    enable_ai_models: bool = True
    action_confidence_threshold: int = 72
    cors_origins: str = "http://localhost:5173"

    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
