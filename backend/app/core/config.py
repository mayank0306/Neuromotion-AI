"""
Application configuration management.

Uses Pydantic Settings for type-safe configuration management.
Environment variables override defaults automatically.
"""

from typing import List, Optional
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables can be set directly or loaded from .env files.
    """

    # ┌────────────────────────────────────────────────────────────┐
    # │ Application Settings                                       │
    # └────────────────────────────────────────────────────────────┘
    APP_NAME: str = "NeuroMotion"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"
    APP_DEBUG: bool = False
    SECRET_KEY: str = "change-this-in-production"

    # ┌────────────────────────────────────────────────────────────┐
    # │ API Configuration                                            │
    # └────────────────────────────────────────────────────────────┘
    API_V1_STR: str = "/api/v1"
    API_DOCS_ENABLED: bool = True

    # ┌────────────────────────────────────────────────────────────┐
    # │ Database Configuration                                       │
    # └────────────────────────────────────────────────────────────┘
    DATABASE_URL: str = "sqlite+aiosqlite:///./neuromotion.db"
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # ┌────────────────────────────────────────────────────────────┐
    # │ Redis Configuration                                          │
    # └────────────────────────────────────────────────────────────┘
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    CACHE_TTL_SECONDS: int = 300

    # ┌────────────────────────────────────────────────────────────┐
    # │ Authentication                                               │
    # └────────────────────────────────────────────────────────────┘
    JWT_SECRET_KEY: str = "change-this-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    BCRYPT_ROUNDS: int = 12

    # ┌────────────────────────────────────────────────────────────┐
    # │ AI Model Configuration                                       │
    # └────────────────────────────────────────────────────────────┘
    POSE_MODEL_PATH: str = "models/blazepose"
    MODEL_CONFIDENCE_THRESHOLD: float = 0.5
    INPUT_SIZE_STR: str = "256,256"
    # Scoring weights for overall quality score
    ANGLE_ACCURACY_WEIGHT: float = 0.4
    SYMMETRY_WEIGHT: float = 0.2
    STABILITY_WEIGHT: float = 0.25
    RANGE_OF_MOTION_WEIGHT: float = 0.15

    @property
    def INPUT_SIZE(self) -> tuple[int, int]:
        """Parse input size from string."""
        parts = self.INPUT_SIZE_STR.split(",")
        return (int(parts[0]), int(parts[1]))

    @property
    def SCORING_WEIGHTS(self) -> dict[str, float]:
        """Get scoring weights for quality score calculation."""
        return {
            "angle_accuracy": self.ANGLE_ACCURACY_WEIGHT,
            "symmetry": self.SYMMETRY_WEIGHT,
            "stability": self.STABILITY_WEIGHT,
            "range_of_motion": self.RANGE_OF_MOTION_WEIGHT
        }

    # ┌────────────────────────────────────────────────────────────┐
    # │ Storage Configuration                                        │
    # └────────────────────────────────────────────────────────────┘
    S3_BUCKET: str = "neuromotion-media"
    S3_REGION: str = "us-east-1"
    S3_ENDPOINT_URL: str = "https://s3.amazonaws.com"
    S3_ACCESS_KEY_ID: Optional[str] = None
    S3_SECRET_ACCESS_KEY: Optional[str] = None
    MAX_UPLOAD_SIZE_MB: int = 100

    # ┌────────────────────────────────────────────────────────────┐
    # │ Email Configuration                                          │
    # └────────────────────────────────────────────────────────────┘
    SMTP_HOST: str = "smtp.mailgun.org"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    SMTP_FROM_EMAIL: str = "NeuroMotion <no-reply@neuromation.ai>"

    # ┌────────────────────────────────────────────────────────────┐
    # │ Payment Configuration                                        │
    # └────────────────────────────────────────────────────────────┘
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    STRIPE_SUCCESS_URL: str = "https://neuromation.ai/success"
    STRIPE_CANCEL_URL: str = "https://neuromation.ai/cancel"

    # ┌────────────────────────────────────────────────────────────┐
    # │ Frontend URLs                                                │
    # └────────────────────────────────────────────────────────────┘
    NEXT_PUBLIC_API_URL: str = "http://localhost:8000/api/v1"
    NEXT_PUBLIC_APP_NAME: str = "NeuroMotion"

    # ┌────────────────────────────────────────────────────────────┐
    # │ Monitoring & Logging                                         │
    # └────────────────────────────────────────────────────────────┘
    LOG_LEVEL: str = "INFO"
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_ENABLED: bool = True
    ENABLE_TRACING: bool = False

    # ┌────────────────────────────────────────────────────────────┐
    # │ Feature Flags                                                │
    # └────────────────────────────────────────────────────────────┘
    ENABLE_TELEMEDICINE: bool = False
    ENABLE_GAIT_ANALYSIS: bool = True
    ENABLE_POSTURE_ASSESSMENT: bool = True
    ENABLE_DATA_EXPORT: bool = True

    # ┌────────────────────────────────────────────────────────────┐
    # │ Development Settings                                         │
    # └────────────────────────────────────────────────────────────┘
    ENABLE_DEBUG_TOOLBAR: bool = False
    ENABLE_CORS: bool = True
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:8000"

    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:
        """Parse CORS origins from string."""
        return self.CORS_ORIGINS.split(",")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.

    Uses LRU cache to avoid re-reading environment variables
    on every request. Clear cache with `get_settings.cache_clear()`
    when configuration changes.
    """
    return Settings()


settings = get_settings()
