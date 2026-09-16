"""
Application configuration.

Loads settings from environment variables and .env file.
Provides a single Settings instance used throughout the application.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API settings
    APP_TITLE: str = "Chest X-ray Finding Analyzer"
    APP_DESCRIPTION: str = (
        "Research prototype for chest X-ray image analysis using PubMedCLIP."
    )
    APP_VERSION: str = "0.1.0"

    # Model settings (used in later phases)
    MODEL_NAME: str = "flaviagiammarino/pubmed-clip-vit-base-patch32"
    DEVICE: str = "auto"  # "auto", "cpu", or "cuda"

    # Inference settings (used in later phases)
    CONFIDENCE_THRESHOLD: float = 0.25
    TOP_K: int = 5

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Single shared settings instance
settings = Settings()
