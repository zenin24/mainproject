"""
Chest X-ray Finding Analyzer — FastAPI Application

Main entry point for the FastAPI application.
Creates the app instance, includes routers, and mounts frontend static files.
"""

from contextlib import asynccontextmanager
import logging
import os
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.models.pubmedclip import pubmedclip_model

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """App lifespan context manager for startup and shutdown tasks."""
    logger.info("Initializing Chest X-ray Finding Analyzer application...")
    try:
        pubmedclip_model.load()
        logger.info("PubMedCLIP model successfully initialized on startup.")
    except Exception as e:
        logger.error("Failed to load PubMedCLIP model on startup: %s", e)
    yield
    logger.info("Shutting down Chest X-ray Finding Analyzer application...")


app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Include API routes first
app.include_router(router)

# Mount frontend static files if frontend directory exists
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir, html=True), name="static")
    logger.info("Mounted frontend static files from '%s'", frontend_dir)
