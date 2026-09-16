"""
Shared pytest fixtures across unit and integration test modules.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    """Test client fixture that triggers FastAPI lifespan startup/shutdown."""
    with TestClient(app) as c:
        yield c
