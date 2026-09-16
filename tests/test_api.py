"""
Tests for API endpoints using FastAPI TestClient.
"""

from io import BytesIO
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app



def create_test_image_bytes(fmt: str = "PNG") -> bytes:
    """Helper to create dummy PNG image bytes."""
    img = Image.new("RGB", (224, 224), color="blue")
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_root_endpoint(client):
    """Test GET / endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "model_loaded" in data


def test_health_endpoint(client):
    """Test GET /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_image_info_endpoint(client):
    """Test POST /image-info endpoint."""
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/image-info",
        files={"file": ("test_xray.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["filename"] == "test_xray.png"
    assert data["width"] == 224
    assert data["height"] == 224


def test_analyze_endpoint_success(client):
    """Test POST /analyze endpoint with valid chest X-ray image."""
    img_bytes = create_test_image_bytes("JPEG")
    response = client.post(
        "/analyze",
        files={"file": ("chest_xray.jpg", img_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["model"]["name"] == "PubMedCLIP"
    assert "findings" in data
    assert isinstance(data["findings"], list)
    assert len(data["findings"]) > 0

    # Phase 19 Schema assertions
    top_finding = data["findings"][0]
    assert "finding" in top_finding
    assert "score" in top_finding
    assert "similarity_score" in top_finding
    assert "rank" in top_finding
    assert top_finding["rank"] == 1
    assert "status" in top_finding
    assert top_finding["status"] in ("candidate", "unlikely")
    assert "explanation_available" in top_finding

    assert "explainability" in data
    assert data["explainability"]["available"] is True
    assert "prototype_confidence" in data
    assert isinstance(data["prototype_confidence"], float)
    assert "disclaimer" in data


def test_explain_endpoint_success(client):
    """Test POST /explain endpoint with valid image and finding name."""
    img_bytes = create_test_image_bytes("PNG")
    response = client.post(
        "/explain",
        files={"file": ("chest_xray.png", img_bytes, "image/png")},
        data={"finding": "pneumonia"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert len(response.content) > 0


def test_analyze_endpoint_invalid_extension(client):
    """Test POST /analyze with an unsupported file extension."""
    response = client.post(
        "/analyze",
        files={"file": ("document.pdf", b"fake content", "application/pdf")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file extension" in data["detail"]

