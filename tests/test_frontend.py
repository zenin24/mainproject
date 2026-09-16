"""
Tests for Phase 20 frontend UI components and static file structure.
"""

import os
import pytest


def test_frontend_files_exist():
    """Verify frontend HTML, CSS, and JS files exist."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    html_path = os.path.join(base_dir, "index.html")
    css_path = os.path.join(base_dir, "styles.css")
    js_path = os.path.join(base_dir, "app.js")

    assert os.path.exists(html_path), "index.html is missing"
    assert os.path.exists(css_path), "styles.css is missing"
    assert os.path.exists(js_path), "app.js is missing"


def test_frontend_html_contents():
    """Verify HTML contains required UI sections and safety disclaimers."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    html_path = os.path.join(base_dir, "index.html")

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Title & Header
    assert "Chest X-ray Finding Analyzer" in content
    # Safety Disclaimer Banner
    assert "Not for Clinical Diagnosis" in content or "Research Prototype" in content
    # Upload Dropzone & File Input
    assert 'id="dropzone"' in content
    assert 'id="fileInput"' in content
    # Analyze Action Button
    assert 'id="analyzeBtn"' in content
    # Findings Table
    assert 'class="findings-table"' in content
    # Explainability Card
    assert 'id="explainabilityCard"' in content


def test_frontend_js_api_integration():
    """Verify app.js references /analyze endpoint and required DOM elements."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
    js_path = os.path.join(base_dir, "app.js")

    with open(js_path, "r", encoding="utf-8") as f:
        js_content = f.read()

    assert "/analyze" in js_content, "app.js must call the /analyze endpoint"
    assert "prototype_confidence" in js_content, "app.js must display prototype_confidence"
    assert "findings" in js_content, "app.js must process findings array"
    assert "status-candidate" in js_content, "app.js must render status-candidate class"
    assert "status-unlikely" in js_content, "app.js must render status-unlikely class"


def test_static_files_route(client):
    """Test that static files are accessible via FastAPI TestClient."""
    response = client.get("/static/index.html")
    assert response.status_code == 200
    assert "Chest X-ray Finding Analyzer" in response.text

    css_response = client.get("/static/styles.css")
    assert css_response.status_code == 200

    js_response = client.get("/static/app.js")
    assert js_response.status_code == 200
    assert "/analyze" in js_response.text

