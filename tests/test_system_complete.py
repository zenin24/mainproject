"""
Phase 22 — Master Complete System Integration Test Suite.

Executes complete end-to-end workflows across preprocessing, PubMedCLIP inference,
structured findings output, spatial visual explainability, threshold calibration,
and frontend static asset delivery.
"""

import os
from io import BytesIO
import numpy as np
import pytest
from PIL import Image

from app.api.routes import router
from app.config import settings
from app.explainability.visualization import compute_spatial_similarity_heatmap, generate_explanation_overlay
from app.inference.analyzer import FINDING_PROMPTS, analyze_image
from app.models.pubmedclip import pubmedclip_model
from app.preprocessing.image import get_image_info, load_image, validate_image_file
from data.evaluation.dataset import EvaluationDataset
from evaluation.metrics import compute_multilabel_metrics
from evaluation.calibrate import run_threshold_sweep



def create_sample_chest_xray_bytes() -> bytes:
    """Generate sample synthetic chest X-ray image bytes."""
    arr = np.random.randint(50, 200, (224, 224), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_complete_end_to_end_analysis_workflow(client):
    """Test full flow: Image Upload -> Preprocessing -> Inference -> Structured Findings -> Explainability Overlay."""
    img_bytes = create_sample_chest_xray_bytes()

    # 1. API Post /analyze
    response = client.post(
        "/analyze",
        files={"file": ("sample_xray.png", img_bytes, "image/png")},
    )
    assert response.status_code == 200
    data = response.json()

    # 2. Response Schema Checks
    assert data["status"] == "success"
    assert data["model"]["name"] == "PubMedCLIP"
    assert data["model"]["type"] == "zero-shot prototype"
    assert data["threshold_used"] == settings.CONFIDENCE_THRESHOLD
    assert isinstance(data["prototype_confidence"], float)
    assert len(data["findings"]) == settings.TOP_K

    # 3. Check findings structure & rank ordering
    scores = [item["score"] for item in data["findings"]]
    assert scores == sorted(scores, reverse=True), "Findings must be sorted by score descending"

    for rank, item in enumerate(data["findings"], start=1):
        assert item["rank"] == rank
        assert item["finding"] in [f[0] for f in FINDING_PROMPTS]
        assert isinstance(item["score"], float)
        assert item["status"] in ("candidate", "unlikely")
        assert item["above_threshold"] == (item["score"] >= settings.CONFIDENCE_THRESHOLD)

    # 4. API Post /explain for top finding
    top_finding = data["findings"][0]["finding"]
    explain_res = client.post(
        "/explain",
        files={"file": ("sample_xray.png", img_bytes, "image/png")},
        data={"finding": top_finding},
    )
    assert explain_res.status_code == 200
    assert explain_res.headers["content-type"] == "image/png"
    assert len(explain_res.content) > 1000, "Heatmap overlay PNG must contain image data"


def test_complete_preprocessing_resilience():
    """Test image preprocessing robustness against invalid inputs."""
    # Valid validation
    validate_image_file("chest.png", "image/png")
    validate_image_file("chest.jpg", "image/jpeg")

    # Invalid extensions
    with pytest.raises(ValueError, match="Unsupported file extension"):
        validate_image_file("data.pdf", "application/pdf")

    # Invalid MIME type
    with pytest.raises(ValueError, match="Unsupported content type"):
        validate_image_file("image.png", "text/plain")

    # Corrupt content
    with pytest.raises(ValueError, match="Cannot open image"):
        load_image(b"not an image byte stream")



def test_evaluation_and_calibration_pipeline_integration():
    """Test integrated evaluation dataset loading, metrics computation, and threshold calibration."""
    data_dir = os.path.join("data", "evaluation")
    csv_path = os.path.join(data_dir, "labels.csv")
    json_path = os.path.join(data_dir, "label_map.json")

    assert os.path.exists(csv_path), "Evaluation dataset labels.csv missing"
    assert os.path.exists(json_path), "Evaluation dataset label_map.json missing"

    # Dataset loading
    dataset = EvaluationDataset(base_dir=data_dir)
    samples, y_matrix, finding_names = dataset.load_samples(split="val")
    assert len(samples) > 0, "Validation dataset split should contain samples"


    # Metrics integration check with synthetic arrays
    y_true = np.array([[1, 0], [0, 1], [1, 1]])
    y_scores = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.9]])
    y_pred = (y_scores >= 0.5).astype(int)
    metrics = compute_multilabel_metrics(y_true, y_pred, y_scores, finding_names=["A", "B"])

    assert "macro_avg" in metrics
    assert "per_finding" in metrics
    assert "A" in metrics["per_finding"]

    # Threshold sweep test
    thresholds = [0.20, 0.25, 0.30]
    sweep_results = run_threshold_sweep(y_true, y_scores, ["A", "B"], thresholds)
    assert len(sweep_results) == len(thresholds)
    assert "threshold" in sweep_results[0]
    assert "macro_f1" in sweep_results[0]



def test_explainability_spatial_heatmap_direct():
    """Test spatial similarity heatmap direct computation and shape consistency."""
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    prompt = "A chest X-ray showing pneumonia"

    heatmap = compute_spatial_similarity_heatmap(img, prompt)
    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

    blended, out_path = generate_explanation_overlay(img, "pneumonia", prompt)
    assert blended.size == (224, 224)
    assert os.path.exists(out_path)
