"""
Tests for Phase 18 explainability and visual attribution.
"""

import os
import numpy as np
import pytest
from PIL import Image

from app.models.pubmedclip import pubmedclip_model
from app.explainability.visualization import (
    compute_spatial_similarity_heatmap,
    generate_explanation_overlay,
)


@pytest.fixture(scope="module")
def loaded_model():
    """Ensure model is loaded for explainability tests."""
    pubmedclip_model.load()
    return pubmedclip_model


def test_spatial_similarity_heatmap_generation(loaded_model):
    """Test generating 2D spatial similarity heatmap from PIL image."""
    img = Image.new("RGB", (224, 224), color="gray")
    prompt = "A chest X-ray showing pneumonia"

    heatmap = compute_spatial_similarity_heatmap(img, prompt)

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.ndim == 2
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_generate_explanation_overlay_file_output(loaded_model, tmp_path):
    """Test overlay generation and saving visual explanation image to disk."""
    img = Image.new("RGB", (256, 256), color=(100, 100, 100))
    prompt = "A chest X-ray showing cardiomegaly"
    out_file = str(tmp_path / "test_explanation.png")

    blended_img, saved_path = generate_explanation_overlay(
        image=img,
        finding_name="cardiomegaly",
        text_prompt=prompt,
        output_path=out_file,
    )

    assert isinstance(blended_img, Image.Image)
    assert blended_img.size == (256, 256)
    assert os.path.exists(saved_path)
