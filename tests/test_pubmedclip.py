"""
Tests for PubMedCLIP model wrapper and similarity calculation.
"""

import pytest
import torch
from PIL import Image

from app.models.pubmedclip import pubmedclip_model


@pytest.fixture(scope="module")
def loaded_model():
    """Fixture ensuring the model is loaded once for tests."""
    pubmedclip_model.load()
    return pubmedclip_model


def test_model_loaded(loaded_model):
    """Verify model reports loaded state."""
    assert loaded_model.is_loaded
    info = loaded_model.get_model_info()
    assert info["loaded"] is True
    assert info["name"] == "PubMedCLIP"


def test_get_image_embedding(loaded_model):
    """Test image embedding generation and normalization."""
    img = Image.new("RGB", (224, 224), color="white")
    emb = loaded_model.get_image_embedding(img)

    assert isinstance(emb, torch.Tensor)
    assert emb.ndim == 2
    assert emb.shape[0] == 1  # batch dimension
    # Check L2 normalization (norm should be ~1.0)
    norm = torch.norm(emb, dim=-1).item()
    assert pytest.approx(norm, abs=1e-4) == 1.0


def test_get_text_embeddings(loaded_model):
    """Test text embedding generation and normalization."""
    prompts = ["A chest X-ray showing pneumonia", "A normal chest X-ray"]
    embs = loaded_model.get_text_embeddings(prompts)

    assert isinstance(embs, torch.Tensor)
    assert embs.shape[0] == 2
    norms = torch.norm(embs, dim=-1)
    for norm in norms:
        assert pytest.approx(norm.item(), abs=1e-4) == 1.0


def test_compute_similarity(loaded_model):
    """Test cosine similarity computation."""
    img = Image.new("RGB", (224, 224), color="gray")
    prompts = ["A chest X-ray showing pneumonia", "A normal chest X-ray"]

    img_emb = loaded_model.get_image_embedding(img)
    text_embs = loaded_model.get_text_embeddings(prompts)

    sims = loaded_model.compute_similarity(img_emb, text_embs)
    assert sims.shape[0] == 2
    # Similarity values for normalized vectors should be between -1.0 and 1.0
    for s in sims.tolist():
        assert -1.0 <= s <= 1.0
