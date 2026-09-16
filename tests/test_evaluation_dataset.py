"""
Tests for Phase 15 evaluation dataset loader and preprocessing consistency.
"""

import os
import numpy as np
import pytest
from PIL import Image

from data.evaluation.dataset import EvaluationDataset
from app.inference.analyzer import get_finding_names


@pytest.fixture(scope="module")
def eval_dataset():
    """Fixture providing initialized EvaluationDataset."""
    return EvaluationDataset()


def test_label_map_structure(eval_dataset):
    """Verify label map contains mapping for all model findings."""
    model_findings = get_finding_names()
    dataset_findings = eval_dataset.get_finding_names()
    assert set(model_findings) == set(dataset_findings)


def test_load_samples_all_splits(eval_dataset):
    """Verify loading validation and test splits."""
    val_images, val_y, val_cols = eval_dataset.load_samples(split="val")
    test_images, test_y, test_cols = eval_dataset.load_samples(split="test")
    all_images, all_y, all_cols = eval_dataset.load_samples(split="all")

    assert len(val_images) == 8
    assert len(test_images) == 8
    assert len(all_images) == 16

    assert val_y.shape == (8, 8)
    assert test_y.shape == (8, 8)
    assert all_y.shape == (16, 8)


def test_preprocessing_consistency(eval_dataset):
    """Verify loaded images are preprocessed to PIL RGB mode."""
    val_images, _, _ = eval_dataset.load_samples(split="val")
    for img_id, pil_img in val_images:
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"
        assert pil_img.size == (256, 256)


def test_ground_truth_binary_values(eval_dataset):
    """Verify ground truth matrix contains binary indicators 0 or 1."""
    _, all_y, _ = eval_dataset.load_samples(split="all")
    unique_vals = set(np.unique(all_y))
    assert unique_vals.issubset({0, 1})
