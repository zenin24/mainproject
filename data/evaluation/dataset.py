"""
Evaluation dataset loader.

Loads ground truth labels and preprocessed images for model evaluation.
Ensures identical preprocessing as the API endpoint via app.preprocessing.image.
"""

import csv
import json
import logging
import os
from typing import Literal

import numpy as np
from PIL import Image

from app.inference.analyzer import get_finding_names
from app.preprocessing.image import load_image

logger = logging.getLogger(__name__)


class EvaluationDataset:
    """Dataset manager for evaluation and threshold calibration."""

    def __init__(
        self,
        base_dir: str = os.path.join("data", "evaluation"),
        labels_csv: str = "labels.csv",
        label_map_json: str = "label_map.json",
    ) -> None:
        self.base_dir = base_dir
        self.img_dir = os.path.join(base_dir, "images")
        self.labels_path = os.path.join(base_dir, labels_csv)
        self.label_map_path = os.path.join(base_dir, label_map_json)

        self.model_findings = get_finding_names()
        self._load_label_map()
        self._load_csv()

    def _load_label_map(self) -> None:
        """Load mapping between CSV header labels and internal finding names."""
        if os.path.exists(self.label_map_path):
            with open(self.label_map_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                self.column_to_finding = content.get("label_mapping", {})
        else:
            self.column_to_finding = {}

    def _load_csv(self) -> None:
        """Parse ground truth annotations CSV."""
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels CSV file not found at '{self.labels_path}'")

        self.records = []
        with open(self.labels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)

        logger.info("Loaded %d evaluation samples from '%s'", len(self.records), self.labels_path)

    def get_finding_names(self) -> list[str]:
        """Return the list of finding names matching the model order."""
        return self.model_findings

    def load_samples(
        self, split: Literal["val", "test", "all"] = "all"
    ) -> tuple[list[tuple[str, Image.Image]], np.ndarray, list[str]]:
        """Load image samples, binary ground-truth matrix, and finding order.

        Args:
            split: Filter samples by split ('val', 'test', or 'all').

        Returns:
            Tuple of:
                - List of (image_id, PIL_RGB_Image)
                - Binary ground truth matrix Y of shape (N_samples, N_findings)
                - List of finding names corresponding to columns of Y
        """
        filtered_records = [
            r for r in self.records if split == "all" or r.get("split") == split
        ]

        images: list[tuple[str, Image.Image]] = []
        y_labels: list[list[int]] = []

        # Map internal finding name to inverse CSV column
        finding_to_column = {v: k for k, v in self.column_to_finding.items()}

        for rec in filtered_records:
            img_id = rec["image_id"]
            img_path = os.path.join(self.img_dir, img_id)

            if not os.path.exists(img_path):
                logger.warning("Image file '%s' missing, skipping sample.", img_path)
                continue

            with open(img_path, "rb") as f:
                img_bytes = f.read()

            # EXACT preprocessing as used in production API
            pil_image = load_image(img_bytes)
            images.append((img_id, pil_image))

            # Build binary ground truth vector matching self.model_findings order
            sample_vector = []
            for finding in self.model_findings:
                csv_col = finding_to_column.get(finding, finding)
                val = int(rec.get(csv_col, 0))
                sample_vector.append(val)

            y_labels.append(sample_vector)

        y_matrix = np.array(y_labels, dtype=int) if y_labels else np.zeros((0, len(self.model_findings)), dtype=int)
        return images, y_matrix, self.model_findings
