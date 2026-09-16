"""
PubMedCLIP model wrapper.

Loads the PubMedCLIP model and processor once, provides methods
for generating image and text embeddings, and computing similarity.
"""

import logging
from typing import Any

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import settings

logger = logging.getLogger(__name__)


def _resolve_device() -> torch.device:
    """Determine the compute device based on configuration and availability.

    Returns:
        torch.device for inference.
    """
    device_setting = settings.DEVICE.lower()

    if device_setting == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    elif device_setting == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)
    return device


class PubMedCLIPModel:
    """Wrapper around PubMedCLIP for image-text similarity.

    The model and processor are loaded once during initialization
    and reused for all subsequent requests.
    """

    def __init__(self) -> None:
        self._model: CLIPModel | None = None
        self._processor: CLIPProcessor | None = None
        self._device: torch.device | None = None
        self._loaded = False

    def load(self) -> None:
        """Load the PubMedCLIP model and processor from Hugging Face.

        This should be called once at application startup.
        """
        if self._loaded:
            logger.info("Model already loaded, skipping.")
            return

        model_name = settings.MODEL_NAME
        logger.info("Loading PubMedCLIP model: %s", model_name)

        self._device = _resolve_device()

        try:
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model = CLIPModel.from_pretrained(model_name)
            self._model.to(self._device)
            self._model.eval()
            self._loaded = True
            logger.info(
                "PubMedCLIP loaded successfully on %s", self._device
            )
        except Exception as e:
            logger.error("Failed to load PubMedCLIP: %s", e)
            raise RuntimeError(f"Failed to load PubMedCLIP model: {e}") from e

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._loaded

    @property
    def device(self) -> torch.device:
        """Get the current compute device."""
        if self._device is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._device

    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """Generate a normalized image embedding.

        Args:
            image: PIL Image in RGB mode.

        Returns:
            Normalized image embedding tensor of shape (1, embed_dim).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            image_features = self._model.get_image_features(**inputs)
            if hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
                image_features = image_features.pooler_output
            elif hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
                image_features = image_features.image_embeds
            elif not isinstance(image_features, torch.Tensor):
                image_features = image_features[1] if len(image_features) > 1 else image_features[0]

        # L2-normalize for cosine similarity
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_text_embeddings(self, texts: list[str]) -> torch.Tensor:
        """Generate normalized text embeddings for a list of prompts.

        Args:
            texts: List of text prompts.

        Returns:
            Normalized text embedding tensor of shape (n_texts, embed_dim).
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        inputs = self._processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self._model.get_text_features(**inputs)
            if hasattr(text_features, "pooler_output") and text_features.pooler_output is not None:
                text_features = text_features.pooler_output
            elif hasattr(text_features, "text_embeds") and text_features.text_embeds is not None:
                text_features = text_features.text_embeds
            elif not isinstance(text_features, torch.Tensor):
                text_features = text_features[1] if len(text_features) > 1 else text_features[0]

        # L2-normalize for cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    def compute_similarity(
        self, image_embedding: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute cosine similarity between image and text embeddings.

        Args:
            image_embedding: Normalized image embedding (1, embed_dim).
            text_embeddings: Normalized text embeddings (n, embed_dim).

        Returns:
            Similarity scores tensor of shape (n,).
        """
        # Cosine similarity (dot product of normalized vectors)
        similarity = (image_embedding @ text_embeddings.T).squeeze(0)
        return similarity

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata."""
        return {
            "name": "PubMedCLIP",
            "checkpoint": settings.MODEL_NAME,
            "device": str(self._device) if self._device else "not loaded",
            "loaded": self._loaded,
        }


# Singleton instance — loaded once, reused across requests
pubmedclip_model = PubMedCLIPModel()
