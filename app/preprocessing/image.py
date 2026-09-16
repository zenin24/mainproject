"""
Image preprocessing for chest X-ray analysis.

Handles image loading, validation, format conversion, and preparation
for PubMedCLIP processing.
"""

import logging
from io import BytesIO
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

# Supported image MIME types
SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/bmp",
    "image/tiff",
}

# Supported file extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def validate_image_file(filename: str, content_type: str | None) -> None:
    """Validate that the uploaded file is a supported image format.

    Args:
        filename: Original filename from the upload.
        content_type: MIME type of the uploaded file.

    Raises:
        ValueError: If the file format is not supported.
    """
    import os

    ext = os.path.splitext(filename)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if content_type and content_type not in SUPPORTED_CONTENT_TYPES:
        # Some clients send 'application/octet-stream', allow it
        if content_type != "application/octet-stream":
            raise ValueError(
                f"Unsupported content type '{content_type}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_CONTENT_TYPES))}"
            )


def load_image(image_bytes: bytes) -> Image.Image:
    """Load an image from raw bytes and convert to RGB.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        PIL Image in RGB mode.

    Raises:
        ValueError: If the image cannot be opened or is corrupt.
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()  # Check for corruption
        # Re-open after verify (verify can close the file)
        image = Image.open(BytesIO(image_bytes))
    except Exception as e:
        raise ValueError(f"Cannot open image: {e}") from e

    # Convert to RGB (handles grayscale, RGBA, palette modes)
    if image.mode != "RGB":
        logger.info("Converting image from %s to RGB", image.mode)
        image = image.convert("RGB")

    return image


def get_image_info(image: Image.Image) -> dict[str, Any]:
    """Extract basic image information.

    Args:
        image: PIL Image object.

    Returns:
        Dictionary with image metadata.
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format or "unknown",
    }
