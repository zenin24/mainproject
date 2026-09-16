"""
Tests for image preprocessing module.
"""

from io import BytesIO
import pytest
from PIL import Image

from app.preprocessing.image import (
    get_image_info,
    load_image,
    validate_image_file,
)


def create_dummy_image(mode: str = "RGB", size: tuple[int, int] = (100, 100), fmt: str = "PNG") -> bytes:
    """Helper to generate a dummy image byte stream."""
    img = Image.new(mode, size, color="gray")
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_validate_image_file_valid():
    """Test validation with valid extensions and MIME types."""
    validate_image_file("chest_xray.png", "image/png")
    validate_image_file("chest_xray.jpg", "image/jpeg")
    validate_image_file("chest_xray.TIFF", "image/tiff")


def test_validate_image_file_invalid_ext():
    """Test validation rejects invalid extensions."""
    with pytest.raises(ValueError, match="Unsupported file extension"):
        validate_image_file("document.pdf", "application/pdf")


def test_validate_image_file_invalid_mime():
    """Test validation rejects invalid MIME types."""
    with pytest.raises(ValueError, match="Unsupported content type"):
        validate_image_file("chest_xray.png", "application/json")


def test_load_image_rgb():
    """Test loading RGB image."""
    img_bytes = create_dummy_image(mode="RGB")
    pil_img = load_image(img_bytes)
    assert pil_img.mode == "RGB"
    assert pil_img.size == (100, 100)


def test_load_image_grayscale_conversion():
    """Test that grayscale image is converted to RGB."""
    img_bytes = create_dummy_image(mode="L")
    pil_img = load_image(img_bytes)
    assert pil_img.mode == "RGB"


def test_load_image_invalid_bytes():
    """Test loading corrupted bytes raises ValueError."""
    with pytest.raises(ValueError, match="Cannot open image"):
        load_image(b"not an image")


def test_get_image_info():
    """Test metadata extraction."""
    img_bytes = create_dummy_image(mode="RGB", size=(256, 128))
    pil_img = load_image(img_bytes)
    info = get_image_info(pil_img)
    assert info["width"] == 256
    assert info["height"] == 128
    assert info["mode"] == "RGB"
