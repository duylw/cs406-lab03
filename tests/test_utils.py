"""Unit tests for UI and image utility functions."""

import numpy as np
import pytest
from core.utils import (
    load_image_from_bytes,
    array_to_png_bytes,
    generate_sample_color_image,
)


def test_generate_sample_color_image() -> None:
    img = generate_sample_color_image(256, 256)
    assert img.shape == (256, 256, 3)
    assert img.dtype == np.uint8
    assert np.min(img) >= 0
    assert np.max(img) <= 255


def test_png_bytes_roundtrip() -> None:
    original = generate_sample_color_image(128, 128)
    png_bytes = array_to_png_bytes(original)
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0

    decoded = load_image_from_bytes(png_bytes)
    assert decoded.shape == original.shape
    assert decoded.dtype == original.dtype
    # PNG compression is lossless for RGB arrays
    np.testing.assert_array_equal(decoded, original)


def test_load_image_invalid_bytes() -> None:
    invalid_bytes = b"not_an_image_file"
    with pytest.raises(ValueError, match="Failed to decode uploaded image data"):
        load_image_from_bytes(invalid_bytes)
