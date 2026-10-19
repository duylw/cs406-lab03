"""Edge cases and robustness test suite."""

import pytest
import numpy as np
from core import (
    apply_mean_filter,
    apply_median_filter,
    apply_laplacian_sharpen,
    apply_unsharp_mask,
    apply_sobel_edge,
    apply_prewitt_edge,
    apply_canny_edge,
    calculate_mse,
    calculate_psnr,
    sanitize_image,
    validate_odd_kernel,
)


def test_rgba_image_sanitization() -> None:
    """Validate that 4-channel RGBA image is composited properly to 3-channel RGB."""
    rgba = np.zeros((64, 64, 4), dtype=np.uint8)
    rgba[:, :, 0] = 200  # Red
    rgba[:, :, 3] = 128  # 50% opacity
    rgb = sanitize_image(rgba)
    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == np.uint8
    # Blended with white background: 200*0.5 + 255*0.5 ~ 227
    assert 225 <= rgb[0, 0, 0] <= 230


def test_grayscale_2d_and_3d_sanitization() -> None:
    """Validate 2D and (H, W, 1) grayscale inputs convert to 3-channel RGB."""
    gray_2d = np.full((32, 32), 120, dtype=np.uint8)
    rgb_from_2d = sanitize_image(gray_2d)
    assert rgb_from_2d.shape == (32, 32, 3)
    assert rgb_from_2d.dtype == np.uint8
    assert np.all(rgb_from_2d == 120)

    gray_3d = np.full((32, 32, 1), 80, dtype=np.uint8)
    rgb_from_3d = sanitize_image(gray_3d)
    assert rgb_from_3d.shape == (32, 32, 3)
    assert rgb_from_3d.dtype == np.uint8
    assert np.all(rgb_from_3d == 80)


def test_float_image_sanitization() -> None:
    """Validate that float images in [0.0, 1.0] are scaled and cast to [0, 255] uint8."""
    float_img = np.random.uniform(0.0, 1.0, size=(32, 32, 3)).astype(np.float32)
    sanitized = sanitize_image(float_img)
    assert sanitized.shape == (32, 32, 3)
    assert sanitized.dtype == np.uint8
    assert np.max(sanitized) <= 255
    assert np.min(sanitized) >= 0


def test_extreme_resolutions() -> None:
    """Test tiny (16x16) and oversized images (downscaling via max_dim)."""
    tiny = np.zeros((16, 16, 3), dtype=np.uint8)
    # Processing tiny image should not crash
    mean_tiny = apply_mean_filter(tiny, kernel_size=3)
    sobel_tiny = apply_sobel_edge(tiny)
    assert mean_tiny.shape == (16, 16, 3)
    assert sobel_tiny.shape == (16, 16)

    # Oversized image with downscaling guard
    oversized = np.zeros((2400, 1800, 3), dtype=np.uint8)
    downscaled = sanitize_image(oversized, max_dim=800)
    assert max(downscaled.shape[:2]) == 800
    assert downscaled.dtype == np.uint8


def test_validate_odd_kernel() -> None:
    """Test kernel size validator enforces positive odd integers >= 3."""
    assert validate_odd_kernel(3) == 3
    assert validate_odd_kernel(5) == 5
    assert validate_odd_kernel(11) == 11

    with pytest.raises(ValueError, match="Kernel size must be an odd integer"):
        validate_odd_kernel(4)
    with pytest.raises(ValueError, match="Kernel size must be at least 3"):
        validate_odd_kernel(2)
    with pytest.raises(ValueError, match="Kernel size must be at least 3"):
        validate_odd_kernel(-1)


def test_invalid_input_exceptions() -> None:
    """Verify exceptions for empty or malformed inputs."""
    with pytest.raises(ValueError, match="Input must be a NumPy array"):
        sanitize_image("not an array")  # type: ignore

    with pytest.raises(ValueError, match="Image array cannot be empty"):
        sanitize_image(np.array([]))

    with pytest.raises(ValueError, match="Image must have 2 or 3 dimensions"):
        sanitize_image(np.zeros((10, 10, 10, 10)))
