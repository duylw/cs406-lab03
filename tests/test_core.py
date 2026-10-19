"""Comprehensive unit test suite for core image processing algorithms."""

import pytest
import numpy as np
from core import (
    add_gaussian_noise,
    add_salt_and_pepper_noise,
    apply_mean_filter,
    apply_median_filter,
    apply_laplacian_sharpen,
    apply_unsharp_mask,
    apply_sobel_edge,
    apply_prewitt_edge,
    apply_canny_edge,
    calculate_mse,
    calculate_psnr,
)


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Fixture providing a synthetic 64x64 RGB image with gradient pattern."""
    h, w = 64, 64
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    r = xx
    g = yy
    b = ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)
    return np.stack([r, g, b], axis=-1)


@pytest.fixture
def sample_gray_image() -> np.ndarray:
    """Fixture providing a synthetic 64x64 Grayscale image."""
    h, w = 64, 64
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    return ((xx.astype(np.uint16) + yy.astype(np.uint16)) // 2).astype(np.uint8)


@pytest.fixture
def step_edge_image() -> np.ndarray:
    """Fixture providing a 64x64 binary vertical step-edge pattern."""
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 255
    return img


# ============================================================================
# 1. Metrics Tests
# ============================================================================

def test_metrics_identical_images(sample_rgb_image: np.ndarray) -> None:
    mse = calculate_mse(sample_rgb_image, sample_rgb_image)
    psnr = calculate_psnr(sample_rgb_image, sample_rgb_image)
    assert mse == 0.0
    assert psnr == float("inf")


def test_metrics_shape_mismatch() -> None:
    img1 = np.zeros((32, 32), dtype=np.uint8)
    img2 = np.zeros((64, 64), dtype=np.uint8)
    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_mse(img1, img2)


def test_metrics_known_difference() -> None:
    img1 = np.full((10, 10), 100, dtype=np.uint8)
    img2 = np.full((10, 10), 110, dtype=np.uint8)
    mse = calculate_mse(img1, img2)
    assert mse == 100.0  # (110 - 100)^2 = 100
    expected_psnr = 10.0 * np.log10((255.0 ** 2) / 100.0)
    assert pytest.approx(calculate_psnr(img1, img2), 0.01) == expected_psnr


# ============================================================================
# 2. Noise Generators Tests
# ============================================================================

def test_gaussian_noise_properties(sample_rgb_image: np.ndarray) -> None:
    noisy = add_gaussian_noise(sample_rgb_image, mean=0.0, sigma=20.0, seed=42)
    assert noisy.shape == sample_rgb_image.shape
    assert noisy.dtype == np.uint8
    assert not np.array_equal(noisy, sample_rgb_image)

    # Re-running with same seed produces identical results
    noisy_repeat = add_gaussian_noise(sample_rgb_image, mean=0.0, sigma=20.0, seed=42)
    np.testing.assert_array_equal(noisy, noisy_repeat)


def test_gaussian_noise_invalid_sigma(sample_rgb_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        add_gaussian_noise(sample_rgb_image, sigma=-1.0)


def test_salt_and_pepper_noise_properties(sample_rgb_image: np.ndarray) -> None:
    noisy = add_salt_and_pepper_noise(sample_rgb_image, amount=0.1, seed=42)
    assert noisy.shape == sample_rgb_image.shape
    assert noisy.dtype == np.uint8

    # Check presence of pure white and pure black pixels
    has_salt = np.any(np.all(noisy == [255, 255, 255], axis=-1))
    has_pepper = np.any(np.all(noisy == [0, 0, 0], axis=-1))
    assert has_salt and has_pepper


def test_salt_and_pepper_zero_amount(sample_rgb_image: np.ndarray) -> None:
    clean = add_salt_and_pepper_noise(sample_rgb_image, amount=0.0)
    np.testing.assert_array_equal(clean, sample_rgb_image)


def test_salt_and_pepper_invalid_bounds(sample_rgb_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="amount must be within"):
        add_salt_and_pepper_noise(sample_rgb_image, amount=1.5)
    with pytest.raises(ValueError, match="salt_vs_pepper must be within"):
        add_salt_and_pepper_noise(sample_rgb_image, salt_vs_pepper=-0.1)


# ============================================================================
# 3. Smoothing / Denoising Filters Tests
# ============================================================================

def test_mean_filter_rgb(sample_rgb_image: np.ndarray) -> None:
    filtered = apply_mean_filter(sample_rgb_image, kernel_size=3)
    assert filtered.shape == sample_rgb_image.shape
    assert filtered.dtype == np.uint8


def test_median_filter_rgb(sample_rgb_image: np.ndarray) -> None:
    filtered = apply_median_filter(sample_rgb_image, kernel_size=3)
    assert filtered.shape == sample_rgb_image.shape
    assert filtered.dtype == np.uint8


def test_denoise_invalid_kernel_size(sample_rgb_image: np.ndarray) -> None:
    # Even kernel sizes must be rejected
    with pytest.raises(ValueError, match="must be an odd integer"):
        apply_mean_filter(sample_rgb_image, kernel_size=4)
    with pytest.raises(ValueError, match="must be an odd integer"):
        apply_median_filter(sample_rgb_image, kernel_size=2)
    # Non-positive sizes must be rejected
    with pytest.raises(ValueError, match="must be positive"):
        apply_mean_filter(sample_rgb_image, kernel_size=0)


def test_median_superiority_on_salt_and_pepper(sample_rgb_image: np.ndarray) -> None:
    """Validate that Median filter significantly outperforms Mean filter on impulse noise."""
    noisy = add_salt_and_pepper_noise(sample_rgb_image, amount=0.08, seed=123)

    mean_denoised = apply_mean_filter(noisy, kernel_size=3)
    median_denoised = apply_median_filter(noisy, kernel_size=3)

    psnr_noisy = calculate_psnr(sample_rgb_image, noisy)
    psnr_mean = calculate_psnr(sample_rgb_image, mean_denoised)
    psnr_median = calculate_psnr(sample_rgb_image, median_denoised)

    # Median filter should restore Salt & Pepper noise with much higher PSNR than Mean filter
    assert psnr_median > psnr_mean + 3.0, (
        f"Expected Median PSNR ({psnr_median:.2f}) to beat Mean PSNR ({psnr_mean:.2f}) by >= 3dB"
    )
    assert psnr_median > psnr_noisy


# ============================================================================
# 4. Sharpening Algorithms Tests
# ============================================================================

def test_laplacian_sharpen_properties(sample_rgb_image: np.ndarray) -> None:
    sharpened = apply_laplacian_sharpen(sample_rgb_image, alpha=1.0)
    assert sharpened.shape == sample_rgb_image.shape
    assert sharpened.dtype == np.uint8
    assert not np.array_equal(sharpened, sample_rgb_image)


def test_laplacian_sharpen_zero_alpha(sample_rgb_image: np.ndarray) -> None:
    result = apply_laplacian_sharpen(sample_rgb_image, alpha=0.0)
    np.testing.assert_array_equal(result, sample_rgb_image)


def test_laplacian_invalid_alpha(sample_rgb_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="alpha must be non-negative"):
        apply_laplacian_sharpen(sample_rgb_image, alpha=-0.5)


def test_unsharp_mask_properties(sample_rgb_image: np.ndarray) -> None:
    sharpened = apply_unsharp_mask(sample_rgb_image, sigma=1.0, strength=1.5)
    assert sharpened.shape == sample_rgb_image.shape
    assert sharpened.dtype == np.uint8
    assert not np.array_equal(sharpened, sample_rgb_image)


def test_unsharp_mask_zero_strength(sample_rgb_image: np.ndarray) -> None:
    result = apply_unsharp_mask(sample_rgb_image, strength=0.0)
    np.testing.assert_array_equal(result, sample_rgb_image)


def test_unsharp_mask_invalid_params(sample_rgb_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sigma must be positive"):
        apply_unsharp_mask(sample_rgb_image, sigma=0.0)
    with pytest.raises(ValueError, match="strength must be non-negative"):
        apply_unsharp_mask(sample_rgb_image, strength=-1.0)


# ============================================================================
# 5. Edge Detection Operators Tests
# ============================================================================

def test_edge_detection_step_edge(step_edge_image: np.ndarray) -> None:
    """Verify that Sobel, Prewitt, and Canny detect the vertical step edge at column 31-32."""
    sobel = apply_sobel_edge(step_edge_image)
    prewitt = apply_prewitt_edge(step_edge_image)
    canny = apply_canny_edge(step_edge_image, low_threshold=50, high_threshold=150)

    for name, edge_map in [("Sobel", sobel), ("Prewitt", prewitt), ("Canny", canny)]:
        assert edge_map.shape == step_edge_image.shape
        assert edge_map.dtype == np.uint8

        # Edge must be strong along the boundary column 31 or 32
        boundary_intensity = np.max(edge_map[:, 31:33])
        background_intensity = np.max(edge_map[:, :25])  # Flat zero region
        assert boundary_intensity > 200, f"{name} failed to detect step edge."
        assert background_intensity == 0, f"{name} detected spurious edges in flat background."


def test_edge_detection_rgb_input(sample_rgb_image: np.ndarray) -> None:
    """Verify edge detectors process 3-channel RGB images and return 2D uint8 maps."""
    sobel = apply_sobel_edge(sample_rgb_image)
    prewitt = apply_prewitt_edge(sample_rgb_image)
    canny = apply_canny_edge(sample_rgb_image)

    expected_2d_shape = sample_rgb_image.shape[:2]
    assert sobel.shape == expected_2d_shape
    assert prewitt.shape == expected_2d_shape
    assert canny.shape == expected_2d_shape


def test_canny_invalid_thresholds(sample_gray_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="low_threshold must be non-negative"):
        apply_canny_edge(sample_gray_image, low_threshold=-10)
    with pytest.raises(ValueError, match="high_threshold .* cannot be less than low_threshold"):
        apply_canny_edge(sample_gray_image, low_threshold=100, high_threshold=50)


def test_sobel_invalid_ksize(sample_gray_image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="ksize must be in"):
        apply_sobel_edge(sample_gray_image, ksize=4)
