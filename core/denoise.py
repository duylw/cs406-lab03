"""Spatial domain smoothing and denoising filters."""

import cv2
import numpy as np


def _validate_odd_kernel_size(kernel_size: int) -> None:
    """Validate that kernel size is a positive odd integer.

    Args:
        kernel_size: Kernel dimension to validate.

    Raises:
        ValueError: If kernel_size is non-positive or even.
    """
    if kernel_size <= 0:
        raise ValueError(f"Kernel size must be positive, got {kernel_size}.")
    if kernel_size % 2 == 0:
        raise ValueError(
            f"Kernel size must be an odd integer, got {kernel_size}. Example valid sizes: 3, 5, 7."
        )


def apply_mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Mean (Box/Averaging) filter to smooth image or reduce Gaussian noise.

    Args:
        image: Input image array of type uint8 (Grayscale or RGB).
        kernel_size: Size of the square sliding window (must be positive odd integer).

    Returns:
        np.ndarray: Denoised image array of type uint8.
    """
    _validate_odd_kernel_size(kernel_size)
    if kernel_size == 1:
        return image.copy()

    denoised = cv2.blur(image, (kernel_size, kernel_size))
    return denoised


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Apply Median filter to eliminate impulse (Salt & Pepper) noise.

    Args:
        image: Input image array of type uint8 (Grayscale or RGB).
        kernel_size: Aperture linear size (must be positive odd integer).

    Returns:
        np.ndarray: Denoised image array of type uint8.
    """
    _validate_odd_kernel_size(kernel_size)
    if kernel_size == 1:
        return image.copy()

    denoised = cv2.medianBlur(image, kernel_size)
    return denoised
