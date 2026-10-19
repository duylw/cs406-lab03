"""Quantitative image quality metrics."""

from typing import Tuple
import numpy as np


def calculate_mse(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE) between two images.

    Args:
        image_true: Ground truth image array.
        image_test: Image array to be evaluated.

    Returns:
        float: Mean squared error value.

    Raises:
        ValueError: If image shapes or dimensions do not match.
    """
    if image_true.shape != image_test.shape:
        raise ValueError(
            f"Shape mismatch: image_true has shape {image_true.shape}, "
            f"image_test has shape {image_test.shape}."
        )

    diff = image_true.astype(np.float64) - image_test.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    return mse


def calculate_psnr(
    image_true: np.ndarray, image_test: np.ndarray, max_val: float = 255.0
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        image_true: Ground truth image array.
        image_test: Image array to be evaluated.
        max_val: Maximum possible pixel intensity value (default: 255.0).

    Returns:
        float: PSNR value in dB, or float('inf') if MSE is zero.

    Raises:
        ValueError: If max_val is non-positive or shapes mismatch.
    """
    if max_val <= 0:
        raise ValueError("max_val must be greater than zero.")

    mse = calculate_mse(image_true, image_test)
    if mse == 0.0:
        return float("inf")

    psnr = 10.0 * np.log10((max_val ** 2) / mse)
    return float(psnr)
