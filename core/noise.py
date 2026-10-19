"""Noise synthesis generators for testing image smoothing algorithms."""

from typing import Optional
import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    mean: float = 0.0,
    sigma: float = 25.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add additive white Gaussian noise to an image.

    Args:
        image: Input image array of type uint8 (Grayscale or RGB).
        mean: Mean of the Gaussian distribution (default: 0.0).
        sigma: Standard deviation of the Gaussian distribution (default: 25.0).
        seed: Optional random seed for reproducible noise generation.

    Returns:
        np.ndarray: Noisy image array in uint8 with values clipped to [0, 255].

    Raises:
        ValueError: If sigma is negative.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=sigma, size=image.shape)

    noisy_image = image.astype(np.float64) + noise
    noisy_clipped = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_clipped


def add_salt_and_pepper_noise(
    image: np.ndarray,
    amount: float = 0.05,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add impulse (salt and pepper) noise to an image.

    Args:
        image: Input image array of type uint8 (Grayscale or RGB).
        amount: Overall noise ratio in range [0.0, 1.0] (default: 0.05).
        salt_vs_pepper: Proportion of salt (white) vs pepper (black) pixels in [0.0, 1.0] (default: 0.5).
        seed: Optional random seed for reproducible noise generation.

    Returns:
        np.ndarray: Noisy image array in uint8.

    Raises:
        ValueError: If amount or salt_vs_pepper is outside [0.0, 1.0].
    """
    if not (0.0 <= amount <= 1.0):
        raise ValueError("amount must be within [0.0, 1.0].")
    if not (0.0 <= salt_vs_pepper <= 1.0):
        raise ValueError("salt_vs_pepper must be within [0.0, 1.0].")

    if amount == 0.0:
        return image.copy()

    rng = np.random.default_rng(seed)
    noisy_image = image.copy()

    spatial_shape = image.shape[:2]
    rand_matrix = rng.uniform(0.0, 1.0, size=spatial_shape)

    pepper_threshold = amount * (1.0 - salt_vs_pepper)
    salt_threshold = amount

    pepper_mask = rand_matrix < pepper_threshold
    salt_mask = (rand_matrix >= pepper_threshold) & (rand_matrix < salt_threshold)

    if image.ndim == 3:
        noisy_image[pepper_mask] = [0, 0, 0]
        noisy_image[salt_mask] = [255, 255, 255]
    else:
        noisy_image[pepper_mask] = 0
        noisy_image[salt_mask] = 255

    return noisy_image
