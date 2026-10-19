"""Spatial domain image sharpening algorithms."""

import cv2
import numpy as np


def apply_laplacian_sharpen(
    image: np.ndarray, alpha: float = 1.0
) -> np.ndarray:
    """Enhance fine details and edges using the Laplacian operator.

    Using the standard center-negative 3x3 Laplacian mask:
    [[ 0,  1,  0],
     [ 1, -4,  1],
     [ 0,  1,  0]]
    The enhanced image is computed as:
    g(x, y) = f(x, y) - alpha * Laplacian(f(x, y))

    Args:
        image: Input image array of type uint8 (Grayscale or RGB).
        alpha: Sharpening scaling factor (default: 1.0).

    Returns:
        np.ndarray: Sharpened image array of type uint8.

    Raises:
        ValueError: If alpha is negative.
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")

    if alpha == 0.0:
        return image.copy()

    image_float = image.astype(np.float64)

    # Standard 3x3 Laplacian kernel with center negative
    laplacian_kernel = np.array(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )

    laplacian = cv2.filter2D(image_float, -1, laplacian_kernel)
    sharpened = image_float - alpha * laplacian
    sharpened_clipped = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened_clipped


def apply_unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 1.5,
    threshold: int = 0,
) -> np.ndarray:
    """Enhance image details using the Unsharp Masking (High-boost filtering) technique.

    The sharpened image is computed as:
    mask = original - blurred
    sharpened = original + strength * mask

    Args:
        image: Input image array of type uint8 (Grayscale or RGB).
        sigma: Standard deviation of Gaussian blur used for unsharp mask (default: 1.0).
        strength: Amount / multiplier for high-frequency details (default: 1.5).
        threshold: Minimum pixel difference required to apply enhancement (default: 0).

    Returns:
        np.ndarray: Sharpened image array of type uint8.

    Raises:
        ValueError: If sigma <= 0 or strength < 0 or threshold < 0.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if strength < 0:
        raise ValueError(f"strength must be non-negative, got {strength}.")
    if threshold < 0:
        raise ValueError(f"threshold must be non-negative, got {threshold}.")

    if strength == 0.0:
        return image.copy()

    image_float = image.astype(np.float64)
    blurred = cv2.GaussianBlur(image_float, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = image_float - blurred

    if threshold > 0:
        low_contrast_mask = np.abs(mask) < threshold
        mask[low_contrast_mask] = 0.0

    sharpened = image_float + strength * mask
    sharpened_clipped = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened_clipped
