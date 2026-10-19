"""Edge detection operators: Sobel, Prewitt, and Canny."""

from typing import Union
import cv2
import numpy as np


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert input image to single-channel grayscale if RGB/BGR."""
    if image.ndim == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        elif image.shape[2] == 1:
            return image[:, :, 0]
    return image


def apply_sobel_edge(
    image: np.ndarray, ksize: int = 3, normalize: bool = True
) -> np.ndarray:
    """Detect edges using the Sobel operator.

    Computes horizontal (Gx) and vertical (Gy) gradients and calculates
    magnitude: M = sqrt(Gx^2 + Gy^2).

    Args:
        image: Input image (RGB or Grayscale) of type uint8.
        ksize: Aperture size for the Sobel kernel (must be 1, 3, 5, or 7). Default: 3.
        normalize: If True, clips gradient magnitude to [0, 255] and casts to uint8.

    Returns:
        np.ndarray: Edge magnitude map (H x W) of type uint8.

    Raises:
        ValueError: If ksize is not an odd positive integer in {1, 3, 5, 7}.
    """
    if ksize not in (1, 3, 5, 7):
        raise ValueError(f"ksize must be in (1, 3, 5, 7), got {ksize}.")

    gray = _to_grayscale(image)
    gray_float = gray.astype(np.float64)

    grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=ksize)

    magnitude = np.hypot(grad_x, grad_y)

    if normalize:
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return magnitude


def apply_prewitt_edge(
    image: np.ndarray, normalize: bool = True
) -> np.ndarray:
    """Detect edges using the Prewitt operator.

    Uses 3x3 horizontal and vertical difference kernels:
    Kx = [[-1, 0, 1],
          [-1, 0, 1],
          [-1, 0, 1]]
    Ky = [[-1, -1, -1],
          [ 0,  0,  0],
          [ 1,  1,  1]]
    Magnitude: M = sqrt(Gx^2 + Gy^2).

    Args:
        image: Input image (RGB or Grayscale) of type uint8.
        normalize: If True, clips gradient magnitude to [0, 255] and casts to uint8.

    Returns:
        np.ndarray: Edge magnitude map (H x W) of type uint8.
    """
    gray = _to_grayscale(image)
    gray_float = gray.astype(np.float64)

    kernel_x = np.array(
        [[-1.0, 0.0, 1.0],
         [-1.0, 0.0, 1.0],
         [-1.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    kernel_y = np.array(
        [[-1.0, -1.0, -1.0],
         [ 0.0,  0.0,  0.0],
         [ 1.0,  1.0,  1.0]],
        dtype=np.float64,
    )

    grad_x = cv2.filter2D(gray_float, -1, kernel_x)
    grad_y = cv2.filter2D(gray_float, -1, kernel_y)

    magnitude = np.hypot(grad_x, grad_y)

    if normalize:
        magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)

    return magnitude


def apply_canny_edge(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    l2_gradient: bool = False,
) -> np.ndarray:
    """Detect edges using the Canny edge detection algorithm.

    Multi-stage pipeline: Gaussian smoothing -> Gradient calculation ->
    Non-maximum suppression -> Hysteresis thresholding.

    Args:
        image: Input image (RGB or Grayscale) of type uint8.
        low_threshold: Lower hysteresis threshold (default: 50).
        high_threshold: Upper hysteresis threshold (default: 150).
        aperture_size: Aperture size for the Sobel operator (default: 3).
        l2_gradient: Whether to use L2 norm for gradient magnitude (default: False).

    Returns:
        np.ndarray: Binary edge map (H x W) of type uint8 with values in {0, 255}.

    Raises:
        ValueError: If low_threshold < 0 or high_threshold < low_threshold.
    """
    if low_threshold < 0:
        raise ValueError("low_threshold must be non-negative.")
    if high_threshold < low_threshold:
        raise ValueError(
            f"high_threshold ({high_threshold}) cannot be less than low_threshold ({low_threshold})."
        )

    gray = _to_grayscale(image)
    edges = cv2.Canny(
        gray,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=aperture_size,
        L2gradient=l2_gradient,
    )
    return edges
