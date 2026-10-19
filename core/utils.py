"""Utility and helper functions for image conversion, validation, and demo data."""

import io
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image


def sanitize_image(
    image: np.ndarray, max_dim: Optional[int] = None
) -> np.ndarray:
    """Normalize and sanitize an image into standard 3-channel RGB uint8 format.

    Handles:
    - 4-channel RGBA images (compositing with white background).
    - 1-channel Grayscale images (2D or 3D H x W x 1 converted to RGB).
    - Float data types normalized and scaled to [0, 255] uint8.
    - Proportional downscaling if image exceeds max_dim.

    Args:
        image: Input NumPy array image.
        max_dim: Optional maximum bounding dimension (width or height).

    Returns:
        np.ndarray: Sanitized RGB image (H x W x 3) in uint8.

    Raises:
        ValueError: If input is not a valid 2D/3D image array or is empty.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a NumPy array.")
    if image.size == 0:
        raise ValueError("Image array cannot be empty.")
    if image.ndim not in (2, 3):
        raise ValueError(f"Image must have 2 or 3 dimensions, got {image.ndim}.")

    # Handle float image types
    if np.issubdtype(image.dtype, np.floating):
        if np.max(image) <= 1.0 and np.min(image) >= 0.0:
            image = (image * 255.0).clip(0, 255).astype(np.uint8)
        else:
            image = image.clip(0, 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)

    # Handle channel dimensions
    if image.ndim == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image_rgb = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA: Alpha composite over white background
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        white_bg = np.ones_like(rgb) * 255.0
        blended = rgb * alpha + white_bg * (1.0 - alpha)
        image_rgb = blended.clip(0, 255).astype(np.uint8)
    elif image.shape[2] == 3:
        image_rgb = image
    else:
        raise ValueError(f"Unsupported channel count: {image.shape[2]}.")

    # Proportional downscaling if exceeding max_dim
    if max_dim is not None and max_dim > 0:
        h, w = image_rgb.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image_rgb


def validate_odd_kernel(kernel_size: int) -> int:
    """Validate that a kernel size is a positive odd integer >= 3.

    Args:
        kernel_size: Kernel dimension to validate.

    Returns:
        int: The validated kernel size.

    Raises:
        ValueError: If kernel_size is not an odd integer >= 3.
    """
    if kernel_size < 3:
        raise ValueError(f"Kernel size must be at least 3, got {kernel_size}.")
    if kernel_size % 2 == 0:
        raise ValueError(
            f"Kernel size must be an odd integer, got {kernel_size}. Examples: 3, 5, 7."
        )
    return kernel_size


def load_image_from_bytes(
    file_bytes: bytes, max_dim: Optional[int] = None
) -> np.ndarray:
    """Decode raw image bytes into a sanitized RGB uint8 NumPy array.

    Args:
        file_bytes: Binary content of uploaded image.
        max_dim: Optional maximum bounding dimension.

    Returns:
        np.ndarray: Decoded and sanitized RGB image array (H x W x 3) in uint8.

    Raises:
        ValueError: If bytes cannot be decoded into a valid image.
    """
    if not file_bytes:
        raise ValueError("Image bytes buffer cannot be empty.")

    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    # Read with UNCHANGED to preserve alpha channel if present
    image_raw = cv2.imdecode(file_array, cv2.IMREAD_UNCHANGED)

    if image_raw is None:
        # Fallback to PIL
        try:
            pil_image = Image.open(io.BytesIO(file_bytes))
            raw_array = np.array(pil_image)
            return sanitize_image(raw_array, max_dim=max_dim)
        except Exception as exc:
            raise ValueError("Failed to decode uploaded image data.") from exc

    # Convert BGR(A) to RGB(A)
    if image_raw.ndim == 3:
        if image_raw.shape[2] == 3:
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        elif image_raw.shape[2] == 4:
            image_raw = cv2.cvtColor(image_raw, cv2.COLOR_BGRA2RGBA)

    return sanitize_image(image_raw, max_dim=max_dim)


def array_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode an image NumPy array into PNG formatted bytes for download.

    Args:
        image: NumPy array in uint8 (H x W, or H x W x 3).

    Returns:
        bytes: Encoded PNG binary data.
    """
    if image.ndim == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", bgr)
    else:
        success, buffer = cv2.imencode(".png", image)

    if not success:
        raise ValueError("Failed to encode array to PNG.")
    return buffer.tobytes()


def generate_sample_color_image(width: int = 512, height: int = 512) -> np.ndarray:
    """Generate a rich synthetic test image containing geometries, gradients, and textures.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        np.ndarray: Synthetic RGB test image in uint8.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Smooth color gradient background
    x = np.linspace(30, 200, width, dtype=np.float32)
    y = np.linspace(30, 200, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    image[:, :, 0] = xx.astype(np.uint8)
    image[:, :, 1] = yy.astype(np.uint8)
    image[:, :, 2] = (((xx + yy) / 2.0)).astype(np.uint8)

    # Crisp geometric shapes for edge detection & sharpening testing
    # Central rectangle
    cv2.rectangle(
        image,
        (width // 8, height // 8),
        (3 * width // 8, 3 * height // 8),
        (240, 240, 60),
        -1,
    )

    # Concentric circles
    center = (3 * width // 4, height // 3)
    cv2.circle(image, center, width // 6, (40, 180, 240), -1)
    cv2.circle(image, center, width // 10, (230, 80, 140), -1)
    cv2.circle(image, center, width // 18, (255, 255, 255), -1)

    # Triangle with sharp edges
    pts = np.array(
        [
            [width // 4, 3 * height // 4],
            [width // 2, 9 * height // 10],
            [width // 8, 9 * height // 10],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [pts], (80, 220, 120))

    # Fine high-frequency pattern lines for sharpening
    for i in range(10):
        cv2.line(
            image,
            (width // 2 + i * 16, height // 2),
            (width // 2 + i * 16, 9 * height // 10),
            (255, 255, 255),
            2,
        )

    # Diagonal test line
    cv2.line(
        image,
        (width // 2, height // 2),
        (width - 40, height - 40),
        (255, 120, 40),
        3,
    )

    return image
