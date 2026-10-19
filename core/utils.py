"""Utility and helper functions for image conversion, validation, and demo data."""

import io
from typing import Tuple
import cv2
import numpy as np
from PIL import Image


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an RGB uint8 NumPy array.

    Args:
        file_bytes: Binary content of uploaded image.

    Returns:
        np.ndarray: Decoded RGB image array (H x W x 3) in uint8.

    Raises:
        ValueError: If bytes cannot be decoded into a valid image.
    """
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if image_bgr is None:
        # Fallback to PIL
        try:
            pil_image = Image.open(io.BytesIO(file_bytes))
            pil_image = pil_image.convert("RGB")
            return np.array(pil_image, dtype=np.uint8)
        except Exception as exc:
            raise ValueError("Failed to decode uploaded image data.") from exc

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


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
