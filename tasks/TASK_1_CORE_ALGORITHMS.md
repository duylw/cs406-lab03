# TASK 1: CORE IMAGE PROCESSING ALGORITHMS

## Metadata
- Task ID: TASK-01
- Title: Core Image Processing Algorithms Implementation
- Status: [DONE]
- Priority: High
- Tracking File: [TRACKING.md](file:///d:/UIT/XLA/lab/lab03/TRACKING.md)
- Reference: [PROBLEM_STATEMENT.md](file:///d:/UIT/XLA/lab/lab03/PROBLEM_STATEMENT.md)

---

## 1. What To Do (Scope & Requirements)
Implement a decoupled, standalone Python library that performs all digital image processing operations specified in the problem statement:

1. Noise Generation:
   - Gaussian noise injection with configurable mean and standard deviation.
   - Salt & Pepper noise injection with configurable noise density ratio.

2. Denoising and Smoothing:
   - Mean Filter (Averaging / Box Filter) with configurable odd kernel size (e.g., 3x3, 5x5, 7x7).
   - Median Filter with configurable odd kernel size (e.g., 3x3, 5x5, 7x7).

3. Image Sharpening:
   - Laplacian-based sharpening operator.
   - Unsharp Masking / High-boost filtering with adjustable blend weight (alpha/strength).

4. Edge Detection:
   - Sobel operator (gradient magnitude combining horizontal Gx and vertical Gy derivatives).
   - Prewitt operator (using 3x3 horizontal and vertical custom convolution kernels).
   - Canny Edge Detection with customizable lower and upper hysteresis thresholds.

5. Image Quality Evaluation:
   - Mean Squared Error (MSE) calculation.
   - Peak Signal-to-Noise Ratio (PSNR) calculation for evaluating denoising effectiveness against ground truth.

---

## 2. How To Do It (Implementation Strategy)
- Language & Environment: Python 3 managed with `uv`.
- Dependencies Setup:
  ```bash
  uv init
  uv add numpy opencv-python pytest
  ```
- Modular Directory Structure:
  ```
  core/
  ├── __init__.py
  ├── noise.py        # Noise synthesis functions
  ├── denoise.py      # Mean and Median smoothing filters
  ├── sharpen.py      # Laplacian and Unsharp Mask algorithms
  ├── edge.py         # Sobel, Prewitt, and Canny edge detectors
  └── metrics.py      # PSNR and MSE calculation utilities
  ```

- Algorithmic Details:
  - Noise injection:
    - Gaussian: Sample from `np.random.normal`, clip values to [0, 255], cast to `np.uint8`.
    - Salt & Pepper: Randomly assign coordinates to 0 (pepper) or 255 (salt) based on density parameter.
  - Smoothing filters:
    - Mean filter: Apply `cv2.blur` or `cv2.boxFilter`.
    - Median filter: Apply `cv2.medianBlur` ensuring odd kernel size.
  - Sharpening:
    - Laplacian: Compute `cv2.Laplacian` with `CV_64F`, subtract/add weighted gradient to the original image, clip to [0, 255].
    - Unsharp Mask: Compute blurred image via Gaussian blur, determine mask `mask = original - blurred`, calculate `sharpened = original + strength * mask`.
  - Edge detection:
    - Prewitt: Define horizontal kernel `Kx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]` and vertical kernel `Ky = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]`. Convolve using `cv2.filter2D`, then compute magnitude `sqrt(Gx^2 + Gy^2)`.
    - Sobel: Use `cv2.Sobel` for Gx and Gy with `CV_64F`, compute gradient magnitude, normalize to `np.uint8`.
    - Canny: Convert input to grayscale (if RGB), apply Gaussian smoothing, run `cv2.Canny(image, low_threshold, high_threshold)`.

---

## 3. How To Test (Verification Protocol)
- Execution Command: `uv run pytest tests/test_core.py`
- Test Script Location: `tests/test_core.py`
- Test Scenarios:
  1. Data Contract & Dimension Assertions:
     - Input: 3-channel RGB image (H x W x 3) and 1-channel Grayscale image (H x W).
     - Expected Output: Denoised and sharpened images must retain identical dimensions and dtype (`np.uint8`). Edge detection maps must return valid 2D or 3D arrays with pixel values bounded within [0, 255].
  2. Edge Detection Boundary Test:
     - Generate a synthetic binary step image (left half black [0], right half white [255]).
     - Apply Sobel, Prewitt, and Canny detectors.
     - Verify that non-zero response is localized precisely along the vertical dividing line.
  3. Denoising Effectiveness Benchmark:
     - Corrupt a reference image with Salt & Pepper noise (e.g., 5% density).
     - Apply Median filter and Mean filter with kernel size 3x3.
     - Measure PSNR against the clean original image.
     - Assertion: PSNR(Median) > PSNR(Mean) by at least 3.0 dB on impulse noise.
  4. Parameter Boundary Test:
     - Ensure functions raise `ValueError` when supplied with even kernel sizes or negative threshold values.

---

## 4. Acceptance Criteria
- [x] Python environment and packages initialized via `uv`.
- [x] All functions in `core/` have comprehensive type hints and docstrings.
- [x] No module imports UI or presentation dependencies.
- [x] Automated test suite via `uv run pytest tests/test_core.py` executes with all tests passing ([PASS]).
