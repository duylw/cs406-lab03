# TASK 3: ROBUSTNESS, EDGE CASE HANDLING, AND AUTOMATED VERIFICATION

## Metadata
- Task ID: TASK-03
- Title: Robustness, Edge Case Handling & Automated Verification
- Status: [TODO]
- Priority: Medium
- Tracking File: [TRACKING.md](file:///d:/UIT/XLA/lab/lab03/TRACKING.md)
- Reference: [PROBLEM_STATEMENT.md](file:///d:/UIT/XLA/lab/lab03/PROBLEM_STATEMENT.md)
- Dependencies: [TASK_1_CORE_ALGORITHMS.md](file:///d:/UIT/XLA/lab/lab03/tasks/TASK_1_CORE_ALGORITHMS.md), [TASK_2_USER_INTERFACE.md](file:///d:/UIT/XLA/lab/lab03/tasks/TASK_2_USER_INTERFACE.md)

---

## 1. What To Do (Scope & Requirements)
Harden the entire image enhancement pipeline against corrupt, non-standard, or extreme inputs, and create an automated end-to-end verification and benchmark runner:

1. Input Sanitization & Edge Case Guardrails:
   - Handle 4-channel RGBA images (strip alpha or composite with white background).
   - Handle single-channel Grayscale images uploaded into color processing pipelines.
   - Handle extreme resolutions: downscale oversized images (> 2000px) for smooth real-time preview while allowing full-resolution export.
   - Enforce parameter boundaries (reject even kernel dimensions, clamp threshold values).

2. Error Handling & User Feedback:
   - Catch file reading errors, corrupted buffers, or unsupported formats with descriptive error messages.
   - Prevent division-by-zero errors in MSE/PSNR computations for identical images (MSE = 0 -> PSNR = Infinity or capped maximum).

3. Automated CLI Verification Runner:
   - Create a standalone CLI script (`verify_pipeline.py`) that runs all algorithms on a reference test image without launching the GUI.
   - Save comparison figures (denoising grid, sharpening comparison, 3-filter edge map comparison) directly to disk for lab reporting.
   - Log execution times and PSNR metrics to console/log file.

---

## 2. How To Do It (Implementation Strategy)
- Module Structure:
  ```
  core/
  └── utils.py             # Input sanitization, normalization, boundary validators
  verify_pipeline.py       # Standalone batch execution and reporting script
  tests/
  ├── test_edge_cases.py   # Test suite for non-standard inputs
  └── test_core.py         # Algorithm tests
  ```

- Implementation Logic:
  - `core/utils.py`:
    - `sanitize_image(image_input) -> np.ndarray`: Checks shape, converts RGBA to RGB using `cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)` or alpha blending, converts 2D grayscale to 3D RGB if required.
    - `validate_odd_kernel(k: int) -> int`: Asserts $k > 1$ and $k \% 2 == 1$. If invalid, raises `ValueError` with clear guidance.
    - `safe_psnr(original, processed) -> float`: Evaluates MSE; if $MSE == 0$, returns `float('inf')` or $100.0$.
  - `verify_pipeline.py`:
    - Loads sample reference image.
    - Executes noise generation (Gaussian + Salt & Pepper).
    - Runs Mean and Median filtering, measures PSNR and execution duration.
    - Runs Laplacian and Unsharp Mask sharpening.
    - Runs Sobel, Prewitt, and Canny edge detection.
    - Assembles Matplotlib comparison figures and saves them to `output/` directory as PNG files.

---

## 3. How To Test (Verification Protocol)
- Execution Commands:
  - Unit & Edge Case Tests: `uv run pytest tests/test_edge_cases.py`
  - Automated Pipeline Runner: `uv run python verify_pipeline.py`
- Test Script Location: `tests/test_edge_cases.py` and `verify_pipeline.py`
- Test Scenarios:
  1. Alpha Channel Image Test:
     - Input: 4-channel RGBA PNG image with transparent sections.
     - Expected: Successfully converted to 3-channel RGB without crash or channel mismatch.
  2. Grayscale Input Test:
     - Input: 2D array (H x W).
     - Expected: Gracefully accepted and processed across all filtering and edge detection modules.
  3. Resolution Extremes Test:
     - Input: Very small image (16x16 pixels) and large image (4096x4096 pixels).
     - Expected: Small image processes without kernel boundary out-of-bounds; large image completes without memory allocation failure.
  4. Invalid Parameter Exception Test:
     - Pass kernel size = 4 (even number) to `apply_median_filter`.
     - Expected: `ValueError` is raised with explicit message: "Kernel size must be an odd positive integer".
  5. CLI Automated Pipeline Test:
     - Run `uv run python verify_pipeline.py`.
     - Expected: Completes with exit code 0, generates all composite comparison images in `output/`, and logs performance summary.

---

## 4. Acceptance Criteria
- [ ] Pipeline gracefully handles RGBA, Grayscale, small, and oversized images.
- [ ] Automated verification script `uv run python verify_pipeline.py` runs end-to-end and outputs all required comparison images.
- [ ] All automated tests via `uv run pytest tests/test_edge_cases.py` pass without errors ([PASS]).
