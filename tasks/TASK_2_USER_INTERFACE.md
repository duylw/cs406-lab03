# TASK 2: USER INTERFACE AND INTERACTIVE DASHBOARD

## Metadata
- Task ID: TASK-02
- Title: User Interface & Interactive Comparison Dashboard
- Status: [TODO]
- Priority: High
- Tracking File: [TRACKING.md](file:///d:/UIT/XLA/lab/lab03/TRACKING.md)
- Reference: [PROBLEM_STATEMENT.md](file:///d:/UIT/XLA/lab/lab03/PROBLEM_STATEMENT.md)
- Dependency: [TASK_1_CORE_ALGORITHMS.md](file:///d:/UIT/XLA/lab/lab03/tasks/TASK_1_CORE_ALGORITHMS.md)

---

## 1. What To Do (Scope & Requirements)
Build an interactive graphical user interface enabling users to interactively test and visually compare all image enhancement and edge detection techniques on personal color images:

1. Image Ingestion:
   - File uploader supporting standard color image formats (JPG, JPEG, PNG, BMP).
   - Display original image preview with metadata (resolution, color channels, file size).

2. Interactive Control Panels:
   - Denoising Controls: Noise type selector (Gaussian vs. Salt & Pepper), noise intensity slider, kernel size selectors (3, 5, 7, 9).
   - Sharpening Controls: Method selector (Laplacian vs. Unsharp Mask), strength/scale slider.
   - Edge Detection Controls: Low threshold and high threshold sliders for Canny hysteresis.

3. Visual Comparison Displays (Grid Layout):
   - Denoising Comparison Matrix: Display side-by-side or 2x2 grid containing:
     - Original Image
     - Noisy Image
     - Mean Filter Output (with live PSNR score badge)
     - Median Filter Output (with live PSNR score badge)
   - Sharpening Comparison: Side-by-side layout comparing Original Image and Sharpened Image.
   - Edge Detection Gallery: 3-column concurrent layout rendering Sobel, Prewitt, and Canny results side-by-side for direct visual assessment.

4. Result Export:
   - Download buttons for individual processed images and composite comparison figures.

---

## 2. How To Do It (Implementation Strategy)
- Framework: Streamlit (`app.py`), chosen for seamless Python image processing integration, fast prototyping, and built-in widget reactivity.
- Dependency Installation:
  ```bash
  uv add streamlit
  ```
- Architecture:
  ```
  app.py              # Main dashboard entrypoint
  ui/
  ├── components.py   # Reusable UI widgets and layout blocks
  └── style.css       # Custom styling for clean, professional layout
  ```
- Workflow:
  - User uploads image via `st.file_uploader`.
  - Image is decoded into RGB format via OpenCV / PIL.
  - Sidebar or top control bar provides parameter adjustments.
  - Processing functions from `core/` are called upon widget value change.
  - Multi-column layouts (`st.columns`) render side-by-side comparisons according to the layout requirements in the problem statement.
  - Live metric computation displays PSNR/MSE metrics directly underneath the respective denoised images.

---

## 3. How To Test (Verification Protocol)
- Execution Command: `uv run streamlit run app.py`
- Test Script / Procedure: Manual inspection checklist and automated UI smoke test via headless runner or script test.
- Test Scenarios:
  1. Upload Format Verification:
     - Upload sample images in JPG, PNG, and BMP formats.
     - Verify images render correctly without color distortion (no inverted BGR/RGB channels).
  2. Reactive Parameter Adjustment:
     - Adjust kernel size slider from 3 to 7. Verify the output updates immediately without application crash.
     - Toggle between Gaussian and Salt & Pepper noise; confirm the comparison grid reflects the correct noise profile.
  3. Concurrent Edge Display Verification:
     - Verify that all three edge detection filters (Sobel, Prewitt, Canny) render simultaneously on screen in 3 distinct columns.
  4. Export Verification:
     - Click download buttons for each processed output.
     - Confirm exported image files can be opened and match the displayed visual result.

---

## 4. Acceptance Criteria
- [ ] Dependency `streamlit` installed and recorded in `pyproject.toml` and `uv.lock`.
- [ ] Application starts cleanly via `uv run streamlit run app.py` without warnings or deprecation errors.
- [ ] User can upload any standard color image and receive real-time visual feedback.
- [ ] All three edge detection outputs are presented simultaneously in a side-by-side layout.
- [ ] Comparison views for denoising clearly display original, noisy, mean-filtered, and median-filtered images with PSNR values.
