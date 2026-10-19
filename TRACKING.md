# PROJECT TASK TRACKING DASHBOARD

## 1. Project Overview
- Subject: Lab-03 Image Enhancing Application
- Specification: [PROBLEM_STATEMENT.md](file:///d:/UIT/XLA/lab/lab03/PROBLEM_STATEMENT.md)
- Agent Guidelines: [AGENT.md](file:///d:/UIT/XLA/lab/lab03/AGENT.md)
- Package Manager: `uv` (Astral) for virtual environment and dependency management.
- Tracking Scope: Centralized management and status tracking for all project tasks.

---

## 2. Tasks Master Directory

| Task ID | Task Title | Specification File | Priority | Status | Owner |
|---|---|---|---|---|---|
| TASK-01 | Core Image Processing Algorithms | [TASK_1_CORE_ALGORITHMS.md](file:///d:/UIT/XLA/lab/lab03/tasks/TASK_1_CORE_ALGORITHMS.md) | High | [DONE] | Agent |
| TASK-02 | User Interface & Comparison Dashboard | [TASK_2_USER_INTERFACE.md](file:///d:/UIT/XLA/lab/lab03/tasks/TASK_2_USER_INTERFACE.md) | High | [DONE] | Agent |
| TASK-03 | Robustness, Edge Cases & Verification | [TASK_3_ROBUSTNESS_TESTING.md](file:///d:/UIT/XLA/lab/lab03/tasks/TASK_3_ROBUSTNESS_TESTING.md) | Medium | [TODO] | Agent |

---

## 3. Environment Setup & Dependency Management (uv)
- Virtual environment and lockfile initialized via `uv`.
- Key dependencies: `numpy`, `opencv-python`, `matplotlib`, `streamlit`, `pytest`.
- Execution rule: All commands must be run with `uv run <command>`.

---

## 4. Detailed Task Status Tracking

### [TASK-01] Core Image Processing Algorithms
- Document: [TASK_1_CORE_ALGORITHMS.md](file:///d:/UIT/XLA/lab/lab03/tasks/TASK_1_CORE_ALGORITHMS.md)
- Current Status: [DONE]
- Deliverables Checklist:
  - [x] Initialize Python environment and dependencies with `uv` (`pyproject.toml`, `uv.lock`)
  - [x] Gaussian noise and Salt & Pepper noise injection (`core/noise.py`)
  - [x] Mean filter and Median filter implementations (`core/denoise.py`)
  - [x] Laplacian sharpening and Unsharp Masking (`core/sharpen.py`)
  - [x] Sobel, Prewitt, and Canny edge detection algorithms (`core/edge.py`)
  - [x] MSE and PSNR metric calculation functions (`core/metrics.py`)
  - [x] Unit test suite passing all assertions via `uv run pytest tests/test_core.py`
- Blockers / Notes: None. All 22 automated tests verified and passing.

---

### [TASK-02] User Interface & Interactive Comparison Dashboard
- Document: [TASK_2_USER_INTERFACE.md](file:///d:/UIT/XLA/lab/lab03/tasks/TASK_2_USER_INTERFACE.md)
- Current Status: [DONE]
- Deliverables Checklist:
  - [x] Add Streamlit dependency via `uv add streamlit`
  - [x] Color image file uploader with metadata inspection (`app.py`)
  - [x] Parameter adjustment widgets for noise, kernel sizes, and thresholds
  - [x] 4-panel Denoising comparison grid (Original vs Noisy vs Mean vs Median with PSNR)
  - [x] Sharpening side-by-side comparison view
  - [x] 3-column concurrent Edge Detection gallery (Sobel, Prewitt, Canny)
  - [x] Processed image and comparison plot export functionality
  - [x] Application launch verified via `uv run streamlit run app.py`
- Blockers / Notes: None. Dashboard verified operational.

---

### [TASK-03] Robustness, Edge Case Handling & Automated Verification
- Document: [TASK_3_ROBUSTNESS_TESTING.md](file:///d:/UIT/XLA/lab/lab03/tasks/TASK_3_ROBUSTNESS_TESTING.md)
- Current Status: [TODO]
- Deliverables Checklist:
  - [ ] Input sanitization for RGBA, Grayscale, and extreme dimensions (`core/utils.py`)
  - [ ] Kernel size boundary validators (odd integer enforcement)
  - [ ] Automated end-to-end CLI batch runner (`verify_pipeline.py`) runnable via `uv run python verify_pipeline.py`
  - [ ] Automated output artifact generation for lab submission
  - [ ] Edge case test suite passing via `uv run pytest tests/test_edge_cases.py`
- Blockers / Notes: Can proceed once initial core algorithms and CLI structure are established.

---

## 5. Execution Dependency Sequence

```
[TASK-01: Core Algorithms]
         |
         +---------------------------------------+
         |                                       |
         v                                       v
[TASK-02: User Interface]               [TASK-03: Robustness & CLI Verification]
         |                                       |
         +-------------------+-------------------+
                             |
                             v
                 [Final Lab Integration]
```

---

## 6. Execution Changelog & Audit Log

- [2026-10-19]: Initialized tracking dashboard.
- [2026-10-19]: Split project into three dedicated task specification documents under `tasks/`.
- [2026-10-19]: Linked all task specification files to the master tracking table.
- [2026-10-19]: Added uv as the mandatory package manager across environment setup, task checklists, and command execution.
- [2026-10-19]: Completed TASK-01 (Core Image Processing Algorithms); verified all 22 automated tests passing under uv.
- [2026-10-19]: Completed TASK-02 (User Interface & Comparison Dashboard); installed Streamlit, implemented interactive comparison layouts, and verified web dashboard.
