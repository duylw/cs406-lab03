# AGENT BEHAVIOR AND OPERATING RULES

## 1. Overview and Core Philosophy
This document defines the strict rules, personality, behavior, and technical standards that any AI agent must follow when developing, refactoring, or maintaining code in this repository.

## 2. Communication and Language Rules
- Language: All code comments, docstrings, technical documentation, tracking logs, and architectural notes must be written in English.
- Tone: Professional, objective, concise, and direct. Avoid conversational filler, pleasantries, or speculative commentary.
- Iconography Constraint: Strictly do NOT use icons, emojis, or pictorial symbols (such as checkmarks, rocket ships, warning signs, stars, etc.) anywhere in documents, code, logs, or commit messages. Use plain text labels instead (e.g., [TODO], [IN PROGRESS], [DONE], [PASS], [FAIL], [WARN], [INFO]).

## 3. Architecture and Code Organization
- Modularity: Code must be decoupled into single-responsibility modules:
  - Core algorithms and mathematical operations must remain independent of UI or delivery frameworks.
  - UI/Presentation layers must only import and call core engine functions without embedding heavy mathematical logic.
  - Test suites must be separated from production code.
- File Structure:
  - Clear, lowercase, snake_case naming for Python files and directories.
  - Explicit function signatures with type hints.
  - Documented inputs, outputs, and exceptions via standard docstrings (Google or NumPy style).
- Robust Input Handling:
  - Always validate image dimensions, channels (RGB vs. RGBA vs. Grayscale), and data types (uint8 vs. float32).
  - Explicitly handle edge cases: zero-size images, unsupported file extensions, extreme kernel dimensions.

## 4. Coding Standards (Python)
- Follow PEP 8 guidelines strictly (naming conventions, line lengths, import order).
- Avoid wildcard imports (`from module import *`).
- Preserve original images during processing; never mutate input buffers in-place unless explicitly designed and documented.
- Use vectorization via NumPy or optimized OpenCV routines instead of nested Python loops for pixel-level calculations.

## 5. Testing and Verification Protocol
- Every algorithm implementation must be accompanied by automated verification tests.
- Tests must assert:
  - Correctness of output array shape and data type.
  - Value bounds (e.g., pixel intensities within [0, 255]).
  - Mathematical correctness for edge detection kernels (e.g., Prewitt, Sobel gradient checks).
  - Resilience against malformed or unusual inputs.
- Manual verification steps must have defined input criteria, expected visual outcomes, and metric thresholds (e.g., PSNR, SSIM where applicable).

## 6. Task Tracking and Discipline
- Consult and maintain `TRACKING.md` systematically.
- Do not mark a task as completed until all acceptance criteria and test cases have passed.
- If an architectural trade-off or technical pivot is necessary, record the rationale clearly in technical notes in English.

## 7. Package and Environment Management (uv)
- Tooling: `uv` is the mandatory package and environment manager for this repository.
- Dependency Declarations:
  - Manage project dependencies via `pyproject.toml` and lockfile `uv.lock`.
  - Add or update packages using `uv add <package>` (or `uv pip install -r requirements.txt`).
- Execution Environment:
  - All test runners, scripts, and applications must be executed via `uv run` (e.g., `uv run pytest`, `uv run streamlit run app.py`, `uv run python verify_pipeline.py`).
  - Direct invocations of system-level `pip` or global Python interpreters are strictly prohibited to ensure isolation and reproducibility.
