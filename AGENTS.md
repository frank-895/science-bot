## Project Goal

Build a life-science data analysis agent that:

1. Receives a natural language question
2. Locates relevant data inside a dataset capsule
3. Executes the required analysis deterministically
4. Returns a machine-checkable answer

The benchmark provides:
- questions
- dataset capsules
- expected answer formats

The system should generalize to unseen questions and capsules.

## Instructions

- Run backend commands via `uv`.
- Write Google-style docstrings for all production code (tests excluded).
- Do not write comments explaining *what* code does; only explain *why*, and only when necessary.
- Raise errors where they occur and log only at boundaries.  
  Services may log only for retry behavior or when intentionally swallowing errors.  
- Do not use `from __future__ import annotations` unless required. Do not manually stringize annotations.
- Place imports at the top of files. Use `if TYPE_CHECKING` to handle circular imports.
- Test `__init__.py` files must be empty.
- Other `__init__.py` files must contain a short docstring (longer only if necessary).
- This system is currently pre-MVP. Backwards compatibility is not necessary.
- Do not add type ignore comments in tests - tests are not type-checked anyway.
- Unit test file structure must exactly mirror the file structure of the source code.

## Pipeline Stages

- Each pipeline stage should live in its own folder under `science_bot/pipeline/`.
- Each stage folder should contain `schemas.py`, `stage.py`, and `__init__.py`.
- `schemas.py` should define the stage-local input and output schemas.
- `stage.py` should contain the actual stage implementation.
- The public stage function must be named `run_<stage_name>_stage`.
- The stage `__init__.py` should export only the input schema, output schema, and `run_<stage_name>_stage`.
