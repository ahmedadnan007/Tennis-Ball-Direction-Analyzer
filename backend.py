"""Compatibility loader for the FastAPI backend.

This lets `uvicorn backend:app` work when launched from the workspace root,
while keeping the real application code inside `tennis-ball-detection-6/`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_BASE_DIR = Path(__file__).resolve().parent / "tennis-ball-detection-6"
_IMPL_PATH = _BASE_DIR / "backend.py"

if not _IMPL_PATH.exists():
    raise ImportError(f"Backend implementation not found: {_IMPL_PATH}")

_SPEC = importlib.util.spec_from_file_location("tennis_ball_backend_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load backend implementation from {_IMPL_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

app = _MODULE.app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)