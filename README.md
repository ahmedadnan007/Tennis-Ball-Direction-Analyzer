# TennuisBallAnalyzer

This workspace contains a tennis ball analysis project with a FastAPI backend and a static frontend.

The root [`backend.py`](backend.py) is a compatibility loader so `uvicorn backend:app` works from the workspace root while the main implementation stays in `tennis-ball-detection-6/`.

## Run

```powershell
python -m pip install -r tennis-ball-detection-6/requirements.txt
uvicorn backend:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/` in your browser.

## Project Docs

For training, inference, and dataset notes, see [`tennis-ball-detection-6/README.md`](tennis-ball-detection-6/README.md).