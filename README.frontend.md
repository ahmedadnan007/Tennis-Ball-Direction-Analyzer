# HTML/CSS/JS Frontend

This project now has a simple web frontend powered by FastAPI.

## Run it

```powershell
python -m pip install -r requirements.txt
uvicorn backend:app --reload --port 8000
```

Then open:

```text
http://127.0.0.1:8000/
```

## What changed

- `backend.py` runs the API and serves the static files.
- `static/index.html` is the new page.
- `static/style.css` controls the layout and design.
- `static/script.js` uploads videos, polls progress, and shows results.
