"""FastAPI backend for the tennis ball analyzer dashboard.

This server accepts video uploads, runs the existing analysis pipeline,
and exposes status/results endpoints for a plain HTML/CSS/JS frontend.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    from imageio_ffmpeg import get_ffmpeg_exe
except Exception:
    get_ffmpeg_exe = None


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUTS_DIR = BASE_DIR / "outputs"
JOBS_DIR = OUTPUTS_DIR / "jobs"
UPLOADS_DIR = OUTPUTS_DIR / "uploads"
WEIGHTS_PATH = BASE_DIR / "runs" / "detect" / "tennis_ball_run" / "weights" / "best.pt"
TRACK_SCRIPT = BASE_DIR / "track_and_analyze.py"
PYTHON_PATH = BASE_DIR / ".venv312" / "Scripts" / "python.exe"

OUTPUTS_DIR.mkdir(exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Tennis Ball Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_jobs: Dict[str, Dict[str, Any]] = {}
_job_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_pixels_per_meter(default_value: float = 22.1) -> float:
    calibration_file = BASE_DIR / "calibration.json"

    if not calibration_file.exists():
        return default_value

    try:
        with open(calibration_file, "r", encoding="utf-8") as file:
            calibration_data = json.load(file)

        pixels_per_meter = calibration_data.get("pixels_per_meter")
        if pixels_per_meter is None:
            pixel_distance = calibration_data.get("pixel_distance")
            real_distance_meters = calibration_data.get("real_distance_meters")
            if pixel_distance and real_distance_meters:
                pixels_per_meter = float(pixel_distance) / float(real_distance_meters)

        return float(pixels_per_meter) if pixels_per_meter else default_value
    except Exception:
        return default_value


def _create_job_record(job_id: str, source_name: str) -> Dict[str, Any]:
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "message": "Waiting to start",
        "logs": [],
        "created_at": _now(),
        "updated_at": _now(),
        "source_name": source_name,
        "job_dir": str(job_dir),
        "input_video": str(job_dir / f"input_{source_name}"),
        "output_video": str(job_dir / "analyzed.mp4"),
        "web_video": None,
        "output_csv": str(job_dir / "shots.csv"),
        "error": None,
    }

    with _job_lock:
        _jobs[job_id] = record

    return record


def _update_job(job_id: str, **changes: Any) -> None:
    with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        job.update(changes)
        job["updated_at"] = _now()


def _append_log(job_id: str, line: str) -> None:
    with _job_lock:
        job = _jobs.get(job_id)
        if not job:
            return
        logs: List[str] = job.setdefault("logs", [])
        logs.append(line)
        job["logs"] = logs[-200:]
        job["updated_at"] = _now()


def _run_job(job_id: str) -> None:
    with _job_lock:
        job = _jobs.get(job_id)

    if not job:
        return

    input_path = Path(job["input_video"])
    output_video = Path(job["output_video"])
    output_csv = Path(job["output_csv"])

    if not WEIGHTS_PATH.exists():
        _update_job(job_id, status="error", message=f"Missing model weights: {WEIGHTS_PATH}", error="weights_missing")
        return

    pixels_per_meter = _load_pixels_per_meter()

    command = [
        str(PYTHON_PATH),
        str(TRACK_SCRIPT),
        "--weights",
        str(WEIGHTS_PATH),
        "--source",
        str(input_path),
        "--output",
        str(output_video),
        "--export-csv",
        str(output_csv),
        "--device",
        "0",
        "--pixels-per-meter",
        str(pixels_per_meter),
        "--conf",
        "0.25",
        "--detect-court",
        "--player-handed",
        "right",
    ]

    _update_job(job_id, status="running", progress=0.0, message="Processing video")
    _append_log(job_id, "Starting analysis")

    try:
        process = subprocess.Popen(
            command,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        progress_pattern = re.compile(r"Progress:\s*([0-9.]+)%")

        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip()
            if not line:
                continue

            # Suppress noisy native Arrow warning that appears on some Windows setups
            # (e.g. ".../arrow/util/cpu_info.cc:104: Failed to get processor information").
            if "Failed to get processor information" in line and "arrow" in line.lower():
                # append a short debug note instead of the full native stack message
                _append_log(job_id, "[info] System CPU info unavailable to Apache Arrow (non-fatal).")
                continue

            _append_log(job_id, line)

            match = progress_pattern.search(line)
            if match:
                try:
                    progress = float(match.group(1))
                    _update_job(job_id, progress=progress, message=f"Processing {progress:.1f}%")
                except ValueError:
                    pass

        process.wait()

        if process.returncode != 0:
            hint = None
            recent_logs = "\n".join(_jobs.get(job_id, {}).get("logs", [])[-40:])
            if "Invalid CUDA 'device=0' requested" in recent_logs:
                hint = "CUDA not available to current PyTorch build. Install CUDA-enabled torch in .venv312."
            _update_job(
                job_id,
                status="error",
                message=hint or f"Processing failed with exit code {process.returncode}",
                error=f"exit_{process.returncode}",
            )
            return

        if not output_video.exists():
            _update_job(job_id, status="error", message="Processing finished but output video was not created", error="missing_output_video")
            return

        # Attempt to create a web-friendly copy using ffmpeg (h264/aac + faststart)
        web_path = output_video.with_name("analyzed_web.mp4")
        transcoded = False
        if get_ffmpeg_exe():
            try:
                ffmpeg_exe = get_ffmpeg_exe()
                transcode_cmd = [
                    str(ffmpeg_exe),
                    "-y",
                    "-i",
                    str(output_video),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    str(web_path),
                ]
                _append_log(job_id, f"Starting transcode to web-friendly MP4: {web_path.name}")
                proc = subprocess.run(transcode_cmd, cwd=str(BASE_DIR), capture_output=True, text=True, timeout=300)
                if proc.returncode == 0 and web_path.exists():
                    transcoded = True
                    _append_log(job_id, "Transcode complete")
                else:
                    _append_log(job_id, f"Transcode failed: {proc.stderr[:200]}")
            except Exception as exc:
                _append_log(job_id, f"Transcode exception: {exc}")

        if transcoded:
            _update_job(job_id, status="done", progress=100.0, message="Processing complete (web copy ready)", web_video=str(web_path))
            _append_log(job_id, "Processing complete (web copy ready)")
        else:
            _update_job(job_id, status="done", progress=100.0, message="Processing complete")
            _append_log(job_id, "Processing complete")

    except Exception as exc:
        _update_job(job_id, status="error", message=str(exc), error=str(exc))


@app.get("/")
def index() -> FileResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_file)


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)) -> JSONResponse:
    job_id = uuid4().hex
    source_name = Path(file.filename or "upload.mp4").name
    job = _create_job_record(job_id, source_name)

    suffix = Path(source_name).suffix or ".mp4"
    input_path = Path(job["input_video"])
    input_path = input_path.with_suffix(suffix)

    with open(input_path, "wb") as target:
        target.write(await file.read())

    _update_job(job_id, input_video=str(input_path), message="Upload complete")

    worker = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    worker.start()

    return JSONResponse(
        {
            "job_id": job_id,
            "status": job["status"],
            "message": job["message"],
        }
    )


@app.get("/api/status/{job_id}")
def get_status(job_id: str) -> JSONResponse:
    with _job_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JSONResponse(
        {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"],
            "logs": job.get("logs", [])[-20:],
            "output_video": f"/api/results/{job_id}/video" if Path(job["output_video"]).exists() else None,
            "output_web_video": f"/api/results/{job_id}/video" if job.get("web_video") and Path(job.get("web_video")).exists() else None,
            "output_csv": f"/api/results/{job_id}/csv" if Path(job["output_csv"]).exists() else None,
            "updated_at": job["updated_at"],
            "error": job.get("error"),
        }
    )


@app.get("/api/results/{job_id}/video")
def download_video(job_id: str, request: Request) -> StreamingResponse:
    with _job_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Prefer web-optimized copy if available
    video_path = Path(job.get("web_video") or job["output_video"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video not ready")

    # Support Range requests so browsers can stream the video inline.
    return _stream_file(video_path, request)


def _stream_file(path: Path, request: Request, chunk_size: int = 1024 * 1024):
    total = path.stat().st_size

    def _range_generator(start: int, end: int):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                read_len = min(chunk_size, remaining)
                data = f.read(read_len)
                if not data:
                    break
                remaining -= len(data)
                yield data

    def full_gen():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    range_header = request.headers.get("range")
    if range_header:
        m = re.match(r"bytes=(\d+)-(\d*)", range_header)
        if m:
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else total - 1
            if end >= total:
                end = total - 1
            content_length = end - start + 1
            headers = {
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{total}",
                "Content-Length": str(content_length),
                "Content-Type": "video/mp4",
                "Content-Disposition": f'inline; filename="{path.name}"',
            }
            return StreamingResponse(_range_generator(start, end), status_code=206, headers=headers, media_type="video/mp4")

    # No Range requested — return full file
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total),
        "Content-Type": "video/mp4",
        "Content-Disposition": f'inline; filename="{path.name}"',
    }
    return StreamingResponse(full_gen(), status_code=200, headers=headers, media_type="video/mp4")


@app.get("/api/results/{job_id}/csv")
def download_csv(job_id: str) -> FileResponse:
    with _job_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    csv_path = Path(job["output_csv"])
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV not ready")

    return FileResponse(csv_path, media_type="text/csv", filename=csv_path.name)


@app.get("/api/results/{job_id}/data")
def get_csv_data(job_id: str) -> JSONResponse:
    with _job_lock:
        job = _jobs.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    csv_path = Path(job["output_csv"])
    if not csv_path.exists():
        return JSONResponse({"job_id": job_id, "count": 0, "rows": []})

    try:
        df = pd.read_csv(csv_path)
        rows = df.fillna("").to_dict(orient="records")
    except Exception:
        rows = []

    return JSONResponse({"job_id": job_id, "count": len(rows), "rows": rows})


@app.get("/api/jobs")
def list_jobs() -> JSONResponse:
    with _job_lock:
        jobs = list(_jobs.values())

    return JSONResponse({"jobs": jobs})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=False)