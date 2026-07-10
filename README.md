# 🎾 Tennis Ball Direction Analyzer

An AI-powered computer vision system that detects, tracks, and analyzes tennis ball trajectories from match video — classifying shot type (forehand / backhand / serve), direction (cross-court / down-the-line), speed, and shot depth, with a FastAPI backend and a web dashboard for uploading videos and reviewing results.

> Final Year Project — built with YOLOv8 object detection, Kalman-filter based tracking, and homography-based court calibration.

---

## ✨ Features

- **Ball detection** — custom-trained YOLOv8 model (`tennis_v3_sinner-2`, 1,881 images, ~92.3% mAP) detects the tennis ball in each frame.
- **Multi-frame tracking** — a Kalman-filter tracker (`ball_tracker.py`) smooths noisy detections, keeps a consistent track ID, filters out false positives by ball size/shape, and survives short occlusions.
- **Court calibration** — `calibration.json` + `court_detector.py` map pixel coordinates to real-world court geometry (8.23 m × 23.77 m singles court) via homography, enabling accurate speed and zone calculations.
- **Shot classification** — `shot_classifier.py` labels every rally shot as a **forehand / backhand / serve**, its **direction** (cross-court left/right, down-the-line), and its **shot type** (groundstroke, lob, drop shot) based on trajectory apex height.
- **Speed & analytics** — per-shot average/max speed (km/h), duration, and a shot confidence score, exported as CSV per job.
- **Web dashboard** — upload a video, watch job progress in real time, then preview the annotated output video and download the results (video + CSV) — no notebook or CLI required.
- **REST API** — a small FastAPI service exposes analysis as async jobs so the frontend (or any other client) can poll for status and stream results.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Tracking | Kalman Filter (`filterpy`) + nearest-neighbor association (`scipy`) |
| Court geometry | OpenCV homography |
| Backend / API | FastAPI, Uvicorn |
| Video processing | OpenCV, `imageio-ffmpeg` |
| Data | pandas, NumPy |
| Frontend | Static HTML / CSS / JavaScript dashboard served by FastAPI |

---

## 📁 Project Structure

```
Tennis-Ball-Direction-Analyzer/
├── backend.py                          # Compatibility loader → forwards to the implementation below
├── README.md                           # You are here
└── tennis-ball-detection-6/            # Main application
    ├── backend.py                      # FastAPI app: endpoints, job queue, video streaming
    ├── ball_tracker.py                 # Kalman-filter multi-object ball tracker
    ├── court_detector.py               # Court line detection + calibration + homography
    ├── shot_classifier.py              # Forehand/backhand/serve + direction + shot-type logic
    ├── convert_coco_to_yolo.py         # Converts COCO-format annotations to YOLO format
    ├── calibrate_court_interactive.py  # Interactive tool to calibrate court geometry on a frame
    ├── run_calibration.py              # Non-interactive calibration runner
    ├── data_exporter.py                # Exports shot analytics
    ├── calibration.json                # Saved court calibration (pixel ↔️ real-world mapping)
    ├── data.yaml                       # YOLO dataset config (1 class: "tennis ball")
    ├── requirements.txt                # Python dependencies
    ├── TRAINING_GUIDE.md               # Step-by-step model training guide
    ├── static/                         # Web dashboard (HTML/CSS/JS) — served at `/`
    ├── outputs/                        # Generated per-job results (video + shots.csv)
    └── runs/detect/.../weights/        # Trained model weights (best.pt) — not committed, see below
```

---

## ⚙️ Setup

**Prerequisites:** Python 3.10+, and an NVIDIA GPU with CUDA for training/inference (the backend currently requires CUDA to be available).

```bash
# 1. Clone the repo
git clone https://github.com/ahmedadnan007/Tennis-Ball-Direction-Analyzer.git
cd Tennis-Ball-Direction-Analyzer

# 2. Install dependencies
python -m pip install -r tennis-ball-detection-6/requirements.txt

# 3. Run the API + dashboard
uvicorn backend:app --reload --port 8000
```

Then open **http://127.0.0.1:8000/** to use the dashboard.

> **Model weights:** the trained YOLOv8 weights (`best.pt`) are not committed to the repository (kept out via `.gitignore` since model files are large). Train your own model using the steps below, then point `WEIGHTS_PATH` in `tennis-ball-detection-6/backend.py` to your `best.pt`, or drop it at the expected path: `tennis-ball-detection-6/runs/detect/models/tennis_v3_sinner-2/weights/best.pt`.

---

## 🧠 Training Your Own Model

Full details are in [`tennis-ball-detection-6/TRAINING_GUIDE.md`](tennis-ball-detection-6/TRAINING_GUIDE.md). Summary:

```bash
cd tennis-ball-detection-6

# 1. Convert the COCO-format dataset annotations to YOLO format
python convert_coco_to_yolo.py

# 2. Train YOLOv8 on the dataset (data.yaml → 1 class: "tennis ball")
python train_yolo.py --data data.yaml --model yolov8m.pt --epochs 100 --imgsz 640 --batch 16 --name tennis_ball_run
```

Trained weights are saved to `runs/train/<name>/weights/best.pt`. On a GPU (RTX 3060+), training takes roughly 30–60 minutes for 100 epochs.

---

## 🎯 Court Calibration

Shot speed, direction, and zone accuracy depend on `calibration.json`, which maps pixels to real court dimensions (8.23 m × 23.77 m). To calibrate for a new camera angle:

```bash
cd tennis-ball-detection-6
python calibrate_court_interactive.py
```

This lets you click the court corners on a sample frame; the resulting geometry is saved to `calibration.json` and used automatically by `court_detector.py` on the next run.

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web dashboard |
| `POST` | `/api/analyze` | Upload a video file, starts an async analysis job → returns `job_id` |
| `GET` | `/api/status/{job_id}` | Poll job progress, logs, and result links |
| `GET` | `/api/results/{job_id}/video` | Stream/download the annotated output video (supports HTTP range requests) |
| `GET` | `/api/results/{job_id}/csv` | Download the per-shot analytics as CSV |
| `GET` | `/api/results/{job_id}/data` | Get per-shot analytics as JSON |
| `GET` | `/api/jobs` | List all analysis jobs |

---

## 📊 Output

Each analyzed video produces:

- **`analyzed.mp4` / `analyzed_web.mp4`** — the source video annotated with ball tracking, court lines, and shot labels.
- **`shots.csv`** — one row per detected shot, including stroke type, direction, shot type, average/max speed (km/h), apex height, duration, and confidence score.

---

## 🗺️ Roadmap / Known Limitations

- CUDA is currently required — CPU-only inference is not yet supported by the backend.
- `train_yolo.py`/training entry point and `track_and_analyze.py` (referenced by the backend as the inference worker) should be included or documented alongside the trained weights before deploying this on a fresh machine.
- Court calibration is per-camera-angle; a new calibration is needed if the camera position changes.
- No automated tests yet.

---

## 🤝 Contributing

Issues and pull requests are welcome. If you're extending this for your own FYP/research, feel free to fork and adapt the tracker, classifier, or dashboard.

## 👤 Author

**Ahmed Adnan** — [@ahmedadnan007](https://github.com/ahmedadnan007)

## 📄 License

No license has been specified yet. Consider adding one (e.g. MIT) if you intend for others to reuse this code.