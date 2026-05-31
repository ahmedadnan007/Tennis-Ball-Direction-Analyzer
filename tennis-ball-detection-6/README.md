# Tennis Ball Detection with YOLOv8

This small toolkit trains YOLOv8 on the provided dataset (YOLO format) and runs detection on videos, saving an output with bounding boxes.

Run the new HTML/CSS/JS dashboard with FastAPI:
```powershell
python -m pip install -r requirements.txt
uvicorn backend:app --reload --port 8000
```

Then open `http://127.0.0.1:8000/` in the browser (root serves the dashboard).

The legacy Streamlit app has been removed — this repository now uses a FastAPI backend with a static HTML/CSS/JS frontend served at `/` and `/static`.

Setup (PowerShell):
```powershell
python -m pip install -r requirements.txt
```

Training
 - Ensure your `data.yaml` is at the repository root and points to `train`, `val`, and `test` folders.
 - Example training command:
```powershell
python train_yolo.py --data data.yaml --model yolov8m.pt --epochs 100 --imgsz 640 --batch 16 --name tennis_ball_run
```
 - Trained runs will be saved under `runs/train/<name>/weights/` by the ultralytics library.

Inference on video
 - After training, run the inference script pointing to `best.pt` (or `last.pt`):
```powershell
python detect_video.py --weights runs/train/tennis_ball_run/weights/best.pt --source input.mp4 --output output.mp4 --conf 0.25
```

Notes & tips
- If you have a GPU, pass `--device 0` to use GPU (or `--device cpu` to force CPU).
-- Use a smaller model like `yolov8n.pt` for fast experiments; default production model is `yolov8m.pt` for higher accuracy.
- If detection is missing small balls, try increasing input `--imgsz` (e.g., 1280) and augmentations or more training epochs.
- Inspect `runs/train/<name>/results.txt` or the ultralytics web UI to monitor mAP and losses.

If you want, I can:
- run a quick check to verify `data.yaml` paths and dataset structure,
- suggest hyperparameters tuned to small-object detection,
- or prepare a small script to extract frames for debugging.
