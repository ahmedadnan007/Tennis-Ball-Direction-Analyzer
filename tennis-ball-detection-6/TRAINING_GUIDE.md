# 🎾 Tennis Ball Model Training Guide

## ✅ Dataset Status
Your tennis ball detection dataset has been restored:
```
tennis dataset/
└── train/
    ├── ~1500 training images (clay, fed, synframe, synthetic)
    └── _annotations.coco.json (COCO format annotations)
```

## 🔧 Training Pipeline

### Step 1: Convert Annotations to YOLO Format
The dataset uses COCO JSON format, but YOLO needs `.txt` files. Convert them:

```bash
cd tennis-ball-detection-6
.\.venv312\Scripts\python.exe convert_coco_to_yolo.py
```

**Expected output:**
```
[OK] Converted 1500 images
[OK] All images have labels!
```

### Step 2: Train the Model
Start the training process (takes 30-60 minutes on GPU):

```bash
.\.venv312\Scripts\python.exe train_model.py
```

**Expected output:**
```
[OK] CUDA available: NVIDIA GeForce RTX...
[INFO] Starting training...
...
[OK] Training completed!
[INFO] Best model saved to: runs/detect/tennis_ball_run/weights/best.pt
```

### Step 3: Verify Backend Works
Once training completes, run the backend:

```bash
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

---

## 🚀 Quick Start (All-in-One)

Run this to do everything automatically:

```bash
cd tennis-ball-detection-6
.\.venv312\Scripts\python.exe convert_coco_to_yolo.py && .\.venv312\Scripts\python.exe train_model.py
```

---

## 📊 What Gets Created

After training, you'll have:
```
runs/
└── detect/
    └── tennis_ball_run/
        ├── weights/
        │   ├── best.pt      ← Used by backend
        │   └── last.pt
        ├── results.csv      ← Training metrics
        └── ...
```

---

## ⚙️ Training Configuration

Edit `train_model.py` to adjust:
- `epochs`: Number of training cycles (default: 100)
- `batch`: Batch size (default: 16, increase if GPU memory allows)
- `imgsz`: Image size (default: 640)

---

## ✨ Notes

- Dataset: ~1500 images from Roboflow
- Model: YOLOv8m (medium size, good balance of speed/accuracy)
- GPU Required: NVIDIA CUDA GPU
- Estimated Time: 30-60 minutes on RTX 3060+
- Output: `best.pt` model for production use

