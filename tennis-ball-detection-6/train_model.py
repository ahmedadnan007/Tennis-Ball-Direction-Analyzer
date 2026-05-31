#!/usr/bin/env python3
"""
Tennis Ball YOLO Model Training Script
Trains YOLOv8 on tennis ball detection dataset
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    try:
        from ultralytics import YOLO
        import torch
    except ImportError as e:
        print(f"[ERROR] Missing required package: {e}")
        print("[INFO] Please install requirements: pip install -r requirements.txt")
        return False
    
    # Verify GPU is available
    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU not available!")
        print("[INFO] This project requires NVIDIA GPU with CUDA support")
        return False
    
    print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Define paths
    DATASET_PATH = PROJECT_ROOT / "tennis dataset"
    DATA_YAML = PROJECT_ROOT / "data.yaml"
    
    if not DATASET_PATH.exists():
        print(f"[ERROR] Dataset not found: {DATASET_PATH}")
        return False
    
    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found: {DATA_YAML}")
        return False
    
    print(f"[OK] Dataset found at: {DATASET_PATH}")
    print(f"[OK] Data config found at: {DATA_YAML}")
    
    # Load pre-trained model
    print("\n[INFO] Loading YOLOv8m model...")
    model = YOLO("yolov8m.pt")
    
    # Train the model
    print("[INFO] Starting training...")
    print("[INFO] This may take 30-60 minutes depending on your GPU...")
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=8,  # Reduced from 16 for Windows GPU memory
        patience=20,
        device=0,  # GPU device 0
        name="tennis_ball_run",
        exist_ok=True,
        save=True,
        cache="disk",  # Use disk cache for Windows stability
        workers=0,  # Disable workers for Windows multiprocessing issue
        mosaic=1.0,  # Data augmentation
        flipud=0.5,
        fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.1,
        resume=True,  # Resume from last checkpoint if exists
    )
    
    print("\n[OK] Training completed!")
    print(f"[INFO] Best model saved to: runs/detect/tennis_ball_run/weights/best.pt")
    print(f"[INFO] Last model saved to: runs/detect/tennis_ball_run/weights/last.pt")
    
    # Verify the trained model exists
    best_pt = PROJECT_ROOT / "runs" / "detect" / "tennis_ball_run" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"[OK] ✓ Trained model verified: {best_pt}")
        return True
    else:
        print(f"[ERROR] Trained model not found at: {best_pt}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
