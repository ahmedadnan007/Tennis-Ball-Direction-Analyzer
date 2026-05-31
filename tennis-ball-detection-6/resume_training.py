#!/usr/bin/env python3
"""Resume YOLOv8 model training from checkpoint"""

import sys
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
        return False
    
    # Verify GPU is available
    if not torch.cuda.is_available():
        print("[ERROR] CUDA GPU not available!")
        return False
    
    print(f"[OK] CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[OK] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Define paths
    LAST_PT = PROJECT_ROOT / "runs" / "detect" / "tennis_ball_run" / "weights" / "last.pt"
    DATA_YAML = PROJECT_ROOT / "data.yaml"
    
    if not LAST_PT.exists():
        print(f"[ERROR] Checkpoint not found: {LAST_PT}")
        return False
    
    if not DATA_YAML.exists():
        print(f"[ERROR] data.yaml not found: {DATA_YAML}")
        return False
    
    print(f"[OK] Resuming from checkpoint: {LAST_PT}")
    print(f"[OK] Data config: {DATA_YAML}")
    
    # Load checkpoint model (this resumes from saved epoch)
    print("\n[INFO] Loading checkpoint model...")
    model = YOLO(str(LAST_PT))
    
    # Resume training
    print("[INFO] Resuming training...")
    print("[INFO] This will continue from epoch 77 to 100...")
    
    results = model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=8,
        patience=20,
        device=0,
        name="tennis_ball_run",
        exist_ok=True,
        save=True,
        cache="disk",
        workers=0,
        resume=True,  # Resume from checkpoint
        mosaic=1.0,
        flipud=0.5,
        fliplr=0.5,
        degrees=10,
        translate=0.1,
        scale=0.1,
    )
    
    print("\n[OK] Training completed!")
    print(f"[INFO] Best model: runs/detect/tennis_ball_run/weights/best.pt")
    
    best_pt = PROJECT_ROOT / "runs" / "detect" / "tennis_ball_run" / "weights" / "best.pt"
    if best_pt.exists():
        print(f"[OK] ✓ Trained model verified!")
        return True
    else:
        print(f"[ERROR] Best model not found at: {best_pt}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
