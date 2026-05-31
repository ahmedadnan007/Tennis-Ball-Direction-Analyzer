"""Auto-label tennis ball images using existing trained model."""

from ultralytics import YOLO
from pathlib import Path
import os
import cv2

# ============================================================
# Settings
# ============================================================
MODEL_PATH = "runs/detect/models/tennis_v2_combined-2/weights/best.pt"
IMAGES_FOLDER = r"sinner_alcaraz_dataset/train"  # Apna folder path
OUTPUT_LABELS = r"sinner_alcaraz_dataset/labels"  # Labels save honge yahan
CONFIDENCE = 0.30  # Kam confidence = zyada balls detect

# ============================================================
# Auto-label
# ============================================================
def main():
    print(f"[INFO] Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    
    os.makedirs(OUTPUT_LABELS, exist_ok=True)
    
    # Get all images
    images = list(Path(IMAGES_FOLDER).glob("*.jpg")) + \
             list(Path(IMAGES_FOLDER).glob("*.png")) + \
             list(Path(IMAGES_FOLDER).glob("*.jpeg"))
    
    print(f"[INFO] Found {len(images)} images")
    
    labeled = 0
    no_detection = 0
    
    for i, img_path in enumerate(images):
        # Read image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Detect
        results = model(str(img_path), conf=CONFIDENCE, verbose=False)
        r = results[0]
        
        # Create label file (same name as image but .txt)
        label_file = Path(OUTPUT_LABELS) / (img_path.stem + ".txt")
        
        with open(label_file, 'w') as f:
            if hasattr(r.boxes, 'xywhn') and len(r.boxes.xywhn) > 0:
                boxes = r.boxes.xywhn.cpu().numpy()  # normalized
                
                for box in boxes:
                    cx, cy, bw, bh = box
                    # YOLO format: class_id cx cy w h
                    # class 0 = tennis ball
                    f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                
                labeled += 1
            else:
                no_detection += 1
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"[INFO] {i+1}/{len(images)} processed ({labeled} labeled, {no_detection} no detection)")
    
    print(f"\n✅ Done!")
    print(f"   Labeled:       {labeled}")
    print(f"   No detection:  {no_detection}")
    print(f"   Total:         {len(images)}")
    print(f"\n📁 Labels saved to: {OUTPUT_LABELS}")
    print(f"💡 Next: Manually verify in LabelImg, then train!")


if __name__ == "__main__":
    main()