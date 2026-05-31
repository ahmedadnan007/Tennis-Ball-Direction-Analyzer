#!/usr/bin/env python3
"""Convert COCO JSON annotations to YOLO format (.txt files)"""

import json
from pathlib import Path

def coco_to_yolo(coco_json_file, output_dir):
    """Convert COCO JSON annotations to YOLO format."""
    
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create mapping from image ID to image info
    image_map = {img['id']: img for img in coco_data['images']}
    
    # Process annotations
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Convert to YOLO format
    count = 0
    for img_id, annotations in annotations_by_image.items():
        img_info = image_map[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Get filename without extension
        img_name = Path(img_info['file_name']).stem
        txt_file = output_dir / f"{img_name}.txt"
        
        # Write YOLO format
        with open(txt_file, 'w') as f:
            for ann in annotations:
                category_id = ann['category_id'] - 1  # YOLO uses 0-based indexing
                bbox = ann['bbox']  # [x, y, width, height]
                
                # Convert to YOLO format: [class_id, x_center, y_center, width_norm, height_norm]
                # Handle both float and string values
                x = float(bbox[0])
                y = float(bbox[1])
                w = float(bbox[2])
                h = float(bbox[3])
                
                x_center = (x + w / 2) / img_width
                y_center = (y + h / 2) / img_height
                width_norm = w / img_width
                height_norm = h / img_height
                
                f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
        
        count += 1
    
    return count

def main():
    dataset_dir = Path(__file__).parent / "tennis dataset"
    train_dir = dataset_dir / "train"
    coco_file = train_dir / "_annotations.coco.json"
    
    if not coco_file.exists():
        print(f"[ERROR] COCO file not found: {coco_file}")
        return False
    
    print(f"[INFO] Converting COCO annotations to YOLO format...")
    print(f"[INFO] Input: {coco_file}")
    
    count = coco_to_yolo(coco_file, train_dir)
    
    print(f"[OK] Converted {count} images")
    print(f"[OK] YOLO label files saved to: {train_dir}")
    
    # Verify
    txt_count = len(list(train_dir.glob("*.txt")))
    jpg_count = len(list(train_dir.glob("*.jpg")))
    
    print(f"[INFO] Images: {jpg_count}, Labels: {txt_count}")
    
    if txt_count == jpg_count:
        print("[OK] ✓ All images have labels!")
        return True
    else:
        print(f"[WARNING] Mismatch: {jpg_count} images but {txt_count} labels")
        return False

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
