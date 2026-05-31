#!/usr/bin/env python
"""Quick calibration runner for tennis court images.

This script helps you calibrate your tennis courts interactively.

Usage:
    python run_calibration.py
    
This will prompt you to select a court image, then open an interactive
calibration tool where you click on court lines to establish exact positions.
"""

import sys
from pathlib import Path
from calibrate_court_interactive import CourtCalibrator


def list_court_images() -> list:
    """Find available court images."""
    court_dir = Path(__file__).parent / 'tennis courts'
    
    if not court_dir.exists():
        print(f"⚠ Tennis courts directory not found: {court_dir}")
        return []
    
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.avif', '*.bmp']:
        images.extend(court_dir.glob(ext))
    
    return sorted(images)


def main():
    print("=" * 70)
    print(" TENNIS COURT CALIBRATION WIZARD")
    print("=" * 70)
    
    # Find available images
    images = list_court_images()
    
    if not images:
        print("\n[WARNING] ERROR: No court images found in 'tennis courts/' directory")
        print("\nMake sure you have court images in:")
        print(f"  {Path('tennis courts').absolute()}")
        return 1
    
    print(f"\n[OK] Found {len(images)} court image(s):\n")
    
    for i, img in enumerate(images, 1):
        size = img.stat().st_size / 1024 / 1024
        print(f"  {i}. {img.name} ({size:.1f} MB)")
    
    while True:
        try:
            choice = input(f"\nSelect image (1-{len(images)}) or 'q' to quit: ").strip()
            
            if choice.lower() == 'q':
                print("Cancelled.")
                return 0
            
            idx = int(choice) - 1
            if 0 <= idx < len(images):
                selected = images[idx]
                break
        except (ValueError, IndexError):
            print("Invalid choice. Try again.")
    
    print(f"\n[OK] Selected: {selected.name}")
    print("\nOpening calibration tool...")
    print("Instructions:")
    print("  1. Click on court lines as prompted (16 points total)")
    print("  2. Be as accurate as possible for best results")
    print("  3. Press 's' to save, 'r' to reset, 'q' to quit")
    print("\nCalibration will be saved to: calibration.json\n")
    
    calibrator = CourtCalibrator(str(selected))
    result = calibrator.run()
    
    if result:
        print("\n" + "=" * 70)
        print(" [SUCCESS] CALIBRATION SAVED SUCCESSFULLY!")
        print("=" * 70)
        print("\nYour calibration data will be automatically used for:")
        print("  [OK] Court line detection")
        print("  [OK] Shot direction classification")
        print("  [OK] Court zone analysis")
        print("\nTo recalibrate, simply run this script again and select a different")
        print("image. The new calibration will overwrite the old one.")
        print("\nGeometry stored in calibration.json:")
        geom = result['court_geometry']
        print(f"  Net Y: {geom.get('net_y', 'N/A'):.0f} px")
        print(f"  Baseline Top: {geom.get('baseline_top_y', 'N/A'):.0f} px")
        print(f"  Baseline Bottom: {geom.get('baseline_bottom_y', 'N/A'):.0f} px")
        print(f"  Court Width: {geom.get('court_width_pixels', 'N/A'):.0f} px")
        print(f"  Court Length: {geom.get('court_length_pixels', 'N/A'):.0f} px")
        print("=" * 70)
        return 0
    else:
        print("\n[CANCELLED] Calibration cancelled.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
