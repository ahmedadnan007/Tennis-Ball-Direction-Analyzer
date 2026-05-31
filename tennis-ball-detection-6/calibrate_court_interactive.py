"""Interactive court calibration tool.

This tool lets you click on tennis court images to manually calibrate:
- Baseline positions (top and bottom)
- Service line positions
- Net position
- Sideline positions
- Center mark

The calibration is saved to calibration.json for use in ball tracking.

Usage:
    python calibrate_court_interactive.py --image "tennis courts/hard.avif"
    
    Then click on court lines as prompted:
    1. Click on TOP baseline (left and right corners)
    2. Click on BOTTOM baseline (left and right corners)
    3. Click on NET position (left and right)
    4. Click on TOP service line (left and right)
    5. Click on BOTTOM service line (left and right)
    6. Click on LEFT sideline (top and bottom)
    7. Click on RIGHT sideline (top and bottom)
    
    Press 's' to save, 'r' to reset, 'q' to quit.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class CourtCalibrator:
    """Interactive tool for court line calibration."""
    
    def __init__(self, image_path: str):
        """Initialize calibrator with an image."""
        self.image_path = Path(image_path)
        self.image = cv2.imread(str(self.image_path))
        
        if self.image is None:
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        # Resize if too large
        height, width = self.image.shape[:2]
        if max(height, width) > 1500:
            scale = 1500 / max(height, width)
            self.image = cv2.resize(self.image, (int(width * scale), int(height * scale)))
        
        self.display_image = self.image.copy()
        self.original_shape = self.image.shape
        
        # Calibration points
        self.calibration_points = {
            'baseline_top_left': None,      # Top left corner of top baseline
            'baseline_top_right': None,     # Top right corner of top baseline
            'baseline_bottom_left': None,   # Bottom left corner of bottom baseline
            'baseline_bottom_right': None,  # Bottom right corner of bottom baseline
            'net_left': None,               # Left side of net
            'net_right': None,              # Right side of net
            'service_line_top_left': None,  # Top service line (left)
            'service_line_top_right': None, # Top service line (right)
            'service_line_bottom_left': None, # Bottom service line (left)
            'service_line_bottom_right': None, # Bottom service line (right)
            'sideline_left_top': None,      # Left sideline (top)
            'sideline_left_bottom': None,   # Left sideline (bottom)
            'sideline_right_top': None,     # Right sideline (top)
            'sideline_right_bottom': None,  # Right sideline (bottom)
            'center_mark_top': None,        # Center mark (top)
            'center_mark_bottom': None,     # Center mark (bottom)
        }
        
        # Calibration steps
        self.calibration_steps = [
            ('baseline_top_left', 'Click on TOP-LEFT corner of top baseline'),
            ('baseline_top_right', 'Click on TOP-RIGHT corner of top baseline'),
            ('baseline_bottom_left', 'Click on BOTTOM-LEFT corner of bottom baseline'),
            ('baseline_bottom_right', 'Click on BOTTOM-RIGHT corner of bottom baseline'),
            ('net_left', 'Click on LEFT side of NET'),
            ('net_right', 'Click on RIGHT side of NET'),
            ('service_line_top_left', 'Click on LEFT end of TOP service line'),
            ('service_line_top_right', 'Click on RIGHT end of TOP service line'),
            ('service_line_bottom_left', 'Click on LEFT end of BOTTOM service line'),
            ('service_line_bottom_right', 'Click on RIGHT end of BOTTOM service line'),
            ('sideline_left_top', 'Click on TOP of LEFT sideline'),
            ('sideline_left_bottom', 'Click on BOTTOM of LEFT sideline'),
            ('sideline_right_top', 'Click on TOP of RIGHT sideline'),
            ('sideline_right_bottom', 'Click on BOTTOM of RIGHT sideline'),
            ('center_mark_top', 'Click on TOP of CENTER mark (on baseline)'),
            ('center_mark_bottom', 'Click on BOTTOM of CENTER mark (on baseline)'),
        ]
        
        self.current_step = 0
        self.window_name = 'Court Calibration'
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks."""
        if event == cv2.EVENT_LBUTTONDOWN:
            point_name, _ = self.calibration_steps[self.current_step]
            self.calibration_points[point_name] = (x, y)
            
            # Draw point on image
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.circle(self.display_image, (x, y), 8, (0, 255, 0), 2)
            
            print(f"[OK] Marked {point_name} at ({x}, {y})")
            
            # Move to next step
            self.current_step += 1
            if self.current_step >= len(self.calibration_steps):
                self.current_step = len(self.calibration_steps)
                print("\n[OK] All points marked! Press 's' to save calibration.")
    
    def _draw_instructions(self):
        """Draw calibration instructions on image."""
        display = self.display_image.copy()
        
        # Add semi-transparent background for text
        overlay = display.copy()
        cv2.rectangle(overlay, (10, 10), (600, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Progress
        progress_step = min(self.current_step + 1, len(self.calibration_steps))
        progress_text = f"Step {progress_step}/{len(self.calibration_steps)}"
        cv2.putText(display, progress_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Current instruction
        if self.current_step < len(self.calibration_steps):
            _, instruction = self.calibration_steps[self.current_step]
        else:
            instruction = "All points marked. Press 's' to save calibration or 'r' to reset."

        lines = [instruction[i:i+40] for i in range(0, len(instruction), 40)]
        for i, line in enumerate(lines):
            cv2.putText(display, line, (20, 75 + i*30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(display, "Keys: 's'=save  'r'=reset  'q'=quit", (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return display
    
    def run(self) -> Optional[Dict]:
        """Run the calibration interface."""
        print("=" * 60)
        print("TENNIS COURT CALIBRATION TOOL")
        print("=" * 60)
        print(f"Image: {self.image_path}")
        print(f"Image size: {self.display_image.shape[1]}x{self.display_image.shape[0]}")
        print("\nInstructions:")
        print("  - Click on court lines as prompted")
        print("  - 's' = Save calibration")
        print("  - 'r' = Reset all points")
        print("  - 'q' = Quit without saving")
        print("=" * 60)
        
        while True:
            display = self._draw_instructions()
            cv2.imshow(self.window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting without saving.")
                break
            elif key == ord('r'):
                print("\nResetting all points...")
                self.calibration_points = {k: None for k in self.calibration_points}
                self.current_step = 0
                self.display_image = self.image.copy()
            elif key == ord('s'):
                if self.current_step < len(self.calibration_steps):
                    print(f"\n⚠ Warning: Not all points marked ({self.current_step + 1}/{len(self.calibration_steps)})")
                    response = input("Save anyway? (y/n): ").lower()
                    if response != 'y':
                        continue
                
                result = self._save_calibration()
                if result:
                    return result
                break
        
        cv2.destroyAllWindows()
        return None
    
    def _save_calibration(self) -> Dict:
        """Save calibration to JSON and compute court geometry."""
        # Calculate derived court geometry
        calibration_data = {
            'image_path': str(self.image_path),
            'image_size': {
                'width': self.display_image.shape[1],
                'height': self.display_image.shape[0]
            },
            'manual_points': self.calibration_points,
            'court_geometry': self._compute_court_geometry(),
            'timestamp': str(Path.cwd())
        }
        
        # Save to calibration.json
        calib_file = Path(__file__).parent / 'calibration.json'
        with open(calib_file, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print(f"\n[OK] Calibration saved to {calib_file}")
        print(json.dumps(calibration_data['court_geometry'], indent=2))
        
        return calibration_data
    
    def _average_point(self, keys, fallback, axis=0):
        """Average a set of calibration point coordinates safely."""
        values = []
        for key in keys:
            point = self.calibration_points.get(key)
            if point is not None:
                values.append(point[axis])
        if not values:
            return fallback
        return sum(values) / len(values)
    
    def _compute_court_geometry(self) -> Dict:
        """Compute court geometry from calibration points."""
        points = self.calibration_points
        width = self.display_image.shape[1]
        height = self.display_image.shape[0]
        
        # Baseline positions (Y coordinates)
        baseline_top_y = self._average_point(
            ['baseline_top_left', 'baseline_top_right'],
            fallback=height * 0.1,
            axis=1
        )
        baseline_bottom_y = self._average_point(
            ['baseline_bottom_left', 'baseline_bottom_right'],
            fallback=height * 0.9,
            axis=1
        )
        
        # Net position (Y coordinate)
        net_y = self._average_point(
            ['net_left', 'net_right'],
            fallback=(baseline_top_y + baseline_bottom_y) / 2,
            axis=1
        )
        
        # Service line positions (Y coordinates)
        service_top_y = self._average_point(
            ['service_line_top_left', 'service_line_top_right'],
            fallback=baseline_top_y + (net_y - baseline_top_y) * 0.5,
            axis=1
        )
        service_bottom_y = self._average_point(
            ['service_line_bottom_left', 'service_line_bottom_right'],
            fallback=net_y + (baseline_bottom_y - net_y) * 0.5,
            axis=1
        )
        
        # Sideline positions (X coordinates)
        sideline_left_x = self._average_point(
            ['sideline_left_top', 'sideline_left_bottom'],
            fallback=width * 0.1,
            axis=0
        )
        sideline_right_x = self._average_point(
            ['sideline_right_top', 'sideline_right_bottom'],
            fallback=width * 0.9,
            axis=0
        )
        
        # Center mark positions (X coordinate)
        center_mark_x = self._average_point(
            ['center_mark_top', 'center_mark_bottom'],
            fallback=(sideline_left_x + sideline_right_x) / 2,
            axis=0
        )
        
        # Court dimensions (in pixels)
        court_width_px = max(1.0, sideline_right_x - sideline_left_x)
        court_length_px = max(1.0, baseline_bottom_y - baseline_top_y)
        
        geometry = {
            'baseline_top_y': float(baseline_top_y),
            'baseline_bottom_y': float(baseline_bottom_y),
            'net_y': float(net_y),
            'service_line_top_y': float(service_top_y),
            'service_line_bottom_y': float(service_bottom_y),
            'sideline_left_x': float(sideline_left_x),
            'sideline_right_x': float(sideline_right_x),
            'center_mark_x': float(center_mark_x),
            'court_width_pixels': float(court_width_px),
            'court_length_pixels': float(court_length_px),
            'court_center_x': float((sideline_left_x + sideline_right_x) / 2),
            'court_center_y': float((baseline_top_y + baseline_bottom_y) / 2),
            'conversion_notes': {
                'width_pixels_to_meters': f"{court_width_px} px = 8.23 m (singles)",
                'length_pixels_to_meters': f"{court_length_px} px = 23.77 m",
                'pixels_to_meters_x': 8.23 / court_width_px,
                'pixels_to_meters_y': 23.77 / court_length_px
            }
        }
        
        return geometry


def main():
    parser = argparse.ArgumentParser(description='Calibrate tennis court lines')
    parser.add_argument('--image', type=str, required=True, help='Path to court image')
    args = parser.parse_args()
    
    calibrator = CourtCalibrator(args.image)
    result = calibrator.run()
    
    if result:
        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)
        print("Court Geometry:")
        for key, value in result['court_geometry'].items():
            if key != 'conversion_notes':
                print(f"  {key}: {value:.1f}")


if __name__ == '__main__':
    main()
