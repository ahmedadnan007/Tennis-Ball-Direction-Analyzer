"""Interactive court calibration tool for accurate speed measurements.

This tool helps you calibrate pixel-to-meter conversion by clicking on known distances
in the tennis court (e.g., baseline to baseline = 23.77m, service box = 6.4m).

Usage:
    python calibrate_court.py --video rfvsrn.mp4
    
Instructions:
    1. The tool will show the first frame of your video
    2. Click two points that represent a known distance
    3. Enter the real-world distance in meters
    4. The tool will calculate pixels-per-meter ratio
    5. Use this value with --pixels-per-meter in track_and_analyze.py
"""

import argparse
import cv2
import numpy as np
import json


class CourtCalibrator:
    def __init__(self, image):
        self.image = image.copy()
        self.display_image = image.copy()
        self.points = []
        self.window_name = "Court Calibration - Click two points on a known distance"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                
                # Draw point
                cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(self.display_image, f"P{len(self.points)}", 
                           (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)
                
                # Draw line between points
                if len(self.points) == 2:
                    cv2.line(self.display_image, self.points[0], self.points[1], 
                            (0, 255, 0), 2)
                    
                    # Calculate pixel distance
                    pixel_dist = np.sqrt(
                        (self.points[1][0] - self.points[0][0])**2 + 
                        (self.points[1][1] - self.points[0][1])**2
                    )
                    
                    # Display distance
                    mid_point = (
                        (self.points[0][0] + self.points[1][0]) // 2,
                        (self.points[0][1] + self.points[1][1]) // 2
                    )
                    cv2.putText(self.display_image, f"{pixel_dist:.1f}px", 
                               mid_point, cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 0, 0), 2)
                
                cv2.imshow(self.window_name, self.display_image)
    
    def calibrate(self):
        """Run the calibration interface."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display instructions
        instructions = self.image.copy()
        cv2.putText(instructions, "CALIBRATION INSTRUCTIONS:", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(instructions, "1. Click two points on a known distance", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "2. Press 'r' to reset points", (10, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "3. Press 'c' to continue after selecting", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "4. Press 'q' to quit", (10, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(instructions, "Press any key to start...", (10, 170), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, instructions)
        cv2.waitKey(0)
        
        cv2.imshow(self.window_name, self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                cv2.destroyAllWindows()
                return None
            
            elif key == ord('r'):
                # Reset
                self.points = []
                self.display_image = self.image.copy()
                cv2.imshow(self.window_name, self.display_image)
            
            elif key == ord('c') and len(self.points) == 2:
                # Continue with calibration
                cv2.destroyAllWindows()
                return self.calculate_calibration()
    
    def calculate_calibration(self):
        """Calculate pixels per meter based on user input."""
        if len(self.points) != 2:
            print("Error: Need exactly 2 points")
            return None
        
        # Calculate pixel distance
        pixel_dist = np.sqrt(
            (self.points[1][0] - self.points[0][0])**2 + 
            (self.points[1][1] - self.points[0][1])**2
        )
        
        print("\n" + "="*60)
        print("CALIBRATION")
        print("="*60)
        print(f"Point 1: {self.points[0]}")
        print(f"Point 2: {self.points[1]}")
        print(f"Pixel distance: {pixel_dist:.2f} pixels")
        print("\nCommon tennis court distances:")
        print("  - Baseline to baseline: 23.77 meters")
        print("  - Service line to baseline: 5.49 meters")
        print("  - Service line to service line: 12.80 meters")
        print("  - Singles sideline to sideline: 8.23 meters")
        print("  - Doubles sideline to sideline: 10.97 meters")
        print("  - Service box width (half court): 6.40 meters")
        print("="*60)
        
        while True:
            try:
                real_distance = float(input("\nEnter the real-world distance in METERS: "))
                if real_distance <= 0:
                    print("Distance must be positive!")
                    continue
                break
            except ValueError:
                print("Invalid input! Please enter a number.")
        
        # Calculate pixels per meter
        pixels_per_meter = pixel_dist / real_distance
        
        print("\n" + "="*60)
        print("CALIBRATION RESULTS")
        print("="*60)
        print(f"Real-world distance: {real_distance:.2f} meters")
        print(f"Pixel distance: {pixel_dist:.2f} pixels")
        print(f"Pixels per meter: {pixels_per_meter:.2f}")
        print("="*60)
        
        # Save calibration
        calibration_data = {
            'point1': self.points[0],
            'point2': self.points[1],
            'pixel_distance': float(pixel_dist),
            'real_distance_meters': float(real_distance),
            'pixels_per_meter': float(pixels_per_meter)
        }
        
        with open('calibration.json', 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        print("\nCalibration saved to 'calibration.json'")
        print("\nTo use this calibration, run:")
        print(f"  py -3.12 track_and_analyze.py --weights <weights> --source <video> --output <output> --pixels-per-meter {pixels_per_meter:.2f}")
        
        return calibration_data


def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate court for accurate speed measurements')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--frame', type=int, default=0, help='Frame number to use for calibration (default: 0)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return
    
    # Set frame position
    if args.frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    
    # Read frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not read frame {args.frame} from video")
        return
    
    print(f"\nLoaded frame {args.frame} from {args.video}")
    print(f"Frame size: {frame.shape[1]}x{frame.shape[0]}")
    
    # Run calibration
    calibrator = CourtCalibrator(frame)
    result = calibrator.calibrate()
    
    if result:
        print("\n✓ Calibration complete!")
    else:
        print("\nCalibration cancelled.")


if __name__ == '__main__':
    main()
