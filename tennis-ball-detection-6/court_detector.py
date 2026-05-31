"""Court line detection module for tennis court analysis.

This module detects tennis court lines to:
- Improve spatial calibration
- Detect shot zones more accurately
- Classify forehand vs backhand based on court position
- Track net position and service boxes

The module can use calibrated court geometry from calibration.json for high accuracy.

Usage:
    detector = CourtDetector()
    court_lines = detector.detect_lines(frame)
    net_position = detector.get_net_position()
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class CourtDetector:
    """Detects tennis court lines and key features."""
    
    def __init__(self, frame_width=1280, frame_height=720, use_calibration=True):
        """Initialize court detector.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
            use_calibration: Load calibration from calibration.json if available
        """
        self.width = frame_width
        self.height = frame_height
        self.court_lines = None
        self.net_position = None
        self.baseline_top = None
        self.baseline_bottom = None
        self.service_line_top = None
        self.service_line_bottom = None
        self.sideline_left = None
        self.sideline_right = None
        self.center_mark_x = None
        
        # Keep old naming for backward compatibility
        self.baseline_left = None
        self.baseline_right = None
        self.service_line = None
        self.sidelines = None
        
        # Calibration data
        self.calibration_data = None
        self.calibration_geometry = None
        self.is_calibrated = False
        
        if use_calibration:
            self._load_calibration()
    
    def _load_calibration(self):
        """Load calibration data from calibration.json if available."""
        calib_file = Path(__file__).parent / 'calibration.json'
        
        if not calib_file.exists():
            return
        
        try:
            with open(calib_file, 'r') as f:
                self.calibration_data = json.load(f)
            
            self.calibration_geometry = self.calibration_data.get('court_geometry', {})
            self.is_calibrated = True
            print(f"[OK] Loaded court calibration from {calib_file}")
            print(f"  Court dimensions: {self.calibration_geometry.get('court_width_pixels', 0):.0f}x{self.calibration_geometry.get('court_length_pixels', 0):.0f} px")
        except Exception as e:
            print(f"[WARNING] Failed to load calibration: {e}")
    
    def _scale_calibration_to_frame(self) -> Dict[str, float]:
        """Scale calibration values to current frame size."""
        if not self.is_calibrated or not self.calibration_geometry:
            return {}
        
        geom = self.calibration_geometry
        calib_width = self.calibration_data['image_size']['width']
        calib_height = self.calibration_data['image_size']['height']
        
        # Calculate scaling factors
        scale_x = self.width / calib_width if calib_width > 0 else 1.0
        scale_y = self.height / calib_height if calib_height > 0 else 1.0
        
        # Scale all court positions
        return {
            'baseline_top_y': geom.get('baseline_top_y', 0) * scale_y,
            'baseline_bottom_y': geom.get('baseline_bottom_y', 0) * scale_y,
            'net_y': geom.get('net_y', self.height / 2) * scale_y,
            'service_line_top_y': geom.get('service_line_top_y', 0) * scale_y,
            'service_line_bottom_y': geom.get('service_line_bottom_y', 0) * scale_y,
            'sideline_left_x': geom.get('sideline_left_x', 0) * scale_x,
            'sideline_right_x': geom.get('sideline_right_x', self.width) * scale_x,
            'center_mark_x': geom.get('center_mark_x', self.width / 2) * scale_x,
            'scale_x': scale_x,
            'scale_y': scale_y
        }
    
    def _apply_calibration(self):
        """Apply calibration data to set court positions."""
        if not self.is_calibrated:
            return False
        
        scaled = self._scale_calibration_to_frame()
        
        # Set positions from calibration
        self.baseline_top = int(scaled.get('baseline_top_y', self.height * 0.1))
        self.baseline_bottom = int(scaled.get('baseline_bottom_y', self.height * 0.9))
        self.net_position = int(scaled.get('net_y', self.height / 2))
        self.service_line_top = int(scaled.get('service_line_top_y', self.height * 0.3))
        self.service_line_bottom = int(scaled.get('service_line_bottom_y', self.height * 0.7))
        self.sideline_left = int(scaled.get('sideline_left_x', self.width * 0.1))
        self.sideline_right = int(scaled.get('sideline_right_x', self.width * 0.9))
        self.center_mark_x = int(scaled.get('center_mark_x', self.width / 2))
        
        # Legacy naming
        self.baseline_left = self.baseline_top
        self.baseline_right = self.baseline_bottom
        self.service_line = (self.service_line_top + self.service_line_bottom) / 2
        self.sidelines = (self.sideline_left, self.sideline_right)
        
        return True

    def get_scaled_court_dimensions(self):
        """Return the scaled court width and length in pixels for the current frame."""
        if not self.is_calibrated:
            return 0.0, 0.0
        scaled = self._scale_calibration_to_frame()
        width_px = max(0.0, float(scaled.get('sideline_right_x', 0) - scaled.get('sideline_left_x', 0)))
        length_px = max(0.0, float(scaled.get('baseline_bottom_y', 0) - scaled.get('baseline_top_y', 0)))
        return width_px, length_px

    def compute_homography(self) -> Optional[np.ndarray]:
        """Compute homography matrix from image court corners to real court coordinates.
        
        Returns:
            3x3 homography matrix, or None if corners cannot be determined
        """
        if not self.is_calibrated:
            return None
        
        scaled = self._scale_calibration_to_frame()
        
        # Get 4 corners of the court in image pixels
        top_left = np.array([
            [scaled.get('sideline_left_x', 0), scaled.get('baseline_top_y', 0)],
        ], dtype=np.float32)
        top_right = np.array([
            [scaled.get('sideline_right_x', self.width), scaled.get('baseline_top_y', 0)],
        ], dtype=np.float32)
        bottom_left = np.array([
            [scaled.get('sideline_left_x', 0), scaled.get('baseline_bottom_y', self.height)],
        ], dtype=np.float32)
        bottom_right = np.array([
            [scaled.get('sideline_right_x', self.width), scaled.get('baseline_bottom_y', self.height)],
        ], dtype=np.float32)
        
        image_corners = np.vstack([top_left, top_right, bottom_left, bottom_right])
        
        # Real court corners in meters (0,0) = top-left, (8.23, 23.77) = bottom-right
        court_width_m = 8.23  # singles court width
        court_length_m = 23.77  # baseline to baseline
        real_corners = np.array([
            [0, 0],  # top-left
            [court_width_m, 0],  # top-right
            [0, court_length_m],  # bottom-left
            [court_width_m, court_length_m]  # bottom-right
        ], dtype=np.float32)
        
        # Compute homography from image to real court
        try:
            H, _ = cv2.findHomography(image_corners, real_corners, cv2.RANSAC, 5.0)
            return H
        except Exception as e:
            print(f"[WARNING] Could not compute homography: {e}")
            return None

    def transform_point_homography(self, point: Tuple[float, float], homography: np.ndarray) -> Tuple[float, float]:
        """Transform a point using homography matrix.
        
        Args:
            point: (x, y) in image pixels
            homography: 3x3 homography matrix
        
        Returns:
            (x, y) in real court meters
        """
        if homography is None:
            return point
        
        # Convert point to homogeneous coordinates
        point_h = np.array([point[0], point[1], 1.0], dtype=np.float32)
        
        # Apply transformation
        transformed_h = homography @ point_h
        
        # Convert back from homogeneous coordinates
        if transformed_h[2] != 0:
            x_real = transformed_h[0] / transformed_h[2]
            y_real = transformed_h[1] / transformed_h[2]
            return (float(x_real), float(y_real))
        else:
            return point
        
    def detect_lines(self, frame, use_cache=True, force_calibration=False):
        """Detect court lines using calibration or edge detection.
        
        Args:
            frame: Input frame
            use_cache: Use cached lines if available
            force_calibration: Force use of calibration data if available
        
        Returns:
            Dictionary of detected lines
        """
        if use_cache and self.court_lines is not None:
            return self.court_lines
        
        # Try calibration first if available
        if self.is_calibrated and (force_calibration or not use_cache):
            if self._apply_calibration():
                self.court_lines = self._get_calibrated_court()
                return self.court_lines
        
        # Fall back to automatic detection
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None:
            # If auto-detection fails and we have calibration, use it
            if self.is_calibrated:
                self._apply_calibration()
                self.court_lines = self._get_calibrated_court()
            else:
                self.court_lines = self._get_default_court()
            return self.court_lines
        
        # Classify lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle
            angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Classify as horizontal or vertical
            if angle < 20 or angle > 160:  # Horizontal
                horizontal_lines.append((x1, y1, x2, y2))
            elif 70 < angle < 110:  # Vertical
                vertical_lines.append((x1, y1, x2, y2))
        
        # Find major horizontal lines (baselines, service lines, net)
        self._find_major_lines(horizontal_lines, vertical_lines)
        
        self.court_lines = {
            'horizontal': horizontal_lines,
            'vertical': vertical_lines,
            'net_y': self.net_position,
            'baseline_top_y': self.baseline_top,
            'baseline_bottom_y': self.baseline_bottom,
            'service_line_top_y': self.service_line_top,
            'service_line_bottom_y': self.service_line_bottom,
            'detection_method': 'hough_transform',
            'is_calibrated': False
        }
        
        return self.court_lines
    
    def _get_calibrated_court(self) -> Dict:
        """Get court lines from calibration data."""
        return {
            'horizontal': [],
            'vertical': [],
            'net_y': self.net_position,
            'baseline_top_y': self.baseline_top,
            'baseline_bottom_y': self.baseline_bottom,
            'service_line_top_y': self.service_line_top,
            'service_line_bottom_y': self.service_line_bottom,
            'sideline_left_x': self.sideline_left,
            'sideline_right_x': self.sideline_right,
            'center_mark_x': self.center_mark_x,
            'detection_method': 'calibration',
            'is_calibrated': True
        }
    
    def _find_major_lines(self, h_lines, v_lines):
        """Find major court lines (net, baselines, service lines)."""
        if not h_lines:
            self._set_default_positions()
            return
        
        # Sort horizontal lines by y position
        h_lines_sorted = sorted(h_lines, key=lambda l: (l[1] + l[3]) / 2)
        
        # Net is typically near middle
        middle_idx = len(h_lines_sorted) // 2
        if h_lines_sorted:
            net_line = h_lines_sorted[middle_idx]
            self.net_position = int((net_line[1] + net_line[3]) / 2)
        
        # Baselines are at top and bottom
        if len(h_lines_sorted) >= 2:
            top_line = h_lines_sorted[0]
            bottom_line = h_lines_sorted[-1]
            self.baseline_left = int((top_line[1] + top_line[3]) / 2)
            self.baseline_right = int((bottom_line[1] + bottom_line[3]) / 2)
        
        # Service line is between net and baseline
        if len(h_lines_sorted) >= 3:
            service_idx = middle_idx // 2 if middle_idx > 0 else 1
            service_line = h_lines_sorted[service_idx]
            self.service_line = int((service_line[1] + service_line[3]) / 2)
        
        if self.net_position is None:
            self._set_default_positions()
    
    def _set_default_positions(self):
        """Set default court positions if detection fails."""
        self.net_position = self.height // 2
        self.baseline_left = int(self.height * 0.1)
        self.baseline_right = int(self.height * 0.9)
        self.service_line = int(self.height * 0.3)
    
    def _get_default_court(self):
        """Get default court layout if detection fails."""
        self._set_default_positions()
        return {
            'horizontal': [],
            'vertical': [],
            'net_y': self.net_position,
            'baseline_left_y': self.baseline_left,
            'baseline_right_y': self.baseline_right,
            'service_line_y': self.service_line
        }
    
    def get_court_side(self, y_position):
        """Determine which side of court (top/bottom relative to net).
        
        Args:
            y_position: Y coordinate
        
        Returns:
            'top' or 'bottom'
        """
        if self.net_position is None:
            return 'unknown'
        
        return 'top' if y_position < self.net_position else 'bottom'
    
    def get_forehand_backhand(self, start_x, end_x, court_side='bottom', 
                             player_handed='right'):
        """Determine if shot is forehand or backhand.
        
        Args:
            start_x: Starting x position of ball
            end_x: Ending x position of ball
            court_side: 'top' or 'bottom' (which side player is on)
            player_handed: 'right' or 'left'
        
        Returns:
            'forehand', 'backhand', or 'unknown'
        """
        # Ball position relative to court center
        court_center_x = self.width / 2
        
        # Determine shot side based on ball starting position
        if start_x < court_center_x - self.width * 0.1:
            # Ball on left side
            ball_side = 'left'
        elif start_x > court_center_x + self.width * 0.1:
            # Ball on right side
            ball_side = 'right'
        else:
            # Ball in center - check movement direction
            if end_x < start_x:
                ball_side = 'left'
            elif end_x > start_x:
                ball_side = 'right'
            else:
                return 'unknown'
        
        # Classify based on handedness
        if player_handed == 'right':
            if ball_side == 'right':
                return 'forehand'
            else:
                return 'backhand'
        else:  # left-handed
            if ball_side == 'left':
                return 'forehand'
            else:
                return 'backhand'
    
    def is_in_service_box(self, x, y):
        """Check if position is in service box.
        
        Args:
            x, y: Position coordinates
        
        Returns:
            Boolean
        """
        if self.net_position is None or self.service_line is None:
            return False
        
        # Service box is between net and service line
        in_y_range = (min(self.net_position, self.service_line) < y < 
                      max(self.net_position, self.service_line))
        
        # Within court width (approximate)
        in_x_range = self.width * 0.2 < x < self.width * 0.8
        
        return in_y_range and in_x_range
    
    def draw_court_lines(self, frame, color=(0, 255, 0), thickness=2):
        """Draw detected court lines on frame.
        
        Args:
            frame: Frame to draw on
            color: Line color
            thickness: Line thickness
        
        Returns:
            Frame with court lines
        """
        if self.court_lines is None:
            return frame
        
        # Draw horizontal lines
        for line in self.court_lines['horizontal']:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw vertical lines
        for line in self.court_lines['vertical']:
            x1, y1, x2, y2 = line
            cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Highlight net position
        if self.net_position:
            cv2.line(frame, (0, self.net_position), 
                    (self.width, self.net_position), (255, 0, 0), 3)
            cv2.putText(frame, "NET", (10, self.net_position - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return frame
    
    def get_shot_zone_advanced(self, start_pos, end_pos):
        """Get advanced shot zone classification using court lines.
        
        Args:
            start_pos: (x, y) starting position
            end_pos: (x, y) ending position
        
        Returns:
            Dictionary with zone information
        """
        start_x, start_y = start_pos
        end_x, end_y = end_pos
        
        # Determine court sides
        start_side = self.get_court_side(start_y)
        end_side = self.get_court_side(end_y)
        
        # Check if in service box
        in_service_box = self.is_in_service_box(end_x, end_y)
        
        # Determine if shot crosses net
        crosses_net = start_side != end_side
        
        return {
            'start_side': start_side,
            'end_side': end_side,
            'crosses_net': crosses_net,
            'in_service_box': in_service_box,
            'net_clearance_pixels': abs(end_y - self.net_position) if self.net_position else None
        }
