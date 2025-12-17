"""Court line detection module for tennis court analysis.

This module detects tennis court lines to:
- Improve spatial calibration
- Detect shot zones more accurately
- Classify forehand vs backhand based on court position
- Track net position and service boxes

Usage:
    detector = CourtDetector()
    court_lines = detector.detect_lines(frame)
    net_position = detector.get_net_position()
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class CourtDetector:
    """Detects tennis court lines and key features."""
    
    def __init__(self, frame_width=1280, frame_height=720):
        """Initialize court detector.
        
        Args:
            frame_width: Video frame width
            frame_height: Video frame height
        """
        self.width = frame_width
        self.height = frame_height
        self.court_lines = None
        self.net_position = None
        self.baseline_left = None
        self.baseline_right = None
        self.service_line = None
        self.sidelines = None
        
    def detect_lines(self, frame, use_cache=True):
        """Detect court lines using edge detection and Hough transform.
        
        Args:
            frame: Input frame
            use_cache: Use cached lines if available
        
        Returns:
            Dictionary of detected lines
        """
        if use_cache and self.court_lines is not None:
            return self.court_lines
        
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
            return self._get_default_court()
        
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
            'baseline_left_y': self.baseline_left,
            'baseline_right_y': self.baseline_right,
            'service_line_y': self.service_line
        }
        
        return self.court_lines
    
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
