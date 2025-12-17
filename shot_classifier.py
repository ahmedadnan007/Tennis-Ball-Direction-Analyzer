"""Shot direction and type classification module for tennis ball analysis.

This module classifies tennis shots based on trajectory analysis:
- Direction: Straight, Cross-court, Down-the-line
- Type: Groundstroke, Lob, Drop shot, Serve
- Court zones: Forehand side, backhand side, center

Usage:
    classifier = ShotClassifier(court_width=1280, court_height=720)
    shot_type = classifier.classify_shot(trajectory_analysis)
"""

import numpy as np
from typing import Dict, Tuple, List, Optional


class ShotClassifier:
    """Classifies tennis shots based on trajectory and movement patterns."""
    
    def __init__(self, court_width=1280, court_height=720, 
                 court_orientation='horizontal', court_detector=None,
                 player_handed='right'):
        """Initialize shot classifier.
        
        Args:
            court_width: Width of video frame in pixels
            court_height: Height of video frame in pixels
            court_orientation: 'horizontal' or 'vertical' court orientation
            court_detector: CourtDetector instance for line detection
            player_handed: 'right' or 'left' for player handedness
        """
        self.court_width = court_width
        self.court_height = court_height
        self.court_orientation = court_orientation
        self.court_detector = court_detector
        self.player_handed = player_handed
        
        # Define court zones (divide court into regions)
        self.zones = self._define_court_zones()
        
    def _define_court_zones(self):
        """Define court zones for shot classification."""
        # Divide court into 9 zones (3x3 grid)
        zones = {}
        
        h_third = self.court_height // 3
        w_third = self.court_width // 3
        
        zone_names = [
            ['back_left', 'back_center', 'back_right'],
            ['mid_left', 'mid_center', 'mid_right'],
            ['front_left', 'front_center', 'front_right']
        ]
        
        for row in range(3):
            for col in range(3):
                zones[zone_names[row][col]] = {
                    'x_min': col * w_third,
                    'x_max': (col + 1) * w_third,
                    'y_min': row * h_third,
                    'y_max': (row + 1) * h_third
                }
        
        return zones
    
    def get_zone(self, position):
        """Get court zone for a given position.
        
        Args:
            position: (x, y) position in pixels
        
        Returns:
            Zone name string
        """
        x, y = position
        
        for zone_name, zone_bounds in self.zones.items():
            if (zone_bounds['x_min'] <= x < zone_bounds['x_max'] and
                zone_bounds['y_min'] <= y < zone_bounds['y_max']):
                return zone_name
        
        return 'unknown'
    
    def classify_direction(self, start_pos, end_pos):
        """Classify shot direction based on start and end positions.
        
        Args:
            start_pos: (x, y) starting position
            end_pos: (x, y) ending position
        
        Returns:
            Direction classification string
        """
        x1, y1 = start_pos
        x2, y2 = end_pos
        
        # Calculate movement
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate angle
        angle = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle)
        
        # Horizontal movement classification
        if abs(dx) < self.court_width * 0.2:
            horizontal = 'straight'
        elif dx > 0:
            horizontal = 'cross_court_right'
        else:
            horizontal = 'cross_court_left'
        
        # Vertical movement classification
        if abs(dy) < self.court_height * 0.15:
            vertical = 'flat'
        elif dy > 0:
            vertical = 'deep'  # Moving toward baseline
        else:
            vertical = 'short'  # Moving toward net
        
        # Combine classifications
        if horizontal == 'straight':
            if vertical == 'flat':
                return 'straight'
            elif vertical == 'deep':
                return 'down_the_line_deep'
            else:
                return 'down_the_line_short'
        else:
            if vertical == 'flat':
                return horizontal
            elif vertical == 'deep':
                return f"{horizontal}_deep"
            else:
                return f"{horizontal}_short"
    
    def classify_shot_type(self, trajectory_analysis):
        """Classify shot type based on apex height.
        
        Classification based on ball apex height:
        - Lob: 6-10m apex (very high trajectory)
        - Drop Shot: 1.2-1.8m apex (very low trajectory)
        - Cross-Court Groundstroke: 2.5-4.0m apex (medium-high)
        - Down-the-Line Groundstroke: 2.0-3.0m apex (medium-low)
        
        Args:
            trajectory_analysis: Output from TrajectoryAnalyzer.analyze_trajectory()
        
        Returns:
            Shot type string and height class
        """
        if not trajectory_analysis:
            return 'unknown', 'unknown'
        
        # Get apex height
        apex_info = trajectory_analysis.get('apex_height', {})
        if not apex_info:
            return 'unknown', 'unknown'
        
        apex_height_m = apex_info.get('apex_height_meters', 0)
        
        # Get direction for groundstroke classification
        direction = trajectory_analysis.get('shot_direction', 'unknown')
        
        # Height-based classification
        # 1. Lob Shot - Very High (6-10m)
        if 6.0 <= apex_height_m <= 10.0:
            return 'lob', 'very_high'
        
        # 2. Drop Shot - Very Low (1.2-1.8m)
        elif 1.2 <= apex_height_m <= 1.8:
            return 'drop_shot', 'very_low'
        
        # 3. Cross-Court Groundstroke - Medium-High (2.5-4.0m)
        elif 2.5 <= apex_height_m <= 4.0:
            if 'cross_court' in direction.lower():
                return 'cross_court_groundstroke', 'medium_high'
            else:
                return 'groundstroke', 'medium_high'
        
        # 4. Down-the-Line Groundstroke - Medium-Low (2.0-3.0m)
        elif 2.0 <= apex_height_m <= 3.0:
            if 'down_the_line' in direction.lower() or 'straight' in direction.lower():
                return 'down_the_line_groundstroke', 'medium_low'
            else:
                return 'groundstroke', 'medium_low'
        
        # Edge cases
        elif apex_height_m > 10.0:
            return 'high_lob', 'extremely_high'
        elif apex_height_m < 1.2:
            return 'low_shot', 'extremely_low'
        elif apex_height_m > 4.0:
            return 'lob', 'high'
        else:
            # 1.8 - 2.5m range (between drop and groundstroke)
            return 'slice', 'low'
    
    def classify_spin(self, trajectory_analysis):
        """Detect spin type based on trajectory curvature.
        
        Args:
            trajectory_analysis: Trajectory analysis dictionary
        
        Returns:
            'topspin', 'slice', 'flat', or 'unknown'
        """
        poly_fit = trajectory_analysis.get('polynomial_fit')
        if not poly_fit or 'coefficients' not in poly_fit:
            return 'unknown'
        
        coeffs = poly_fit['coefficients']
        if len(coeffs) < 3:
            return 'flat'
        
        # Quadratic coefficient indicates curvature
        a = coeffs[0]
        
        if abs(a) < 0.0001:
            return 'flat'
        elif a < -0.001:  # Downward curve
            return 'topspin'
        elif a > 0.001:   # Upward curve
            return 'slice'
        else:
            return 'flat'
    
    def classify_court_side(self, position):
        """Classify which side of court (forehand/backhand from camera view).
        
        Args:
            position: (x, y) position
        
        Returns:
            'left', 'center', or 'right'
        """
        x, y = position
        
        if x < self.court_width * 0.33:
            return 'left'
        elif x < self.court_width * 0.67:
            return 'center'
        else:
            return 'right'
    
    def classify_shot(self, trajectory_analysis):
        """Complete shot classification with height-based type and direction.
        
        Args:
            trajectory_analysis: Output from TrajectoryAnalyzer.analyze_trajectory()
        
        Returns:
            Dictionary with complete shot classification
        """
        if not trajectory_analysis or len(trajectory_analysis.get('positions_pixels', [])) < 2:
            return {
                'type': 'unknown',
                'direction': 'unknown',
                'start_zone': 'unknown',
                'end_zone': 'unknown',
                'side': 'unknown',
                'confidence': 0.0
            }
        
        positions = trajectory_analysis['positions_pixels']
        start_pos = positions[0]
        end_pos = positions[-1]
        
        # Get zones
        start_zone = self.get_zone(start_pos)
        end_zone = self.get_zone(end_pos)
        
        # Classify direction
        direction = self.classify_direction(start_pos, end_pos)
        
        # Store direction in trajectory_analysis for shot type classification
        trajectory_analysis['shot_direction'] = direction
        
        # Classify shot type (returns tuple: shot_type, height_class)
        shot_type, height_class = self.classify_shot_type(trajectory_analysis)
        
        # Classify court side
        side = self.classify_court_side(start_pos)
        
        # Get apex height info
        apex_info = trajectory_analysis.get('apex_height', {})
        apex_height_m = apex_info.get('apex_height_meters', 0)
        
        # Detect spin
        spin_type = self.classify_spin(trajectory_analysis)
        
        # Forehand/Backhand classification
        stroke = 'unknown'
        if self.court_detector:
            stroke = self.court_detector.get_forehand_backhand(
                start_pos[0], end_pos[0], 
                player_handed=self.player_handed
            )
        else:
            # Fallback: simple position-based classification
            court_center = self.court_width / 2
            if start_pos[0] < court_center - self.court_width * 0.1:
                stroke = 'backhand' if self.player_handed == 'right' else 'forehand'
            elif start_pos[0] > court_center + self.court_width * 0.1:
                stroke = 'forehand' if self.player_handed == 'right' else 'backhand'
        
        # Advanced court zone info
        advanced_zone = {}
        if self.court_detector:
            advanced_zone = self.court_detector.get_shot_zone_advanced(start_pos, end_pos)
        
        # Calculate confidence based on trajectory quality
        confidence = self._calculate_confidence(trajectory_analysis)
        
        return {
            'type': shot_type,
            'height_class': height_class,
            'apex_height_meters': apex_height_m,
            'direction': direction,
            'stroke': stroke,  # forehand/backhand
            'spin': spin_type,  # topspin/slice/flat
            'start_zone': start_zone,
            'end_zone': end_zone,
            'side': side,
            'confidence': confidence,
            'avg_speed_kmh': trajectory_analysis.get('avg_speed_kmh', 0),
            'max_speed_kmh': trajectory_analysis.get('max_speed_kmh', 0),
            'duration_seconds': trajectory_analysis.get('duration_seconds', 0),
            'advanced_zone': advanced_zone  # Court line-based zones
        }
    
    def _calculate_confidence(self, trajectory_analysis):
        """Calculate classification confidence score.
        
        Args:
            trajectory_analysis: Trajectory analysis dictionary
        
        Returns:
            Confidence score (0-1)
        """
        num_frames = trajectory_analysis.get('num_frames', 0)
        
        # More frames = higher confidence
        if num_frames < 5:
            return 0.3
        elif num_frames < 10:
            return 0.5
        elif num_frames < 20:
            return 0.7
        elif num_frames < 40:
            return 0.85
        else:
            return 0.95
    
    def get_shot_summary(self, classification):
        """Get human-readable shot summary.
        
        Args:
            classification: Output from classify_shot()
        
        Returns:
            String summary
        """
        if classification['type'] == 'unknown':
            return "Unknown shot"
        
        # Format shot type with height class
        shot_type = classification['type'].replace('_', ' ').title()
        height_class = classification.get('height_class', '').replace('_', ' ').title()
        apex_height = classification.get('apex_height_meters', 0)
        
        summary = f"{shot_type}"
        
        # Add height information
        if apex_height > 0:
            summary += f" ({height_class}, {apex_height:.1f}m apex)"
        
        # Add direction if significant
        if 'cross_court' in classification['direction']:
            summary += f" - {classification['direction'].replace('_', ' ').title()}"
        elif classification['direction'] != 'straight':
            summary += f" ({classification['direction'].replace('_', ' ')})"
        
        # Add speed info
        if classification['avg_speed_kmh'] > 0:
            summary += f" @ {classification['avg_speed_kmh']:.1f} km/h"
        
        return summary
    
    def analyze_rally(self, trajectory_analyses):
        """Analyze a sequence of shots (rally).
        
        Args:
            trajectory_analyses: List of trajectory analysis dictionaries
        
        Returns:
            Rally statistics dictionary
        """
        if not trajectory_analyses:
            return {}
        
        classifications = [self.classify_shot(traj) for traj in trajectory_analyses]
        
        # Count shot types
        shot_types = {}
        directions = {}
        
        for cls in classifications:
            shot_type = cls['type']
            direction = cls['direction']
            
            shot_types[shot_type] = shot_types.get(shot_type, 0) + 1
            directions[direction] = directions.get(direction, 0) + 1
        
        # Calculate rally statistics
        speeds = [cls['avg_speed_kmh'] for cls in classifications if cls['avg_speed_kmh'] > 0]
        
        return {
            'num_shots': len(classifications),
            'shot_types': shot_types,
            'directions': directions,
            'avg_speed': np.mean(speeds) if speeds else 0,
            'max_speed': np.max(speeds) if speeds else 0,
            'min_speed': np.min(speeds) if speeds else 0,
            'classifications': classifications
        }
