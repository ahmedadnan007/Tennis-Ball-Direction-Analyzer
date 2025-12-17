"""Export shot data to CSV for machine learning training.

This module exports trajectory and shot classification data to CSV format
for training machine learning models.

Usage:
    exporter = ShotDataExporter('shots_data.csv')
    exporter.add_shot(trajectory_analysis, shot_classification)
    exporter.save()
"""

import csv
import pandas as pd
from typing import Dict, List
import os


class ShotDataExporter:
    """Exports shot data to CSV for ML training."""
    
    def __init__(self, output_file='shot_data.csv'):
        """Initialize exporter.
        
        Args:
            output_file: Output CSV filename
        """
        self.output_file = output_file
        self.shots = []
        
        # Define CSV columns
        self.columns = [
            # Identifiers
            'track_id', 'frame_start', 'frame_end', 'num_frames',
            
            # Trajectory features
            'apex_height_meters', 'total_distance_meters', 'duration_seconds',
            'avg_speed_kmh', 'max_speed_kmh', 'avg_speed_mps', 'max_speed_mps',
            
            # Position features
            'start_x', 'start_y', 'end_x', 'end_y',
            'start_zone', 'end_zone', 'court_side',
            
            # Movement features
            'displacement_x', 'displacement_y', 'displacement_angle_deg',
            'horizontal_direction', 'vertical_direction',
            
            # Curvature features
            'poly_coef_a', 'poly_coef_b', 'poly_coef_c',
            'trajectory_r2', 'num_bounces',
            
            # Advanced features
            'crosses_net', 'in_service_box', 'net_clearance_pixels',
            
            # Labels
            'shot_type', 'height_class', 'direction', 'stroke', 'spin',
            'confidence'
        ]
    
    def add_shot(self, trajectory_analysis, shot_classification):
        """Add a shot to the dataset.
        
        Args:
            trajectory_analysis: Trajectory analysis dictionary
            shot_classification: Shot classification dictionary
        """
        if not trajectory_analysis or not shot_classification:
            return
        
        # Extract trajectory features
        positions = trajectory_analysis.get('positions_pixels', [])
        if len(positions) < 2:
            return
        
        start_pos = positions[0]
        end_pos = positions[-1]
        
        # Calculate displacement
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        
        # Get polynomial coefficients
        poly_fit = trajectory_analysis.get('polynomial_fit', {})
        coeffs = poly_fit.get('coefficients', [0, 0, 0])
        if len(coeffs) < 3:
            coeffs = list(coeffs) + [0] * (3 - len(coeffs))
        
        # Advanced zone info
        adv_zone = shot_classification.get('advanced_zone', {})
        
        # Create data row
        shot_data = {
            # Identifiers
            'track_id': trajectory_analysis.get('track_id', 0),
            'frame_start': trajectory_analysis.get('frames', [0])[0],
            'frame_end': trajectory_analysis.get('frames', [0])[-1],
            'num_frames': trajectory_analysis.get('num_frames', 0),
            
            # Trajectory features
            'apex_height_meters': shot_classification.get('apex_height_meters', 0),
            'total_distance_meters': trajectory_analysis.get('total_distance_meters', 0),
            'duration_seconds': trajectory_analysis.get('duration_seconds', 0),
            'avg_speed_kmh': trajectory_analysis.get('avg_speed_kmh', 0),
            'max_speed_kmh': trajectory_analysis.get('max_speed_kmh', 0),
            'avg_speed_mps': trajectory_analysis.get('avg_speed_mps', 0),
            'max_speed_mps': trajectory_analysis.get('max_speed_mps', 0),
            
            # Position features
            'start_x': start_pos[0],
            'start_y': start_pos[1],
            'end_x': end_pos[0],
            'end_y': end_pos[1],
            'start_zone': shot_classification.get('start_zone', 'unknown'),
            'end_zone': shot_classification.get('end_zone', 'unknown'),
            'court_side': shot_classification.get('side', 'unknown'),
            
            # Movement features
            'displacement_x': dx,
            'displacement_y': dy,
            'displacement_angle_deg': trajectory_analysis.get('direction_angle_deg', 0),
            'horizontal_direction': self._get_horizontal_direction(shot_classification.get('direction', '')),
            'vertical_direction': self._get_vertical_direction(shot_classification.get('direction', '')),
            
            # Curvature features
            'poly_coef_a': coeffs[0],
            'poly_coef_b': coeffs[1],
            'poly_coef_c': coeffs[2],
            'trajectory_r2': poly_fit.get('r2_score', 0),
            'num_bounces': trajectory_analysis.get('num_bounces', 0),
            
            # Advanced features
            'crosses_net': adv_zone.get('crosses_net', False),
            'in_service_box': adv_zone.get('in_service_box', False),
            'net_clearance_pixels': adv_zone.get('net_clearance_pixels', 0),
            
            # Labels
            'shot_type': shot_classification.get('type', 'unknown'),
            'height_class': shot_classification.get('height_class', 'unknown'),
            'direction': shot_classification.get('direction', 'unknown'),
            'stroke': shot_classification.get('stroke', 'unknown'),
            'spin': shot_classification.get('spin', 'unknown'),
            'confidence': shot_classification.get('confidence', 0)
        }
        
        self.shots.append(shot_data)
    
    def _get_horizontal_direction(self, full_direction):
        """Extract horizontal component from direction."""
        if 'cross_court_left' in full_direction:
            return 'cross_court_left'
        elif 'cross_court_right' in full_direction:
            return 'cross_court_right'
        elif 'straight' in full_direction or 'down_the_line' in full_direction:
            return 'straight'
        else:
            return 'unknown'
    
    def _get_vertical_direction(self, full_direction):
        """Extract vertical component from direction."""
        if 'deep' in full_direction:
            return 'deep'
        elif 'short' in full_direction:
            return 'short'
        elif 'flat' in full_direction:
            return 'flat'
        else:
            return 'unknown'
    
    def save(self):
        """Save collected shots to CSV file."""
        if not self.shots:
            print("[WARNING] No shots to save")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.shots, columns=self.columns)
        
        # Save to CSV
        df.to_csv(self.output_file, index=False)
        print(f"[SUCCESS] Saved {len(self.shots)} shots to {self.output_file}")
        
        # Print summary statistics
        self._print_summary(df)
    
    def _print_summary(self, df):
        """Print summary statistics of collected data."""
        print("\n" + "="*60)
        print("DATASET SUMMARY")
        print("="*60)
        
        print(f"\nTotal Shots: {len(df)}")
        
        print("\nShot Types:")
        for shot_type, count in df['shot_type'].value_counts().items():
            print(f"  {shot_type}: {count}")
        
        print("\nStrokes:")
        for stroke, count in df['stroke'].value_counts().items():
            print(f"  {stroke}: {count}")
        
        print("\nSpin Types:")
        for spin, count in df['spin'].value_counts().items():
            print(f"  {spin}: {count}")
        
        print("\nDirections:")
        for direction, count in df['direction'].value_counts().head(5).items():
            print(f"  {direction}: {count}")
        
        print("\nAverage Speed: {:.1f} km/h".format(df['avg_speed_kmh'].mean()))
        print("Average Apex Height: {:.2f} m".format(df['apex_height_meters'].mean()))
        print("Average Confidence: {:.2f}".format(df['confidence'].mean()))
        
        print("="*60)
    
    def load(self, input_file=None):
        """Load existing CSV data.
        
        Args:
            input_file: CSV file to load (uses self.output_file if None)
        
        Returns:
            pandas DataFrame
        """
        file_path = input_file or self.output_file
        
        if not os.path.exists(file_path):
            print(f"[ERROR] File not found: {file_path}")
            return None
        
        df = pd.read_csv(file_path)
        print(f"[INFO] Loaded {len(df)} shots from {file_path}")
        
        return df
