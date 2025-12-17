"""Trajectory analysis module for tennis ball motion.

This module analyzes ball trajectories to:
- Calculate smooth ball paths using polynomial curve fitting
- Estimate ball speed using pixel-to-meter conversion and frame rates
- Store and analyze trajectory history for shot classification
- Detect bounce points and trajectory changes

Usage:
    analyzer = TrajectoryAnalyzer(
        fps=30, 
        court_length_meters=23.77,
        court_length_pixels=800
    )
    
    # Analyze a trajectory
    analysis = analyzer.analyze_trajectory(track_history)
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter, medfilt
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class TrajectoryAnalyzer:
    """Analyzes tennis ball trajectories for speed, direction, and motion patterns."""
    
    def __init__(self, fps=30, court_length_meters=23.77, court_length_pixels=800,
                 court_width_meters=8.23, court_width_pixels=400):
        """Initialize trajectory analyzer.
        
        Args:
            fps: Video frame rate
            court_length_meters: Real tennis court length in meters (baseline to baseline)
            court_length_pixels: Court length in video pixels
            court_width_meters: Real tennis court width in meters (singles court)
            court_width_pixels: Court width in video pixels
        """
        self.fps = fps
        self.dt = 1.0 / fps  # Time between frames in seconds
        
        # Pixel to meter conversion factors
        self.px_to_m_x = court_length_meters / court_length_pixels
        self.px_to_m_y = court_width_meters / court_width_pixels
        
        # Storage for analyzed trajectories
        self.trajectories = []
    
    def pixels_to_meters(self, px_x, px_y):
        """Convert pixel coordinates to meters.
        
        Args:
            px_x: X coordinate in pixels
            px_y: Y coordinate in pixels
        
        Returns:
            (x_meters, y_meters)
        """
        return px_x * self.px_to_m_x, px_y * self.px_to_m_y
    
    def calculate_velocity(self, positions, times=None):
        """Calculate velocity from position history.
        
        Args:
            positions: Array of (x, y) positions
            times: Array of time points (optional, uses frame intervals if None)
        
        Returns:
            Array of (vx, vy) velocities
        """
        if len(positions) < 2:
            return np.array([])
        
        positions = np.array(positions)
        
        if times is None:
            times = np.arange(len(positions)) * self.dt
        
        # Calculate velocity using central differences
        velocities = np.zeros_like(positions)
        
        # Forward difference for first point
        velocities[0] = (positions[1] - positions[0]) / self.dt
        
        # Central difference for middle points
        for i in range(1, len(positions) - 1):
            velocities[i] = (positions[i+1] - positions[i-1]) / (2 * self.dt)
        
        # Backward difference for last point
        velocities[-1] = (positions[-1] - positions[-2]) / self.dt
        
        return velocities
    
    def calculate_speed(self, velocities):
        """Calculate speed magnitude from velocity vectors.
        
        Args:
            velocities: Array of (vx, vy) velocity vectors
        
        Returns:
            Array of speed magnitudes with outliers filtered
        """
        if len(velocities) == 0:
            return np.array([])
        
        velocities = np.array(velocities)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Apply median filter to remove spikes (window size 5 frames)
        if len(speeds) >= 5:
            speeds = medfilt(speeds, kernel_size=5)
        
        # Remove outliers using IQR method
        if len(speeds) >= 4:
            q1 = np.percentile(speeds, 25)
            q3 = np.percentile(speeds, 75)
            iqr = q3 - q1
            upper_bound = q3 + 2.0 * iqr  # More conservative: 2.0 instead of 1.5
            
            # Cap speeds at upper bound (realistic max tennis ball speed ~70 m/s or 252 km/h)
            max_realistic_speed = min(upper_bound, 70.0)  # 70 m/s = 252 km/h
            speeds = np.clip(speeds, 0, max_realistic_speed)
        
        return speeds
    
    def fit_trajectory_polynomial(self, positions, degree=2):
        """Fit polynomial curve to trajectory.
        
        Args:
            positions: Array of (x, y) positions
            degree: Polynomial degree (2 for parabolic, 3 for cubic)
        
        Returns:
            Dictionary with fitted parameters and predictions
        """
        if len(positions) < degree + 1:
            return None
        
        positions = np.array(positions)
        x = positions[:, 0]
        y = positions[:, 1]
        
        # Fit polynomial y = f(x)
        try:
            # Use polyfit for x->y mapping
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            
            # Generate smooth predictions
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = poly(x_smooth)
            
            # Calculate R² score
            y_pred = poly(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'coefficients': coeffs,
                'polynomial': poly,
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'r2_score': r2,
                'degree': degree
            }
        except:
            return None
    
    def fit_trajectory_spline(self, positions, smoothing=None):
        """Fit smooth spline to trajectory.
        
        Args:
            positions: Array of (x, y) positions
            smoothing: Smoothing parameter (None for auto)
        
        Returns:
            Dictionary with spline predictions
        """
        if len(positions) < 4:
            return None
        
        positions = np.array(positions)
        
        # Create parameter t (arc length approximation)
        t = np.zeros(len(positions))
        for i in range(1, len(positions)):
            t[i] = t[i-1] + np.linalg.norm(positions[i] - positions[i-1])
        
        if t[-1] == 0:
            return None
        
        try:
            # Fit spline for x(t) and y(t)
            k = min(3, len(positions) - 1)  # Spline degree
            
            spline_x = UnivariateSpline(t, positions[:, 0], k=k, s=smoothing)
            spline_y = UnivariateSpline(t, positions[:, 1], k=k, s=smoothing)
            
            # Generate smooth trajectory
            t_smooth = np.linspace(t[0], t[-1], 100)
            x_smooth = spline_x(t_smooth)
            y_smooth = spline_y(t_smooth)
            
            return {
                'spline_x': spline_x,
                'spline_y': spline_y,
                't_smooth': t_smooth,
                'x_smooth': x_smooth,
                'y_smooth': y_smooth,
                'arc_length': t[-1]
            }
        except:
            return None
    
    def calculate_apex_height(self, positions_px, positions_m):
        """Calculate apex (maximum) height of ball trajectory.
        
        Args:
            positions_px: Array of (x, y) positions in pixels
            positions_m: Array of (x, y) positions in meters
        
        Returns:
            Dictionary with apex information:
                - apex_height_meters: Maximum vertical height in meters
                - apex_height_pixels: Maximum vertical height in pixels
                - apex_index: Index of apex point
                - apex_position: (x, y) position at apex
        """
        if len(positions_px) < 2:
            return None
        
        # Find minimum Y coordinate (top of screen = low Y value in image coordinates)
        # Since y-axis is inverted in images (0 at top), minimum y = highest point
        y_coords_px = positions_px[:, 1]
        apex_idx = np.argmin(y_coords_px)
        
        # Calculate height relative to lowest point in trajectory
        lowest_y_px = np.max(y_coords_px)
        apex_y_px = y_coords_px[apex_idx]
        apex_height_px = lowest_y_px - apex_y_px  # Height above lowest point
        
        # Convert to meters
        apex_height_m = apex_height_px * self.px_to_m_y
        
        return {
            'apex_height_meters': float(apex_height_m),
            'apex_height_pixels': float(apex_height_px),
            'apex_index': int(apex_idx),
            'apex_position_px': positions_px[apex_idx].tolist(),
            'apex_position_m': positions_m[apex_idx].tolist(),
            'lowest_y_pixels': float(lowest_y_px),
            'apex_y_pixels': float(apex_y_px)
        }
    
    def detect_bounce(self, positions, velocities, threshold_factor=2.0):
        """Detect potential bounce points in trajectory.
        
        Args:
            positions: Array of (x, y) positions
            velocities: Array of (vx, vy) velocities
            threshold_factor: Factor for detecting sudden direction changes
        
        Returns:
            List of bounce point indices
        """
        if len(velocities) < 3:
            return []
        
        velocities = np.array(velocities)
        
        # Calculate velocity direction changes
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        bounce_points = []
        for i in range(1, len(angles) - 1):
            # Detect sudden angle changes
            angle_change = abs(angles[i+1] - angles[i-1])
            
            # Normalize to [0, pi]
            angle_change = min(angle_change, 2*np.pi - angle_change)
            
            # If angle changes significantly (e.g., > 45 degrees)
            if angle_change > np.pi / 4:
                bounce_points.append(i)
        
        return bounce_points
    
    def analyze_trajectory(self, track_history, track_id=None):
        """Comprehensive trajectory analysis.
        
        Args:
            track_history: List of dicts with 'frame', 'center', 'bbox', 'confidence'
            track_id: Optional track identifier
        
        Returns:
            Dictionary with complete trajectory analysis
        """
        if len(track_history) < 3:
            return None
        
        # Extract positions and metadata
        frames = [h['frame'] for h in track_history]
        positions_px = np.array([h['center'] for h in track_history])
        
        # Smooth positions using Savitzky-Golay filter to reduce noise
        if len(positions_px) >= 7:
            positions_px[:, 0] = savgol_filter(positions_px[:, 0], window_length=5, polyorder=2)
            positions_px[:, 1] = savgol_filter(positions_px[:, 1], window_length=5, polyorder=2)
        
        # Convert to meters
        positions_m = np.array([
            self.pixels_to_meters(p[0], p[1]) for p in positions_px
        ])
        
        # Calculate velocities and speeds (in meters/second)
        velocities_m = self.calculate_velocity(positions_m)
        speeds_m = self.calculate_speed(velocities_m) if len(velocities_m) > 0 else []
        
        # Calculate velocities in pixels (for visualization)
        velocities_px = self.calculate_velocity(positions_px)
        
        # Fit trajectory curves
        poly_fit = self.fit_trajectory_polynomial(positions_px, degree=2)
        spline_fit = self.fit_trajectory_spline(positions_px)
        
        # Detect bounces
        bounce_indices = self.detect_bounce(positions_px, velocities_px)
        
        # Calculate apex height
        apex_info = self.calculate_apex_height(positions_px, positions_m)
        
        # Calculate statistics
        avg_speed = np.mean(speeds_m) if len(speeds_m) > 0 else 0
        max_speed = np.max(speeds_m) if len(speeds_m) > 0 else 0
        
        # Calculate total distance traveled
        distances = np.linalg.norm(np.diff(positions_m, axis=0), axis=1)
        total_distance = np.sum(distances) if len(distances) > 0 else 0
        
        # Duration
        duration = (frames[-1] - frames[0]) / self.fps
        
        # Overall direction (start to end vector)
        direction_vector = positions_m[-1] - positions_m[0] if len(positions_m) >= 2 else [0, 0]
        direction_angle = np.arctan2(direction_vector[1], direction_vector[0])
        
        analysis = {
            'track_id': track_id,
            'num_frames': len(track_history),
            'duration_seconds': duration,
            'frames': frames,
            
            # Positions
            'positions_pixels': positions_px.tolist(),
            'positions_meters': positions_m.tolist(),
            
            # Velocities and speeds
            'velocities_meters_per_sec': velocities_m.tolist() if len(velocities_m) > 0 else [],
            'speeds_meters_per_sec': speeds_m.tolist() if len(speeds_m) > 0 else [],
            'avg_speed_mps': float(avg_speed),
            'max_speed_mps': float(max_speed),
            'avg_speed_kmh': float(avg_speed * 3.6),
            'max_speed_kmh': float(max_speed * 3.6),
            
            # Distance
            'total_distance_meters': float(total_distance),
            
            # Direction
            'direction_vector': direction_vector.tolist(),
            'direction_angle_rad': float(direction_angle),
            'direction_angle_deg': float(np.degrees(direction_angle)),
            
            # Curve fitting
            'polynomial_fit': poly_fit,
            'spline_fit': spline_fit,
            
            # Apex height
            'apex_height': apex_info,
            
            # Bounce detection
            'bounce_points': bounce_indices,
            'num_bounces': len(bounce_indices),
        }
        
        # Store for later access
        self.trajectories.append(analysis)
        
        return analysis
    
    def classify_trajectory_shape(self, analysis):
        """Classify trajectory shape (flat, lob, topspin, etc.).
        
        Args:
            analysis: Output from analyze_trajectory()
        
        Returns:
            String classification
        """
        if not analysis or not analysis.get('polynomial_fit'):
            return 'unknown'
        
        poly_fit = analysis['polynomial_fit']
        
        if poly_fit['degree'] >= 2:
            # Get the quadratic coefficient (curvature)
            a = poly_fit['coefficients'][0]
            
            # Analyze curvature
            if abs(a) < 0.0001:
                return 'flat'
            elif a > 0:
                return 'lob'  # Upward curving
            else:
                return 'topspin'  # Downward curving
        
        return 'linear'
    
    def get_trajectory_summary(self, analysis):
        """Get human-readable summary of trajectory.
        
        Args:
            analysis: Output from analyze_trajectory()
        
        Returns:
            String summary
        """
        if not analysis:
            return "No trajectory data"
        
        shape = self.classify_trajectory_shape(analysis)
        
        summary = f"""
Trajectory Analysis (Track {analysis['track_id']}):
- Duration: {analysis['duration_seconds']:.2f}s ({analysis['num_frames']} frames)
- Average Speed: {analysis['avg_speed_kmh']:.1f} km/h ({analysis['avg_speed_mps']:.1f} m/s)
- Max Speed: {analysis['max_speed_kmh']:.1f} km/h ({analysis['max_speed_mps']:.1f} m/s)
- Total Distance: {analysis['total_distance_meters']:.2f} m
- Direction: {analysis['direction_angle_deg']:.1f}°
- Shape: {shape}
- Bounces: {analysis['num_bounces']}
        """
        
        return summary.strip()
    
    def smooth_positions(self, positions, window_length=5, poly_order=2):
        """Apply Savitzky-Golay filter to smooth positions.
        
        Args:
            positions: Array of (x, y) positions
            window_length: Window size (must be odd)
            poly_order: Polynomial order
        
        Returns:
            Smoothed positions
        """
        if len(positions) < window_length:
            return positions
        
        positions = np.array(positions)
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        try:
            smoothed_x = savgol_filter(positions[:, 0], window_length, poly_order)
            smoothed_y = savgol_filter(positions[:, 1], window_length, poly_order)
            return np.column_stack([smoothed_x, smoothed_y])
        except:
            return positions
