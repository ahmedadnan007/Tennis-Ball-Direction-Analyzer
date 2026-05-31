"""Trajectory analysis module for tennis ball motion.

This module analyzes ball trajectories to:
- Calculate smooth ball paths using polynomial curve fitting
- Estimate ball speed using pixel-to-meter conversion and frame rates
- Store and analyze trajectory history for shot classification
- Detect bounce points and trajectory changes
- Detect serve shots (NEW)

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
                 court_width_meters=8.23, court_width_pixels=400, homography=None):
        self.fps = fps
        self.dt = 1.0 / fps
        self.homography = homography
        self.px_to_m_x = court_length_meters / court_length_pixels
        self.px_to_m_y = court_width_meters / court_width_pixels
        self.trajectories = []
    
    def set_homography(self, homography):
        self.homography = homography
    
    def transform_point(self, point):
        if self.homography is not None:
            point_h = np.array([point[0], point[1], 1.0], dtype=np.float32)
            transformed_h = self.homography @ point_h
            if transformed_h[2] != 0:
                return np.array([
                    transformed_h[0] / transformed_h[2],
                    transformed_h[1] / transformed_h[2]
                ])
            else:
                return np.array([point[0] * self.px_to_m_x, point[1] * self.px_to_m_y])
        else:
            return np.array([point[0] * self.px_to_m_x, point[1] * self.px_to_m_y])
    
    def pixels_to_meters(self, px_x, px_y):
        return tuple(self.transform_point((px_x, px_y)))
    
    def calculate_velocity(self, positions, times=None):
        if len(positions) < 2:
            return np.array([])
        
        positions = np.array(positions)
        
        if times is None:
            times = np.arange(len(positions)) * self.dt
        
        velocities = np.zeros_like(positions)
        velocities[0] = (positions[1] - positions[0]) / self.dt
        
        for i in range(1, len(positions) - 1):
            velocities[i] = (positions[i+1] - positions[i-1]) / (2 * self.dt)
        
        velocities[-1] = (positions[-1] - positions[-2]) / self.dt
        
        return velocities
    
    def calculate_speed(self, velocities):
        if len(velocities) == 0:
            return np.array([])
        
        velocities = np.array(velocities)
        speeds = np.linalg.norm(velocities, axis=1)
        
        if len(speeds) >= 5:
            speeds = medfilt(speeds, kernel_size=5)
        
        if len(speeds) >= 4:
            q1 = np.percentile(speeds, 25)
            q3 = np.percentile(speeds, 75)
            iqr = q3 - q1
            upper_bound = q3 + 2.0 * iqr
            max_realistic_speed = min(upper_bound, 70.0)
            speeds = np.clip(speeds, 0, max_realistic_speed)
        
        return speeds
    
    def fit_trajectory_polynomial(self, positions, degree=2):
        if len(positions) < degree + 1:
            return None
        
        positions = np.array(positions)
        x = positions[:, 0]
        y = positions[:, 1]
        
        try:
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = poly(x_smooth)
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
        if len(positions) < 4:
            return None
        
        positions = np.array(positions)
        t = np.zeros(len(positions))
        for i in range(1, len(positions)):
            t[i] = t[i-1] + np.linalg.norm(positions[i] - positions[i-1])
        
        if t[-1] == 0:
            return None
        
        try:
            k = min(3, len(positions) - 1)
            spline_x = UnivariateSpline(t, positions[:, 0], k=k, s=smoothing)
            spline_y = UnivariateSpline(t, positions[:, 1], k=k, s=smoothing)
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
        if len(positions_px) < 2:
            return None
        
        y_coords_px = positions_px[:, 1]
        apex_idx = np.argmin(y_coords_px)
        lowest_y_px = np.max(y_coords_px)
        apex_y_px = y_coords_px[apex_idx]
        apex_height_px = lowest_y_px - apex_y_px
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
        if len(velocities) < 3:
            return []
        
        velocities = np.array(velocities)
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        bounce_points = []
        
        for i in range(1, len(angles) - 1):
            angle_change = abs(angles[i+1] - angles[i-1])
            angle_change = min(angle_change, 2*np.pi - angle_change)
            if angle_change > np.pi / 4:
                bounce_points.append(i)
        
        return bounce_points

    # ================================================================
    # NEW: Serve Detection Function
    # Serve ki 4 properties hoti hain:
    # 1. Ball upar se neeche aati hai (high Y movement)
    # 2. Fast speed (100+ km/h)
    # 3. Ball court ke end se start hoti hai
    # 4. Mostly vertical movement hota hai
    # ================================================================
    def detect_serve(self, positions_px, speeds_kmh, frame_width, frame_height):
        """
        Serve detect karo trajectory se.

        Serve ki pehchan:
        - Ball upar se neeche aati hai (dy > 0, kyunki image mein y upar se badta hai)
        - Speed 100+ km/h hoti hai
        - Mostly vertical movement (dy > dx)
        - Ball court ke peeche se start hoti hai (top ya bottom 30% of frame)

        Args:
            positions_px: Ball positions pixels mein
            speeds_kmh: Speed array km/h mein
            frame_width: Video width
            frame_height: Video height

        Returns:
            True agar serve hai, False agar nahi
        """
        if len(positions_px) < 4:
            return False

        positions_px = np.array(positions_px)

        start_pos = positions_px[0]
        end_pos = positions_px[-1]

        dx = abs(end_pos[0] - start_pos[0])   # Horizontal movement
        dy = end_pos[1] - start_pos[1]          # Vertical movement (+ = downward)

        # Serve mein ball NEECHE aati hai
        if dy <= 0:
            return False

        # Vertical movement horizontal se zyada hona chahiye (serve mostly neeche aati)
        if dy < dx * 0.5:
            return False

        # Speed check: serve 80+ km/h hoti hai
        avg_speed = np.mean(speeds_kmh) if len(speeds_kmh) > 0 else 0
        if avg_speed < 80:
            return False

        # Ball start position: court ke upar ya neeche 35% mein honi chahiye
        start_y_ratio = start_pos[1] / frame_height
        if not (start_y_ratio < 0.35 or start_y_ratio > 0.65):
            return False

        # Ball ka starting X court ke side pe hona chahiye (not center)
        start_x_ratio = start_pos[0] / frame_width
        if 0.35 < start_x_ratio < 0.65:
            # Center se serve possible hai but less likely
            # Allow if speed is very high
            if avg_speed < 120:
                return False

        return True

    def analyze_trajectory(self, track_history, track_id=None):
        """Comprehensive trajectory analysis with serve detection."""
        if len(track_history) < 3:
            return None
        
        frames = [h['frame'] for h in track_history]
        positions_px = np.array([h['center'] for h in track_history])
        
        # Smooth positions
        if len(positions_px) >= 7:
            positions_px[:, 0] = savgol_filter(positions_px[:, 0], window_length=5, polyorder=2)
            positions_px[:, 1] = savgol_filter(positions_px[:, 1], window_length=5, polyorder=2)
        
        # Convert to meters
        positions_m = np.array([
            self.pixels_to_meters(p[0], p[1]) for p in positions_px
        ])
        
        # Calculate velocities and speeds
        velocities_m = self.calculate_velocity(positions_m)
        speeds_m = self.calculate_speed(velocities_m) if len(velocities_m) > 0 else []
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
        
        # Speeds in km/h for serve detection
        speeds_kmh_arr = np.array(speeds_m) * 3.6 if len(speeds_m) > 0 else np.array([])

        # Frame dimensions (estimate from positions if not available)
        # Use max observed coordinates as rough frame size
        frame_width = float(np.max(positions_px[:, 0]) * 2.5)
        frame_height = float(np.max(positions_px[:, 1]) * 2.5)

        # NEW: Serve detect karo
        is_serve = self.detect_serve(
            positions_px,
            speeds_kmh_arr,
            frame_width,
            frame_height
        )

        # Total distance
        distances = np.linalg.norm(np.diff(positions_m, axis=0), axis=1)
        total_distance = np.sum(distances) if len(distances) > 0 else 0
        
        # Duration
        duration = (frames[-1] - frames[0]) / self.fps
        
        # Direction
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

            # NEW: Serve detection
            'is_serve': is_serve,
        }
        
        self.trajectories.append(analysis)
        return analysis
    
    def classify_trajectory_shape(self, analysis):
        if not analysis or not analysis.get('polynomial_fit'):
            return 'unknown'
        
        # NEW: Serve check pehle karo
        if analysis.get('is_serve', False):
            return 'serve'

        poly_fit = analysis['polynomial_fit']
        
        if poly_fit['degree'] >= 2:
            a = poly_fit['coefficients'][0]
            if abs(a) < 0.0001:
                return 'flat'
            elif a > 0:
                return 'lob'
            else:
                return 'topspin'
        
        return 'linear'
    
    def get_trajectory_summary(self, analysis):
        if not analysis:
            return "No trajectory data"
        
        shape = self.classify_trajectory_shape(analysis)

        # NEW: Serve ko summary mein show karo
        serve_label = " [SERVE DETECTED]" if analysis.get('is_serve', False) else ""
        
        summary = f"""
Trajectory Analysis (Track {analysis['track_id']}){serve_label}:
- Duration: {analysis['duration_seconds']:.2f}s ({analysis['num_frames']} frames)
- Average Speed: {analysis['avg_speed_kmh']:.1f} km/h ({analysis['avg_speed_mps']:.1f} m/s)
- Max Speed: {analysis['max_speed_kmh']:.1f} km/h ({analysis['max_speed_mps']:.1f} m/s)
- Total Distance: {analysis['total_distance_meters']:.2f} m
- Direction: {analysis['direction_angle_deg']:.1f}°
- Shape: {shape}
- Bounces: {analysis['num_bounces']}
- Is Serve: {analysis.get('is_serve', False)}
        """
        
        return summary.strip()
    
    def smooth_positions(self, positions, window_length=5, poly_order=2):
        if len(positions) < window_length:
            return positions
        
        positions = np.array(positions)
        
        if window_length % 2 == 0:
            window_length += 1
        
        try:
            smoothed_x = savgol_filter(positions[:, 0], window_length, poly_order)
            smoothed_y = savgol_filter(positions[:, 1], window_length, poly_order)
            return np.column_stack([smoothed_x, smoothed_y])
        except:
            return positions