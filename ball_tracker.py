"""Ball tracking module using Kalman Filter for smooth trajectory estimation.

This module maintains consistent ball IDs across frames, handles missing detections,
and provides smooth trajectory predictions using a Kalman Filter.

Usage:
    tracker = BallTracker(max_missing_frames=10, min_confidence=0.3)
    for frame_detections in video_detections:
        tracked_balls = tracker.update(frame_detections)
"""

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.spatial.distance import cdist


class BallTrack:
    """Represents a single tracked ball with Kalman Filter."""
    
    _id_counter = 0
    
    def __init__(self, detection, frame_number):
        """Initialize a ball track.
        
        Args:
            detection: dict with keys 'bbox' (x1,y1,x2,y2), 'confidence', 'center'
            frame_number: Current frame number
        """
        self.id = BallTrack._id_counter
        BallTrack._id_counter += 1
        
        self.kf = self._init_kalman_filter(detection['center'])
        self.age = 0  # How many frames this track has existed
        self.time_since_update = 0  # Frames since last detection
        self.hits = 1  # Number of successful detections
        self.hit_streak = 1  # Consecutive detections
        
        self.history = []  # Store trajectory history
        self.confidence_history = []
        
        # Store current detection
        self.last_detection = detection
        self.last_frame = frame_number
        self.history.append({
            'frame': frame_number,
            'center': detection['center'],
            'bbox': detection['bbox'],
            'confidence': detection['confidence']
        })
        self.confidence_history.append(detection['confidence'])
    
    def _init_kalman_filter(self, initial_position):
        """Initialize Kalman Filter for 2D position and velocity tracking.
        
        State vector: [x, y, vx, vy] where (x,y) is position and (vx,vy) is velocity
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (assumes constant velocity model)
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position, not velocity)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise covariance
        kf.R *= 10.0
        
        # Process noise covariance
        kf.Q = np.eye(4) * 0.1
        kf.Q[2:, 2:] *= 5.0  # Higher uncertainty in velocity
        
        # Initial state covariance
        kf.P *= 100.0
        kf.P[2:, 2:] *= 1000.0  # Higher uncertainty in initial velocity
        
        # Initialize state with position (velocity starts at 0)
        kf.x = np.array([initial_position[0], initial_position[1], 0, 0])
        
        return kf
    
    def predict(self):
        """Predict the next state using Kalman Filter.
        
        Returns:
            Predicted center position (x, y)
        """
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:2]  # Return predicted position
    
    def update(self, detection, frame_number):
        """Update the track with a new detection.
        
        Args:
            detection: dict with keys 'bbox', 'confidence', 'center'
            frame_number: Current frame number
        """
        self.kf.update(np.array(detection['center']))
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        self.last_detection = detection
        self.last_frame = frame_number
        
        # Store in history
        self.history.append({
            'frame': frame_number,
            'center': detection['center'],
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'predicted': self.kf.x[:2].copy()
        })
        self.confidence_history.append(detection['confidence'])
    
    def get_state(self):
        """Get current state of the track.
        
        Returns:
            dict with current position, velocity, bbox, and metadata
        """
        pos = self.kf.x[:2]
        vel = self.kf.x[2:]
        
        # Estimate bbox size based on last detection
        if self.last_detection:
            bbox = self.last_detection['bbox']
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            predicted_bbox = [
                pos[0] - w/2,
                pos[1] - h/2,
                pos[0] + w/2,
                pos[1] + h/2
            ]
        else:
            predicted_bbox = [pos[0]-10, pos[1]-10, pos[0]+10, pos[1]+10]
        
        return {
            'id': self.id,
            'center': pos.tolist(),
            'velocity': vel.tolist(),
            'bbox': predicted_bbox,
            'age': self.age,
            'hits': self.hits,
            'hit_streak': self.hit_streak,
            'time_since_update': self.time_since_update,
            'avg_confidence': np.mean(self.confidence_history) if self.confidence_history else 0.0
        }
    
    def mark_missed(self):
        """Mark that this track was not matched in current frame."""
        self.time_since_update += 1
        self.hit_streak = 0


class BallTracker:
    """Multi-object tracker for tennis balls using Kalman Filter."""
    
    def __init__(self, max_missing_frames=10, min_confidence=0.25, 
                 max_distance=100, min_hits=3):
        """Initialize the ball tracker.
        
        Args:
            max_missing_frames: Maximum frames a track can be missing before deletion
            min_confidence: Minimum detection confidence to consider
            max_distance: Maximum pixel distance for association
            min_hits: Minimum hits before a track is considered confirmed
        """
        self.max_missing_frames = max_missing_frames
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        self.min_hits = min_hits
        
        self.tracks = []
        self.frame_count = 0
        
    def update(self, detections):
        """Update tracks with new detections.
        
        Args:
            detections: List of dicts with keys 'bbox', 'confidence'
                       bbox format: [x1, y1, x2, y2]
        
        Returns:
            List of tracked ball states (confirmed tracks only)
        """
        self.frame_count += 1
        
        # Filter detections by confidence
        valid_detections = [
            d for d in detections if d.get('confidence', 0) >= self.min_confidence
        ]
        
        # Add center to detections
        for det in valid_detections:
            bbox = det['bbox']
            det['center'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        
        # Predict next positions for existing tracks
        predictions = []
        for track in self.tracks:
            predictions.append(track.predict())
        
        # Associate detections with tracks
        matched_tracks, unmatched_detections, unmatched_tracks = \
            self._associate(valid_detections, predictions)
        
        # Update matched tracks
        for track_idx, det_idx in matched_tracks:
            self.tracks[track_idx].update(valid_detections[det_idx], self.frame_count)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            new_track = BallTrack(valid_detections[det_idx], self.frame_count)
            self.tracks.append(new_track)
        
        # Remove dead tracks (missing for too long)
        self.tracks = [
            t for t in self.tracks 
            if t.time_since_update < self.max_missing_frames
        ]
        
        # Return confirmed tracks (tracks with enough hits)
        confirmed_tracks = [
            t.get_state() for t in self.tracks 
            if t.hits >= self.min_hits or t.hit_streak >= 1
        ]
        
        return confirmed_tracks
    
    def _associate(self, detections, predictions):
        """Associate detections with predicted tracks using Hungarian algorithm.
        
        Args:
            detections: List of detection dicts
            predictions: List of predicted positions from Kalman Filter
        
        Returns:
            matched: List of (track_idx, detection_idx) tuples
            unmatched_dets: List of unmatched detection indices
            unmatched_tracks: List of unmatched track indices
        """
        if len(predictions) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(predictions)))
        
        # Compute cost matrix (Euclidean distance)
        det_centers = np.array([d['center'] for d in detections])
        pred_centers = np.array(predictions)
        
        cost_matrix = cdist(pred_centers, det_centers, metric='euclidean')
        
        # Use simple greedy matching (for more sophisticated, use scipy.optimize.linear_sum_assignment)
        matched_tracks = []
        matched_dets = set()
        
        # Greedy assignment: for each track, find closest detection
        for track_idx in range(len(predictions)):
            if len(matched_dets) >= len(detections):
                break
            
            # Find closest unmatched detection
            min_dist = float('inf')
            best_det_idx = -1
            
            for det_idx in range(len(detections)):
                if det_idx in matched_dets:
                    continue
                
                dist = cost_matrix[track_idx, det_idx]
                if dist < min_dist and dist < self.max_distance:
                    min_dist = dist
                    best_det_idx = det_idx
            
            if best_det_idx != -1:
                matched_tracks.append((track_idx, best_det_idx))
                matched_dets.add(best_det_idx)
        
        # Find unmatched detections and tracks
        matched_track_indices = {m[0] for m in matched_tracks}
        unmatched_tracks = [i for i in range(len(predictions)) if i not in matched_track_indices]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_dets]
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def get_all_trajectories(self):
        """Get complete trajectory history for all tracks.
        
        Returns:
            List of track histories with metadata
        """
        trajectories = []
        for track in self.tracks:
            if len(track.history) > 0:
                trajectories.append({
                    'id': track.id,
                    'history': track.history,
                    'hits': track.hits,
                    'age': track.age
                })
        return trajectories
    
    def reset(self):
        """Reset the tracker."""
        self.tracks = []
        self.frame_count = 0
        BallTrack._id_counter = 0
