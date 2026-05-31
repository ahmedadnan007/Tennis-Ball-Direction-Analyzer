"""Ball tracking module using Kalman Filter for smooth trajectory estimation.

This module maintains consistent ball IDs across frames, handles missing detections,
and provides smooth trajectory predictions using a Kalman Filter.

Usage:
    tracker = BallTracker(max_missing_frames=10, min_confidence=0.60)
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
        self.age = 0
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        
        self.history = []
        self.confidence_history = []
        
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
        """Initialize Kalman Filter for 2D position and velocity tracking."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        kf.R *= 10.0
        kf.Q = np.eye(4) * 0.1
        kf.Q[2:, 2:] *= 5.0
        
        kf.P *= 100.0
        kf.P[2:, 2:] *= 1000.0
        
        kf.x = np.array([initial_position[0], initial_position[1], 0, 0])
        
        return kf
    
    def predict(self):
        """Predict the next state using Kalman Filter."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.kf.x[:2]
    
    def update(self, detection, frame_number):
        """Update the track with a new detection."""
        self.kf.update(np.array(detection['center']))
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        
        self.last_detection = detection
        self.last_frame = frame_number
        
        self.history.append({
            'frame': frame_number,
            'center': detection['center'],
            'bbox': detection['bbox'],
            'confidence': detection['confidence'],
            'predicted': self.kf.x[:2].copy()
        })
        self.confidence_history.append(detection['confidence'])
    
    def get_state(self):
        """Get current state of the track."""
        if self.last_detection is not None and self.time_since_update == 0:
            pos = np.array(self.last_detection['center'])
        else:
            pos = self.kf.x[:2]
        vel = self.kf.x[2:]
        
        if self.last_detection is not None:
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
    
    def __init__(self, max_missing_frames=10, min_confidence=0.60,
                 max_distance=80, min_hits=5):
        """Initialize the ball tracker.
        
        Args:
            max_missing_frames: Maximum frames a track can be missing before deletion
            min_confidence: Minimum detection confidence (FIXED: 0.25 -> 0.60)
            max_distance: Maximum pixel distance for association (FIXED: 100 -> 80)
            min_hits: Minimum hits before confirmed (FIXED: 3 -> 5)
        """
        self.max_missing_frames = max_missing_frames
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        self.min_hits = min_hits
        
        self.tracks = []
        self.frame_count = 0

    # =====================================================
    # FIX: Size + Shape filter for tennis ball
    # =====================================================
    def _is_valid_ball_size(self, bbox):
        """
        Tennis ball size check - fake detections hatao.
        Ball 5px se 45px ke beech honi chahiye screen pe.
        """
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1

        # Size range check
        if w < 5 or w > 45:
            return False
        if h < 5 or h > 45:
            return False

        # Shape check: ball round hoti hai (width ≈ height)
        aspect_ratio = w / (h + 1e-6)
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            return False

        return True

    def update(self, detections):
        """Update tracks with new detections.
        
        Args:
            detections: List of dicts with keys 'bbox', 'confidence'
                       bbox format: [x1, y1, x2, y2]
        
        Returns:
            List of tracked ball states (confirmed tracks only)
        """
        self.frame_count += 1
        
        # Filter detections by confidence (FIXED: was 0.25, now 0.60)
        valid_detections = [
            d for d in detections if d.get('confidence', 0) >= self.min_confidence
        ]

        # FIX: Size + shape filter lagao - fake detections hatao
        valid_detections = [
            d for d in valid_detections
            if self._is_valid_ball_size(d['bbox'])
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
        
        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks 
            if t.time_since_update < self.max_missing_frames
        ]
        
        # FIX: Strict condition - OR hata ke AND lagaya + hit_streak >= 2
        confirmed_tracks = [
            t.get_state() for t in self.tracks 
            if t.hits >= self.min_hits and t.hit_streak >= 2
        ]
        
        return confirmed_tracks
    
    def _associate(self, detections, predictions):
        """Associate detections with predicted tracks."""
        if len(predictions) == 0:
            return [], list(range(len(detections))), []
        
        if len(detections) == 0:
            return [], [], list(range(len(predictions)))
        
        det_centers = np.array([d['center'] for d in detections])
        pred_centers = np.array(predictions)
        
        cost_matrix = cdist(pred_centers, det_centers, metric='euclidean')
        
        matched_tracks = []
        matched_dets = set()
        
        for track_idx in range(len(predictions)):
            if len(matched_dets) >= len(detections):
                break
            
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
        
        matched_track_indices = {m[0] for m in matched_tracks}
        unmatched_tracks = [i for i in range(len(predictions)) if i not in matched_track_indices]
        unmatched_detections = [i for i in range(len(detections)) if i not in matched_dets]
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def get_all_trajectories(self):
        """Get complete trajectory history for all tracks."""
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