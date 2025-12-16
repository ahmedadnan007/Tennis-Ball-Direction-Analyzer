"""Utility helper functions for tennis ball analysis project."""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def bbox_to_center(bbox):
    """Convert bounding box to center point.
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        (center_x, center_y)
    """
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def center_to_bbox(center, width, height):
    """Convert center point and size to bounding box.
    
    Args:
        center: (x, y)
        width: Box width
        height: Box height
    
    Returns:
        [x1, y1, x2, y2]
    """
    x, y = center
    return [x - width/2, y - height/2, x + width/2, y + height/2]


def bbox_area(bbox):
    """Calculate bounding box area.
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        Area in pixels
    """
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_iou(bbox1, bbox2):
    """Calculate Intersection over Union of two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        IoU score (0-1)
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def euclidean_distance(point1, point2):
    """Calculate Euclidean distance between two points.
    
    Args:
        point1: (x1, y1)
        point2: (x2, y2)
    
    Returns:
        Distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def angle_between_vectors(v1, v2):
    """Calculate angle between two vectors in radians.
    
    Args:
        v1: [x1, y1]
        v2: [x2, y2]
    
    Returns:
        Angle in radians
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    return np.arccos(cos_angle)


def draw_trajectory_on_frame(frame, positions, color=(0, 255, 255), thickness=2):
    """Draw trajectory path on frame.
    
    Args:
        frame: Image to draw on
        positions: List of (x, y) positions
        color: BGR color tuple
        thickness: Line thickness
    
    Returns:
        Frame with trajectory drawn
    """
    if len(positions) < 2:
        return frame
    
    for i in range(len(positions) - 1):
        pt1 = (int(positions[i][0]), int(positions[i][1]))
        pt2 = (int(positions[i+1][0]), int(positions[i+1][1]))
        cv2.line(frame, pt1, pt2, color, thickness)
    
    return frame


def draw_bbox_with_label(frame, bbox, label, color=(0, 255, 0), thickness=2):
    """Draw bounding box with label on frame.
    
    Args:
        frame: Image to draw on
        bbox: [x1, y1, x2, y2]
        label: Text label
        color: BGR color tuple
        thickness: Box thickness
    
    Returns:
        Frame with bbox drawn
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label background
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    
    # Draw label text
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return frame


def draw_velocity_arrow(frame, position, velocity, scale=10, color=(255, 0, 0), thickness=2):
    """Draw velocity vector as arrow.
    
    Args:
        frame: Image to draw on
        position: (x, y) starting point
        velocity: (vx, vy) velocity vector
        scale: Arrow length scale factor
        color: BGR color tuple
        thickness: Arrow thickness
    
    Returns:
        Frame with arrow drawn
    """
    start_point = (int(position[0]), int(position[1]))
    end_point = (int(position[0] + velocity[0] * scale), 
                 int(position[1] + velocity[1] * scale))
    
    cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=0.3)
    
    return frame


def draw_info_panel(frame, info_dict, position=(10, 30), font_scale=0.6, 
                    color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw information panel on frame.
    
    Args:
        frame: Image to draw on
        info_dict: Dictionary of label: value pairs
        position: Starting position (x, y)
        font_scale: Font size scale
        color: Text color
        bg_color: Background color
    
    Returns:
        Frame with info panel
    """
    x, y = position
    line_height = 25
    
    for i, (label, value) in enumerate(info_dict.items()):
        text = f"{label}: {value}"
        y_pos = y + i * line_height
        
        # Draw background
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                        font_scale, 1)
        cv2.rectangle(frame, (x - 5, y_pos - text_height - 5), 
                     (x + text_width + 5, y_pos + 5), bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale, color, 1, cv2.LINE_AA)
    
    return frame


def interpolate_missing_positions(positions, frames, max_gap=5):
    """Interpolate missing positions in trajectory.
    
    Args:
        positions: List of (x, y) positions
        frames: List of frame numbers corresponding to positions
        max_gap: Maximum gap to interpolate
    
    Returns:
        (interpolated_positions, interpolated_frames)
    """
    if len(positions) < 2:
        return positions, frames
    
    interpolated_pos = []
    interpolated_frames = []
    
    for i in range(len(positions) - 1):
        interpolated_pos.append(positions[i])
        interpolated_frames.append(frames[i])
        
        gap = frames[i+1] - frames[i]
        
        if gap > 1 and gap <= max_gap:
            # Interpolate
            for j in range(1, gap):
                alpha = j / gap
                interp_pos = (
                    positions[i][0] + alpha * (positions[i+1][0] - positions[i][0]),
                    positions[i][1] + alpha * (positions[i+1][1] - positions[i][1])
                )
                interpolated_pos.append(interp_pos)
                interpolated_frames.append(frames[i] + j)
    
    # Add last position
    interpolated_pos.append(positions[-1])
    interpolated_frames.append(frames[-1])
    
    return interpolated_pos, interpolated_frames


def get_video_properties(video_path):
    """Get video properties.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video properties
    """
    cap = cv2.VideoCapture(video_path)
    
    properties = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    return properties


def create_color_from_id(track_id, saturation=255, value=255):
    """Create unique color for track ID.
    
    Args:
        track_id: Integer track ID
        saturation: HSV saturation (0-255)
        value: HSV value/brightness (0-255)
    
    Returns:
        BGR color tuple
    """
    # Use golden ratio for hue distribution
    golden_ratio = 0.618033988749895
    hue = int((track_id * golden_ratio % 1.0) * 180)
    
    # Create HSV color
    hsv_color = np.uint8([[[hue, saturation, value]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    
    return tuple(map(int, bgr_color))


def smooth_trajectory(positions, window_size=5):
    """Apply moving average smoothing to trajectory.
    
    Args:
        positions: List of (x, y) positions
        window_size: Window size for moving average
    
    Returns:
        Smoothed positions
    """
    if len(positions) < window_size:
        return positions
    
    positions = np.array(positions)
    smoothed = np.zeros_like(positions)
    
    for i in range(len(positions)):
        start = max(0, i - window_size // 2)
        end = min(len(positions), i + window_size // 2 + 1)
        smoothed[i] = np.mean(positions[start:end], axis=0)
    
    return smoothed


def estimate_court_homography(court_points_image, court_points_real):
    """Estimate homography matrix for court perspective transformation.
    
    Args:
        court_points_image: List of 4+ (x, y) points in image coordinates
        court_points_real: List of 4+ (x, y) points in real-world coordinates
    
    Returns:
        Homography matrix (3x3)
    """
    court_points_image = np.array(court_points_image, dtype=np.float32)
    court_points_real = np.array(court_points_real, dtype=np.float32)
    
    H, _ = cv2.findHomography(court_points_image, court_points_real, cv2.RANSAC, 5.0)
    
    return H


def transform_point_with_homography(point, homography_matrix):
    """Transform point using homography matrix.
    
    Args:
        point: (x, y) point in image coordinates
        homography_matrix: 3x3 homography matrix
    
    Returns:
        Transformed (x, y) point
    """
    point_h = np.array([point[0], point[1], 1.0])
    transformed_h = homography_matrix @ point_h
    
    return (transformed_h[0] / transformed_h[2], transformed_h[1] / transformed_h[2])
