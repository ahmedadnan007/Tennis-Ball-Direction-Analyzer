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


def draw_broadcast_overlay(frame, shot_data, position='top_left'):
    """Draw professional tennis broadcast-style overlay.
    
    Args:
        frame: Image to draw on
        shot_data: Dictionary with shot information
        position: 'top_left', 'top_right', 'bottom_left', 'bottom_right'
    
    Returns:
        Frame with broadcast overlay
    """
    h, w = frame.shape[:2]
    
    # Determine position
    if position == 'top_right':
        x, y = w - 320, 20
    elif position == 'bottom_left':
        x, y = 20, h - 150
    elif position == 'bottom_right':
        x, y = w - 320, h - 150
    else:  # top_left
        x, y = 20, 20
    
    # Create overlay
    overlay = frame.copy()
    
    # Main panel background
    panel_width = 300
    panel_height = 130
    
    # Gradient background effect
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), 
                 (20, 20, 20), -1)
    
    # Colored accent bar (tennis green/blue)
    accent_color = (100, 200, 50)  # Tennis court green
    cv2.rectangle(overlay, (x, y), (x + 8, y + panel_height), accent_color, -1)
    
    # Border
    cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), 
                 (80, 80, 80), 2)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    
    # Draw content
    text_x = x + 20
    text_y = y + 30
    
    # Track ID (small header)
    track_id = shot_data.get('track_id', 0)
    cv2.putText(frame, f"TRACK #{track_id}", (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
    
    # Shot type (large, prominent)
    shot_type = shot_data.get('shot_type', 'Unknown')
    text_y += 28
    cv2.putText(frame, shot_type.upper(), (text_x, text_y), 
               cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Apex height with icon
    apex = shot_data.get('apex_height', 0)
    height_class = shot_data.get('height_class', '')
    text_y += 26
    height_text = f"HEIGHT: {apex:.1f}m ({height_class})"
    cv2.putText(frame, height_text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1, cv2.LINE_AA)
    
    # Speed with icon
    speed = shot_data.get('speed', 0)
    text_y += 24
    speed_text = f"SPEED: {speed:.0f} km/h"
    cv2.putText(frame, speed_text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 2, cv2.LINE_AA)
    
    # Direction
    direction = shot_data.get('direction', 'Unknown')
    text_y += 24
    direction_text = f"DIR: {direction}"
    cv2.putText(frame, direction_text, (text_x, text_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    
    return frame


def draw_reel_style_shot_overlay(frame, shot_data, position='top_right'):
    """Draw a social-media/reel style shot card with mini-court + speed label.

    Args:
        frame: Image to draw on
        shot_data: Dict with keys like speed, shot_label, trail_points
        position: 'top_left' or 'top_right'

    Returns:
        Frame with overlay
    """
    h, w = frame.shape[:2]

    # Compact card similar to the reference image
    card_w = 185
    card_h = 210
    margin = 14

    if position == 'top_left':
        x1, y1 = margin, margin
    else:  # top_right
        x1, y1 = w - card_w - margin, margin

    x2, y2 = x1 + card_w, y1 + card_h

    # Main translucent card (clean and minimal)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (28, 28, 28), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (85, 85, 85), 1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)

    # Mini court area
    court_pad_x = 25
    court_x1 = x1 + court_pad_x
    court_y1 = y1 + 14
    court_x2 = x2 - court_pad_x
    court_y2 = y1 + 125

    # Court lines using real tennis proportions
    # Court dimensions (meters): length=23.77, doubles width=10.97, singles width=8.23
    # Net at center. Service line is 6.40m from net on each side.
    cv2.rectangle(frame, (court_x1, court_y1), (court_x2, court_y2), (230, 230, 230), 1)

    court_w = court_x2 - court_x1
    court_h = court_y2 - court_y1

    # Singles sideline inset ratio from doubles court
    singles_inset_ratio = (10.97 - 8.23) / (2 * 10.97)  # ~0.1249
    inset_x = int(court_w * singles_inset_ratio)

    singles_left_x = court_x1 + inset_x
    singles_right_x = court_x2 - inset_x

    # Key horizontal lines by regulation ratios
    # baseline -> service line distance = 11.885 - 6.40 = 5.485m
    service_from_top_ratio = 5.485 / 23.77
    service_from_bottom_ratio = 1.0 - service_from_top_ratio

    net_y = court_y1 + court_h // 2
    svc_top_y = court_y1 + int(court_h * service_from_top_ratio)
    svc_bottom_y = court_y1 + int(court_h * service_from_bottom_ratio)

    # Singles sidelines
    cv2.line(frame, (singles_left_x, court_y1), (singles_left_x, court_y2), (185, 185, 185), 1)
    cv2.line(frame, (singles_right_x, court_y1), (singles_right_x, court_y2), (185, 185, 185), 1)

    # Net line (full width)
    cv2.line(frame, (court_x1, net_y), (court_x2, net_y), (205, 205, 205), 1)

    # Service lines (only between singles sidelines)
    cv2.line(frame, (singles_left_x, svc_top_y), (singles_right_x, svc_top_y), (160, 160, 160), 1)
    cv2.line(frame, (singles_left_x, svc_bottom_y), (singles_right_x, svc_bottom_y), (160, 160, 160), 1)

    # Center service line (between service lines only)
    center_x = court_x1 + court_w // 2
    cv2.line(frame, (center_x, svc_top_y), (center_x, svc_bottom_y), (175, 175, 175), 1)

    # Plot recent trajectory points on mini court
    trail_points = shot_data.get('trail_points', [])
    court_bounds = shot_data.get('court_bounds', None)
    trajectory_bounds = shot_data.get('trajectory_bounds', None)
    map_mode = shot_data.get('map_mode', 'court_ref')

    # Prefer trajectory-reference mapping so mini-court dots follow
    # the same path reference as the on-frame red tracking line.
    source_bounds = trajectory_bounds if map_mode == 'trajectory_ref' and isinstance(trajectory_bounds, dict) else court_bounds

    if isinstance(source_bounds, dict):
        min_x = float(source_bounds.get('min_x', 0.0))
        max_x = float(source_bounds.get('max_x', float(w - 1)))
        min_y = float(source_bounds.get('min_y', 0.0))
        max_y = float(source_bounds.get('max_y', float(h - 1)))
    else:
        min_x, max_x = 0.0, float(w - 1)
        min_y, max_y = 0.0, float(h - 1)

    # Fallback when detector bounds are invalid
    if max_x - min_x < 6:
        min_x, max_x = 0.0, float(w - 1)
    if max_y - min_y < 6:
        min_y, max_y = 0.0, float(h - 1)

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    if len(trail_points) > 0:
        # Single live dot only (latest tracked ball position)
        px, py = trail_points[-1]
        norm_x = max(0.0, min(1.0, (float(px) - min_x) / span_x))
        norm_y = max(0.0, min(1.0, (float(py) - min_y) / span_y))

        cx = int(court_x1 + norm_x * court_w)
        cy = int(court_y1 + norm_y * court_h)

        # Bright green ball marker
        cv2.circle(frame, (cx, cy), 4, (70, 220, 90), -1)

    # Shot label + speed section
    shot_label = shot_data.get('shot_label', 'Flat Backhand')
    speed = float(shot_data.get('speed', 0.0))

    # Split shot label across 2 lines like reference (e.g., Flat / Backhand)
    words = shot_label.split()
    if len(words) >= 2:
        line1 = words[0]
        line2 = " ".join(words[1:])
    else:
        line1 = shot_label
        line2 = ""

    label_y = y1 + 156
    cv2.putText(frame, line1, (x1 + 14, label_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.56, (30, 140, 255), 2, cv2.LINE_AA)
    if line2:
        cv2.putText(frame, line2, (x1 + 14, label_y + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 0.56, (30, 140, 255), 2, cv2.LINE_AA)

    speed_x = x2 - 52
    cv2.putText(frame, f"{int(round(speed))}", (speed_x, label_y + 8),
                cv2.FONT_HERSHEY_DUPLEX, 0.78, (242, 242, 242), 2, cv2.LINE_AA)
    cv2.putText(frame, "KM/H", (speed_x - 2, label_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1, cv2.LINE_AA)

    return frame


def draw_info_panel(frame, info_dict, position=(10, 30), font_scale=0.6, 
                    color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw broadcast-style information panel with professional styling.
    
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
    line_height = 28
    padding_x = 15
    padding_y = 10
    
    if len(info_dict) == 0:
        return frame
    
    # Calculate panel dimensions
    max_width = 0
    total_height = padding_y
    
    texts = []
    for label, value in info_dict.items():
        text = f"{label}: {value}"
        texts.append(text)
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 
                                                        font_scale, 2)
        max_width = max(max_width, text_width)
        total_height += line_height
    
    total_height += padding_y
    panel_width = max_width + 2 * padding_x
    
    # Create semi-transparent overlay
    overlay = frame.copy()
    
    # Draw rounded rectangle background with gradient effect
    panel_bg = (25, 25, 25)  # Dark background
    cv2.rectangle(overlay, 
                 (x - padding_x, y - padding_y), 
                 (x + panel_width - padding_x, y + total_height - padding_y),
                 panel_bg, -1)
    
    # Add subtle border
    border_color = (60, 180, 255)  # Tennis blue
    cv2.rectangle(overlay, 
                 (x - padding_x, y - padding_y), 
                 (x + panel_width - padding_x, y + total_height - padding_y),
                 border_color, 2)
    
    # Blend overlay with original frame (70% overlay, 30% original)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # Draw text with enhanced styling
    for i, text in enumerate(texts):
        y_pos = y + i * line_height + padding_y
        
        # Draw text shadow for better readability
        cv2.putText(frame, text, (x + 2, y_pos + 2), cv2.FONT_HERSHEY_DUPLEX, 
                   font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        
        # Draw main text
        cv2.putText(frame, text, (x, y_pos), cv2.FONT_HERSHEY_DUPLEX, 
                   font_scale, color, 2, cv2.LINE_AA)
    
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
