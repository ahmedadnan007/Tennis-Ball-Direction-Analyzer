"""Utility helper functions for tennis ball analysis project."""

import cv2
import numpy as np
from typing import Tuple, List, Optional


def bbox_to_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def center_to_bbox(center, width, height):
    x, y = center
    return [x - width/2, y - height/2, x + width/2, y + height/2]


def bbox_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_iou(bbox1, bbox2):
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
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def angle_between_vectors(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def draw_trajectory_on_frame(frame, positions, color=(0, 255, 255), thickness=2):
    if len(positions) < 2:
        return frame
    for i in range(len(positions) - 1):
        pt1 = (int(positions[i][0]), int(positions[i][1]))
        pt2 = (int(positions[i+1][0]), int(positions[i+1][1]))
        cv2.line(frame, pt1, pt2, color, thickness)
    return frame


def draw_bbox_with_label(frame, bbox, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return frame


def draw_velocity_arrow(frame, position, velocity, scale=10, color=(255, 0, 0), thickness=2):
    start_point = (int(position[0]), int(position[1]))
    end_point   = (int(position[0] + velocity[0] * scale),
                   int(position[1] + velocity[1] * scale))
    cv2.arrowedLine(frame, start_point, end_point, color, thickness, tipLength=0.3)
    return frame


def draw_broadcast_overlay(frame, shot_data, position='top_left'):
    h, w = frame.shape[:2]
    if position == 'top_right':
        x, y = w - 320, 20
    elif position == 'bottom_left':
        x, y = 20, h - 150
    elif position == 'bottom_right':
        x, y = w - 320, h - 150
    else:
        x, y = 20, 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + 300, y + 130), (20, 20, 20), -1)
    cv2.rectangle(overlay, (x, y), (x + 8, y + 130), (100, 200, 50), -1)
    cv2.rectangle(overlay, (x, y), (x + 300, y + 130), (80, 80, 80), 2)
    cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
    return frame


# ============================================================
# PROFESSIONAL OVERLAY - Shows Forehand/Backhand/Serve + Direction + Speed
# Spin (topspin/slice) REMOVED
# FIX 1: Ball ab sirf SINGLES court ke andar plot hoti hai (side galleries mein nahi)
# FIX 2: Trail LINE hata di gayi - ab sirf ball ka dot dikhta hai
# ============================================================
def draw_reel_style_shot_overlay(frame, shot_data, position='top_right'):
    h, w = frame.shape[:2]

    card_w = 220
    card_h = 250
    margin = 16

    if position == 'top_left':
        x1, y1 = margin, margin
    else:
        x1, y1 = w - card_w - margin, margin

    x2, y2 = x1 + card_w, y1 + card_h

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 20), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y1 + 4), (0, 210, 255), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 50, 60), 1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    # Mini court
    pad_x    = 18
    court_x1 = x1 + pad_x
    court_y1 = y1 + 14
    court_x2 = x2 - pad_x
    court_y2 = y1 + 138
    court_w  = court_x2 - court_x1
    court_h  = court_y2 - court_y1

    cv2.rectangle(frame, (court_x1, court_y1), (court_x2, court_y2), (40, 60, 90), -1)
    cv2.rectangle(frame, (court_x1, court_y1), (court_x2, court_y2), (100, 130, 160), 1)

    singles_inset = int(court_w * 0.125)
    sl_left  = court_x1 + singles_inset
    sl_right = court_x2 - singles_inset

    cv2.line(frame, (sl_left,  court_y1), (sl_left,  court_y2), (180, 200, 220), 1)
    cv2.line(frame, (sl_right, court_y1), (sl_right, court_y2), (180, 200, 220), 1)

    net_y = court_y1 + court_h // 2
    cv2.line(frame, (court_x1, net_y), (court_x2, net_y), (220, 220, 220), 2)
    cv2.line(frame, (court_x1, net_y - 2), (court_x1, net_y + 2), (255, 255, 255), 2)
    cv2.line(frame, (court_x2, net_y - 2), (court_x2, net_y + 2), (255, 255, 255), 2)

    svc_top_y    = court_y1 + int(court_h * 0.23)
    svc_bottom_y = court_y1 + int(court_h * 0.77)
    cv2.line(frame, (sl_left, svc_top_y),    (sl_right, svc_top_y),    (150, 170, 190), 1)
    cv2.line(frame, (sl_left, svc_bottom_y), (sl_right, svc_bottom_y), (150, 170, 190), 1)

    cx = court_x1 + court_w // 2
    cv2.line(frame, (cx, svc_top_y), (cx, svc_bottom_y), (140, 160, 180), 1)

    # Position data
    trail_points      = shot_data.get('trail_points', [])
    court_bounds      = shot_data.get('court_bounds', None)
    trajectory_bounds = shot_data.get('trajectory_bounds', None)
    source_bounds     = trajectory_bounds if isinstance(trajectory_bounds, dict) else court_bounds

    if isinstance(source_bounds, dict):
        min_x = float(source_bounds.get('min_x', 0.0))
        max_x = float(source_bounds.get('max_x', float(w - 1)))
        min_y = float(source_bounds.get('min_y', 0.0))
        max_y = float(source_bounds.get('max_y', float(h - 1)))
    else:
        min_x, max_x = 0.0, float(w - 1)
        min_y, max_y = 0.0, float(h - 1)

    if max_x - min_x < 10:
        min_x, max_x = 0.0, float(w - 1)
    if max_y - min_y < 10:
        min_y, max_y = 0.0, float(h - 1)

    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    # ── FIX 1 ──────────────────────────────────────────────
    # Ball ki x-position ko sirf SINGLES court (sl_left..sl_right)
    # ke andar map karo, poori court width pe nahi. Isse ball
    # kabhi side galleries (doubles tramlines) mein nahi jaayegi.
    singles_w = max(sl_right - sl_left, 1)

    def map_to_court(px, py):
        norm_x = max(0.0, min(1.0, (float(px) - min_x) / span_x))
        norm_y = max(0.02, min(0.98, (float(py) - min_y) / span_y))
        bx = int(sl_left + norm_x * singles_w)
        by = int(court_y1 + norm_y * court_h)
        return bx, by

    # ── FIX 2 ──────────────────────────────────────────────
    # Trail LINE aur start-dot hata diya. Ab sirf current ball
    # ka dot draw hota hai - koi line/trail nahi.
    if len(trail_points) > 0:
        bx, by = map_to_court(trail_points[-1][0], trail_points[-1][1])
        cv2.circle(frame, (bx, by), 7, (0, 255, 200), -1)
        cv2.circle(frame, (bx, by), 4, (255, 255, 255), -1)

    # ── TEXT SECTION ──
    divider_y = court_y2 + 8
    cv2.line(frame, (x1 + 10, divider_y), (x2 - 10, divider_y), (50, 50, 60), 1)

    # ============================================================
    # SIMPLIFIED LABEL:
    # Line 1: Stroke (Forehand / Backhand / Serve)
    # Line 2: Direction (Cross Court / Down The Line / Straight)
    # Right:  Speed km/h
    # ============================================================
    stroke_label    = shot_data.get('stroke',    'Forehand').title()
    direction_label = shot_data.get('direction', '').replace('_', ' ').title()
    speed           = float(shot_data.get('speed', 0.0))

    # Clean up direction label
    direction_label = (direction_label
                       .replace('Cross Court Right', 'Cross Court')
                       .replace('Cross Court Left',  'Cross Court')
                       .replace('Down The Line Deep',  'Down The Line')
                       .replace('Down The Line Short', 'Down The Line'))

    label_y = divider_y + 24

    # Stroke - big cyan
    cv2.putText(frame, stroke_label, (x1 + 12, label_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.62, (0, 210, 255), 2, cv2.LINE_AA)

    # Direction - smaller white
    if direction_label:
        cv2.putText(frame, direction_label, (x1 + 12, label_y + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 200, 200), 1, cv2.LINE_AA)

    # Speed - right side big
    speed_str = str(int(round(speed)))
    speed_x   = x2 - 60
    cv2.putText(frame, speed_str, (speed_x, label_y + 16),
                cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'KM/H', (speed_x + 2, label_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 170), 1, cv2.LINE_AA)

    return frame


def draw_info_panel(frame, info_dict, position=(10, 30), font_scale=0.6,
                    color=(255, 255, 255), bg_color=(0, 0, 0)):
    x, y         = position
    line_height  = 28
    padding_x    = 15
    padding_y    = 10
    if not info_dict:
        return frame
    max_width    = 0
    total_height = padding_y
    texts        = []
    for label, value in info_dict.items():
        text = f"{label}: {value}"
        texts.append(text)
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
        max_width   = max(max_width, tw)
        total_height += line_height
    total_height += padding_y
    panel_width   = max_width + 2 * padding_x
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - padding_x, y - padding_y),
                  (x + panel_width - padding_x, y + total_height - padding_y), (25, 25, 25), -1)
    cv2.rectangle(overlay, (x - padding_x, y - padding_y),
                  (x + panel_width - padding_x, y + total_height - padding_y), (60, 180, 255), 2)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    for i, text in enumerate(texts):
        y_pos = y + i * line_height + padding_y
        cv2.putText(frame, text, (x + 2, y_pos + 2), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y_pos), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, color, 2, cv2.LINE_AA)
    return frame


def interpolate_missing_positions(positions, frames, max_gap=5):
    if len(positions) < 2:
        return positions, frames
    interpolated_pos    = []
    interpolated_frames = []
    for i in range(len(positions) - 1):
        interpolated_pos.append(positions[i])
        interpolated_frames.append(frames[i])
        gap = frames[i+1] - frames[i]
        if 1 < gap <= max_gap:
            for j in range(1, gap):
                alpha = j / gap
                interpolated_pos.append((
                    positions[i][0] + alpha * (positions[i+1][0] - positions[i][0]),
                    positions[i][1] + alpha * (positions[i+1][1] - positions[i][1])
                ))
                interpolated_frames.append(frames[i] + j)
    interpolated_pos.append(positions[-1])
    interpolated_frames.append(frames[-1])
    return interpolated_pos, interpolated_frames


def get_video_properties(video_path):
    cap         = cv2.VideoCapture(video_path)
    fps         = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not fps or fps <= 0:
        fps = 30.0
    if frame_count <= 0:
        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)

    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return {'fps': fps, 'width': width, 'height': height,
            'frame_count': frame_count, 'duration': duration}


def create_color_from_id(track_id, saturation=255, value=255):
    golden_ratio = 0.618033988749895
    hue          = int((track_id * golden_ratio % 1.0) * 180)
    hsv_color    = np.uint8([[[hue, saturation, value]]])
    bgr_color    = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return tuple(map(int, bgr_color))


def smooth_trajectory(positions, window_size=5):
    if len(positions) < window_size:
        return positions
    positions = np.array(positions)
    smoothed  = np.zeros_like(positions)
    for i in range(len(positions)):
        start = max(0, i - window_size // 2)
        end   = min(len(positions), i + window_size // 2 + 1)
        smoothed[i] = np.mean(positions[start:end], axis=0)
    return smoothed


def estimate_court_homography(court_points_image, court_points_real):
    H, _ = cv2.findHomography(
        np.array(court_points_image, dtype=np.float32),
        np.array(court_points_real,  dtype=np.float32),
        cv2.RANSAC, 5.0
    )
    return H


def transform_point_with_homography(point, homography_matrix):
    point_h       = np.array([point[0], point[1], 1.0])
    transformed_h = homography_matrix @ point_h
    return (transformed_h[0] / transformed_h[2], transformed_h[1] / transformed_h[2])