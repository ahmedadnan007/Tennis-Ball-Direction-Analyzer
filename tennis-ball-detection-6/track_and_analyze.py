"""Integrated demo: Ball detection, tracking, and trajectory analysis.

Usage:
    python track_and_analyze.py --weights best.pt --source video.mp4 --output analyzed.mp4
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from ball_tracker import BallTracker
from trajectory_analyzer import TrajectoryAnalyzer
from shot_classifier import ShotClassifier
from court_detector import CourtDetector
from data_exporter import ShotDataExporter
from utils.helpers import (
    draw_trajectory_on_frame,
    draw_bbox_with_label,
    draw_info_panel,
    draw_reel_style_shot_overlay,
    create_color_from_id,
    get_video_properties
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights',          type=str,   required=True)
    parser.add_argument('--source',           type=str,   required=True)
    parser.add_argument('--output',           type=str,   default='tracked_output.mp4')
    parser.add_argument('--conf',             type=float, default=0.40)
    parser.add_argument('--device',           type=str,   default='0')
    parser.add_argument('--show',             action='store_true')
    parser.add_argument('--fps',              type=float, default=None)
    parser.add_argument('--court-length',     type=float, default=1200)
    parser.add_argument('--court-width',      type=float, default=600)
    parser.add_argument('--pixels-per-meter', type=float, default=None)
    parser.add_argument('--trail-length',     type=int,   default=30)
    parser.add_argument('--detect-court',     action='store_true')
    parser.add_argument('--export-csv',       type=str,   default=None)
    parser.add_argument('--player-handed',    type=str,   default='right',
                        choices=['right', 'left'])
    return parser.parse_args()


# ============================================================
# TUNED: Looser thresholds = more valid shots accepted (18-19)
# ============================================================
def is_valid_shot(analysis, shot_class, min_speed=10.0):
    speed_kmh   = analysis.get('avg_speed_kmh', 0)
    num_bounces = analysis.get('num_bounces', 0)
    apex_info   = analysis.get('apex_height', {})
    apex_m      = apex_info.get('apex_height_meters', 0) if apex_info else 0
    num_frames  = analysis.get('num_frames', 0)
    total_dist  = analysis.get('total_distance_meters', 0)

    # Speed check - lowered to 10 km/h
    if speed_kmh < min_speed:
        return False, f"speed {speed_kmh:.1f} km/h below {min_speed} km/h"

    # Bounce check - fake stationary objects
    if num_bounces > 5:
        return False, f"{num_bounces} bounces → fake detection"

    # Apex check - lowered to 0.2m (was 0.4m)
    if apex_m < 0.2:
        return False, f"apex {apex_m:.2f}m too low (min 0.2m)"

    # Stationary object check
    if num_frames > 30 and total_dist < 0.5:
        return False, f"{num_frames} frames but only {total_dist:.2f}m moved"

    return True, "ok"


def fix_shot_type(shot_class, analysis):
    shot_type = shot_class.get('type', 'unknown')
    speed_kmh = analysis.get('avg_speed_kmh', 0)
    if shot_type == 'drop_shot' and speed_kmh > 60:
        shot_class['type'] = 'groundstroke'
        print(f"[FIX] drop_shot at {speed_kmh:.1f} km/h → groundstroke")
    if shot_type == 'lob' and speed_kmh < 30:
        shot_class['type'] = 'groundstroke'
        print(f"[FIX] lob at {speed_kmh:.1f} km/h → groundstroke")
    if shot_type == 'lob' and speed_kmh > 150:
        shot_class['type'] = 'groundstroke'
        print(f"[FIX] lob at {speed_kmh:.1f} km/h → groundstroke")
    return shot_class


def safe_destroy_windows():
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def build_display_data(analysis, shot_class, history, width, height,
                       args, court_detector, speed_kmh):
    """Build shot display dictionary - shows Forehand/Backhand/Serve + Direction."""

    stroke_label    = shot_class.get('stroke',    'unknown').title()
    direction_label = shot_class.get('direction', '').replace('_', ' ').title()

    # Fallback stroke label
    if stroke_label.lower() == 'unknown':
        stroke_label = 'Forehand'

    # Clean direction labels
    direction_label = (direction_label
                       .replace('Cross Court Right', 'Cross Court')
                       .replace('Cross Court Left',  'Cross Court')
                       .replace('Down The Line Deep',  'Down The Line')
                       .replace('Down The Line Short', 'Down The Line'))

    court_bounds = {
        'min_x': 0.0, 'max_x': float(width - 1),
        'min_y': 0.0, 'max_y': float(height - 1)
    }

    if args.detect_court and court_detector and court_detector.court_lines:
        lines = court_detector.court_lines
        vertical_lines = lines.get('vertical', []) if isinstance(lines, dict) else []
        if vertical_lines:
            x_samples = sorted([(x1 + x2) / 2.0 for x1, _, x2, _ in vertical_lines])
            if len(x_samples) >= 2:
                court_bounds['min_x'] = float(max(0.0, x_samples[0]))
                court_bounds['max_x'] = float(min(width - 1, x_samples[-1]))
        top_y    = lines.get('baseline_left_y',  int(height * 0.1))
        bottom_y = lines.get('baseline_right_y', int(height * 0.9))
        if top_y is not None and bottom_y is not None:
            court_bounds['min_y'] = float(max(0.0,        min(top_y, bottom_y)))
            court_bounds['max_y'] = float(min(height - 1, max(top_y, bottom_y)))

    trail_points   = analysis.get('positions_pixels', [])
    selected_trail = trail_points[-24:] if trail_points else [h['center'] for h in history[-24:]]

    trajectory_bounds = None
    if selected_trail:
        xs = [p[0] for p in selected_trail]
        ys = [p[1] for p in selected_trail]
        trajectory_bounds = {
            'min_x': float(min(xs)), 'max_x': float(max(xs)),
            'min_y': float(min(ys)), 'max_y': float(max(ys))
        }

    return {
        'speed':             speed_kmh,
        'shot_type':         shot_class['type'].replace('_', ' ').title(),
        'stroke':            stroke_label,
        'direction':         direction_label,
        'shot_label':        stroke_label,  # Only stroke in big text
        'trail_points':      selected_trail,
        'court_bounds':      court_bounds,
        'trajectory_bounds': trajectory_bounds,
        'map_mode':          'trajectory_ref'
    }


def main():
    args = parse_args()

    print(f"[INFO] Loading video: {args.source}")
    video_props = get_video_properties(args.source)
    print(f"[INFO] Video: {video_props['width']}x{video_props['height']} @ {video_props['fps']:.1f} FPS")
    print(f"[INFO] Duration: {video_props['duration']:.1f}s ({video_props['frame_count']} frames)")

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.source}")
        return

    used_fps = args.fps if args.fps is not None else video_props['fps']
    if args.fps is not None:
        print(f"[INFO] Overriding video FPS with: {args.fps:.1f}")

    width  = video_props['width']
    height = video_props['height']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(args.output, fourcc, used_fps, (width, height))

    print(f"[INFO] Loading YOLO model: {args.weights}")
    model = YOLO(args.weights)

    tracker = BallTracker(
        max_missing_frames=15,
        min_confidence=0.40,
        max_distance=100,
        min_hits=3
    )

    court_detector = None
    if args.detect_court:
        print("[INFO] Initializing court line detector...")
        court_detector = CourtDetector(width, height)

    if args.pixels_per_meter:
        px_per_m = args.pixels_per_meter
        analyzer = TrajectoryAnalyzer(
            fps=used_fps,
            court_length_meters=1.0,
            court_length_pixels=px_per_m,
            court_width_meters=1.0,
            court_width_pixels=px_per_m
        )
    else:
        court_length_px = args.court_length
        court_width_px  = args.court_width
        homography = None

        if court_detector and hasattr(court_detector, 'is_calibrated') and court_detector.is_calibrated:
            calibrated_width, calibrated_length = court_detector.get_scaled_court_dimensions()
            if calibrated_width > 0 and calibrated_length > 0:
                court_width_px  = calibrated_width
                court_length_px = calibrated_length
            homography = court_detector.compute_homography()

        analyzer = TrajectoryAnalyzer(
            fps=used_fps,
            court_length_pixels=court_length_px,
            court_width_pixels=court_width_px,
            homography=homography
        )

    classifier = ShotClassifier(
        court_width=width,
        court_height=height,
        court_detector=court_detector,
        player_handed=args.player_handed
    )

    csv_exporter = None
    if args.export_csv:
        print(f"[INFO] Will export shot data to: {args.export_csv}")
        csv_exporter = ShotDataExporter(args.export_csv)

    print("[INFO] Starting tracking and analysis...")

    frame_idx             = 0
    track_histories       = {}
    court_lines_detected  = False
    active_track_ids_prev = set()
    processed_shot_ids    = set()
    current_shot_display  = None
    shot_display_frames   = 0
    # Shot card ab agle shot tak (ya video end tak) screen pe rahega,
    # sirf 3 sec ke liye nahi. Naya live shot aate hi wo replace ho jaayega.
    shot_display_duration = 100000
    total_accepted        = 0
    total_rejected        = 0

    live_display        = {}
    live_last_update    = {}
    locked_speed        = {}   # NEW: per-track locked (peak) speed - ek shot = ek stable number
    LIVE_UPDATE_EVERY   = 8
    MIN_FRAMES_FOR_LIVE = 6

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.detect_court and not court_lines_detected and frame_idx == 0:
            print("[INFO] Detecting court lines...")
            court_detector.detect_lines(frame)
            court_lines_detected = True
            print(f"[INFO] Net position detected at y={court_detector.net_position}")

        results = model(frame, conf=0.40, device=args.device, verbose=False)
        r = results[0]

        detections = []
        if hasattr(r.boxes, 'xyxy') and len(r.boxes.xyxy) > 0:
            boxes  = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                if w < 5 or w > 45 or h < 5 or h > 45:
                    continue
                aspect = w / (h + 1e-6)
                if aspect < 0.4 or aspect > 2.5:
                    continue
                detections.append({'bbox': box.tolist(), 'confidence': float(score)})

        tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            track_id = obj['id']
            if track_id not in track_histories:
                track_histories[track_id] = []
            track_histories[track_id].append({
                'frame':      frame_idx,
                'center':     obj['center'],
                'bbox':       obj['bbox'],
                'confidence': obj['avg_confidence']
            })

        for obj in tracked_objects:
            track_id = obj['id']
            color    = create_color_from_id(track_id)
            # ID label + bounding box hata diya (debug clutter tha).
            # Ab ball pe sirf saaf colored dot + trail dikhega.
            if track_id in track_histories:
                positions = [h['center'] for h in track_histories[track_id][-args.trail_length:]]
                draw_trajectory_on_frame(frame, positions, color, 2)
                center = (int(obj['center'][0]), int(obj['center'][1]))
                cv2.circle(frame, center, 5, color, -1)

        active_track_ids = {obj['id'] for obj in tracked_objects}

        # Live analysis
        for track_id in active_track_ids:
            if track_id in processed_shot_ids:
                continue
            history = track_histories.get(track_id, [])
            if len(history) < MIN_FRAMES_FOR_LIVE:
                continue
            last_update = live_last_update.get(track_id, -999)
            if frame_idx - last_update < LIVE_UPDATE_EVERY:
                continue
            try:
                analysis = analyzer.analyze_trajectory(history, track_id)
                if analysis:
                    speed_kmh = analysis.get('avg_speed_kmh', 0)
                    if speed_kmh >= 10.0:
                        # ── SPEED LOCK ──────────────────────────────────
                        # Har shot ka PEAK speed pakad ke hold karo. Ye ball
                        # ki hit-speed (racket se nikalne ke baad) hoti hai
                        # aur poore shot ke liye stable rehti hai - number
                        # ab 80->49 ke beech fluctuate nahi karega.
                        prev_locked   = locked_speed.get(track_id, 0.0)
                        display_speed = max(prev_locked, speed_kmh)
                        locked_speed[track_id] = display_speed

                        shot_class = classifier.classify_shot(analysis)
                        shot_class = fix_shot_type(shot_class, analysis)
                        display = build_display_data(
                            analysis, shot_class, history,
                            width, height, args, court_detector, display_speed
                        )
                        live_display[track_id]     = display
                        live_last_update[track_id] = frame_idx
            except Exception:
                pass

        for tid in list(live_display.keys()):
            if tid not in active_track_ids and tid in processed_shot_ids:
                live_display.pop(tid, None)
                live_last_update.pop(tid, None)
                locked_speed.pop(tid, None)

        completed_tracks = active_track_ids_prev - active_track_ids

        for completed_id in completed_tracks:
            if completed_id in processed_shot_ids:
                continue
            if completed_id not in track_histories:
                continue

            history     = track_histories[completed_id]
            frame_count = len(history)
            print(f"[DEBUG] Track {completed_id} completed with {frame_count} frames")

            if frame_count < 6:
                print(f"[DEBUG] ✗ Track {completed_id} too short: {frame_count} frames (need 6+)")
                processed_shot_ids.add(completed_id)
                track_histories.pop(completed_id, None)
                live_display.pop(completed_id, None)
                live_last_update.pop(completed_id, None)
                locked_speed.pop(completed_id, None)
                continue

            analysis = analyzer.analyze_trajectory(history, completed_id)
            if not analysis:
                processed_shot_ids.add(completed_id)
                track_histories.pop(completed_id, None)
                live_display.pop(completed_id, None)
                locked_speed.pop(completed_id, None)
                continue

            shot_class = classifier.classify_shot(analysis)
            speed_kmh  = analysis['avg_speed_kmh']
            shot_class = fix_shot_type(shot_class, analysis)

            print(f"[DEBUG] Shot speed: {speed_kmh:.1f} km/h, Stroke: {shot_class.get('stroke', '?')}, Dir: {shot_class.get('direction', '?')}")

            # TUNED: min_speed=10.0 (was 15.0)
            valid, reason = is_valid_shot(analysis, shot_class, min_speed=10.0)

            if not valid:
                print(f"[DEBUG] ✗ Shot rejected: {reason}")
                total_rejected += 1
                processed_shot_ids.add(completed_id)
                track_histories.pop(completed_id, None)
                live_display.pop(completed_id, None)
                live_last_update.pop(completed_id, None)
                locked_speed.pop(completed_id, None)
                continue

            total_accepted += 1
            apex_m = analysis.get('apex_height', {}).get('apex_height_meters', 0)

            # Final speed = jo live dikhaya tha wahi (locked peak), taake
            # shot khatam hote hi number jump na kare.
            final_speed = max(locked_speed.get(completed_id, 0.0), speed_kmh)

            print(f"[INFO] ✓ Shot: {shot_class.get('stroke','?')} - {shot_class.get('direction','?')} @ {final_speed:.1f} km/h | Apex: {apex_m:.1f}m")

            if csv_exporter:
                csv_exporter.add_shot(analysis, shot_class)

            current_shot_display = build_display_data(
                analysis, shot_class, history,
                width, height, args, court_detector, final_speed
            )
            shot_display_frames = 0

            live_display.pop(completed_id, None)
            live_last_update.pop(completed_id, None)
            locked_speed.pop(completed_id, None)
            processed_shot_ids.add(completed_id)
            track_histories.pop(completed_id, None)

        active_track_ids_prev = active_track_ids.copy()

        # Display
        best_live_id  = None
        best_live_len = 0
        for tid in active_track_ids:
            if tid in live_display and tid not in processed_shot_ids:
                tlen = len(track_histories.get(tid, []))
                if tlen > best_live_len:
                    best_live_len = tlen
                    best_live_id  = tid

        if best_live_id is not None:
            draw_reel_style_shot_overlay(frame, live_display[best_live_id], position='top_right')
            shot_display_frames += 1
        elif current_shot_display and shot_display_frames < shot_display_duration:
            draw_reel_style_shot_overlay(frame, current_shot_display, position='top_right')
            shot_display_frames += 1

        # Frame counter
        counter_text = f"FRAME {frame_idx}/{video_props['frame_count']}"
        (text_w, text_h), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        ov = frame.copy()
        cv2.rectangle(ov, (width - text_w - 25, height - text_h - 25),
                      (width - 5, height - 5), (25, 25, 25), -1)
        cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (width - text_w - 25, height - text_h - 25),
                      (width - 5, height - 5), (80, 80, 80), 1)
        cv2.putText(frame, counter_text, (width - text_w - 15 + 1, height - 15 + 1),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, counter_text, (width - text_w - 15, height - 15),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 2)

        out.write(frame)

        if args.show:
            try:
                cv2.imshow('Tennis Ball Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass

        frame_idx += 1
        if frame_idx % 30 == 0:
            progress = (frame_idx / video_props['frame_count']) * 100
            print(f"[INFO] Progress: {progress:.1f}% ({frame_idx}/{video_props['frame_count']})")

    cap.release()
    out.release()
    safe_destroy_windows()

    print(f"\n[SUCCESS] Processing complete!")
    print(f"[INFO] Output saved to: {args.output}")
    print(f"[INFO] ✓ Shots accepted: {total_accepted} | ✗ Rejected: {total_rejected}")

    if csv_exporter:
        csv_exporter.save()

    print("\n" + "=" * 60)
    print("SHOT STATISTICS")
    print("=" * 60)

    all_classifications = []
    for track_id, history in track_histories.items():
        if len(history) >= 5:
            analysis = analyzer.analyze_trajectory(history, track_id)
            if analysis:
                shot_class = classifier.classify_shot(analysis)
                shot_class = fix_shot_type(shot_class, analysis)
                valid, _ = is_valid_shot(analysis, shot_class)
                if not valid:
                    continue
                all_classifications.append(shot_class)

    if all_classifications:
        strokes    = {}
        directions = {}
        for cls in all_classifications:
            strokes[cls.get('stroke', 'unknown')]    = strokes.get(cls.get('stroke', 'unknown'), 0) + 1
            directions[cls.get('direction', 'unknown')] = directions.get(cls.get('direction', 'unknown'), 0) + 1

        print("\nStrokes:")
        for k, v in sorted(strokes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k.title()}: {v}")
        print("\nDirections:")
        for k, v in sorted(directions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {k.replace('_',' ').title()}: {v}")
        print("=" * 60)

    print(f"\nTotal frames processed: {frame_idx}")


if __name__ == '__main__':
    main()