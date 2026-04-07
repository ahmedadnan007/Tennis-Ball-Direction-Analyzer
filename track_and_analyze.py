"""Integrated demo: Ball detection, tracking, and trajectory analysis.

This script demonstrates the complete pipeline:
1. Detect tennis balls using YOLO
2. Track balls across frames using Kalman Filter
3. Analyze trajectories (speed, direction, shape)
4. Visualize results with overlays

Usage:
    python track_and_analyze.py --weights runs/detect/tennis_ball_run/weights/best.pt --source video.mp4 --output analyzed.mp4
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
    parser = argparse.ArgumentParser(description='Tennis ball tracking and trajectory analysis')
    parser.add_argument('--weights', type=str, required=True, help='Path to YOLO weights (.pt)')
    parser.add_argument('--source', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='tracked_output.mp4', help='Output video path')
    parser.add_argument('--conf', type=float, default=0.25, help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='0', help='Device (0 for GPU, cpu for CPU)')
    parser.add_argument('--show', action='store_true', help='Display video while processing')
    parser.add_argument('--court-length', type=float, default=1200, 
                       help='Court length in pixels (for calibration)')
    parser.add_argument('--court-width', type=float, default=600,
                       help='Court width in pixels (for calibration)')
    parser.add_argument('--pixels-per-meter', type=float, default=None,
                       help='Direct pixel to meter ratio (overrides court calibration)')
    parser.add_argument('--trail-length', type=int, default=30, 
                       help='Number of points to show in trajectory trail')
    parser.add_argument('--detect-court', action='store_true',
                       help='Enable court line detection')
    parser.add_argument('--export-csv', type=str, default=None,
                       help='Export shot data to CSV file for ML training')
    parser.add_argument('--player-handed', type=str, default='right',
                       choices=['right', 'left'],
                       help='Player handedness for forehand/backhand classification')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Get video properties
    print(f"[INFO] Loading video: {args.source}")
    video_props = get_video_properties(args.source)
    print(f"[INFO] Video: {video_props['width']}x{video_props['height']} @ {video_props['fps']:.1f} FPS")
    print(f"[INFO] Duration: {video_props['duration']:.1f}s ({video_props['frame_count']} frames)")
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.source}")
        return
    
    # Initialize video writer
    fps = video_props['fps']
    width = video_props['width']
    height = video_props['height']
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Load YOLO model
    print(f"[INFO] Loading YOLO model: {args.weights}")
    model = YOLO(args.weights)
    
    # Initialize tracker and trajectory analyzer
    tracker = BallTracker(
        max_missing_frames=10,
        min_confidence=args.conf,
        max_distance=100,
        min_hits=3
    )
    
    # Use manual pixels-per-meter if provided, otherwise use court dimensions
    if args.pixels_per_meter:
        # Direct ratio
        px_per_m = args.pixels_per_meter
        analyzer = TrajectoryAnalyzer(
            fps=fps,
            court_length_meters=1.0,  # Dummy value
            court_length_pixels=px_per_m,
            court_width_meters=1.0,  # Dummy value
            court_width_pixels=px_per_m
        )
    else:
        analyzer = TrajectoryAnalyzer(
            fps=fps,
            court_length_pixels=args.court_length,
            court_width_pixels=args.court_width
        )
    
    # Initialize shot classifier
    court_detector = None
    if args.detect_court:
        print("[INFO] Initializing court line detector...")
        court_detector = CourtDetector(width, height)
    
    classifier = ShotClassifier(
        court_width=width,
        court_height=height,
        court_detector=court_detector,
        player_handed=args.player_handed
    )
    
    # Initialize CSV exporter
    csv_exporter = None
    if args.export_csv:
        print(f"[INFO] Will export shot data to: {args.export_csv}")
        csv_exporter = ShotDataExporter(args.export_csv)
    
    print("[INFO] Starting tracking and analysis...")
    
    frame_idx = 0
    track_histories = {}  # Store trajectory history for each track ID
    court_lines_detected = False  # Flag for one-time court detection
    
    # Track completed shots for display
    active_track_ids_prev = set()
    current_shot_display = None  # Current shot to display
    shot_display_frames = 0  # Frames to keep displaying shot
    shot_display_duration = 100  # Show for 100 frames (2 seconds at 50fps)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect court lines (only once, on first frame)
        if args.detect_court and not court_lines_detected and frame_idx == 0:
            print("[INFO] Detecting court lines...")
            court_detector.detect_lines(frame)
            court_lines_detected = True
            print(f"[INFO] Net position detected at y={court_detector.net_position}")
        
        # Draw court lines if detected
        if args.detect_court and court_detector.court_lines and args.show:
            # Only draw on display, not on saved video
            display_frame = frame.copy()
            court_detector.draw_court_lines(display_frame, color=(0, 255, 0), thickness=1)
        else:
            display_frame = frame
        
        # Run YOLO detection
        results = model(frame, conf=args.conf, device=args.device, verbose=False)
        r = results[0]
        
        # Extract detections
        detections = []
        if hasattr(r.boxes, 'xyxy') and len(r.boxes.xyxy) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            
            for box, score in zip(boxes, scores):
                detections.append({
                    'bbox': box.tolist(),
                    'confidence': float(score)
                })
        
        # Update tracker
        tracked_objects = tracker.update(detections)
        
        # Update trajectory histories
        for obj in tracked_objects:
            track_id = obj['id']
            
            if track_id not in track_histories:
                track_histories[track_id] = []
            
            track_histories[track_id].append({
                'frame': frame_idx,
                'center': obj['center'],
                'bbox': obj['bbox'],
                'confidence': obj['avg_confidence']
            })
        
        # Draw visualizations
        for obj in tracked_objects:
            track_id = obj['id']
            color = create_color_from_id(track_id)
            
            # Draw bounding box
            label = f"ID:{track_id} ({obj['avg_confidence']:.2f})"
            draw_bbox_with_label(frame, obj['bbox'], label, color, 2)
            
            # Draw trajectory trail
            if track_id in track_histories:
                history = track_histories[track_id]
                positions = [h['center'] for h in history[-args.trail_length:]]
                draw_trajectory_on_frame(frame, positions, color, 2)
                
                # Draw current position dot
                center = (int(obj['center'][0]), int(obj['center'][1]))
                cv2.circle(frame, center, 5, color, -1)
        
        # Analyze and display trajectory info for ACTIVE shots only
        overlay_positions = ['top_left', 'top_right', 'bottom_left']
        overlay_idx = 0
        
        # Get currently active track IDs (only show overlays for balls in current frame)
        active_track_ids = set([obj['id'] for obj in tracked_objects])
        
        # Detect completed tracks (tracks that were active but are no longer)
        completed_tracks = active_track_ids_prev - active_track_ids
        
        # If a track completed, save shot info for display and export to CSV
        for completed_id in completed_tracks:
            if completed_id in track_histories and len(track_histories[completed_id]) >= 10:
                history = track_histories[completed_id]
                analysis = analyzer.analyze_trajectory(history, completed_id)
                if analysis:
                    shot_class = classifier.classify_shot(analysis)
                    
                    # Filter out stationary/slow balls (minimum 10 km/h)
                    min_speed_threshold = 10.0  # km/h
                    
                    # Export to CSV (only valid shots with sufficient speed)
                    if csv_exporter and analysis['avg_speed_kmh'] >= min_speed_threshold:
                        csv_exporter.add_shot(analysis, shot_class)
                    
                    # Create shot display data (only for valid shots)
                    if analysis['avg_speed_kmh'] >= min_speed_threshold:
                        spin_label = shot_class.get('spin', 'unknown').title()
                        stroke_label = shot_class.get('stroke', 'unknown').title()
                        if stroke_label.lower() == 'unknown':
                            stroke_label = shot_class['type'].replace('_', ' ').title()

                        # Estimate court bounds for accurate mini-court projection
                        court_bounds = {
                            'min_x': 0.0,
                            'max_x': float(width - 1),
                            'min_y': 0.0,
                            'max_y': float(height - 1)
                        }

                        if args.detect_court and court_detector and court_detector.court_lines:
                            lines = court_detector.court_lines

                            # X bounds from vertical lines (if available)
                            vertical_lines = lines.get('vertical', []) if isinstance(lines, dict) else []
                            if vertical_lines:
                                x_samples = []
                                for x1, _, x2, _ in vertical_lines:
                                    x_samples.append((x1 + x2) / 2.0)
                                if len(x_samples) >= 2:
                                    x_samples = sorted(x_samples)
                                    court_bounds['min_x'] = float(max(0.0, x_samples[0]))
                                    court_bounds['max_x'] = float(min(width - 1, x_samples[-1]))

                            # Y bounds from detected baselines
                            top_y = lines.get('baseline_left_y', int(height * 0.1))
                            bottom_y = lines.get('baseline_right_y', int(height * 0.9))
                            if top_y is not None and bottom_y is not None:
                                court_bounds['min_y'] = float(max(0.0, min(top_y, bottom_y)))
                                court_bounds['max_y'] = float(min(height - 1, max(top_y, bottom_y)))

                        # Use analyzed trajectory points (smoothed) for better stability
                        trail_points = analysis.get('positions_pixels', [])
                        selected_trail_points = trail_points[-24:] if trail_points else [h['center'] for h in history[-24:]]

                        # Bounds from the exact trail used by the red tracking path
                        if selected_trail_points:
                            xs = [p[0] for p in selected_trail_points]
                            ys = [p[1] for p in selected_trail_points]
                            trajectory_bounds = {
                                'min_x': float(min(xs)),
                                'max_x': float(max(xs)),
                                'min_y': float(min(ys)),
                                'max_y': float(max(ys))
                            }
                        else:
                            trajectory_bounds = None

                        current_shot_display = {
                            'speed': analysis['avg_speed_kmh'],
                            'shot_type': shot_class['type'].replace('_', ' ').title(),
                            'spin': spin_label,
                            'stroke': stroke_label,
                            'direction': shot_class['direction'].replace('_', ' ').title(),
                            'shot_label': f"{spin_label} {stroke_label}",
                            'trail_points': selected_trail_points,
                            'court_bounds': court_bounds,
                            'trajectory_bounds': trajectory_bounds,
                            'map_mode': 'trajectory_ref'
                        }
                        shot_display_frames = 0  # Reset counter
        
        # Update previous active tracks
        active_track_ids_prev = active_track_ids.copy()
        
        for obj in tracked_objects:
            track_id = obj['id']
            
            if track_id in track_histories and len(track_histories[track_id]) >= 10:
                history = track_histories[track_id]
                
                # Analyze trajectory
                analysis = analyzer.analyze_trajectory(history, track_id)
                
                if analysis:
                    # Classify shot
                    shot_class = classifier.classify_shot(analysis)
                    
                    # Only show overlays for first 3 active tracks
                    if overlay_idx < 3:
                        # Get apex height and format shot type for display
                        apex_height = shot_class.get('apex_height_meters', 0)
                        height_class = shot_class.get('height_class', '').replace('_', ' ').title()
                        shot_type_display = shot_class['type'].replace('_', ' ').title()
                        direction_display = shot_class['direction'].replace('_', ' ').title()
                        
                        # Add stroke and spin to display
                        stroke = shot_class.get('stroke', 'Unknown').title()
                        spin = shot_class.get('spin', 'Unknown').title()
                        
                        # Prepare data for broadcast overlay
                        shot_data = {
                            'track_id': track_id,
                            'shot_type': f"{shot_type_display} ({stroke} {spin})",
                            'apex_height': apex_height,
                            'height_class': height_class,
                            'speed': analysis['avg_speed_kmh'],
                            'direction': direction_display
                        }
                        
                        
                        overlay_idx += 1
        
        # Draw shot analytics when available
        if current_shot_display and shot_display_frames < shot_display_duration:
            draw_reel_style_shot_overlay(frame, current_shot_display, position='top_right')
            
            shot_display_frames += 1
        
        # Draw professional frame counter in bottom-right corner
        counter_text = f"FRAME {frame_idx}/{video_props['frame_count']}"
        (text_w, text_h), _ = cv2.getTextSize(counter_text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 2)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (width - text_w - 25, height - text_h - 25), 
                     (width - 5, height - 5), (25, 25, 25), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Border
        cv2.rectangle(frame, (width - text_w - 25, height - text_h - 25), 
                     (width - 5, height - 5), (80, 80, 80), 1)
        
        # Text with shadow
        cv2.putText(frame, counter_text, (width - text_w - 15 + 1, height - 15 + 1),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(frame, counter_text, (width - text_w - 15, height - 15),
                   cv2.FONT_HERSHEY_DUPLEX, 0.6, (200, 200, 200), 2)
        
        # Write frame
        out.write(frame)
        
        # Display if requested (with court lines if enabled)
        if args.show:
            if args.detect_court and court_detector.court_lines:
                display_with_lines = frame.copy()
                court_detector.draw_court_lines(display_with_lines, color=(0, 255, 0), thickness=1)
                cv2.imshow('Tennis Ball Tracking', display_with_lines)
            else:
                cv2.imshow('Tennis Ball Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_idx += 1
        
        # Progress update
        if frame_idx % 30 == 0:
            progress = (frame_idx / video_props['frame_count']) * 100
            print(f"[INFO] Progress: {progress:.1f}% ({frame_idx}/{video_props['frame_count']})")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n[SUCCESS] Processing complete!")
    print(f"[INFO] Output saved to: {args.output}")
    
    # Save CSV export if enabled
    if csv_exporter:
        csv_exporter.save()
    
    # Final trajectory analysis
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("="*60)
    
    all_classifications = []
    for track_id, history in track_histories.items():
        if len(history) >= 5:
            analysis = analyzer.analyze_trajectory(history, track_id)
            if analysis:
                shot_class = classifier.classify_shot(analysis)
                all_classifications.append(shot_class)
                
                print(analyzer.get_trajectory_summary(analysis))
                print(f"Shot Classification: {classifier.get_shot_summary(shot_class)}")
                print(f"  - Type: {shot_class['type']}")
                print(f"  - Stroke: {shot_class.get('stroke', 'unknown')}")
                print(f"  - Spin: {shot_class.get('spin', 'unknown')}")
                print(f"  - Height Class: {shot_class.get('height_class', 'unknown')}")
                print(f"  - Apex Height: {shot_class.get('apex_height_meters', 0):.2f}m")
                print(f"  - Direction: {shot_class['direction']}")
                print(f"  - Zone: {shot_class['start_zone']} → {shot_class['end_zone']}")
                print(f"  - Court Side: {shot_class['side']}")
                print(f"  - Confidence: {shot_class['confidence']:.2f}")
                print("-" * 60)
    
    # Rally statistics
    if all_classifications:
        print("\n" + "="*60)
        print("SHOT STATISTICS")
        print("="*60)
        
        # Count shot types
        shot_types = {}
        directions = {}
        strokes = {}
        spins = {}
        for cls in all_classifications:
            shot_types[cls['type']] = shot_types.get(cls['type'], 0) + 1
            directions[cls['direction']] = directions.get(cls['direction'], 0) + 1
            strokes[cls.get('stroke', 'unknown')] = strokes.get(cls.get('stroke', 'unknown'), 0) + 1
            spins[cls.get('spin', 'unknown')] = spins.get(cls.get('spin', 'unknown'), 0) + 1
        
        print("\nShot Types:")
        for shot_type, count in sorted(shot_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {shot_type.replace('_', ' ').title()}: {count}")
        
        print("\nStrokes (Forehand/Backhand):")
        for stroke, count in sorted(strokes.items(), key=lambda x: x[1], reverse=True):
            print(f"  {stroke.title()}: {count}")
        
        print("\nSpin Types:")
        for spin, count in sorted(spins.items(), key=lambda x: x[1], reverse=True):
            print(f"  {spin.title()}: {count}")
        
        print("\nShot Directions:")
        for direction, count in sorted(directions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {direction.replace('_', ' ').title()}: {count}")
        
        print("="*60)
    
    print(f"\nTotal tracks analyzed: {len(track_histories)}")
    print(f"Total frames processed: {frame_idx}")


if __name__ == '__main__':
    main()
