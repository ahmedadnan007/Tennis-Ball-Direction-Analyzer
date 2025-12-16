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
from utils.helpers import (
    draw_trajectory_on_frame, 
    draw_bbox_with_label, 
    draw_info_panel,
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
    
    print("[INFO] Starting tracking and analysis...")
    
    frame_idx = 0
    track_histories = {}  # Store trajectory history for each track ID
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
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
        
        # Analyze and display trajectory info
        info_y = 30
        for track_id, history in track_histories.items():
            if len(history) >= 10:  # Only analyze if enough points
                # Analyze trajectory
                analysis = analyzer.analyze_trajectory(history, track_id)
                
                if analysis:
                    # Display info on frame
                    info = {
                        f"Track {track_id}": f"{analysis['avg_speed_kmh']:.1f} km/h",
                    }
                    draw_info_panel(frame, info, position=(10, info_y))
                    info_y += 35
        
        # Draw frame counter
        cv2.putText(frame, f"Frame: {frame_idx}", (width - 150, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Display if requested
        if args.show:
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
    
    # Final trajectory analysis
    print("\n" + "="*60)
    print("TRAJECTORY ANALYSIS SUMMARY")
    print("="*60)
    
    for track_id, history in track_histories.items():
        if len(history) >= 5:
            analysis = analyzer.analyze_trajectory(history, track_id)
            if analysis:
                print(analyzer.get_trajectory_summary(analysis))
                print("-" * 60)
    
    print(f"\nTotal tracks analyzed: {len(track_histories)}")
    print(f"Total frames processed: {frame_idx}")


if __name__ == '__main__':
    main()
