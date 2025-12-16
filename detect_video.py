"""Run YOLOv8 detection on a video and save the output with bounding boxes.

Usage examples (PowerShell):
python detect_video.py --weights runs/train/tennis_ball_run/weights/best.pt --source input.mp4 --output out.mp4 --conf 0.25
"""
import argparse
import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to trained weights (.pt)')
    parser.add_argument('--source', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='out.mp4', help='Output video path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--device', type=str, default=None, help='Device string e.g. "0" for GPU or "cpu"')
    parser.add_argument('--show', action='store_true', help='Show video while processing')
    return parser.parse_args()


def draw_boxes(frame, boxes, scores, classes, class_names=None):
    for (x1, y1, x2, y2), conf, cls in zip(boxes.astype(int), scores, classes):
        label = f"{int(cls)} {conf:.2f}"
        if class_names and int(cls) < len(class_names):
            label = f"{class_names[int(cls)]} {conf:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return frame


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print('Error opening video:', args.source)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    model = YOLO(args.weights)

    # Try to get class names if available in model
    try:
        class_names = model.model.names
    except Exception:
        class_names = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run prediction on the frame
        results = model(frame, conf=args.conf, device=args.device)
        # results is a list-like; get first
        r = results[0]

        # Extract boxes, confidences, classes
        try:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.empty((0, 4))
            scores = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.empty((0,))
            classes = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, 'cls') else np.empty((0,))
        except Exception:
            # Fallback if attributes differ
            xyxy = []
            scores = []
            classes = []
            for box in r.boxes:
                xyxy.append(box.xyxy)
                scores.append(float(box.conf))
                classes.append(int(box.cls))
            boxes = np.array(xyxy)
            scores = np.array(scores)
            classes = np.array(classes)

        if boxes.shape[0] > 0:
            frame = draw_boxes(frame, boxes, scores, classes, class_names)

        out.write(frame)

        if args.show:
            cv2.imshow('out', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f'Processed {frame_idx} frames. Output saved to {args.output}')


if __name__ == '__main__':
    main()
