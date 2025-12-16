"""Train YOLOv8 on the provided dataset.

Usage examples (PowerShell):
python train_yolo.py --data data.yaml --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16
"""
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data.yaml', help='Path to dataset yaml')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Backbone or weights to start from')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--name', type=str, default='tennis_ball_run', help='Run name for outputs')
    parser.add_argument('--device', type=str, default=None, help='Device string, eg. "0" or "cpu"')
    return parser.parse_args()


def main():
    args = parse_args()
    print('Training YOLOv8 with these arguments:', args)

    model = YOLO(args.model)

    # Train - ultralytics YOLO API accepts these kwargs
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
    )


if __name__ == '__main__':
    main()
