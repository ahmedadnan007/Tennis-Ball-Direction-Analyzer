from ultralytics import YOLO

if __name__ == '__main__':

    # ✅ 570 wala purana model load karo
    model = YOLO('runs/detect/tennis_ball_run/weights/best.pt')

    model.train(
        data    = 'tennis-ball-detection-6/data.yaml',
        epochs  = 100,
        imgsz   = 640,
        batch   = 8,
        device  = 0,
        workers = 4,
        amp     = True,
        cache   = True,

        flipud  = 0.3,
        fliplr  = 0.5,
        mosaic  = 1.0,
        mixup   = 0.15,
        hsv_h   = 0.015,
        hsv_s   = 0.7,
        hsv_v   = 0.4,

        project = 'models',
        name    = 'tennis_v2_combined',
    )

    print("\n✅ Training Complete!")
    print("Model: models/tennis_v2_combined/weights/best.pt")