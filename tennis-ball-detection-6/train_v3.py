from ultralytics import YOLO

if __name__ == '__main__':
    
    # Naya combined model base
    model = YOLO('runs/detect/models/tennis_v2_combined-2/weights/best.pt')

    model.train(
        data    = 'sinner_alcaraz_dataset/data.yaml',
        epochs  = 60,        # Kam epochs - already trained hai
        imgsz   = 640,
        batch   = 8,
        device  = 0,
        workers = 4,
        amp     = True,
        cache   = True,

        flipud  = 0.3,
        fliplr  = 0.5,
        mosaic  = 1.0,
        mixup   = 0.1,

        project = 'models',
        name    = 'tennis_v3_sinner',
    )

    print("\n✅ Training Complete!")
    print("Model: models/tennis_v3_sinner/weights/best.pt")