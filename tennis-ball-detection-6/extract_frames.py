import cv2
import os

video = r"input\Sinner vs Alcaraz   4K Highli.MOV"
out_folder = r"sinner_alcaraz_dataset\train"  # Updated path
os.makedirs(out_folder, exist_ok=True)

cap = cv2.VideoCapture(video)
count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Har 10th frame save karo
    if count % 10 == 0:
        path = os.path.join(out_folder, f"sinner_{saved:04d}.jpg")
        cv2.imwrite(path, frame)
        saved += 1
    count += 1

print(f"✅ Saved {saved} frames!")
print(f"📁 Location: {out_folder}")
cap.release()