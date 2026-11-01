"""Live pothole detection + severity classification.

Stage 1: YOLO finds potholes in each frame.
Stage 2: each detected pothole is cropped and rated by a ResNet18 classifier.

Run:  python live.py      (press q in the video window to quit)
Needs: runs/detect/pothole_runs/v1/weights/best.pt and severity_layer4.pt
"""
import time

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

# ---- Settings ---------------------------------------------------------------
SOURCE = 0            # 0 = webcam, or a video path such as "road_video.mp4"
CONF = 0.5            # YOLO confidence threshold (lower values give false positives)
YOLO_WEIGHTS = "runs/detect/pothole_runs/v1/weights/best.pt"
CLASSIFIER_WEIGHTS = "severity_layer4.pt"

# ---- Load models (once, before the loop) -------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO(YOLO_WEIGHTS)

classifier = models.resnet18(weights=None)
classifier.fc = nn.Linear(classifier.fc.in_features, 3)
classifier.load_state_dict(torch.load(CLASSIFIER_WEIGHTS, map_location=device))
classifier = classifier.to(device)
classifier.eval()

class_names = ["1_minor", "2_moderate", "3_severe"]
colors = {  # BGR, because OpenCV uses blue-green-red order
    "1_minor": (0, 255, 0),
    "2_moderate": (0, 165, 255),
    "3_severe": (0, 0, 255),
}
crop_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- Video loop ----------------------------------------------------------------
cap = cv2.VideoCapture(SOURCE)
if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {SOURCE}")

prev_time = time.time()

while True:
    ok, frame = cap.read()
    if not ok:  # camera failed or the video ended
        break

    results = yolo_model(frame, conf=CONF, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(x1, 0), max(y1, 0)  # keep the crop inside the frame

        crop_bgr = frame[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        crop = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        x = crop_transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            scores = classifier(x)
        label = class_names[scores.argmax(1).item()]

        color = colors[label]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    now = time.time()
    fps = 1 / max(now - prev_time, 1e-6)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Pothole detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
