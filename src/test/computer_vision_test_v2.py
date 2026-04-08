# yolov8
# python -m src.test.computer_vision_test_v2

import cv2
import csv
import time

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.constants.config import VIDEO_PATH_1

model = YOLO("yolov8n.pt")
# yolov8n.pt   (nano — очень быстро)
# yolov8s.pt   (small)
# yolov8m.pt   (medium)
# yolov8l.pt   (large)
# yolov8x.pt   (extra large)

tracker = DeepSort(max_age=30)
cap = cv2.VideoCapture(VIDEO_PATH_1)

# Маппинг классов COCO: 2-car, 3-motorcycle, 5-bus, 7-truck
target_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

with open("data/logs/cv_detections_v2.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "car", "bus", "truck", "motorcycle", "total_count"])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame)
        detections = []

        counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in target_classes and conf > 0.5:
                    label = target_classes[cls]
                    counts[label] += 1

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

        tracks = tracker.update_tracks(detections, frame=frame)

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            counts["car"],
            counts["bus"],
            counts["truck"],
            counts["motorcycle"],
            sum(counts.values())
        ])

        for track in tracks:
            if not track.is_confirmed(): continue
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(frame, f"{track.track_id} {track.get_det_class()}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("UrbanFlow CV v2", frame)
        if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()