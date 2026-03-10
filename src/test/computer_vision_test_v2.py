# yolov8

import cv2

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

vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    detections = []

    for r in results:
        boxes = r.boxes

        for box in boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls in vehicle_classes and conf > 0.5:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                w = x2 - x1
                h = y2 - y1

                detections.append(([x1, y1, w, h], conf, "car"))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (l, t - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()