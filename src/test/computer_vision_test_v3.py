import cv2
import csv
import time

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.constants.config import VIDEO_PATH_1, LOG_FILE_PATH_V3

model = YOLO("yolov8n-seg.pt")
tracker = DeepSort(max_age=30)
cap = cv2.VideoCapture(VIDEO_PATH_1)

target_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


with open(LOG_FILE_PATH_V3, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "total_vehicles",
        "lane_left", "lane_center", "lane_right",
        "vehicles_details"
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        height, width, _ = frame.shape
        lane_w = 1600 // 3

        results = model(frame, verbose=False)
        detections = []

        annotated_frame = results[0].plot()

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in target_classes and conf > 0.5:
                    label = target_classes[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, label))

        tracks = tracker.update_tracks(detections, frame=annotated_frame)

        lane_counts = {"left": 0, "center": 0, "right": 0}
        current_vehicles = []

        for track in tracks:
            if not track.is_confirmed(): continue

            track_id = track.track_id
            v_class = track.get_det_class()
            l, t, r, b = map(int, track.to_ltrb())

            cx = (l + r) // 2

            if cx < lane_w:
                lane = "left"
            elif cx < 2 * lane_w:
                lane = "center"
            else:
                lane = "right"

            lane_counts[lane] += 1
            current_vehicles.append(f"ID_{track_id}({v_class})->{lane}")

            cv2.putText(annotated_frame, f"ID:{track_id} {lane}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # Черная обводка
            cv2.putText(annotated_frame, f"ID:{track_id} {lane}", (l, t - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)  # Белый текст

        total = sum(lane_counts.values())
        details_str = "; ".join(current_vehicles)

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"), total,
            lane_counts["left"], lane_counts["center"], lane_counts["right"],
            details_str
        ])

        cv2.line(annotated_frame, (lane_w, 0), (lane_w, height), (0, 255, 255), 2)
        cv2.line(annotated_frame, (2 * lane_w, 0), (2 * lane_w, height), (0, 255, 255), 2)

        cv2.imshow("UrbanFlow - Segmentation + Tracking", annotated_frame)
        if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()