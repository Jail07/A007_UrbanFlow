import cv2
import json
import numpy as np
import time
import csv

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.constants.config import VIDEO_PATH, CAMERA_CONFIG_JSON_PATH_v1, LOG_FILE_PATH_V1, YOLOV8_CFG_PATH


def load_lanes_config(config_path, intersection_id, camera_id):
    with open(config_path, 'r') as f:
        config = json.load(f)

    raw_lanes = config[intersection_id][camera_id]

    lanes = {}
    for lane_name, points in raw_lanes.items():
        if not isinstance(points, list):
            continue
        lanes[lane_name] = np.array(points, np.int32)

    return lanes


model = YOLO(YOLOV8_CFG_PATH)
tracker = DeepSort(max_age=30)
cap = cv2.VideoCapture(VIDEO_PATH)

LANES = load_lanes_config(
    CAMERA_CONFIG_JSON_PATH_v1,
    'intersection_001',
    'camera_south'
)

target_classes = {
    2: "car"
}

LANE_COLORS = {
    "lane_left": (0, 0, 255),
    "lane_center": (0, 255, 0),
    "lane_right": (255, 0, 0)
}

LANE_LABELS = {
    "lane_left": "left",
    "lane_center": "center",
    "lane_right": "right"
}



with open(LOG_FILE_PATH_V1, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "total_vehicles",
        "lane_left", "lane_center", "lane_right",
        "vehicles_details"
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        overlay = frame.copy()
        for name, poly in LANES.items():
            if name in LANE_COLORS:
                cv2.fillPoly(overlay, [poly], LANE_COLORS[name])

        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        results = model(frame, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls in target_classes and conf > 0.5:
                    label = target_classes[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    detections.append((
                        [x1, y1, x2 - x1, y2 - y1],
                        conf,
                        label
                    ))

        tracks = tracker.update_tracks(detections, frame=frame)

        lane_counts = {k: 0 for k in LANE_LABELS.keys()}
        current_vehicles = []

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            v_class = track.get_det_class()

            l, t, r, b = map(int, track.to_ltrb())

            cx = (l + r) // 2
            cy = b

            vehicle_lane = "unknown"

            for lane_name, poly in LANES.items():
                if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                    vehicle_lane = lane_name
                    lane_counts[lane_name] += 1
                    break

            if vehicle_lane in LANE_LABELS:
                current_vehicles.append(
                    f"ID_{track_id}({v_class})->{vehicle_lane}"
                )

            lane_label = LANE_LABELS.get(vehicle_lane, "unknown")

            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)
            cv2.rectangle(frame, (l, t), (r, b), (255, 255, 255), 2)

            cv2.putText(
                frame,
                f"ID:{track_id} {lane_label}",
                (l, t - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        total = sum(lane_counts.values())
        details_str = "; ".join(current_vehicles)

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            total,
            lane_counts["lane_left"],
            lane_counts["lane_center"],
            lane_counts["lane_right"],
            details_str
        ])

        cv2.putText(
            frame,
            f"TOTAL: {total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("UrbanFlow - Polygons FIXED", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()