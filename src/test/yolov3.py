import cv2
import numpy as np
import csv
import time

from src.constants.config import VIDEO_PATH, YOLOV3_WEIGHTS_PATH, YOLOV3_CFG_PATH

net = cv2.dnn.readNet(YOLOV3_WEIGHTS_PATH, YOLOV3_CFG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

cap = cv2.VideoCapture(VIDEO_PATH)

LOG_FILE = "data/logs/yolov3_polygons.csv"

with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "timestamp", "total_vehicles",
        "lane_left", "lane_center", "lane_right"
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        lane_w = width // 3

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416),
                                     (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = center_x - w // 2
                    y = center_y - h // 2

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        lane_counts = {"left": 0, "center": 0, "right": 0}

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]

                cx = x + w // 2

                if cx < lane_w:
                    lane = "left"
                elif cx < 2 * lane_w:
                    lane = "center"
                else:
                    lane = "right"

                lane_counts[lane] += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                cv2.putText(
                    frame,
                    f"{lane}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        total = sum(lane_counts.values())

        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            total,
            lane_counts["left"],
            lane_counts["center"],
            lane_counts["right"]
        ])

        cv2.line(frame, (lane_w, 0), (lane_w, height), (0, 255, 255), 2)
        cv2.line(frame, (2 * lane_w, 0), (2 * lane_w, height), (0, 255, 255), 2)

        cv2.putText(frame, f"TOTAL: {total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("YOLOv3 + Lanes + CSV", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()