# yolov3
# python -m src.test.computer_vision_test

import cv2
import numpy as np
import csv
import os
import time
from src.constants.config import VIDEO_PATH, YOLOV3_WEIGHTS_PATH, YOLOV3_CFG_PATH

net = cv2.dnn.readNet(YOLOV3_WEIGHTS_PATH, YOLOV3_CFG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]

target_classes = [2, 3, 5, 7]

cap = cv2.VideoCapture(VIDEO_PATH)
LOG_FILE = "data/logs/traffic_log_v1.csv"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


def init_csv():
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "class_id", "label", "confidence", "x", "y", "w", "h"])


init_csv()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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

            if confidence > 0.3 and class_id in target_classes:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if len(indexes) > 0:
            flat_indexes = np.array(indexes).flatten()
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                conf = confidences[i]
                ts = time.time()

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                writer.writerow([ts, class_ids[i], label, conf, x, y, w, h])

    cv2.imshow("UrbanFlow Vision V1 (YOLOv3)", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print(f"Данные успешно сохранены в {LOG_FILE}")