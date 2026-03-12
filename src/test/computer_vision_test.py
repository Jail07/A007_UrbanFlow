# yolov3

import cv2
import numpy as np

# from src.constants.config import VIDEO_PATH, VIDEO_PATH_1, YOLOV3_WEIGHTS_PATH, YOLOV3_CFG_PATH


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
VIDEO_DIR = DATA_DIR / "video"
YOLOV3_DIR = DATA_DIR / "yolov3"

VIDEO_PATH = VIDEO_DIR / "video_0.mp4"
VIDEO_PATH_1 = VIDEO_DIR / "video_1.mp4"

YOLOV3_CFG_PATH = YOLOV3_DIR / "yolov3.cfg"
YOLOV3_WEIGHTS_PATH = YOLOV3_DIR / "yolov3.weights"


net = cv2.dnn.readNet(YOLOV3_WEIGHTS_PATH, YOLOV3_CFG_PATH)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# cap = cv2.VideoCapture(VIDEO_PATH)
cap = cv2.VideoCapture(VIDEO_PATH_1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

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

            print(f"Class ID: {class_id}, confidence: {confidence}")

            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = center_x - w // 2
                y = center_y - h // 2

                print(f"Center X: {x}, Center Y: {y}, W: {w}, H: {h}")

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(f"Class IDs: {len(indexes)}")
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            print(f"Label: {label}, confidence: {confidences[i]}, coordinates: {(x, y, w, h)}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()