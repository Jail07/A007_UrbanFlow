import cv2
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        for i in range(len(ids)):
            c = corners[i][0]
            cx = int((c[0][0] + c[2][0]) / 2)
            cy = int((c[0][1] + c[2][1]) / 2)

            track_id = ids[i][0]

            cv2.polylines(frame, [c.astype(int)], True, (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id}", (cx, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("UrbanFlow - ArUco Tracking", frame)
    if cv2.waitKey(1) == 27: break