# pylint: disable=no-member,no-name-in-module
from ultralytics import YOLO
import cv2
import numpy as np


# load the trained model
model = YOLO("weights/best.pt")



# open cv configuration to read media files
# create a named window for display and set it to be resizable
cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
# width, height = 800, 600
cv2.resizeWindow("Video", 800, 600)
media = cv2.VideoCapture("test_videos/fire_game.mp4")
if not media.isOpened():
    print("Error opening video stream or file")
else:
    print("Video file opened successfully")

ret, frame = media.read()
while True:
    ret, frame = media.read()

    if not ret:
        break
    results = model(frame)
    annotated_frame = results[0].plot()
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cx = int((x1 + x2)/2)
        cy = int((y1 + y2)/2)
        cv2.circle(annotated_frame,(cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(
            annotated_frame,
            (f'Center: ({cx}, {cy})'),
            (cx+5, cy-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )
    cv2.imshow("Video", annotated_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
media.release()
cv2.destroyAllWindows()