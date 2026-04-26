# pylint: disable=no-member,no-name-in-module
from ultralytics import YOLO
import cv2


# load the trained model
model = YOLO("weights/best.pt")



# open cv configuration to read media files
# create a named window for display and set it to be resizable
cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
# width, height = 800, 600
cv2.resizeWindow("Video", 800, 600)
media = cv2.VideoCapture("test_videos/fire.mp4")
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
    cv2.imshow("Video", annotated_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
media.release()
cv2.destroyAllWindows()