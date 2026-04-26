# pylint: disable=no-member,no-name-in-module
import cv2

def test_opencv_installed():
    assert cv2.__version__ is not None

def test_video_capture():
    cap = cv2.VideoCapture(0)
    assert cap.isOpened() is True
    cap.release()