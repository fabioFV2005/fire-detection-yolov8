from ultralytics import YOLO
import os
import pytest

MODEL_PATH = "runs/detect/train/weights/best.pt"
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Model not found")
def test_trained_model_inference():
    model = YOLO(MODEL_PATH)
    results = model("test.jpg")
    assert results is not None