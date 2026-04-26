from ultralytics import YOLO

def test_model_load():
    model = YOLO("yolov8n.pt")
    assert model is not None

def test_inference_image():
    model = YOLO("yolov8n.pt")
    results = model("https://ultralytics.com/images/bus.jpg")
    assert len(results) > 0