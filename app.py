from ultralytics import YOLO
from roboflow import Roboflow
import ultralytics
from IPython.display import Image

if __name__ == "__main__":
    rf = Roboflow(api_key="Qi37FpOq1jfVKwxNKLCl")
    project = rf.workspace("-jwzpw").project("continuous_fire")
    version = project.version(1)
    dataset = version.download("yolov8")
    print(dataset.location)
    model = YOLO("yolov8n.pt")
    data = dataset.location + "/data.yaml"
    results = model.train(data=data, epochs=1, imgsz=640, plots=True, device=0, task="detect")
    Image(filename="./content/runs/detect/train3/results.png", width=600)