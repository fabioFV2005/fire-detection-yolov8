from ultralytics import YOLO
from roboflow import Roboflow
from IPython.display import Image
import os
from dotenv import load_dotenv
# load environment variables from .env file
load_dotenv()
api_key = os.getenv("DATA_SET_API_KEY")
workspace = os.getenv("ROBOFLOW_WORKSPACE")
project_name = os.getenv("ROBOFLOW_PROJECT")
version_number = os.getenv("ROBOFLOW_VERSION")

if __name__ == "__main__":
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_number)
    dataset = version.download("yolov8")
    print(dataset.location)
    model = YOLO("yolov8n.pt")
    data = dataset.location + "/data.yaml"
    results = model.train(data=data, epochs=50, imgsz=640, plots=True, device=0, task="detect")
    Image(filename="./content/runs/detect/train3/results.png", width=600)