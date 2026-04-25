!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="Qi37FpOq1jfVKwxNKLCl")
project = rf.workspace("-jwzpw").project("continuous_fire")
version = project.version(1)
dataset = version.download("yolov8")
                