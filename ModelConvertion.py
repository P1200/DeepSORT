import cv2
from ultralytics import YOLO

onnx_model_path = "./Model/yolo11n.onnx"

model = YOLO("Model/yolo11n.pt")
model.export(format="onnx")

# The Magic:
net = cv2.dnn.readNetFromONNX(onnx_model_path)
