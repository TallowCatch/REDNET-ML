from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])  # allow YOLO checkpoint class to load

model = YOLO("yolov8n.yaml")
model.train(data="yolo/rednet_det.yaml", imgsz=640, epochs=50,
            batch=16, project="runs/yolo", name="hab_yolov8n") # should change this to hab_yolov8n3

model.val(data="yolo/rednet_det.yaml", imgsz=640)

model.export(format="onnx", imgsz=640, opset=12)  # saves to runs/yolo/hab_yolov8n/weights/best.onnx
