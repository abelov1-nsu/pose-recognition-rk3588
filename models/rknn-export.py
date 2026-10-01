from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")
model.export(
    format="rknn",
    name="rk3588"
)