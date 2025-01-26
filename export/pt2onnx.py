from ultralytics import YOLO

model = YOLO(r'../trains_det/hands_25_01_10_s/train2/weights/best.pt')
model.export(format="onnx", simplify=True, opset=11)
