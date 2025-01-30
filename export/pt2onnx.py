from ultralytics import YOLO


# model = YOLO(r'../trains_det/hands_s_250110/train/weights/best.pt')
model = YOLO(r'../trains_det/kx_s_250126/S2_prune/step_pre_val/last_x0.8.pt')

model.export(format="onnx", simplify=True, opset=11)
