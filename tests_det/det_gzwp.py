import cv2
import os
from ops_image import draw_detections_pipeline
from ultralytics.models.yolo.detect.predict import DetectionPredictor

overrides = {"task": "det",
             "model": r'../trains_det/gzwp_09_04_s/train/weights/best.onnx',
             # "model": r'../trains_det/gzwp_08_12_m/train/weights/best.pt',
             "verbose": False,
             "iou": 0.5,
             "conf": 0.5,
             "save": False,
             }

predictor = DetectionPredictor(overrides=overrides)

img_root = r"/home/chenjun/code/datasets/内部数据/bank2406-抽帧标注-钱与手/ATM遗留物品/dataset/JPEGImages"
class_name = {
    0: "phone",
    1: "wallet",
    2: "card",
    3: "money",
    4: "zbm",
}
image_list = os.listdir(img_root)
for image_name in image_list:
    img_path = os.path.join(img_root, image_name)
    print(img_path)
    image_bgr = cv2.imread(img_path)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gzwp = predictor(image_bgr)[0].cpu().numpy()
    image_show = draw_detections_pipeline(image_bgr, gzwp.boxes.xyxy, gzwp.boxes.conf, gzwp.boxes.cls, class_name)
    cv2.imshow("Image", image_show)

    key = cv2.waitKey(0) & 0xFF  # Wait for interval seconds
    if key == ord('q'):
        break

cv2.destroyAllWindows()
