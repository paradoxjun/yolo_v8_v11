import cv2
import os
import yaml
from tests_det.test_utils.ops_image import draw_detections_pipeline
from ultralytics.models.yolo.detect.predict import DetectionPredictor

model_cfg_path = "setting.yaml"
img_root = r"G:\datasets\bank_scene\00-task\det_kx_1101\val\images"
class_name = {
    0: "kx",
    1: "kx_dk",
}

with open(model_cfg_path) as f:
    overrides_cfg = yaml.safe_load(f.read())

predictor = DetectionPredictor(overrides=overrides_cfg)

image_list = os.listdir(img_root)
for image_name in image_list:
    img_path = os.path.join(img_root, image_name)
    print(img_path)
    image_bgr = cv2.imread(img_path)
    # image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    det_res = predictor(image_bgr)[0].cpu().numpy()
    image_show = draw_detections_pipeline(image_bgr, det_res.boxes.xyxy, det_res.boxes.conf, det_res.boxes.cls,
                                          class_name)
    cv2.imshow("Image", image_show)

    key = cv2.waitKey(0) & 0xFF  # Wait for interval seconds
    if key == ord('q'):
        break

cv2.destroyAllWindows()
