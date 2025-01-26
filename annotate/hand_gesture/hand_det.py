import cv2
from tests_det.test_utils.ops_image import draw_detections_pipeline, resize_image
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from annotate.hand_gesture.det_test_utils import plot_bbox, expand_bbox

overrides_1 = {"task": "det",
               "mode": "predict",
               # "model": r'../trains_det/hands_07_11_m/train/weights/best.pt',
               # "model": r"../export/output/best.onnx",
               "model": r"../export/best_int32.onnx",
               "verbose": False,
               "iou": 0.5,
               "conf": 0.1,
               "save": False
               }

overrides_2 = {"task": "det",
               "mode": "predict",
               "model": r'../trains_det/hands_08_12_m/train/weights/best.pt',
               "verbose": False,
               "iou": 0.5,
               "conf": 0.1,
               "save": False
               }

overrides_3 = {"task": "det",
               "mode": "predict",
               "model": r'../weights/yolov8m.pt',
               "verbose": False,
               "classes": [0],
               "iou": 0.5,
               "conf": 0.1,
               "save": False
               }

pred_shou_1 = DetectionPredictor(overrides=overrides_1)
pred_shou_2 = DetectionPredictor(overrides=overrides_2)
pred_ren = DetectionPredictor(overrides=overrides_3)


# img_path = "../ultralytics/assets/zidane.jpg"
# img_path = "/home/chenjun/code/ultralytics_YOLOv8/ultralytics/assets/img_3.png"
# img_path = r"/home/chenjun/code/ultralytics_YOLOv8/export/paml_1.jpg"
# img_path = r"/home/chenjun/code/ultralytics_YOLOv8/ultralytics/assets/bus.jpg"
img_path = r"/home/chenjun/code/datasets/bank_monitor/原始数据/2024-4-25/3_monitor_far/34311525_2228395034.jpg"

image = cv2.imread(img_path)
h, w, _ = image.shape

shou_1 = pred_shou_1(image)[0].cpu().numpy()
image_1 = draw_detections_pipeline(image, shou_1.boxes.xyxy, shou_1.boxes.conf, shou_1.boxes.cls, {0: "hand"})
shou_2 = pred_shou_2(image)[0].cpu().numpy()
image_2 = draw_detections_pipeline(image, shou_2.boxes.xyxy, shou_2.boxes.conf, shou_2.boxes.cls, {0: "hand"})

det_ren = pred_ren(image)[0].cpu().numpy()
# image_3 = plot_bbox(image, det_ren)

# cv2.imshow('Image', image_1)
# cv2.waitKey(0)
# cv2.imshow('Image', image_2)
# cv2.waitKey(0)

image_copy_1 = image.copy()
image_copy_2 = image.copy()
for ren in det_ren.boxes.xyxy:
    x1, y1, x2, y2 = expand_bbox(ren, w, h, 0.05)
    image_ren = image[y1:y2, x1:x2].copy()
    shou_1 = pred_shou_1(image_ren)[0].cpu().numpy()
    shou_2 = pred_shou_2(image_ren)[0].cpu().numpy()
    image_patch_1 = draw_detections_pipeline(image_ren, shou_1.boxes.xyxy, shou_1.boxes.conf, shou_1.boxes.cls, {0: "hand"})
    image_patch_2 = draw_detections_pipeline(image_ren, shou_2.boxes.xyxy, shou_2.boxes.conf, shou_2.boxes.cls, {0: "hand"})

    image_copy_1[y1:y2, x1:x2] = image_patch_1
    image_copy_2[y1:y2, x1:x2] = image_patch_2

    # cv2.imshow('Image', image_copy_1)
    # cv2.waitKey(0)

image_copy_1 = plot_bbox(image_copy_1, det_ren)
image_copy_2 = plot_bbox(image_copy_2, det_ren)
# cv2.imshow('Image', image_copy_1)
# cv2.waitKey(0)
# cv2.imshow('Image', image_copy_2)
# cv2.waitKey(0)


first_row = cv2.hconcat([image_1, image_2])
second_row = cv2.hconcat([image_copy_1, image_copy_2])
final_image = cv2.vconcat([first_row, second_row])
final_image, _ = resize_image(final_image, 720)

cv2.imshow("Final Image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
