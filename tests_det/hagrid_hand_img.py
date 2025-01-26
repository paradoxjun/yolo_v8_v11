import cv2
from ultralytics.values.color import colors_dict_ch_bgr
from ultralytics.task_bank.utils.ops import resize_and_pad
from tests_det.det_test_utils import (get_video, plot_bbox, plot_keypoints, predictor_ren_det, predictor_ren_pose,
                                      predictor_shou_det, predictor_shou_pose, expand_bbox)

# img_path = r"/home/chenjun/code/datasets/bank_monitor/2024-05-28-柜台内结账场景-collect/bad_2/267869299_1943473399.jpg"
# img_path = r"/home/chenjun/code/ultralytics_YOLOv8/export/paml_1.jpg"
img_path = r"../ultralytics/assets/zidane.jpg"
# img_path = r"/home/chenjun/code/datasets/hagrid/hagrid_yolo_det/test/images/one/0b33a988-c20d-48f1-9abe-2a42c3d9231d.jpg"
# img_path = r"/home/chenjun/code/datasets/hagrid/hagrid_yolo_det/test/images/ok/0a799e8d-77fe-4dd7-ab63-4b2fbc6ac574.jpg"
# img_path = r"/home/chenjun/code/datasets/hagrid/hagrid_yolo_det/test/images/stop/1d6d0d5a-be5f-4afa-b7fa-46110721c454.jpg"
# img_path = r"/home/chenjun/code/ultralytics_YOLOv8/ultralytics/assets/bus.jpg"
# img_path = r'/home/chenjun/code/ultralytics_YOLOv8/ultralytics/assets/img_3.png'

img_bgr = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
det_ren = predictor_ren_pose(img_rgb)[0]

h, w, _ = img_rgb.shape

# 画出人的检测框并显示
image_show = plot_bbox(img_bgr, det_ren)
cv2.imshow('Image', image_show)
cv2.waitKey(0)

# 画出人的关键点并显示
# image_show = plot_keypoints(image_show, det_ren.keypoints, connections=((9, 7), (7, 5), (5, 6), (6, 8), (8, 10)))
# cv2.imshow('Image', image_show)
# cv2.waitKey(0)

for ren in det_ren.boxes.xyxy:
    x1, y1, x2, y2 = expand_bbox(ren, w, h, 0.0)
    image_ren = img_rgb[y1:y2, x1:x2]

    det_shou = predictor_shou_det(image_ren)[0]
    print(det_shou.boxes.xyxy)
    # print(det_shou.boxes.xyxy)
    image_show = plot_bbox(img_bgr, det_shou, color=colors_dict_ch_bgr["橙"], offset=(x1, y1))
    cv2.imshow('Image', image_show)
    cv2.waitKey(0)

    for shou in det_shou.boxes.xyxy:
        print(det_shou.boxes.xyxy)
        x11, y11, x22, y22 = expand_bbox(shou, w, h, 2/3)
        image_shou = image_ren[y11:y22, x11:x22]
        # image_shou_bgr = cv2.cvtColor(image_shou, cv2.COLOR_BGR2RGB)
        # cv2.imshow('Image', image_shou_bgr)
        # cv2.waitKey(0)
        pose_shou = predictor_shou_pose(image_shou)[0]
        image_show = plot_keypoints(image_show, pose_shou.keypoints, color=colors_dict_ch_bgr["深天蓝"],
                                    offset=(x1 + x11, y1 + y11))

cv2.imshow('Image', image_show)
cv2.waitKey(0)
cv2.destroyAllWindows()
