import os
import cv2
from ultralytics.task_bank.detection.predict import BankDetectionPredictor
from ultralytics.task_bank.pose_body.predict import PosePredictor


def expand_bbox(xyxy, img_width, img_height, scale=0.1):
    # 计算宽度和高度，和中心点
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    center_x = xyxy[0] + width / 2
    center_y = xyxy[1] + height / 2

    # 增加10%的宽度和高度
    new_width = width * (1 + scale)
    new_height = height * (1 + scale)

    # 计算新的边界框坐标，并确保新的边界框坐标不超过图片的边界
    new_x1 = max(2, int(center_x - new_width / 2))
    new_y1 = max(2, int(center_y - new_height / 2))
    new_x2 = min(int(img_width) - 2, int(center_x + new_width / 2))
    new_y2 = min(int(img_height), int(center_y + new_height / 2))

    return new_x1, new_y1, new_x2, new_y2


def get_video(video_path, read_from_camera=False):
    if read_from_camera:  # 使用摄像头获取视频
        v_cap = cv2.VideoCapture(0)
    else:
        assert os.path.isfile(video_path), "Video path in method get_video() is error. "
        v_cap = cv2.VideoCapture(video_path)

    return v_cap


def plot_bbox(image_ori, det_res, color=(0, 0, 255), offset=(0, 0)):
    # 根据检测结果绘制图像
    image = image_ori.copy()
    for i, bbox in enumerate(det_res.boxes.xyxy):
        x1, y1, x2, y2 = list(map(int, bbox))
        conf = det_res.boxes.conf[i]
        cls = det_res.boxes.cls[i]
        label = f'{det_res.names[int(cls)]} {float(conf):.2f}'

        # 绘制边界框和标签
        cv2.rectangle(image, (x1 + offset[0], y1 + offset[1]), (x2 + offset[0], y2 + offset[1]), color, 2)
        cv2.putText(image, label, (x1 + offset[0], y1 + offset[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


_connections = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19),
                (19, 20))


def plot_keypoints(image, keypoints, color=(0, 255, 0), offset=(0, 0), connections=_connections):
    if keypoints is not None:
        for data in keypoints.xy:
            if len(data) == 0:
                continue

            if connections is not None:
                for start_idx, end_idx in connections:
                    sta_point = data[start_idx]
                    end_point = data[end_idx]
                    if (sta_point[0] > 0 or sta_point[1] > 0) and (end_point[0] > 0 and end_point[1] > 0):  # 忽略无效点
                        cv2.line(image, (int(sta_point[0] + offset[0]), int(sta_point[1] + offset[1])),
                                 (int(end_point[0] + offset[0]), int(end_point[1] + offset[1])), (60, 179, 113), 2)

            for point in data:
                x, y = point[:2]
                if x > 0 or y > 0:  # 忽略无效点
                    cv2.circle(image, (int(x + offset[0]), int(y + offset[1])), 3, color, -1)

    return image


_overrides_ren_pose = {"task": "pose",
                       "mode": "predict",
                       "model": r'../weights/yolov8l-pose.pt',
                       "verbose": False,
                       "classes": [0],
                       "iou": 0.5,
                       "conf": 0.3
                       }

_overrides_ren_det = {"task": "det",
                      "mode": "predict",
                      "model": r'../weights/yolov8m.pt',
                      "verbose": False,
                      "classes": [0],
                      "iou": 0.5,
                      "conf": 0.3
                      }

_overrides_shou_pose = {"task": "pose",
                        "mode": "predict",
                        # "model": r'../trains_pose/hands_08_01_s/train/weights/best.pt',
                        "model": r'../trains_pose/hands_08_08_s/train2/weights/best.pt',
                        # "model": r'../trains_pose/hands_08_08_m/train/weights/best.pt',
                        "verbose": False,
                        "classes": [0],
                        "iou": 0.9,
                        "conf": 0.2
                        }

_overrides_shou_det = {"task": "det",
                       "mode": "predict",
                       "model": r'../trains_det/hands_08_12_m/train/weights/best.pt',
                       # "model": r'../trains_det/hands_07_11_m/train/weights/best.pt',
                       "verbose": False,
                       # "classes": [0],
                       "iou": 0.6,
                       "conf": 0.1
                       }

predictor_ren_pose = PosePredictor(overrides=_overrides_ren_pose)
predictor_ren_det = BankDetectionPredictor(overrides=_overrides_ren_det)
predictor_shou_pose = PosePredictor(overrides=_overrides_shou_pose)
predictor_shou_det = BankDetectionPredictor(overrides=_overrides_shou_det)
