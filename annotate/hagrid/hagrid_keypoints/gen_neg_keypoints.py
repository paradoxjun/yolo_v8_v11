import os
import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
ancestor_directory = current_file_path.parents[2]
sys.path.insert(0, str(ancestor_directory))     #  必须转成str

import cv2
import random
import numpy as np
from ultralytics.models.yolo.detect.predict import DetectionPredictor

mode_file = ["val", "test", "train"]
cate_file = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted", "rock",
             "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]

sample_each_cate = {
    "train": 7500,  # 原始每个类别，2.0w~2.5w
    "val": 1000,  # 原始每个类别，3k
    "test": 1500,  # 原始每个类别，5k
}

overrides_1 = {"task": "detect",
               "mode": "predict",
               "model": r'../../weights/yolov8m.pt',
               "verbose": False,
               "save": False,
               "plots": False,
               "classes": [0],
               "iou": 0.7,
               "conf": 0.8,
               }

predictor = DetectionPredictor(overrides=overrides_1)


def check_hagrid_all_path_valid(img_root, mode_file, cate_file):
    # 测试hagrid检测数据集所有路径是否有效
    n = len(cate_file)
    assert n == 18, f"ERROR：数据集类别文件夹数量错误：{n}，总共应该有18个。"

    for mode in mode_file:
        for cate in cate_file:
            directory = os.path.join(img_root, mode, "images", cate)
            is_exist = os.path.isdir(directory)

            if not is_exist:
                raise FileNotFoundError(f"缺少文件夹：{directory}")

    print(f"SUCCESS：数据集文件夹没有缺失。")


def draw_hand_mask(image, label_path):
    """
    根据手部标签，给手部打码。
    Args:
        image: 用cv2读取的图片张量。
        label_path: 标签列表。
    Returns:
    """
    height, width, _ = image.shape

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        x_center, y_center, w, h = list(map(float, parts[1:5]))
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

        image[y1:y2, x1:x2] = np.random.randint(0, 256, (y2 - y1, x2 - x1, 3), dtype=np.uint8)


def sample_and_predictor(image_cate_root, ren_predictor, sample_num, seed=2024, batch_size=64):
    files = [os.path.join(image_cate_root, f) for f in os.listdir(image_cate_root)
             if os.path.isfile(os.path.join(image_cate_root, f))]

    if sample_num < len(files):
        random.seed(seed)
        files = random.sample(files, sample_num)

    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]

        yield ren_predictor(batch_files)


def pipeline(img_root, mode_file, cate_file, ren_predictor, save_root="yolo_pose_neg", batch_size=2560):
    check_hagrid_all_path_valid(img_root, mode_file, cate_file)

    for mode in mode_file:
        for cate in cate_file:
            img_cate_root = os.path.join(img_root, mode, "images", cate)
            pred_batch = sample_and_predictor(img_cate_root, ren_predictor, int(sample_each_cate[mode] * 1.2), batch_size)

            valid_num = 0   # 有些可能检测失败

            for pred_result in pred_batch:
                for pred in pred_result:
                    if len(pred.boxes.cls) == 0:
                        continue

                    label_path = pred.path.replace("images", "labels").replace('jpg', 'txt')
                    img_save_path = pred.path.replace('yolo_det', save_root)

                    draw_hand_mask(pred.orig_img, label_path)

                    x1, y1, x2, y2 = list(map(int, pred.boxes.xyxy[0].cpu().numpy()))
                    pred.orig_img = pred.orig_img[y1:y2, x1:x2]

                    save_dir = os.path.dirname(img_save_path)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)

                    cv2.imwrite(img_save_path, pred.orig_img)

                    valid_num += 1
                    if valid_num >= sample_each_cate[mode]:
                        break

                if valid_num >= sample_each_cate[mode]:
                    break

            print(f"Finish: {mode}/{cate}")


if __name__ == '__main__':
    img_root = r'datasets/hagrid/yolo_det'
    pipeline(img_root, mode_file, cate_file, predictor)

