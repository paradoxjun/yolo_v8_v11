import os
import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
ancestor_directory = current_file_path.parents[3]
sys.path.insert(0, str(ancestor_directory))     #  必须转成str

import cv2
import random
import mediapipe as mp
from tqdm import tqdm
from annotate.process_image import get_img_files
from annotate.process_compute import calculate_iou
from annotate.hagrid.hagrid_keypoints.hand_keypoints import (process_mp_result, process_hagrid_bbox, rectify_keypoints,
                                                             rotate_yolo_keypoints_90_clockwise,
                                                             rotate_keypoints_90_clockwise)
# from annotate.hagrid.hagrid_keypoints.plot_pose_hand import plot_bbox, plot_keypoints

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, model_complexity=0, min_detection_confidence=0.3)
mp_drawing = mp.solutions.drawing_utils


def get_hagrid_patch_keypoints(det_image_root, det_dir='yolo_det', pose_dir='yolo_pose_origin', limit_min_size=96,
                               base_factor=1.5, expand_factor=0.5, max_det_num=6, iou_thre=0.66):
    image_files = get_img_files(det_image_root)  # 获取所有文件路径为列表

    for det_image_path in tqdm(image_files, desc='Processing images', ncols=100, ascii=True):
        image = cv2.imread(det_image_path)

        # 按yolo存储方式，获取检测标签路径：只有存储文件夹和文件后缀不一样
        det_label_path = (os.path.splitext(det_image_path)[0] + '.txt').replace('images', 'labels')
        pose_image_path = det_image_path.replace(det_dir, pose_dir)     # 在同级根目录下，det和pose只有这一级不同
        pose_label_path = det_label_path.replace(det_dir, pose_dir)     # 同理获取pose的存储路径
        pose_image_rotate_path = pose_image_path.replace('origin', 'rotate')
        pose_label_rotate_path = pose_label_path.replace('origin', 'rotate')

        pose_image_dir = os.path.dirname(pose_image_path)  # 创建pose存储的多级目录
        pose_image_rotate_dir = os.path.dirname(pose_image_rotate_path)
        pose_label_dir = os.path.dirname(pose_label_path)  # 创建pose存储的多级目录
        pose_label_rotate_dir = os.path.dirname(pose_label_rotate_path)  # 创建pose存储的多级目录

        if not os.path.exists(pose_image_dir):
            os.makedirs(pose_image_dir)

        if not os.path.exists(pose_image_rotate_dir):
            os.makedirs(pose_image_rotate_dir)

        if not os.path.exists(pose_label_dir):
            os.makedirs(pose_label_dir)

        if not os.path.exists(pose_label_rotate_dir):
            os.makedirs(pose_label_rotate_dir)

        with open(det_label_path, 'r') as file:
            lines = file.readlines()

        # 一张图片有多只手，用“图片名_str(order+1)”来命名
        for order, line in enumerate(lines):
            is_stop = False     # 是否停止：已经获取有效一次，且本次小于下一次
            best_iou = 0    # 记录最好一次IoU结果
            best_keypoints = None   # 记录最好一次关键结果
            offset = [0, 0]

            parts = line.strip().split()
            if len(parts) != 5:
                raise ValueError(f"获取的标签长度错误：{parts}")

            # xywhn -> xywh
            x_center, y_center, width, height = map(float, parts[1:5])  # yolo归一化检测框的xywhn
            img_h, img_w, _ = image.shape   # 原始图片的尺寸
            xc, yc, w, h = x_center * img_w, y_center * img_h, width * img_w, height * img_h    # 实际图片上检测框xywh

            if w < limit_min_size or h < limit_min_size:
                continue

            # 手部检测框区域的实际坐标xyxy
            x_min = int(max(0, xc - w / 2))
            y_min = int(max(0, yc - h / 2))
            x_max = int(min(img_w, xc + w / 2))
            y_max = int(min(img_h, yc + h / 2))

            for i in range(int(max(1, max_det_num))):
                if is_stop:
                    break

                # 获取手部区域，并把右上角平移到(0, 0)，并裁剪出手部区域
                bbox_new = process_hagrid_bbox(xc, yc, w, h, img_w, img_h, base_factor, expand_factor, i)
                x1, y1, x2, y2 = list(map(int, bbox_new))   # 检测框放大后的 左上+右下
                hand_region_enlarge = image[y1:y2, x1:x2]   # 绘制局部区域需要生成拷贝：加上.copy()
                hand_region_enlarge_rgb = cv2.cvtColor(hand_region_enlarge, cv2.COLOR_BGR2RGB)

                # 处理手部区域并获取手部关键点
                results = hands.process(hand_region_enlarge_rgb)

                if results.multi_hand_landmarks is not None:
                    # 根据mediapipe结果获取：在放大的手部检测区域中的手部检测框，和手部关键点。
                    bbox_pred, keypoints = process_mp_result(results, y2 - y1, x2 - x1)
                    # 根据实际标签检测框，获取gt在放大区域中的位置。
                    bbox_gt_in_hand_region_enlarge = [x_min-x1, y_min-y1, x_max-x1, y_max-y1]
                    # 在放大区域上计算gt和pred的iou来衡量检测到关键点对目标框的覆盖程度。
                    iou = calculate_iou(bbox_gt_in_hand_region_enlarge, bbox_pred)

                    # 当前检测满足IoU阈值
                    if iou > iou_thre:
                        # 满足阈值，但检测结果变差了，就停止
                        if iou < best_iou:
                            is_stop = True
                        # 满足阈值，且检测结果还能更好，可以继续检测，希望变得更好
                        else:
                            best_iou = iou
                            best_keypoints = keypoints
                            offset = [x_min - x1, y_min - y1]
                    # 当前检测不满足，但上一次满足过，说明检测变差了，可以停止了。
                    elif best_iou > iou_thre:
                        is_stop = True

            if best_keypoints is not None:
                # 扩大手部区域
                new_w = w * 5 // 3
                new_h = h * 5 // 3

                # 扩大后的区域随机抖动
                new_x1 = int(max(0, xc - new_w / 2 + random.randint(-5, 5)))
                new_y1 = int(max(0, yc - new_h / 2 + random.randint(-5, 5)))
                new_x2 = int(min(img_w, xc + new_w / 2 + random.randint(-5, 5)))
                new_y2 = int(min(img_h, yc + new_h / 2 + random.randint(-5, 5)))

                new_w = new_x2 - new_x1
                new_h = new_y2 - new_y1

                # 扩大区域转成yolo
                norm_xc = (xc - new_x1) / new_w
                norm_yc = (yc - new_y1) / new_h
                norm_w = w / new_w
                norm_h = h / new_h

                # 获取手部区域，并对关键点进行偏移
                image_save = image[new_y1:new_y2, new_x1:new_x2]
                # image_save = image[y_min:y_max, x_min:x_max]
                keypoints_in_hand_box = rectify_keypoints(best_keypoints, w, h, offset=offset, norm=False)

                enlarge_offset = (new_x1 - x_min, new_y1 - y_min)
                keypoints = rectify_keypoints(keypoints_in_hand_box, new_w, new_h, offset=enlarge_offset, norm=True)
                label_save = [0, norm_xc, norm_yc, norm_w, norm_h] + keypoints.flatten().tolist()

                image_save_rotate = cv2.rotate(image_save, cv2.ROTATE_90_CLOCKWISE)
                label_save_rotate = rotate_yolo_keypoints_90_clockwise(label_save)

                # 显示手部关键点，需要把norm=True, wh=(w, h); norm=Fasle, wh=(1, 1)
                # image_plot = plot_keypoints(image_save.copy(), keypoints, wh=(new_w, new_h))
                # cv2.imshow('Hand Detection and Keypoints', image_plot)
                # cv2.waitKey(0)

                # keypoints_rotate = rotate_keypoints_90_clockwise(keypoints)
                # image_plot = plot_keypoints(image_save_rotate.copy(), keypoints_rotate, wh=(new_h, new_w))
                # cv2.imshow('Hand Detection and Keypoints', image_plot)
                # cv2.waitKey(0)

                # save_name_part = pose_image_path.rpartition('.')  # 都是.jpg格式，简化了
                pose_image_path_save = pose_image_path[:-4] + f'_{str(order+1)}.jpg'
                pose_label_path_save = pose_label_path[:-4] + f'_{str(order+1)}.txt'

                # 保存图片和标签
                cv2.imwrite(pose_image_path_save, image_save)
                with open(pose_label_path_save, 'w') as file:
                    labels_str = ' '.join(map(str, label_save))
                    file.write(labels_str)

                # print(f"文件已处理：{pose_image_path_save}")

                pose_image_rotate_path_save = pose_image_rotate_path[:-4] + f'_{str(order + 1)}.jpg'
                pose_label_rotate_path_save = pose_label_rotate_path[:-4] + f'_{str(order + 1)}.txt'

                # 保存图片和标签
                cv2.imwrite(pose_image_rotate_path_save, image_save_rotate)
                with open(pose_label_rotate_path_save, 'w') as file:
                    labels_rotate_str = ' '.join(map(str, label_save_rotate))
                    file.write(labels_rotate_str)

                # print(f"文件已处理：{pose_image_path_save}")

        # # 显示图像，按键 'q' 退出循环
        # cv2.imshow('Hand Detection and Keypoints', image)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break

    # return


if __name__ == '__main__':
    det_image_root = r'datasets/hagrid/yolo_det'
    get_hagrid_patch_keypoints(det_image_root)
