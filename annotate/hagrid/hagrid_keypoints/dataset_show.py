import os
import random
import cv2
import numpy as np
from annotate.process_image import get_img_files
from annotate.hagrid.hagrid_keypoints.plot_pose_hand import plot_bbox, plot_keypoints, draw_detections_pipeline, load_labels


# hagrid_cate_file = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted",
#                     "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted", "no_gesture"]

# hagrid_cate_file = ["boardgame", "diy", "drink", "food", "furniture", "gardening", "housework", "packing", "puzzle",
#                     "repair", "study", "vlog", ""]

hagrid_cate_file = ["test/images", "train/images", "val/images", ""]
class_name = {i: hagrid_cate_file[i] for i in range(len(hagrid_cate_file))}


def show_yolo_det(root_dir, output_size=(72, 72), per_row=6, per_folder=3):
    random.seed(2025)
    all_images = []

    for hagrid_cate in hagrid_cate_file[:-1]:
        folder_path = os.path.join(root_dir, hagrid_cate)
        files = get_img_files(folder_path)

        selected_files = random.sample(files, min(len(files), per_folder))
        # 加载和调整大小
        images = [cv2.resize(cv2.imread(file), output_size) for file in selected_files]
        all_images.extend(images)

        # 每行显示6个图片，总共9行
    num_images = len(all_images)
    num_rows = (num_images + per_row - 1) // per_row  # 向上取整得到总行数
    combined_image = np.zeros((output_size[1] * num_rows, output_size[0] * per_row, 3), dtype=np.uint8)

    for index, image in enumerate(all_images):
        row = index // per_row
        col = index % per_row
        x = col * output_size[0]
        y = row * output_size[1]
        combined_image[y:y + output_size[1], x:x + output_size[0]] = image

    # 显示合成的图片
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)  # 按任意键继续
    cv2.destroyAllWindows()


def show_yolo_det_with_label(root_dir, output_size=(180, 150), per_row=4, per_folder=4):
    random.seed(225)
    all_images = []

    for hagrid_cate in hagrid_cate_file[:-1]:
        folder_path = os.path.join(root_dir, hagrid_cate)
        files = get_img_files(folder_path)
        selected_files = random.sample(files, min(len(files), per_folder))
        # 加载和调整大小
        for file in selected_files:
            image = cv2.imread(file)
            labels, bboxes, _ = load_labels(file, image.shape)
            image = draw_detections_pipeline(image, bboxes, labels, class_names={0: "hand"})
            all_images.append(cv2.resize(image, output_size))

        # 每行显示6个图片，总共9行
    num_images = len(all_images)
    num_rows = (num_images + per_row - 1) // per_row  # 向上取整得到总行数
    combined_image = np.zeros((output_size[1] * num_rows, output_size[0] * per_row, 3), dtype=np.uint8)

    for index, image in enumerate(all_images):
        row = index // per_row
        col = index % per_row
        x = col * output_size[0]
        y = row * output_size[1]
        combined_image[y:y + output_size[1], x:x + output_size[0]] = image

    # 显示合成的图片
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)  # 按任意键继续
    cv2.destroyAllWindows()


def show_yolo_pose_with_label(root_dir, output_size=(96, 96), per_row=6, per_folder=2):
    random.seed(202)
    all_images = []

    for hagrid_cate in hagrid_cate_file[:-1]:
        folder_path = os.path.join(root_dir, hagrid_cate)
        files = get_img_files(folder_path)
        selected_files = random.sample(files, min(len(files), per_folder))
        # 加载和调整大小
        for file in selected_files:
            image = cv2.imread(file)
            h, w, _ = image.shape
            labels, bboxes, keypoints = load_labels(file, image.shape)
            image = plot_bbox(image, bboxes)
            image = plot_keypoints(image, keypoints, wh=(w, h))

            all_images.append(cv2.resize(image, output_size))

        # 每行显示6个图片，总共9行
    num_images = len(all_images)
    num_rows = (num_images + per_row - 1) // per_row  # 向上取整得到总行数
    combined_image = np.zeros((output_size[1] * num_rows, output_size[0] * per_row, 3), dtype=np.uint8)

    for index, image in enumerate(all_images):
        row = index // per_row
        col = index % per_row
        x = col * output_size[0]
        y = row * output_size[1]
        combined_image[y:y + output_size[1], x:x + output_size[0]] = image

    # 显示合成的图片
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)  # 按任意键继续
    cv2.destroyAllWindows()


def show_handpose_yolo_det_with_label(root_dir, output_size=(128, 128), per_row=4, per_col=4):
    random.seed(202)
    all_images = []

    files = get_img_files(root_dir)
    random.shuffle(files)
    total = 0

    for file in files:
        image = cv2.imread(file)
        h, w, _ = image.shape
        if h < 224 or h < 224:
            print(f"尺寸不行。已收集：{total}")
            continue

        labels, bboxes, keypoints = load_labels(file, image.shape)
        image = plot_bbox(image, bboxes)
        image = plot_keypoints(image, keypoints, wh=(w, h))

        all_images.append(cv2.resize(image, output_size))

        total += 1
        if total >= per_col * per_row:
            break

        # 每行显示6个图片，总共9行
    num_images = len(all_images)
    num_rows = (num_images + per_row - 1) // per_row  # 向上取整得到总行数
    combined_image = np.zeros((output_size[1] * num_rows, output_size[0] * per_row, 3), dtype=np.uint8)

    for index, image in enumerate(all_images):
        row = index // per_row
        col = index % per_row
        x = col * output_size[0]
        y = row * output_size[1]
        combined_image[y:y + output_size[1], x:x + output_size[0]] = image

    # 显示合成的图片
    cv2.imshow('Combined Image', combined_image)
    cv2.waitKey(0)  # 按任意键继续
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_yolo_det_with_label(r'atasets\hand_detection\hand_det_ump')
    # show_handpose_yolo_det_with_label(r'datasets/handpose/handpose_v2_yolov8_pose/val/images')
    # show_yolo_pose_with_label(r'datasets/hagrid/yolo_pose/val/images')
