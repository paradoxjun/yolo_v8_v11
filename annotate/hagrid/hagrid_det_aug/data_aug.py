import cv2
import os
import random
from annotate.hagrid.hagrid_det_aug.aug_utils import mosaic_aug_row, mosaic_aug_col, load_labels
from annotate.process_image import get_img_files


mode_file = ["val", "test", "train"]
hagrid_det_root = 'datasets/hagrid/yolo_det'
save_mosaic_root = 'datasets/hagrid/yolo_det_mosaic'

# num_order = [1, 2, 3, 4, 6, 8, 9, 9, 9, 9, 9, 9, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 16]
num_order = [1, 2, 3, 4, 6, 8, 9, 9, 12, 12, 12, 12, 12, 15, 15, 15, 15, 16, 16, 16, 16, 20, 20, 20, 20, 24, 24, 24, 24]
# row_shape = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 6: (2, 3), 9: (3, 3), 8: (2, 4), 12: (3, 4), -12: (4, 3), 16: (4, 4)}
row_shape = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 6: (2, 3), 8: (2, 4), 9: (3, 3), 12: (3, 4), 15: (3, 5),
             16: (4, 4), 20: (4, 5), 24: (4, 6)}
# col_shape = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 6: (3, 2), 9: (3, 3), 8: (4, 2), 12: (4, 3), -12: (3, 4), 16: (4, 4)}
col_shape = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 6: (3, 2), 8: (4, 2), 9: (3, 3), 12: (4, 3), 15: (5, 3),
             16: (4, 4), 20: (5, 4), 24: (6, 4)}

n = len(num_order)
file_number = 0     # 生成图片索引序号
random.seed(2024)

for mode in mode_file:
    all_images_path = get_img_files(os.path.join(hagrid_det_root, mode))
    random.shuffle(all_images_path)
    total_image_num = len(all_images_path)
    save_root = os.path.join(save_mosaic_root, mode)

    cur = 0         # 指示每个文件夹下图片索引序号
    row_order = 0   # 宽度一样，为512；在num_order中进行到哪一步
    col_order = 0   # 高度一样，为512；在num_order中进行到哪一步
    now_row = 0     # 需要多少张宽度为512的图片，已经获取了
    now_col = 0     # 需要多少张高度为512的图片，已经获取了

    row_images = []  # 宽度要一样，为512
    row_labels = []
    col_images = []  # 高度要一样，为512
    col_labels = []

    while cur < total_image_num:
        row_or_col = 0     # 1：row满足，-1：col满足，0：均不满足
        row_order %= n
        col_order %= n

        while cur < total_image_num and now_row < num_order[row_order] and now_col < num_order[col_order]:
            image_path = all_images_path[cur]
            cur += 1
            image = cv2.imread(image_path)
            label = load_labels(image_path)
            h, w, _ = image.shape

            if w == 512:
                row_images.append(image)
                row_labels.append(label)
                now_row += 1
                if now_row == num_order[row_order]:
                    row_or_col = 1
            else:
                col_images.append(image)
                col_labels.append(label)
                now_col += 1
                if now_col == num_order[col_order]:
                    row_or_col = -1

        if row_or_col == 1:     # row满足
            mosaic_image, mosaic_label = mosaic_aug_row(row_images, row_labels, row_shape[num_order[row_order]])
            file_number += 1
            image_name = str(file_number).zfill(8) + '.jpg'
            label_name = str(file_number).zfill(8) + '.txt'
            save_image_path = os.path.join(save_root, 'images', image_name)
            print(f"INFO: save at: {save_image_path}")

            # 保存图片和标签
            cv2.imwrite(save_image_path, mosaic_image)
            with open(os.path.join(save_root, 'labels', label_name), 'w') as file:
                for item in mosaic_label:
                    labels_str = ' '.join(map(str, item)) + '\n'
                    file.write(labels_str)

            row_images = []  # 宽度要一样，为512
            row_labels = []
            now_row = 0
            row_order += 1

        elif row_or_col == -1:      # col满足
            mosaic_image, mosaic_label = mosaic_aug_col(col_images, col_labels, col_shape[num_order[col_order]])
            file_number += 1
            image_name = str(file_number).zfill(8) + '.jpg'
            label_name = str(file_number).zfill(8) + '.txt'
            save_image_path = os.path.join(save_root, 'images', image_name)
            print(f"INFO: save at: {save_image_path}")

            # 保存图片和标签
            cv2.imwrite(save_image_path, mosaic_image)
            with open(os.path.join(save_root, 'labels', label_name), 'w') as file:
                for item in mosaic_label:
                    labels_str = ' '.join(map(str, item)) + '\n'
                    file.write(labels_str)

            col_images = []  # 宽度要一样，为512
            col_labels = []
            now_col = 0
            col_order += 1
        else:       # 便利完了，剩下的图片不足以拼接，就用最大拼接
            break
