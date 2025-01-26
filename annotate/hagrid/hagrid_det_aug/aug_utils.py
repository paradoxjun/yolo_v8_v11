import cv2
import random
import numpy as np
from annotate.process_image import get_img_files
from ultralytics.values.color import colors_bgr as colors

random.seed(2024)

neg_img = get_img_files(r'datasets/COCO2017/det_neg/images')


def mosaic_aug_row(images, labels, shape, image_pad=None, target_width=512):
    """
    Perform mosaic augmentation on a list of images and their corresponding labels.

    Args:
        images (list): List of images to be augmented.
        labels (list): List of labels corresponding to the images.
        shape (tuple): The shape of the mosaic (rows, cols).
        image_pad (numpy array): Image used for padding.
        target_width (int): Target width for each image (default 512).

    Returns:
        mosaic_image: The augmented mosaic image.
        mosaic_labels: The adjusted labels for the mosaic image.
    """

    rows, cols = shape
    assert len(images) == rows * cols, "Number of images should match the shape dimensions."

    max_heights = [0] * rows
    row_widths = [[] for _ in range(rows)]
    padded_images = []
    padded_labels = []

    for i, image in enumerate(images):
        h, w = image.shape[:2]
        row_idx = i // cols
        max_heights[row_idx] = max(max_heights[row_idx], h)
        row_widths[row_idx].append(w)

    max_widths = [max(row_widths[row_idx][col_idx] for row_idx in range(rows)) for col_idx in range(cols)]
    # max_widths = [max(widths) for widths in row_widths]

    for i, (image, label) in enumerate(zip(images, labels)):
        h, w = image.shape[:2]
        row_idx = i // cols
        max_height = max_heights[row_idx]
        pad_height = max_height - h
        if pad_height > 0:
            image_pad = cv2.imread(random.choice(neg_img))
            pad = get_random_crop(image_pad, (pad_height, w))
            padded_image = np.vstack((pad, image))  # Padding on top
        else:
            padded_image = image

        new_labels = []
        for cls, cx, cy, bw, bh in label:
            cx *= w
            cy = cy * h + pad_height
            bw *= w
            bh *= h
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            new_labels.append([cls, x1, y1, x2, y2])
        padded_labels.append(new_labels)
        padded_images.append(padded_image)

    target_height = sum(max_heights)
    target_width = sum(max_widths)
    target_size = (target_height, target_width)

    mosaic_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    mosaic_labels = []

    y_offsets = [sum(max_heights[:i]) for i in range(rows)]
    for i, (padded_image, label) in enumerate(zip(padded_images, padded_labels)):
        row_idx = i // cols
        col_idx = i % cols
        y_offset = y_offsets[row_idx]
        x_offset = sum(max_widths[:col_idx])
        ph, pw = padded_image.shape[:2]

        mosaic_image[y_offset:y_offset + ph, x_offset:x_offset + pw] = padded_image

        for cls, x1, y1, x2, y2 in label:
            x1 += x_offset
            y1 += y_offset
            x2 += x_offset
            y2 += y_offset
            mosaic_labels.append([cls, x1, y1, x2, y2])

    final_scale = 512 / min(target_size)
    final_size = (int(target_size[1] * final_scale), int(target_size[0] * final_scale))
    final_mosaic_image = cv2.resize(mosaic_image, final_size)

    scaled_labels = []
    for cls, x1, y1, x2, y2 in mosaic_labels:
        x1_scaled = x1 * final_scale
        y1_scaled = y1 * final_scale
        x2_scaled = x2 * final_scale
        y2_scaled = y2 * final_scale
        cx = (x1_scaled + x2_scaled) / 2 / final_size[0]
        cy = (y1_scaled + y2_scaled) / 2 / final_size[1]
        bw = (x2_scaled - x1_scaled) / final_size[0]
        bh = (y2_scaled - y1_scaled) / final_size[1]
        scaled_labels.append([cls, cx, cy, bw, bh])

    return final_mosaic_image, scaled_labels


def mosaic_aug_col(images, labels, shape, image_pad=None, target_height=512):
    """
    Perform mosaic augmentation on a list of images and their corresponding labels.

    Args:
        images (list): List of images to be augmented.
        labels (list): List of labels corresponding to the images.
        shape (tuple): The shape of the mosaic (rows, cols).
        image_pad (numpy array): Image used for padding.
        target_height (int): Target height for each image (default 512).

    Returns:
        mosaic_image: The augmented mosaic image.
        mosaic_labels: The adjusted labels for the mosaic image.
    """

    rows, cols = shape
    assert len(images) == rows * cols, "Number of images should match the shape dimensions."

    max_widths = [0] * cols
    col_heights = [[] for _ in range(cols)]
    padded_images = []
    padded_labels = []

    for i, image in enumerate(images):
        h, w = image.shape[:2]
        col_idx = i % cols
        max_widths[col_idx] = max(max_widths[col_idx], w)
        col_heights[col_idx].append(h)

    max_heights = [sum(col_heights[col_idx]) for col_idx in range(cols)]

    for i, (image, label) in enumerate(zip(images, labels)):
        h, w = image.shape[:2]
        col_idx = i % cols
        max_width = max_widths[col_idx]
        pad_width = max_width - w
        if pad_width > 0:
            image_pad = cv2.imread(random.choice(neg_img))
            pad = get_random_crop(image_pad, (h, pad_width))
            padded_image = np.hstack((pad, image))  # Padding on the left
        else:
            padded_image = image

        new_labels = []
        for cls, cx, cy, bw, bh in label:
            cx = cx * w + pad_width
            cy *= h
            bw *= w
            bh *= h
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
            new_labels.append([cls, x1, y1, x2, y2])
        padded_labels.append(new_labels)
        padded_images.append(padded_image)

    target_height = max(max_heights)
    target_width = sum(max_widths)
    target_size = (target_height, target_width)

    mosaic_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    mosaic_labels = []

    x_offsets = [sum(max_widths[:i]) for i in range(cols)]
    for i, (padded_image, label) in enumerate(zip(padded_images, padded_labels)):
        col_idx = i % cols
        row_idx = i // cols
        x_offset = x_offsets[col_idx]
        y_offset = sum(col_heights[col_idx][:row_idx])
        ph, pw = padded_image.shape[:2]

        mosaic_image[y_offset:y_offset + ph, x_offset:x_offset + pw] = padded_image

        for cls, x1, y1, x2, y2 in label:
            x1 += x_offset
            y1 += y_offset
            x2 += x_offset
            y2 += y_offset
            mosaic_labels.append([cls, x1, y1, x2, y2])

    final_scale = 512 / min(target_size)
    final_size = (int(target_size[1] * final_scale), int(target_size[0] * final_scale))
    final_mosaic_image = cv2.resize(mosaic_image, final_size)

    scaled_labels = []
    for cls, x1, y1, x2, y2 in mosaic_labels:
        x1_scaled = x1 * final_scale
        y1_scaled = y1 * final_scale
        x2_scaled = x2 * final_scale
        y2_scaled = y2 * final_scale
        cx = (x1_scaled + x2_scaled) / 2 / final_size[0]
        cy = (y1_scaled + y2_scaled) / 2 / final_size[1]
        bw = (x2_scaled - x1_scaled) / final_size[0]
        bh = (y2_scaled - y1_scaled) / final_size[1]
        scaled_labels.append([cls, cx, cy, bw, bh])

    return final_mosaic_image, scaled_labels


def get_random_crop(image, crop_size):
    """
    Get a random crop from the given image.

    Args:
        image (numpy array): The image to crop from.
        crop_size (tuple): The size of the crop (height, width).

    Returns:
        cropped_image: The randomly cropped image.
    """
    h, w = image.shape[:2]
    ch, cw = crop_size

    if ch > h or cw > w:
        return cv2.resize(image, (crop_size[1], crop_size[0]))

    y = np.random.randint(0, h - ch + 1)
    x = np.random.randint(0, w - cw + 1)
    return image[y:y + ch, x:x + cw]


def load_labels(image_path):
    label_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
    labels = []

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        bbox = list(map(float, parts[1:]))
        labels.append([int(parts[0]), *bbox])

    return labels


def draw_boxes(image, labels, colors=colors):
    """
    根据标签列表画出检测框。
    Args:
        image: 用cv2读取的图片张量。
        labels: 标签列表。
        label_map: 标签映射成名称的哈希表。
        colors: 颜色列表。
    Returns:
    """
    height, width, _ = image.shape
    for item in labels:
        label = int(item[0])
        bbox = item[1:5]
        x_center, y_center, w, h = bbox
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
        color = colors[label % len(colors)]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)


if __name__ == '__main__':
    from annotate.process_image import image_show_cv2

    img_1 = r'datasets/hagrid/yolo_det/test/images/like/0c9d0712-0fdb-46ae-8019-9c0b7bd4d647.jpg'
    img_2 = r'datasets/hagrid/yolo_det/test/images/four/0a79f9a5-494c-47ad-8f70-20473bd860c3.jpg'
    img_3 = r'datasets/hagrid/yolo_det/test/images/four/0a591585-5814-4115-9a00-29dd546a4537.jpg'
    img_4 = r'datasets/hagrid/yolo_det/test/images/four/0aa03a0d-80b8-4133-9ac4-c7f98f6e4460.jpg'
    img_5 = r'datasets/hagrid/yolo_det/test/images/palm/0b1a1bd2-fca0-438a-8b9d-fbd7af71d574.jpg'

    img_pad = r'datasets/COCO2017/det_neg/images/val2017/000000000285.jpg'

    images = []
    labels = []

    for im in [img_2]:
        images.append(cv2.imread(im))
        labels.append(load_labels(im))

        image_aug, label_aug = mosaic_aug_row(images, labels, (1, 1), cv2.imread(img_pad))
    image_show_cv2(image_aug)

    draw_boxes(image_aug, label_aug)
    image_show_cv2(image_aug)

    img_11 = r'datasets/hagrid/yolo_det/test/images/palm/0afdcfbb-9056-4e56-9d45-2b986d5bb0b3.jpg'
    img_22 = r'datasets/hagrid/yolo_det/test/images/palm/0a51908a-961a-4d8a-88ba-966a5cac598c.jpg'
    img_33 = r'datasets/hagrid/yolo_det/test/images/palm/0e0ef3a6-792a-49e9-9058-1170c562d5b6.jpg'
    img_44 = r'datasets/hagrid/yolo_det/test/images/palm/0e78c638-fadb-4d91-beb7-8238cd4eb68b.jpg'

    images_2 = []
    labels_2 = []

    for im in [img_11, img_22, img_33, img_44]:
        image_read = cv2.imread(im)
        print(image_read.shape)
        images_2.append(image_read)
        labels_2.append(load_labels(im))

    image_aug, label_aug = mosaic_aug_col(images_2, labels_2, (2, 2), cv2.imread(img_pad))
    print(label_aug)
    image_show_cv2(image_aug)

    draw_boxes(image_aug, label_aug)
    image_show_cv2(image_aug)
