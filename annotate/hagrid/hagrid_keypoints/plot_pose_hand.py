import cv2
from ultralytics.values.color import colors_bgr as colors


_connections = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19),
                (19, 20))


def plot_bbox(image, xyxy, color=(0, 0, 255), offset=(0, 0)):
    for i, bbox in enumerate(xyxy):
        x1, y1, x2, y2 = list(map(int, bbox))
        cv2.rectangle(image, (x1 + offset[0], y1 + offset[1]), (x2 + offset[0], y2 + offset[1]), color, 2)

    return image


def plot_keypoints(image, keypoints, color=(0, 255, 0), offset=(0, 0), connections=_connections, wh=(1, 1)):
    if len(keypoints) == 21:
        for start_idx, end_idx in connections:
            sta_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            if (sta_point[0] > 0 or sta_point[1] > 0) and (end_point[0] > 0 and end_point[1] > 0):  # 忽略无效点
                cv2.line(image, (int(sta_point[0] * wh[0] + offset[0]), int(sta_point[1] * wh[1] + offset[1])),
                         (int(end_point[0] * wh[0] + offset[0]), int(end_point[1] * wh[1] + offset[1])),
                         (60, 179, 113), 2)

        for point in keypoints:
            x, y = point[:2]
            x *= wh[0]
            y *= wh[1]
            if x > 0 or y > 0:  # 忽略无效点
                cv2.circle(image, (int(x + offset[0]), int(y + offset[1])), 3, color, -1)

    return image


def draw_detections_pipeline(image_det, boxes, class_ids, class_names=None, mask_alpha=0.3):
    image = image_det.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.005
    text_thickness = int(min([img_height, img_width]) * 0.008)

    for box, class_id in zip(boxes, class_ids):
        # 必须先画，否则多次叠加胡导致不同目标颜色深浅不同
        color = colors[class_id]
        x1, y1, x2, y2 = box

        # Draw fill rectangle in mask image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

    image = cv2.addWeighted(image, mask_alpha, image_det, 1 - mask_alpha, 0)  # 叠加两张图

    for box, class_id in zip(boxes, class_ids):
        color = colors[class_id]
        x1, y1, x2, y2 = box

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw text
        label = class_names[class_id] if class_names is not None else str(int(class_id))
        caption = f'{label}'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.2)
        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), text_thickness, cv2.LINE_AA)

    return image


def load_labels(image_path, image_shape):
    label_path = image_path.replace('images', 'labels').replace('jpg', 'txt')
    height, width, _ = image_shape

    labels, bboxes, keypoints = [], [], []

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split()
        labels.append(int(parts[0]))

        bbox = list(map(float, parts[1:]))
        x_center, y_center, w, h = bbox[:4]
        x_center, y_center, w, h = x_center * width, y_center * height, w * width, h * height
        x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
        x2, y2 = int(x_center + w / 2), int(y_center + h / 2)

        bboxes.append([x1, y1, x2, y2])

    if len(bbox[4:]) < 0:
        for i in range(21):
            keypoints.append([bbox[4+2*i], bbox[4+2*i+1]])

    return labels, bboxes, keypoints
