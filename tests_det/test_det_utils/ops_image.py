import cv2
import os
import sys
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, project_root)

from ultralytics.values.color import colors_bgr as colors


def resize_image(image, new_width=None, new_height=None, keep_wh_ratio=True):
    """
    Resize an image while keeping the aspect ratio intact if required.

    :param image: Path to the input image or numpy array.
    :param new_width: Desired width of the resized image.
    :param new_height: Desired height of the resized image.
    :param keep_wh_ratio: Boolean to keep width-height ratio.
    :return: Resized image and the (width_ratio, height_ratio).
    """
    is_pil = False

    if isinstance(image, np.ndarray):
        original_height, original_width = image.shape[:2]
    else:
        original_width, original_height = image.size
        is_pil = True

    if new_width and not new_height:
        new_height = int((new_width / original_width) * original_height) if keep_wh_ratio else int(original_height)
    elif new_height and not new_width:
        new_width = int((new_height / original_height) * original_width) if keep_wh_ratio else int(original_width)
    elif new_width and new_height:
        if keep_wh_ratio:
            width_ratio = new_width / original_width
            height_ratio = new_height / original_height
            if width_ratio < height_ratio:
                new_height = int(width_ratio * original_height)
            else:
                new_width = int(height_ratio * original_width)
    elif not new_width and not new_height:
        raise ValueError("Either new_width or new_height must be provided.")

    if is_pil:
        resized_img = image.resize((new_width, new_height), Image.ANTIALIAS)
    else:
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    width_ratio = new_width / original_width
    height_ratio = new_height / original_height

    return resized_img, (width_ratio, height_ratio)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scalefill=False, scaleup=True, stride=32):
    """
    调整图片大小并填充以适应目标尺寸。
    :param im:输入图片。
    :param new_shape:目标形状，默认 (640, 640)。
    :param color:填充颜色，默认 (114, 114, 114)。
    :param auto:自动调整填充，保持最小矩形。True会让图片宽高是stride的最小整数倍，比如32，可以方便卷积。
    :param scalefill:是否拉伸填充。在auto是False时，True会让图片拉伸变形。
    :param scaleup:是否允许放大。False让图片只能缩小。
    :param stride:步幅大小，默认 32。
    :return:返回调整后的图片，缩放比例(宽，高)和填充值。
    """
    shape = im.shape[:2]  # 获取当前图片的形状 [高度, 宽度]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])  # 缩放比例 (新尺寸 / 旧尺寸)
    if not scaleup:  # 如果不允许放大，只进行缩小 (提高验证的 mAP)
        r = min(r, 1.0)

    ratio = r, r  # 计算填充宽度和高度的缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # 新的未填充尺寸 (宽度, 高度)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # 计算宽高方向的填充值
    if auto:  # 如果设置为自动，保持最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # 使填充值是步幅的倍数
    elif scalefill:  # 如果拉伸填充，完全填充
        dw, dh = 0.0, 0.0  # 不进行填充
        new_unpad = (new_shape[1], new_shape[0])  # 未填充的尺寸就是目标尺寸
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 计算宽高的缩放比例

    dw /= 2  # 将填充值均分到两侧
    dh /= 2  # 将填充值均分到上下

    if shape[::-1] != new_unpad:  # 如果当前形状和新的未填充形状不同，则调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下填充的像素数
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右填充的像素数
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加填充边框，填充值为指定颜色

    return im, ratio, (dw, dh)  # 返回调整后的图片，缩放比例和填充值


def draw_detections_on_raw_image(image, boxes, scores, class_ids, class_names=None):
    # more fast, but lack draw mask
    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.001
    text_thickness = int(min([img_height, img_width]) * 0.002)

    for box, class_id, score in zip(boxes, class_ids, scores):
        color = colors[class_id % len(colors)]

        x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label = str(int(class_id)) if class_names is None else class_names[class_id]
        caption = f'{label} {int(score * 100)}%'

        # Draw text
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.2)
        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), text_thickness, cv2.LINE_AA)


def draw_detections_pipeline(image_det, boxes, scores, class_ids, class_names=None, mask_alpha=0.3):
    image = image_det.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.001
    font_size = 0.4
    text_thickness = int(min([img_height, img_width]) * 0.002)

    for box, class_id in zip(boxes, class_ids):
        # 必须先画，否则多次叠加胡导致不同目标颜色深浅不同
        color = colors[-5]
        x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)

    image = cv2.addWeighted(image, mask_alpha, image_det, 1 - mask_alpha, 0)  # 叠加两张图

    for box, class_id, score in zip(boxes, class_ids, scores):
        color = colors[-5]
        x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

        # Draw bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw text
        label = class_names[int(class_id)] if class_names is not None else str(int(class_id))
        caption = f'{label} {int(score * 100)}%'
        (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.2)
        th = 12
        cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

        cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                    (255, 255, 255), text_thickness, cv2.LINE_AA)

    return image


def draw_detections(image, boxes, scores, class_ids, class_names=None, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id] if class_names is not None else str(int(class_id))
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box(image: np.ndarray, box: np.ndarray, color=(0, 0, 255), thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color=(0, 0, 255), font_size: float = 0.001,
              text_thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1), (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness,
                       cv2.LINE_AA)


def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3) -> np.ndarray:
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = list(map(int, box)) if isinstance(box, (list, tuple)) else box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
