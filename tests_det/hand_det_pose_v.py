import cv2
import datetime
import numpy as np
from PIL import Image
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.pose.predict import PosePredictor


colors_bgr = ((0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0), (203, 192, 255),
              (147, 20, 255), (214, 112, 218), (221, 160, 221), (128, 0, 128), (130, 0, 75), (219, 112, 147),
              (139, 0, 0), (255, 105, 65), (255, 144, 30), (250, 206, 135), (255, 191, 0), (160, 158, 95),
              (170, 255, 127), (113, 179, 60), (0, 128, 0), (0, 255, 127), (0, 128, 128), (205, 250, 255),
              (170, 232, 238), (0, 215, 255), (32, 165, 218), (0, 165, 255), (30, 105, 210), (19, 69, 139),
              (114, 128, 250), (0, 0, 128))

colors_name_ch = ("纯红", "酸橙绿", "纯黄", "纯蓝", "紫红", "青", "粉红", "深粉", "兰花紫", "李子", "紫", "靛青", "中紫", "深蓝",
                  "皇家蓝", "道奇蓝", "淡蓝", "深天蓝", "军校蓝", "碧绿", "春天绿", "纯绿", "查特酒绿", "橄榄", "柠檬薄纱", "灰秋麒麟",
                  "金", "秋麒麟", "橙", "巧克力", "马鞍棕", "鲑鱼肉", "栗")

colors_dict_ch_bgr = dict(zip(colors_name_ch, colors_bgr))


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


overrides_1 = {"task": "pose",
               "mode": "predict",
               "model": r'../trains_pose/hands_09_20_s/train/weights/best.pt',
               "imgsz": 320,
               "verbose": False,
               "iou": 0.25,
               "save": False,
               "conf": 0.5,
               }

overrides_2 = {"task": "detect",
               "mode": "predict",
               "model": r'../trains_det/hands_10_25_s/train/weights/best.pt',
               "imgsz": 480,
               "verbose": False,
               "save": False,
               "conf": 0.5,
               "iou": 0.5
               }


def bbox_offset(bbox, offset_x, offset_y):
    for box in bbox:
        box[0] += offset_x
        box[1] += offset_y
        box[2] += offset_x
        box[3] += offset_y

    return bbox


def kpt_offset(kpt, offset_x, offset_y, dim=2):
    for i in range(0, len(kpt), dim):
        kpt[i] += offset_x
        kpt[i + 1] += offset_y

    return kpt


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


def draw_bboxes_and_keypoints(image, bboxes, scores, class_ids, kpts=None, class_names=None, cat_order=None, kpt_dim=2,
                              box_offset=(0, 0), cls_color=((0, 0, 255),), kpt_offset=(0, 0), kpt_color=((255, 0, 0),),
                              line_color=((0, 255, 0), ), kpt_norm=False):
    """
    BGR通道处理：对检测或关键点检测画出检测框和关键点。

    Args:
        image: np.ndarray 输入的图片张量；
        bboxes: list(list) 一组检测框；
        scores: list 检测框对应的置信度；
        class_ids: list 检测框对应的类别标签；
        kpts: list 一组关键点；
        class_names: dict 类别标签对应的名称，不给就输出标签序号；
        cat_order: tuple(tuple)，关键点连续的顺序；
        kpt_dim: int 2或3，关键点的维度（2就是没有置信度，3有置信度）
        box_offset: tuple 检测框的偏移量
        cls_color: tuple(tuple)，类别颜色
        kpt_offset: tuple 关键点的偏移量
        kpt_color: tuple(tuple)，关键点的颜色
        line_color: tuple(tuple) 关键点连线的颜色
        kpt_norm: bool 关键点是否归一化了

    Returns:
        绘制后的图像
    """
    if isinstance(bboxes, np.ndarray):
        bboxes.tolist()

    if isinstance(scores, np.ndarray):
        scores.tolist()

    if isinstance(class_ids, np.ndarray):
        class_ids.tolist()
    elif isinstance(class_ids, int):
        class_ids = [class_ids]

    if len(class_ids) == 1:
        class_ids = class_ids * len(scores)

    if isinstance(kpts, np.ndarray):
        kpts.tolist()


    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.002
    text_thickness = int(min([img_height, img_width]) * 0.002)

    # for box, score, class_id in zip(bboxes, scores, class_ids):
    #     class_id = int(class_id)
    #     det_color = cls_color[class_id % len(cls_color)]
    #     label = class_names.get(class_id, str(int(class_id))) if class_names is not None else str(int(class_id))
    #     caption = f'{label} {int(score * 100)}%'
    #
    #     (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_size,
    #                                   thickness=text_thickness)
    #     th = int(th * 1.2)
    #
    #     x1, y1, x2, y2 = list(map(int, box[:4]))
    #     x1, y1, x2, y2 = x1 + box_offset[0], y1 + box_offset[1], x2 + box_offset[0], y2 + box_offset[1]
    #
    #     cv2.rectangle(det_img, (x1, y1), (x2, y2), det_color, 2)
    #     cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), det_color, -1)
    #
    #     det_img = cv2.putText(det_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255),
    #                           text_thickness, cv2.LINE_AA)

    if kpts is not None:
        w, h = 1, 1
        if kpt_norm:
            w, h = img_width, img_height

        for kpt in kpts:
            if kpt and cat_order is not None:
                for i, (k1, k2) in enumerate(cat_order):
                    k1x = k1 * kpt_dim
                    k2x = k2 * kpt_dim

                    if (kpt[k1x] > 0 or kpt[k1x + 1] > 0) and (kpt[k2x] > 0 or kpt[k2x + 1] > 0):   # 忽略无效点
                        cv2.line(det_img, (int(kpt[k1x] * w + kpt_offset[0]), int(kpt[k1x + 1] * h + kpt_offset[1])),
                                 (int(kpt[k2x] * w + kpt_offset[0]), int(kpt[k2x + 1] * h + kpt_offset[1])),
                                 line_color[i % len(line_color)], 2)

            for i in range(0, len(kpt), kpt_dim):
                x, y = kpt[i] * w, kpt[i + 1] * h

                if x > 0 or y > 0:
                    cv2.circle(det_img, (int(x + kpt_offset[0]), int(y + kpt_offset[1])), 3,
                               kpt_color[i // 2 % len(kpt_color)], -1)

    return det_img


class_name = {0: 'hand'}

_connections = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
                (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (13, 14), (14, 15), (15, 16), (17, 18), (18, 19),
                (19, 20))


line_color = [colors_dict_ch_bgr["橙"] for _ in range(4)] + [colors_dict_ch_bgr["深天蓝"] for _ in range(4)] + \
             [colors_dict_ch_bgr["紫红"]] + [colors_dict_ch_bgr["查特酒绿"] for _ in range(3)] + \
             [colors_dict_ch_bgr["紫红"]] + [colors_dict_ch_bgr["青"] for _ in range(3)] + \
             [colors_dict_ch_bgr["紫红"]] + [colors_dict_ch_bgr["深粉"] for _ in range(3)] + \
             [colors_dict_ch_bgr["纯黄"] for _ in range(3)]

pose_shou = PosePredictor(overrides=overrides_1)
det_shou = DetectionPredictor(overrides=overrides_2)

video_path = r"../ultralytics/assets/t1.mp4"
cap = cv2.VideoCapture(video_path)

# 初始化视频保存相关变量
save_video = False
video_writer = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame, _ = resize_image(frame, 960)
    img_height, img_width, _ = frame.shape

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_all = det_shou(img)[0]

    for i, bbox in enumerate(hand_all.boxes.xyxy):
        x1, y1, x2, y2 = list(map(int, bbox))
        x11, y11, x22, y22 = expand_bbox(bbox, img_width, img_height, scale=0)
        x111, y111, x222, y222 = expand_bbox(bbox, img_width, img_height, scale=2/3)
        conf = hand_all.boxes.conf[i]
        cls = hand_all.boxes.cls[i]
        label = f'{hand_all.names[int(cls)]} {float(conf):.2f}'

        image_shou = frame[y1:y2, x1:x2].copy()
        image_shou_l = frame[y111:y222, x111:x222]
        # cv2.imshow('a', image_shou_l)
        # cv2.waitKey(0)

        shou_all = pose_shou(image_shou_l)[0].cpu().numpy()

        if len(shou_all.boxes.conf) > 0:
            sx1, sy1, sx2, sy2 = list(map(int, shou_all.boxes.xyxy[0]))
            ww, hh = sx2 - sx1, sy2 - sy1

            kpts = [list(map(int, shou_all.keypoints.xy[0].reshape(1, 42)[0].tolist()))]
            result = [
                (value - sx1) / ww if i % 2 == 0 else (value - sy1) / hh
                for i, value in enumerate(kpts[0])
            ]

            image_shou_l = draw_bboxes_and_keypoints(image_shou_l, [0, 0, 0, 0], shou_all.boxes.conf,
                                                   shou_all.boxes.cls,
                                                   kpts=kpts, cat_order=_connections, line_color=line_color)

            frame[y111:y222, x111:x222] = image_shou_l

        # 绘制边界框和标签
        cv2.rectangle(frame, (x11, y11), (x22, y22), (0, 255, 0), 2)
        cv2.putText(frame, label, (x11, y11 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # 按 's' 键开始保存视频
        if not save_video:
            save_video = True
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            output_filename = f"video_plot_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_filename, fourcc, 20.0, (img_width, img_height))
            print(f"开始保存视频到 {output_filename}")

    if save_video:
        video_writer.write(frame)

    if key == ord('q'):
        # 按 'q' 键结束保存并退出
        if save_video:
            save_video = False
            video_writer.release()
            print(f"视频保存完成并存储在当前目录下。")
        break

cap.release()
cv2.destroyAllWindows()
