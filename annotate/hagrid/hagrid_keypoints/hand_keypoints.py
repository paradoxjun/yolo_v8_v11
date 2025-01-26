import numpy as np


def process_mp_result(results, region_h, region_w, offset=(0, 0)):
    """
    处理mediapipe对手部关键点的预测结果。
    Args:
        results: mp_hands.Hands.process(img)的结果。
        region_h: 手部区域的高度。
        region_w: 手部区域的宽度。
        offset: 偏移量，为手部区域在实际图片左上角点的(x, y)。
    Returns:
        xyxy，keypoints：原图中刚好能包住手部区域的矩形框，原图中的关键点。
    """
    if results.multi_hand_landmarks is not None:
        list_lms = []  # 采集所有关键点坐标
        x_min, y_min, x_max, y_max = 1e6, 1e6, 0, 0  # 初始化bbox的初始值
        hand = results.multi_hand_landmarks[0]

        for i in range(21):
            # 获取手部区域的坐标（限制在该区域内），并映射回原始图片
            pos_x = min(max(2, int(hand.landmark[i].x * region_w)), int(region_w) - 2) + int(offset[0])
            pos_y = min(max(2, int(hand.landmark[i].y * region_h)), int(region_h) - 2) + int(offset[1])

            # 保留全部点
            list_lms.append((pos_x, pos_y))

            # 获取边界框
            x_min = min(x_min, pos_x)
            y_min = min(y_min, pos_y)
            x_max = max(x_max, pos_x)
            y_max = max(y_max, pos_y)

        return (np.array([x_min - 1, y_min - 1, x_max + 1, y_max + 1], dtype=np.float32),
                np.array(list_lms, dtype=np.float32))

    return np.empty((0, 4), dtype=np.dtype), np.empty((0, 2), dtype=np.float32)


def process_hagrid_bbox(xc, yc, w, h, w_img, h_img, base_factor=1.2, expand_factor=0.4, iter_num=1, offset=(0, 0)):
    """
    将原始的手部目标框区域进行放大。
    Args:
        xc: 检测框中心点横坐标
        yc: 检测框中心点纵坐标
        w: 检测框的宽度
        h: 检测框的高度
        w_img: 图片的宽度
        h_img: 图片的高度
        base_factor: 基本放大系数
        expand_factor: 递增放大系数
        iter_num: 放大次数
        offset: x和y的平移量。
    Returns:
        放大后的检测框坐标xyxy
    """
    w_new = w * (base_factor + expand_factor * iter_num)
    h_new = h * (base_factor + expand_factor * iter_num)

    x_min_new = int(max(2, xc - w_new / 2 + offset[0]))
    y_min_new = int(max(2, yc - h_new / 2 + offset[1]))
    x_max_new = int(min(w_img - 3, xc + w_new / 2 + offset[0]))
    y_max_new = int(min(h_img - 3, yc + h_new / 2 + offset[1]))

    return np.array([x_min_new, y_min_new, x_max_new, y_max_new], dtype=np.float32)


def rectify_keypoints(keypoints, w, h, offset=(0, 0), norm=False):
    """
        继续操作，对减去后的值操作：
        如果第1个维度的第1个值v，满足1<v<w-1：不修改；
        如果第1个维度的第1个值v，满足w-1 <= v < w+10：修改v为w-2;
        如果第1个维度的第1个值v，满足-10 < v <= 1：修改v为2;
        其余情况修改为0；
        如果第1个维度的第2个值v，满足1<v<h-1：不修改；
        如果第1个维度的第2个值v，满足h-1 <= v < h+5：修改v为h-2;
        如果第1个维度的第2个值v，满足-5 < h <= 1：修改v为2;
        其余情况修改为0；
        上述修改好后，第一个维度的第一or第二个值为0，则将两个数值都修改为0.
    Args:
        keypoints: 关键点
        w: 实际检测框的宽度
        h: 实际检测框的高度
        offset: x,y （高和宽）上的偏移量
        norm: 进行归一化
    Returns:
        校正后的关键点。
    """
    keypoints[:, 0] -= int(offset[0])
    keypoints[:, 1] -= int(offset[1])

    v1 = keypoints[:, 0]
    v2 = keypoints[:, 1]

    # 预先创建全零数组并使用布尔索引进行条件修改，可以减少重复计算。将边缘、图片外点的值置为0。
    # Apply conditions for v1
    v1_mod = np.zeros_like(v1)
    mask1 = (v1 >= 1) & (v1 <= w - 2)
    mask2 = (v1 > w - 2) & (v1 < w + 15)
    mask3 = (v1 > -16) & (v1 < 1)
    v1_mod[mask1] = v1[mask1]
    v1_mod[mask2] = int(w - 2)
    v1_mod[mask3] = 1

    # Apply conditions for v2
    v2_mod = np.zeros_like(v2)
    mask1 = (v2 >= 1) & (v2 <= h - 2)
    mask2 = (v2 > h - 2) & (v2 < h + 15)
    mask3 = (v2 > -16) & (v2 < 1)
    v2_mod[mask1] = v2[mask1]
    v2_mod[mask2] = int(h - 2)
    v2_mod[mask3] = 1

    # 更新数组
    if norm:
        keypoints[:, 0] = v1_mod / w
        keypoints[:, 1] = v2_mod / h
    else:
        keypoints[:, 0] = v1_mod
        keypoints[:, 1] = v2_mod

    # 如果第 1 维度的第一个或第二个值为 0，则将两个数值都修改为 0
    mask = (keypoints[:, 0] == 0) | (keypoints[:, 1] == 0)
    keypoints[mask] = 0

    return keypoints


def rotate_keypoints_90_clockwise(keypoints):
    """
    将关键点顺时针旋转90度

    :param keypoints: 形状为 [n, 2] 的关键点数组
    :return: 旋转后的关键点数组
    """
    rotated_keypoints = np.zeros_like(keypoints)
    rotated_keypoints[:, 0] = 1 - keypoints[:, 1]  # x' = 1 - y
    rotated_keypoints[:, 1] = keypoints[:, 0]      # y' = x
    return rotated_keypoints


def rotate_yolo_keypoints_90_clockwise(yolo_box):
    """
    将 YOLO 关键点检测框顺时针旋转90度后的结果

    :param yolo_box: 归一化的 YOLO 关键点检测框 [类别, xc, yc, w, h, x1, y1, x2, y2,...]
    :return: 旋转后的 YOLO 关键点检测框
    """
    rotated_box = yolo_box.copy()

    # 旋转中心点和宽高（归一化的）
    rotated_box[1] = 1 - yolo_box[2]  # xc' = 1 - yc
    rotated_box[2] = yolo_box[1]  # yc' = xc
    rotated_box[3] = yolo_box[4]  # w' = h
    rotated_box[4] = yolo_box[3]  # h' = w

    # 旋转关键点
    for i in range(5, len(yolo_box), 2):
        rotated_box[i] = 1 - yolo_box[i + 1]  # x' = 1 - y
        rotated_box[i + 1] = yolo_box[i]  # y' = x

    return rotated_box
