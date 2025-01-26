import numpy as np


def calculate_iou(box1, box2):
    """
    计算两个矩形框的交并比（IoU）
    :param box1: 第一个矩形框的坐标 [x_min, y_min, x_max, y_max]
    :param box2: 第二个矩形框的坐标 [x_min, y_min, x_max, y_max]
    :return: 交并比（IoU）
    """
    # 计算交集的坐标
    inter_x_min = max(box1[0], box2[0])
    inter_y_min = max(box1[1], box2[1])
    inter_x_max = min(box1[2], box2[2])
    inter_y_max = min(box1[3], box2[3])

    # 计算交集的面积
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # 计算两个矩形框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并集的面积
    union_area = box1_area + box2_area - inter_area

    # 计算IoU
    iou = inter_area / union_area if union_area > 0 else 0

    return iou


def calculate_iou_np(boxes1, boxes2):
    """
    计算一组框和另一组框之间的 IoU。
    boxes1 和 boxes2 是形状为 (N, 4) 的 NumPy 数组，其中 N 是框的数量，4 对应于 [x1, y1, x2, y2]。
    返回一个形状为 (N,) 的 NumPy 数组，其中包含每个框之间的 IoU。
    """
    x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
    y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
    x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
    y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])

    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = boxes1_area + boxes2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou
