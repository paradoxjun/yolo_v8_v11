import cv2
import torch
import numpy as np


def get_arm_keypoint(data, keypoint_indices=(9, 10)):
    # 检查数据是否为空或大小为零
    if data is None or data.shape[0] == 0 or data.shape[1] == 0:
        # 返回一个空的数组，形状为 (0, len(keypoint_indices), 3)
        return np.empty((0, len(keypoint_indices), 3))      # 如果不需要使用置信度，长度设为2

    # 检查 keypoint_indices 是否超出数据的范围
    if max(keypoint_indices) >= data.shape[1]:
        raise IndexError("Keypoint indices are out of bounds for the given data")

    return data[:, keypoint_indices, :]


def plot_keypoint(image, data, connections=((0, 0), (1, 1)), point_color=(0, 0, 255), point_radius=4,
                  line_color=(0, 255, 0), line_thickness=2):
    """
    在图片上绘制关键点和连线。
    Args:
        image: 图片源
        data: YOLOv8姿态检测结果
        connections: 连线顺讯
        point_color: 关键点的颜色
        point_radius: 关键点的大小
        line_color: 连线的颜色
        line_thickness: 连线的粗细

    Returns:
        绘制了关键点的图片。
    """
    data = data.cpu().numpy() if torch.is_tensor(data) else np.array(data)  # 将张量移动到CPU并转换为numpy数组

    # 绘制关键点
    for person in data:
        # 绘制连接线
        if connections:
            for start_idx, end_idx in connections:
                sta_point = person[start_idx]
                end_point = person[end_idx]
                if (sta_point[0] > 0 or sta_point[1] > 0) and (end_point[0] > 0 and end_point[1] > 0):  # 忽略无效点
                    cv2.line(image, (int(sta_point[0]), int(sta_point[1])),
                             (int(end_point[0]), int(end_point[1])), line_color, line_thickness)

        # 绘制关键点
        for point in person:
            x, y = point[:2]
            if x > 0 or y > 0:  # 忽略无效点
                cv2.circle(image, (int(x), int(y)), point_radius, point_color, -1)

    return image
