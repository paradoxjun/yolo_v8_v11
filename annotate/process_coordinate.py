"""
方法参考自：yolov5/utils/general
"""
import numpy as np


def xyxy2xywh(x, precision=6):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right."""
    y = [0.0] * 4
    y[0] = round((x[0] + x[2]) / 2, precision)  # x center
    y[1] = round((x[1] + x[3]) / 2, precision)  # y center
    y[2] = round(x[2] - x[0], precision)  # width
    y[3] = round(x[3] - x[1], precision)  # height

    return y


def xywh2xyxy(x, precision=6):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right."""
    y = [0.0] * 4
    y[0] = round(x[0] - x[2] / 2, precision)  # top left x
    y[1] = round(x[1] - x[3] / 2, precision)  # top left y
    y[2] = round(x[0] + x[2] / 2, precision)  # bottom right x
    y[3] = round(x[1] + x[3] / 2, precision)  # bottom right y

    return y


def xyxy2xywhn(xyxy, shape, precision=6):
    """
    计算归一化边界框的中心点坐标、宽度和高度。
    :param xyxy:
    :param shape: （高度，宽度）
    :param precision:
    :return:
    """
    xmin, ymin, xmax, ymax = xyxy[:4]

    x_center = round((xmin + xmax) / (2.0 * shape[1]), precision)
    y_center = round((ymin + ymax) / (2.0 * shape[0]), precision)
    box_width = round((xmax - xmin) / shape[1], precision)
    box_height = round((ymax - ymin) / shape[0], precision)

    return [x_center, y_center, box_width, box_height]


def xywhn2xyxy(coordinate, shape, precision=6):
    x_center, y_center, box_width, box_height = coordinate[:4]

    x_min = round((2.0 * x_center - box_width) * shape[1] / 2.0, precision)
    y_min = round((2.0 * y_center - box_height) * shape[0] / 2.0, precision)
    x_max = round((2.0 * x_center + box_width) * shape[1] / 2.0, precision)
    y_max = round((2.0 * y_center + box_height) * shape[0] / 2.0, precision)

    return [x_min, y_min, x_max, y_max]


def tlwh2xywhn(xywh, shape, precision=6):
    """左上+宽高 => 归一化的中心+宽高"""
    x, y, w, h = xywh[:4]
    x_center = round((x + w / 2.0) / shape[1], precision)
    y_center = round((y + h / 2.0) / shape[0], precision)
    box_width = round(w / shape[1], precision)
    box_height = round(h / shape[0], precision)

    return [x_center, y_center, box_width, box_height]
