import numpy as np


def apply_mask_body(image, bboxes, keypoints, offset=(0, 0)):
    """
    根据身体区域给定的检测框和关键点在图像上应用马赛克。
    :param image: 输入图像
    :param bboxes: 一组人的检测框 [[x11, y11, x12, y12], [x21, y21, x22, y22], ...]
    :param keypoints: 对应于人的手腕和手肘关键点 [[左手肘1, 右手肘1, 左手腕1, 右手腕1], ]
    :param offset: x和y方向上的偏移量
    :return: 是否进行了修改
    """
    is_modify = False

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = list(map(int, bbox[:4]))

        for j in range(2):
            point = keypoints[i][j]

            if point[0] == 0 and point[1] == 0:
                continue
            else:
                hand_width = (x2 - x1) / 3.0
                hand_height = (y2 - y1) / 3.0

                min_x = max(int(point[0] - hand_width / 2), x1) + offset[0]
                max_x = min(int(point[0] + hand_width / 2), x2) + offset[0]
                min_y = max(int(point[1] - hand_height / 2), y1) + offset[1]
                max_y = min(int(point[1] + hand_height / 2), y2) + offset[1]

                if min_y >= max_y or min_x >= max_x:  # 关键点在人的检测框外
                    continue

                is_modify = True

            image[min_y:max_y, min_x:max_x] = np.random.randint(0, 256, (max_y - min_y, max_x - min_x, 3),
                                                                dtype=np.uint8)

    return is_modify


def apply_mask_hand(image, bboxes, offset=(0, 0)):
    """
    将图片上指定检测框部分修改为随机高斯噪声。
    :param image: 输入图像
    :param bboxes: 检测框数组 [[x_min, y_min, x_max, y_max], ...]
    :param offset: x和y方向上的偏移量
    :return: 是都进行了修改
    """
    is_modify = False

    for bbox in bboxes:
        min_x, min_y, max_x, max_y = map(int, bbox)
        min_x += offset[0]
        max_x += offset[0]
        min_y += offset[1]
        max_y += offset[1]
        image[min_y:max_y, min_x:max_x] = np.random.randint(0, 256, (max_y - min_y, max_x - min_x, 3), dtype=np.uint8)
        is_modify = True

    return is_modify
