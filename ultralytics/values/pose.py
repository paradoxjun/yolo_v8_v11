COCO_keypoint_indexes = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

MPII_keypoint_indexes = {
    0: 'right_ankle',
    1: 'right_knee',
    2: 'right_hip',
    3: 'left_hip',
    4: 'left_knee',
    5: 'left_ankle',
    6: 'pelvis',
    7: 'thorax',
    8: 'upper_neck',
    9: 'head_top',
    10: 'right_wrist',
    11: 'right_elbow',
    12: 'right_shoulder',
    13: 'left_shoulder',
    14: 'left_elbow',
    15: 'left_wrist'
}

COCO_DEFAULT_UPPER_BODY_KEYPOINT_INDICES = (5, 6, 7, 8, 9, 10)          # 上半身的关键点索引
COCO_DEFAULT_CONNECTIONS = ((4, 2), (2, 0), (0, 1), (1, 3), (3, 5))     # 关键点连接顺序（例如：0连接1，1连接2，依此类推）

default_up_body_indices = {
    0: 'left_shoulder',
    1: 'right_shoulder',
    2: 'left_elbow',
    3: 'right_elbow',
    4: 'left_wrist',
    5: 'right_wrist',
}
