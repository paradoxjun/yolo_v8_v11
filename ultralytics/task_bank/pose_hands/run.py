import mediapipe as mp
import cv2
import numpy as np
from utils import *

cap = cv2.VideoCapture('/home/chenjun/code/datasets/内部数据/bank2406-柜台垂直视角1/城东柜员1/城东柜员1全景_20240201161000-20240201162000_1.mp4')
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=True, max_num_hands=10, min_detection_confidence=0.2, min_tracking_confidence=0.2)
mpDraw = mp.solutions.drawing_utils

while True:
    # 读取一帧图像
    ret, img = cap.read()
    height, width, channels = img.shape
    # 转换为RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 得到检测结果
    results = hands.process(imgRGB)

    # if results.multi_hand_landmarks:
    #     for hand in results.multi_hand_landmarks:
    #         mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
    #         # draw_hand(img, width, height, hand)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            # hand = results.multi_hand_landmarks[0]
            # mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

            # 采集所有关键点坐标
            list_lms = []
            for i in range(21):
                pos_x = int(hand.landmark[i].x * width)
                pos_y = int(hand.landmark[i].y * height)
                list_lms.append([pos_x, pos_y])

            # 构造凸包点
            list_lms = np.array(list_lms, dtype=np.int32)
            hull_index = [0, 1, 2, 3, 6, 10, 14, 19, 18, 17]
            hull = cv2.convexHull(list_lms[hull_index], True)
            # cv2.polylines(img, [hull], True, (0, 255, 0), 2)

            # 查找外部的点数
            ll = [4, 8, 12, 16, 20]
            out_fingers = []
            for i in ll:
                pt = (int(list_lms[i][0]), int(list_lms[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < 0:
                    out_fingers.append(i)

            str_gesture = get_str_gesture(out_fingers, list_lms)
            cv2.putText(img, str_gesture, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 4, cv2.LINE_AA)
            for i in ll:
                pos_x = int(hand.landmark[i].x * width)
                pos_y = int(hand.landmark[i].y * height)
                cv2.circle(img, (pos_x, pos_y), 3, (0, 255, 255), -1)

    cv2.imshow('hands', img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
