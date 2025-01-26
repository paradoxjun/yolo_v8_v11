import mediapipe as mp
import cv2
import os
from ultralytics.task_bank.pose_hands.utils import draw_hand

# 设置环境变量来抑制日志信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TensorFlow Lite 日志级别
os.environ['GLOG_minloglevel'] = '2'  # GLOG 日志级别
os.environ['MEDIAPIPE_LOGLEVEL'] = '2'  # MediaPipe 日志级别


class HandsKeyPoints:
    def __init__(self, static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3,
                 min_tracking_confidence=0.3):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def __call__(self, img, *args, is_rgb=False, **kwargs):
        height, width, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if not is_rgb else img
        result = self.hands.process(img_rgb)

        return result.multi_handedness, result.multi_hand_landmarks


get_hand = HandsKeyPoints(static_image_mode=True)


def plot_hands(img, det_res):
    for bbox in det_res:
        x1, y1, x2, y2 = list(map(int, bbox[:4]))
        img_patch = img[y1:y2, x1:x2]
        hands = get_hand(img_patch)[1]
        draw_hand(img, x2 - x1, y2 - y1, hands, offset=(x1, y1))


if __name__ == '__main__':
    img_path = r'../../assets/bus.jpg'

    hands = HandsKeyPoints()
    img = cv2.imread(img_path)

    keys = hands(img)
    # print(keys)

    h, w, _ = img.shape
    draw_hand(img, w, h, keys[1])
    cv2.imshow("hands", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

