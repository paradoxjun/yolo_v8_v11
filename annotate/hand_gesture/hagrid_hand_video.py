import cv2

from ultralytics.task_bank.detection.predict import BankDetectionPredictor
from ultralytics.task_bank.pose_body.predict import PosePredictor
from ultralytics.task_bank.utils.ops import resize_and_pad


overrides_1 = {"task": "pose",
               "mode": "predict",
               "model": r'../weights/yolov8m-pose.pt',
               "verbose": False,
               "classes": [0],
               "iou": 0.25
               }

overrides_2 = {"task": "detect",
               "mode": "predict",
               "model": r'../trains_det/hands_07_09_m/train2/weights/best.pt',
               "verbose": False,
               }

predictor_ren = PosePredictor(overrides=overrides_1)
predictor_shou = BankDetectionPredictor(overrides=overrides_2)

# video_path = r"/home/chenjun/code/datasets/bank_monitor/save_video/c3.mp4"
# video_path = r"/home/chenjun/下载/bank2406-柜台垂直视角1/城东柜员1/城东柜员1全景_20240201161000-20240201162000_1.mp4"
video_path = r"/home/chenjun/下载/bank2406-柜台垂直视角1/城东柜员3/城东柜员3全景_20240321145001-20240321150001_1.mp4"
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_and_pad(frame)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    ren_all = predictor_ren(img)[0]

    for i, bbox in enumerate(ren_all.boxes.xyxy):
        x1, y1, x2, y2 = list(map(int, bbox))
        conf = ren_all.boxes.conf[i]
        cls = ren_all.boxes.cls[i]
        label = f'{ren_all.names[int(cls)]} {float(conf):.2f}'

        # 绘制边界框和标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        image_ren = frame[y1:y2, x1:x2]
        shou_all = predictor_shou(image_ren)[0]

        for j, bbox_s in enumerate(shou_all.boxes.xyxy):
            x1_s, y1_s, x2_s, y2_s = list(map(int, bbox_s))
            conf = shou_all.boxes.conf[j]
            cls = shou_all.boxes.cls[j]
            label = f'{shou_all.names[int(cls)]} {float(conf):.2f}'

            # 绘制边界框和标签
            cv2.rectangle(frame, (x1 + x1_s, y1 + y1_s), (x1 + x2_s, y1 + y2_s), (255, 255, 0), 2)
            cv2.putText(frame, label, (x1 + x1_s, y1 + y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (255, 255, 0), 2)

    cv2.imshow('Frame', frame)
    # time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()