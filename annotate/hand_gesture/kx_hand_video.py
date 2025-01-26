import cv2

from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.task_bank.utils.ops import resize_and_pad


overrides_1 = {"task": "detect",
               "mode": "predict",
               "model": r'../trains_det/kx_11_18_s/train2/weights/best.pt',
               "verbose": False,
               }
predictor_kx = DetectionPredictor(overrides=overrides_1)

# video_path = r"/home/chenjun/code/datasets/bank_monitor/save_video/c3.mp4"
# video_path = r"/home/chenjun/下载/bank2406-柜台垂直视角1/城东柜员1/城东柜员1全景_20240201161000-20240201162000_1.mp4"
video_path = r"../../tests_det/kx/kx.mp4"
cap = cv2.VideoCapture(video_path)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = resize_and_pad(frame)

    # img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = frame
    kx_all = predictor_kx(img)[0]

    for i, bbox in enumerate(kx_all.boxes.xyxy):
        x1, y1, x2, y2 = list(map(int, bbox))
        conf = kx_all.boxes.conf[i]
        cls = kx_all.boxes.cls[i]
        label = f'{kx_all.names[int(cls)]} {float(conf):.2f}'

        # 绘制边界框和标签
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Frame', frame)
    # time.sleep(0.1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
