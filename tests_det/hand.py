import cv2

from ultralytics.task_bank.detection.predict import BankDetectionPredictor
from ultralytics.task_bank.utils.ops import resize_and_pad

overrides_2 = {"task": "detect",
               "mode": "predict",
               "model": r'../trains_det/hands_07_11_m/runs/train/weights/best.pt',
               "verbose": False,
               }

predictor_shou = BankDetectionPredictor(overrides=overrides_2)


img_path = r"/home/chenjun/code/ultralytics_YOLOv8/export/paml_1.jpg"
#img_path = r"/home/chenjun/code/deploy/test_data/zidane.jpg"
frame = cv2.imread(img_path)

img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = resize_and_pad(img)

ren_all = predictor_shou(img)[0]


for i, bbox in enumerate(ren_all.boxes.xyxy):
    print(bbox)
    x1, y1, x2, y2 = list(map(int, bbox))
    conf = ren_all.boxes.conf[i]
    print(conf)
    cls = ren_all.boxes.cls[i]
    label = f'{ren_all.names[int(cls)]} {float(conf):.2f}'

    # 绘制边界框和标签
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    image_ren = frame[y1:y2, x1:x2]
    shou_all = predictor_shou(image_ren)[0]

cv2.imshow('Frame', frame)
# time.sleep(0.1)
cv2.waitKey(0)
cv2.destroyAllWindows()
