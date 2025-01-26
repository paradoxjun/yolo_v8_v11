from ultralytics.task_bank.detection.predict import BankDetectionPredictor
from ultralytics.task_bank.pose_body.predict import PosePredictor


overrides_1 = {"task": "pose",
               "mode": "predict",
               "model": r'../../weights/yolov8s-pose.pt',
               "verbose": False,
               "classes": [0],
               "iou": 0.1
               }

overrides_2 = {"task": "detect",
               "mode": "predict",
               "model": r'../../trains_det/hands_07_09_m/train2/weights/best.pt',
               # "model": r'../../trains_det/hands_07_11_m/runs/train/weights/best.pt',
               "verbose": False,
               }

predictor_ren = PosePredictor(overrides=overrides_1)
predictor_shou = BankDetectionPredictor(overrides=overrides_2)
