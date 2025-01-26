import sys
import os
import yaml
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


current_file_path = Path(__file__).resolve()
ancestor_directory = current_file_path.parents[2]
sys.path.insert(0, str(ancestor_directory))     #  必须转成str

from ultralytics import YOLO

sys.path.pop(0)


model = YOLO("./train3/weights/best.pt")
# model = YOLO("../../weights/yolov8s-pose.pt")

with open('./setting.yaml') as f:
    overrides = yaml.safe_load(f.read())

model.train(**overrides)
