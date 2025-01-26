import sys
import os
import yaml
from pathlib import Path


current_file_path = Path(__file__).resolve()
ancestor_directory = current_file_path.parents[2]
sys.path.insert(0, str(ancestor_directory))     #  必须转成str

from ultralytics import YOLO

sys.path.pop(0)


# model = YOLO("../hands_10_15_s/train2/weights/best.pt")
model = YOLO("../hands_25_01_02_s/train/weights/best.pt")


with open('./setting.yaml') as f:
    overrides = yaml.safe_load(f.read())

model.train(**overrides)
