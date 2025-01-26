import sys
import os
import yaml
from pathlib import Path


current_file_path = Path(__file__).resolve()
ancestor_directory = current_file_path.parents[2]
sys.path.insert(0, str(ancestor_directory))     #  必须转成str

from ultralytics import YOLO

sys.path.pop(0)


model = YOLO("../../weights/yolov8n-pose.pt")

with open('./setting.yaml') as f:
    overrides = yaml.safe_load(f.read())

model.train(**overrides)
