import sys
import yaml
import torch
from pathlib import Path

current_file_path = Path(__file__).resolve()
ancestor_directory = current_file_path.parents[2]
sys.path.insert(0, str(ancestor_directory))  # 必须转成str

from ultralytics import YOLO

sys.path.pop(0)


# 加载剪枝后的模型权重文件
# model_path = r'G:\code\yolo_v8_v11\torch_pruning\step_0_pre_val/best.pt'
# model_path = r'../kx_s_241118/train/weights/best.pt'
model_path = r'./S2_prune/step_pre_val/last_x0.8.pt'
model = YOLO(model_path)


with open('./setting.yaml') as f:
    overrides = yaml.safe_load(f.read())

if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    model.train(**overrides)
