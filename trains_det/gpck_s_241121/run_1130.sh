#!/bin/bash
python train.py

cd /data/chenjun/code/ultralytics_YOLOv8/trains_det/kx_11_18_s

python train.py

# 添加项目路径到 PYTHONPATH
export PYTHONPATH=/data/chenjun/code/classfication_net:$PYTHONPATH

# 可选：输出确认路径
echo "PYTHONPATH set to: $PYTHONPATH"
cd /data/chenjun/code/classfication_net/start_file
python train.py
cd /data/chenjun/code/ultralytics_YOLOv8/trains_det/gpck_11_21_s

