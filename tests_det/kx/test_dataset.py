import os
import yaml
from ultralytics import YOLO
from multiprocessing import freeze_support


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    freeze_support()

    model_cfg_path = "setting.yaml"

    with open(model_cfg_path) as f:
        overrides_cfg = yaml.safe_load(f.read())
    fs = 20

    model = YOLO(model=overrides_cfg["model"], task=overrides_cfg["task"])  # load an official model
    metrics = model.val(**overrides_cfg)  # no arguments needed, dataset and settings remembered

    box_map50 = round(float(metrics.box.map50), 4)  # map50
    box_map75 = round(float(metrics.box.map75), 4)  # map75
    box_map50_95 = round(float(metrics.box.map), 4)  # map50-95

    print("box metric:".rjust(fs), "mAP50".rjust(fs), "mAP75".rjust(fs), "mAP50-95".rjust(fs))
    print(overrides_cfg["model"].rpartition('/')[-1].rjust(fs),
          str(box_map50).rjust(fs), str(box_map75).rjust(fs), str(box_map50_95).rjust(fs))
