from ultralytics import YOLO
from multiprocessing import freeze_support
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


if __name__ == "__main__":
    freeze_support()

    model_path = "../trains_det/hands_10_25_s/train2/weights/best.pt"
    # model_path = "../trains_det/kx_11_18_s/train/weights/best.pt"
    # model_path = "../export/best.onnx"
    data_path = "./test.yaml"
    fs = 20

    model = YOLO(model_path, task="detect")  # load an official model
    metrics = model.val(data=data_path, batch=16)  # no arguments needed, dataset and settings remembered

    box_map50 = round(float(metrics.box.map50), 4)  # map50
    box_map75 = round(float(metrics.box.map75), 4)  # map75
    box_map50_95 = round(float(metrics.box.map), 4)  # map50-95

    print("box metric:".rjust(fs), "mAP50".rjust(fs), "mAP75".rjust(fs), "mAP50-95".rjust(fs))
    print(model_path.rpartition('/')[-1].rjust(fs),
          str(box_map50).rjust(fs), str(box_map75).rjust(fs), str(box_map50_95).rjust(fs))
