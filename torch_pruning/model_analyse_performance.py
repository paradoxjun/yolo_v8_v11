import torch
import torch_pruning as tp
from ultralytics import YOLO


def analyse_yolo_model_performance(model_path, zero_threshold=1e-6, img_size_hw=(640, 640)):
    model = YOLO(model_path)
    example_inputs = torch.randn(1, 3, img_size_hw[0], img_size_hw[1]).to(model.device)
    model_macs, model_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    print(f"\033[1;34m模型路径\033[0m: {model_path}")
    print(f"\033[1;34m输入参数\033[0m: zero_threshold: {zero_threshold}, img_size_hw: {img_size_hw}")
    print(f"\033[1;34m模型参数\033[0m: {model_nparams:,} parameters")
    print(f"\033[1;34m模型GFLOPs\033[0m: {model_macs / 1e9: .2f} G flops ")

    # 统计 BN 层和所有层的稀疏度信息
    bn_weights, bn_biases, all_weights, all_biases, non_bn_weights, non_bn_biases = [], [], [], [], [], []

    for layer in model.model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            bn_weights.append(layer.weight.detach().cpu().numpy().flatten())  # BN 权重
            bn_biases.append(layer.bias.detach().cpu().numpy().flatten())  # BN 偏置
        elif hasattr(layer, 'weight') and layer.weight is not None:
            non_bn_weights.append(layer.weight.detach().cpu().numpy().flatten())  # 非 BN 层权重
        if hasattr(layer, 'bias') and layer.bias is not None:
            non_bn_biases.append(layer.bias.detach().cpu().numpy().flatten())  # 非 BN 层偏置

    # 合并所有数据
    bn_weights = np.concatenate(bn_weights) if bn_weights else np.array([])
    bn_biases = np.concatenate(bn_biases) if bn_biases else np.array([])
    non_bn_weights = np.concatenate(non_bn_weights) if non_bn_weights else np.array([])
    non_bn_biases = np.concatenate(non_bn_biases) if non_bn_biases else np.array([])
    all_weights = np.concatenate((bn_weights, non_bn_weights)) if bn_weights.size or non_bn_weights.size else np.array([])
    all_biases = np.concatenate((bn_biases, non_bn_biases)) if bn_biases.size or non_bn_biases.size else np.array([])

    # 计算稀疏度（0 的占比）
    def compute_sparsity(tensor):
        return 100.0 * (np.sum(abs(tensor) <= zero_threshold) / tensor.size) if tensor.size else 0.0

    print(f"\033[1;35mBN 层权重稀疏度:\033[0m {compute_sparsity(bn_weights):.2f}%")
    print(f"\033[1;35mBN 层偏置稀疏度:\033[0m {compute_sparsity(bn_biases):.2f}%")
    print(f"\033[1;35mBN 层所有参数稀疏度:\033[0m {compute_sparsity(np.concatenate((bn_weights, bn_biases))):.2f}%")

    print(f"\033[1;36m非 BN 层权重稀疏度:\033[0m {compute_sparsity(non_bn_weights):.2f}%")
    print(f"\033[1;36m非 BN 层偏置稀疏度:\033[0m {compute_sparsity(non_bn_biases):.2f}%")
    print(f"\033[1;36m非 BN 层所有参数稀疏度:\033[0m {compute_sparsity(np.concatenate((non_bn_weights, non_bn_biases))):.2f}%")

    print(f"\033[1;33m所有层权重稀疏度:\033[0m {compute_sparsity(all_weights):.2f}%")
    print(f"\033[1;33m所有层偏置稀疏度:\033[0m {compute_sparsity(all_biases):.2f}%")
    print(f"\033[1;33m所有层参数总稀疏度:\033[0m {compute_sparsity(np.concatenate((all_weights, all_biases))):.2f}%")


if __name__ == '__main__':
    import os
    import numpy as np

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # YOLO 权重路径
    # weight_path = r"../trains_det/kx_s_241118/train/weights/last.pt"
    # weight_path = r"../trains_det/kx_s_250126/S1_L1/weights/last.pt"
    weight_path = r"../trains_det/kx_s_250126/S2_prune/step_pre_val/last_x0.8.pt"
    zero_threshold = 1e-6

    analyse_yolo_model_performance(weight_path, zero_threshold=zero_threshold)
