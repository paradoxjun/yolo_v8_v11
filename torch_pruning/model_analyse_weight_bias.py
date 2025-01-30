import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from ultralytics import YOLO


def visualize_bn_weight_bias_distribution(model, save_name="bn_weight_bias_distribution.jpg"):
    """
    分析 YOLO 模型中 BN 层的权重 (gamma) 和偏置 (beta) 分布，并绘制直方图。

    1. 提取所有 BatchNorm2d 层的 weight (gamma) 和 bias (beta)。
    2. 计算 80%、90%、95%、99%、99.9%、99.99% 的百分位数。
    3. 记录并显示绝对值最大的 5 个异常值（不取绝对值，保留正负信息）。
    4. **异常值和 "Top 5 Outliers" 文字共用一个背景框，防止遮挡**。
    5. **不同百分比线使用不同颜色**。

    Args:
        model: 已加载的 YOLO 模型。
        save_name: 保存图像的路径。
    """
    gamma_weights, beta_biases = [], []

    # 遍历模型，提取 BatchNorm2d 层的 gamma (weight) 和 beta (bias)
    for layer in model.model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            gamma_weights.append(layer.weight.detach().cpu().numpy())  # weight (gamma)
            beta_biases.append(layer.bias.detach().cpu().numpy())  # bias (beta)

    # 将权重和偏置展平成一维数组
    gamma_weights = np.concatenate(gamma_weights)
    beta_biases = np.concatenate(beta_biases)

    # 计算百分位数
    percentiles = [80, 90, 95, 99, 99.9, 99.99]
    colors = ["blue", "orange", "green", "red", "purple", "black"]  # 不同百分位数颜色
    gamma_quantiles = np.percentile(gamma_weights, percentiles)
    beta_quantiles = np.percentile(beta_biases, percentiles)

    # 计算最大的 5 个异常值（包含负数）
    top_gamma_outliers = sorted(gamma_weights, key=lambda x: abs(x), reverse=True)[:5]
    top_beta_outliers = sorted(beta_biases, key=lambda x: abs(x), reverse=True)[:5]

    # 创建画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 绘制 gamma (权重) 分布（不取绝对值）
    sns.histplot(gamma_weights, bins=100, kde=True, ax=axes[0], color='blue', alpha=0.7)
    axes[0].set_title("Distribution of Gamma Weights (BN Layers)")
    axes[0].set_xlabel("Gamma Value")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(True)

    # 绘制 beta (偏置) 分布（不取绝对值）
    sns.histplot(beta_biases, bins=100, kde=True, ax=axes[1], color='green', alpha=0.7)
    axes[1].set_title("Distribution of Beta Biases (BN Layers)")
    axes[1].set_xlabel("Beta Value")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True)

    # 在直方图上添加百分位数的虚线
    for i, (gamma_q, beta_q) in enumerate(zip(gamma_quantiles, beta_quantiles)):
        axes[0].axvline(x=gamma_q, color=colors[i], linestyle='--', label=f"{percentiles[i]}%: {gamma_q:.4f}")
        axes[1].axvline(x=beta_q, color=colors[i], linestyle='--', label=f"{percentiles[i]}%: {beta_q:.4f}")

    axes[0].legend()
    axes[1].legend()

    # 右下角标记异常值，所有文本在同一个背景框内
    text_x, text_y = 0.95, 0.05  # 右下角
    bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7)  # 背景框

    # 组合异常值文本（权重 + 偏置）
    gamma_outliers_text = "Top 5 Gamma Outliers:\n" + "\n".join(
        [f"{i + 1}: {value:.6f}" for i, value in enumerate(top_gamma_outliers)])
    beta_outliers_text = "Top 5 Beta Outliers:\n" + "\n".join(
        [f"{i + 1}: {value:.6f}" for i, value in enumerate(top_beta_outliers)])

    axes[0].text(text_x, text_y, gamma_outliers_text, transform=axes[0].transAxes,
                 fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=bbox_props)

    axes[1].text(text_x, text_y, beta_outliers_text, transform=axes[1].transAxes,
                 fontsize=9, verticalalignment='bottom', horizontalalignment='right', bbox=bbox_props)

    # 保存并显示
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.show()
    print(f"\nBN 层参数分布已保存为 {save_name}")


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # YOLO 权重路径
    weight_path = "../trains_det/kx_s_250126/Step_1_L1/weights/best.pt"
    save_name = "bn_weight_bias_distribution_best.jpg"

    # 加载模型, 分析 BN 层参数分布
    visualize_bn_weight_bias_distribution(YOLO(weight_path), save_name)
