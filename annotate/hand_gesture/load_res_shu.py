import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights, ShuffleNet_V2_X1_0_Weights, resnet18, shufflenet_v2_x1_0
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F


# 加载类别标签
hagrid_cate_file = ["call", "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace", "peace_inverted",
                    "rock", "stop", "stop_inverted", "three", "three2", "two_up", "two_up_inverted"]

# 定义模型和权重文件路径
model1_name = "../weights/best_model_resnet18.pth"
model2_name = "../weights/best_model_shuffle_net_v2.pth"

# 加载并挂载 ResNet18 模型
model1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
num_ftrs1 = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs1, len(hagrid_cate_file))
model1.load_state_dict(torch.load(model1_name, map_location=torch.device('cpu')))
model1.eval()

# 加载并挂载 ShuffleNet 模型
model2 = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
num_ftrs2 = model2.fc.in_features
model2.fc = nn.Linear(num_ftrs2, len(hagrid_cate_file))
model2.load_state_dict(torch.load(model2_name, map_location=torch.device('cpu')))
model2.eval()


# 图像预处理函数
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 判断输入类型，如果是路径则使用 PIL 打开图像
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    elif isinstance(image, np.ndarray):
        # 如果是 ndarray，则将其转换为 PIL 图像
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # 添加批次维度
    return input_batch


# ResNet18 预测函数
def resnet_pred(image_path):
    input_batch = preprocess_image(image_path)
    with torch.no_grad():
        output = model1(input_batch)
        probabilities = F.softmax(output, dim=1)  # 计算 softmax 概率
        _, predicted_idx = torch.max(output, 1)
        confidence = probabilities[0, predicted_idx.item()].item()  # 获取预测类别的置信度
    return hagrid_cate_file[predicted_idx.item()], confidence


# ShuffleNet 预测函数
def shufflenet_pred(image_path):
    input_batch = preprocess_image(image_path)
    with torch.no_grad():
        output = model2(input_batch)
        probabilities = F.softmax(output, dim=1)  # 计算 softmax 概率
        _, predicted_idx = torch.max(output, 1)
        confidence = probabilities[0, predicted_idx.item()].item()  # 获取预测类别的置信度
    return hagrid_cate_file[predicted_idx.item()], confidence


if __name__ == "__main__":
    # 图片路径
    image_path = r'F:\datasets\hagrid\yolo_cls\val\palm/00bac9c3-c40f-405c-9a98-174b3d72b604_1.jpg'  # 替换为你要预测的图像路径

    # 使用 ResNet18 预测
    predicted_label1 = resnet_pred(image_path)
    print(f"ResNet18 预测的类别: {predicted_label1}")

    # 使用 ShuffleNet 预测
    predicted_label2 = shufflenet_pred(image_path)
    print(f"ShuffleNet 预测的类别: {predicted_label2}")
