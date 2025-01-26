import torch
import torch.nn as nn
import numpy as np

class KPNet(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super(KPNet, self).__init__()

        mlp1_layers = [2, 64, 128, 256]  # M
        mlp2_layers = [256, 128, 64, num_classes]  # M

        self.mlp1 = nn.ModuleList()
        self.mlp2 = nn.ModuleList()

        for i in range(len(mlp1_layers) - 1):
            self.mlp1.append(nn.Conv1d(mlp1_layers[i], mlp1_layers[i + 1], 1))
            self.mlp1.append(nn.BatchNorm1d(mlp1_layers[i + 1]))
            self.mlp1.append(nn.ReLU())

        for i in range(len(mlp2_layers) - 2):
            self.mlp2.append(nn.Linear(mlp2_layers[i], mlp2_layers[i + 1]))
            self.mlp2.append(nn.BatchNorm1d(mlp2_layers[i + 1]))
            self.mlp2.append(nn.ReLU())
            if i >= 1:
                self.mlp2.append(nn.Dropout(p=dropout_rate))

        self.mlp2.append(nn.Linear(mlp2_layers[-2], mlp2_layers[-1]))

    def forward(self, x):
        x = x.transpose(2, 1)
        for layer in self.mlp1:
            x = layer(x)
        x = torch.max(x, 2)[0]

        for layer in self.mlp2:
            x = layer(x)

        return x

class KPNetPredictor:
    def __init__(self, model_path=None, num_classes=14):
        self.model = KPNet(num_classes)
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.label_dict = {0: 'six', 1: 'dislike', 2: 'fist', 3: 'four', 4: 'like', 5: 'one', 6: 'ok', 7: 'palm', 8: 'two',
                           9: 'rock', 10: 'stop', 11: 'three', 12: 'three2', 13: 'two_up'}

    def preprocess_input(self, input_data):
        if isinstance(input_data, list):
            input_data = np.array(input_data)
        if isinstance(input_data, np.ndarray):
            input_data = input_data.reshape(-1, 2)
            input_data = torch.tensor(input_data, dtype=torch.float32)
        elif not isinstance(input_data, torch.Tensor):
            raise ValueError("输入必须是列表、numpy 数组或 PyTorch 张量")

        return input_data.unsqueeze(0)

    def predict(self, input_data):
        input_tensor = self.preprocess_input(input_data)
        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = nn.functional.softmax(output, dim=1)
            confidence, predicted_idx = torch.max(probabilities, 1)
            predicted_label = self.label_dict[predicted_idx.item()]
            return predicted_label, confidence.item()


kp_predictor = KPNetPredictor(model_path='F:/code/KPNet/model_save/20240816-172047/best_model.pth', num_classes=14)
# 示例用法
if __name__ == "__main__":
    # 创建预测器实例，加载模型


    # 示例输入：列表
    example_input_list = [0.46934974193573, 0.9505019187927246, 0.642944872379303, 0.8652004599571228, 0.7651044130325317, 0.7555271983146667, 0.8486872315406799, 0.6519468426704407, 0.9194111824035645, 0.5483664870262146, 0.6043681502342224, 0.5422735214233398, 0.7072393298149109, 0.4326002299785614, 0.7908222079277039, 0.45087912678718567, 0.8551166653633118, 0.49962282180786133, 0.4950675368309021, 0.49962282180786133, 0.527214765548706, 0.31074100732803345, 0.5850798487663269, 0.19497475028038025, 0.6300859451293945, 0.09139441698789597, 0.3921963572502136, 0.5057157874107361, 0.33433133363723755, 0.33511286973953247, 0.31504297256469727, 0.22543956339359283, 0.31504297256469727, 0.1279521882534027, 0.29575464129447937, 0.5544594526290894, 0.1928834617137909, 0.42650729417800903, 0.13501842319965363, 0.3290199041366577, 0.09001228213310242, 0.2437184453010559]
    predicted_label, confidence = kp_predictor.predict(example_input_list)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.4f}")

    # 示例输入：numpy 数组
    example_input_np = np.array([0.3375, 0.8416666666666667, 0.45625, 0.8333333333333334, 0.559375, 0.7583333333333333, 0.63125, 0.6958333333333333, 0.696875, 0.6625, 0.509375, 0.525, 0.48125, 0.37083333333333335, 0.4625, 0.4375, 0.45625, 0.5333333333333333, 0.4375, 0.5125, 0.415625, 0.36666666666666664, 0.40625, 0.45416666666666666, 0.403125, 0.5458333333333333, 0.359375, 0.5166666666666667, 0.346875, 0.37916666666666665, 0.346875, 0.45416666666666666, 0.284375, 0.5375, 0.284375, 0.5375, 0.284375, 0.4125, 0.290625, 0.43333333333333335, 0.303125, 0.475])
    predicted_label, confidence = kp_predictor.predict(example_input_np)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.4f}")

    # 示例输入：PyTorch 张量
    example_input_tensor = np.array([0.21296296296296297, 0.38472222222222224, 0.26851851851851855, 0.3638888888888889, 0.2833333333333333, 0.3263888888888889, 0.24259259259259258, 0.29583333333333334, 0.1814814814814815, 0.2763888888888889, 0.29074074074074074, 0.2611111111111111, 0.3277777777777778, 0.2152777777777778, 0.35, 0.18472222222222223, 0.3648148148148148, 0.15833333333333333, 0.2462962962962963, 0.24861111111111112, 0.25, 0.19027777777777777, 0.24259259259259258, 0.15138888888888888, 0.2388888888888889, 0.12222222222222222, 0.2, 0.25277777777777777, 0.18703703703703703, 0.19583333333333333, 0.17962962962962964, 0.1597222222222222, 0.17592592592592593, 0.13194444444444445, 0.15925925925925927, 0.26805555555555555, 0.14074074074074075, 0.2263888888888889, 0.13148148148148148, 0.1986111111111111, 0.12222222222222222, 0.175]
)
    predicted_label, confidence = kp_predictor.predict(example_input_tensor)
    print(f"Predicted label: {predicted_label}, Confidence: {confidence:.4f}")
