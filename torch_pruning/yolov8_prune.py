# This code is adapted from Issue [#147](https://github.com/VainF/Torch-Pruning/issues/147), implemented by @Hyunseok-Kim0.
import math
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, C2f, C2f_v2, Conv, Bottleneck
from ultralytics.nn.tasks import attempt_load_one_weight
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.torch_utils import initialize_weights, de_parallel

import torch_pruning as tp


def save_pruning_performance_graph(x, y1, y2, y3, plt_save_name='pruning_perf_change.png'):
    """
    保存剪枝性能变化图。
    Args:
        x: (List) 表示所有剪枝步骤（或剪枝率）对应的“横坐标”信息。
        y1: (List) 表示剪枝并进行微调（fine-tuning）后得到的 mAP。在绘图时会当作“恢复后的mAP”。
        y2: (List) 表示剪枝后对应的 MACs 值（计算复杂度的一种度量，越小通常表示模型越快）。
        y3: (List) 表示剪枝后但尚未微调时的 mAP（往往会比微调后的 mAP 低一些）。
        plt_save_name: (str) 保存的绘制图像名称。
    Returns:
        输出一张图像并保存为 'pruning_perf_change.png'。
    """
    try:
        plt.style.use("ggplot")
    except:
        pass

    # 转换数组并计算 MACs 的归一化比率
    x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
    y2_ratio = y2 / y2[0]

    # create the figure and the axis object
    fig, ax = plt.subplots(figsize=(8, 6))

    # plot the pruned mAP and recovered mAP
    ax.set_xlabel('Pruning Ratio')
    ax.set_ylabel('mAP')
    ax.plot(x, y1, label='recovered mAP')
    ax.scatter(x, y1)
    ax.plot(x, y3, color='tab:gray', label='pruned mAP')
    ax.scatter(x, y3, color='tab:gray')

    # create a second axis that shares the same x-axis
    ax2 = ax.twinx()

    # plot the second set of data
    ax2.set_ylabel('MACs')
    ax2.plot(x, y2_ratio, color='tab:orange', label='MACs')
    ax2.scatter(x, y2_ratio, color='tab:orange')

    # add a legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='best')

    ax.set_xlim(105, -5)
    ax.set_ylim(0, max(y1) + 0.05)
    ax2.set_ylim(0.05, 1.05)

    # calculate the highest and lowest points for each set of data
    max_y1_idx = np.argmax(y1)
    min_y1_idx = np.argmin(y1)
    max_y2_idx = np.argmax(y2)
    min_y2_idx = np.argmin(y2)
    max_y1 = y1[max_y1_idx]
    min_y1 = y1[min_y1_idx]
    max_y2 = y2_ratio[max_y2_idx]
    min_y2 = y2_ratio[min_y2_idx]

    # add text for the highest and lowest values near the points
    ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {max_y1:.2f}', fontsize=10)
    ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {min_y1:.2f}', fontsize=10)
    ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max MACs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
    ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min MACs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

    plt.title('Comparison of mAP and MACs with Pruning Ratio')
    plt.savefig(plt_save_name)


def infer_shortcut(bottleneck):
    # 判定 bottleneck 有没有残差连接
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels

    # a) c1 == c2        表示输入通道数与输出通道数相同
    # b) hasattr(bottleneck, 'add')  检查 bottleneck 对象是否有名为 'add' 的属性
    # c) bottleneck.add  该属性为 True 时表示确实存在捷径连接
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


def transfer_weights(c2f, c2f_v2):
    # 1. 直接把 c2f 的 cv2 和 m (ModuleList) 复制给 c2f_v2
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m

    # 2. 获取旧模型和新模型的 state_dict（权重字典）
    state_dict = c2f.state_dict()   # 包含 c2f 的所有参数(key)->tensor 的映射
    state_dict_v2 = c2f_v2.state_dict()     # 包含 c2f_v2 的所有参数(key)->tensor 的映射

    # 3. 处理 cv1 中的卷积权重，并拆分到 c2f_v2 的 cv0、cv1
    # Transfer cv1 weights from C2f to cv0 and cv1 in C2f_v2
    old_weight = state_dict['cv1.conv.weight']  # 取出旧模型 c2f 的 'cv1.conv.weight'
    half_channels = old_weight.shape[0] // 2
    state_dict_v2['cv0.conv.weight'] = old_weight[:half_channels]   # 把前一半的卷积核(通道维度)放到新模型的 'cv0.conv.weight'
    state_dict_v2['cv1.conv.weight'] = old_weight[half_channels:]   # 把后一半的卷积核(通道维度)放到新模型的 'cv1.conv.weight'

    # 4. 同理，把 cv1 的 BatchNorm 参数(权重、偏置、均值、方差)也拆分到 cv0.bn 和 cv1.bn
    # Transfer cv1 batchnorm weights and buffers from C2f to cv0 and cv1 in C2f_v2
    for bn_key in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = state_dict[f'cv1.bn.{bn_key}']
        state_dict_v2[f'cv0.bn.{bn_key}'] = old_bn[:half_channels]  # 前 half_channels 个用于 cv0.bn
        state_dict_v2[f'cv1.bn.{bn_key}'] = old_bn[half_channels:]  # 后 half_channels 个用于 cv1.bn

    # 5. 把剩余的权重直接复制过去(所有不是以 'cv1.' 开头的 key)，因为 'cv1.' 是我们刚才专门拆分处理的，其它的保持原样即可
    # Transfer remaining weights and buffers
    for key in state_dict:
        if not key.startswith('cv1.'):
            state_dict_v2[key] = state_dict[key]

    # 6. 复制所有非方法(non-method)的属性给新的 c2f_v2，例如一些标量、列表、或者其他需要保留的成员
    # Transfer all non-method attributes
    for attr_name in dir(c2f):
        attr_value = getattr(c2f, attr_name)
        if not callable(attr_value) and '_' not in attr_name:
            setattr(c2f_v2, attr_name, attr_value)

    # 7. 用修改完的 state_dict_v2 为 c2f_v2 加载权重，实际完成全部参数赋值
    c2f_v2.load_state_dict(state_dict_v2)


def replace_c2f_with_c2f_v2(module):
    # 1. 遍历当前 module 的所有子模块
    for name, child_module in module.named_children():
        # 2. 如果发现子模块是 C2f 类型，就要把它替换成 C2f_v2
        if isinstance(child_module, C2f):
            # 2.1 通过第一个 Bottleneck 推断它是否存在 shortcut
            # Replace C2f with C2f_v2 while preserving its parameters
            shortcut = infer_shortcut(child_module.m[0])

            # 2.2 根据旧 C2f 的输入输出通道数、Bottleneck 数量等，创建一个新的 C2f_v2
            c2f_v2 = C2f_v2(child_module.cv1.conv.in_channels,
                            child_module.cv2.conv.out_channels,
                            n=len(child_module.m),
                            shortcut=shortcut,
                            g=child_module.m[0].cv2.conv.groups,
                            e=child_module.c / child_module.cv2.conv.out_channels)

            # 2.3 调用 transfer_weights，把旧的 C2f 参数(权重、BN等)拷贝/转换到新的 c2f_v2
            transfer_weights(child_module, c2f_v2)

            # 2.4 用 setattr 把模块本身替换成新创建的 c2f_v2
            setattr(module, name, c2f_v2)
        else:
            # 3. 如果这个子模块不是 C2f，就递归地继续往下找
            replace_c2f_with_c2f_v2(child_module)


def replace_silu_with_relu(module):
    """
    Recursively replace all SiLU activation functions in the given module
    with ReLU activation functions.

    Args:
        module (nn.Module): The module to process.
    """
    # 1. 遍历当前 module 的所有子模块
    for name, child_module in module.named_children():
        # 2. 如果发现子模块是 SiLU 类型，就把它替换成 ReLU
        if isinstance(child_module, nn.SiLU):
            # 用 ReLU 替换 SiLU
            setattr(module, name, nn.ReLU(inplace=True))
        else:
            # 3. 如果子模块不是 SiLU，递归地继续处理子模块
            replace_silu_with_relu(child_module)


def replace_module_in_dict(model_dict, original_module, new_module):
    """
    Replace a specified module type in a model dictionary (e.g., 'C2f' -> 'C2f_v2').

    Args:
        model_dict (dict): Model configuration dictionary containing 'backbone' and 'head' keys.
        original_module (str): The name of the module to replace (e.g., 'C2f').
        new_module (str): The new module name to replace with (e.g., 'C2f_v2').

    Returns:
        dict: Updated model dictionary with the specified module replaced.
    """
    def replace_modules(layers, original, new):
        for layer in layers:
            if layer[2] == original:
                layer[2] = new
        return layers

    # Replace in backbone and head
    if 'backbone' in model_dict:
        model_dict['backbone'] = replace_modules(model_dict['backbone'], original_module, new_module)
    if 'head' in model_dict:
        model_dict['head'] = replace_modules(model_dict['head'], original_module, new_module)

    return model_dict


def save_model_v2(self: BaseTrainer):
    """
    Disabled half precision saving. originated from ultralytics/yolo/engine/trainer.py
    保存训练中模型状态的函数，与 YOLO 代码中的保存逻辑类似，使用 torch.save 保存在本地文件（.pt）里。
    将训练过程中的关键信息（模型参数、优化器状态、当前 epoch、最佳精度等）打包成一个 checkpoint 对象。
    """
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt


def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Disabled half precision saving. originated from ultralytics/yolo/utils/torch_utils.py
    精简训练好的 PyTorch 模型文件，移除只在继续训练时才需要的优化器、EMA、更新次数等信息，从而得到一个更小、更适合推理部署的 .pt 文件。
    """
    x = torch.load(f, map_location=torch.device('cpu'))
    args = {**DEFAULT_CFG_DICT, **x['train_args']}  # combine model args with default args, preferring model args
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def final_eval_v2(self: BaseTrainer):
    """
    originated from ultralytics/yolo/engine/trainer.py
    对保存的权重文件进行最终的整理（移除优化器等冗余信息）并在“best.pt”上跑一次验证，得到最终评估指标。
    """
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)  # strip optimizers
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')


def train_v2(self: YOLO, pruning=False, **kwargs):
    """
    Disabled loading new model when pruning flag is set.
    originated from ultralytics/engine/model.py/train(self, trainer=None, **kwargs: Any)
    用于训练 YOLO 模型，并在需要时支持剪枝模式（pruning=True），大体上基于 ultralytics/yolo 的 YOLO().train() 方法做了修改或增强。
    """

    self._check_is_pytorch_model()
    if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args    # overwrite kwargs

    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs['cfg']))

    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")

    # custom = {
    #     # NOTE: handle the case when 'cfg' includes 'data'.
    #     "data": overrides.get("data"),
    #     "model": self.overrides["model"],
    #     "task": self.task,
    # }
    # args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right

    if overrides.get("resume"):
        overrides["resume"] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    self.trainer = self.task_map[self.task]["trainer"](overrides=overrides, _callbacks=self.callbacks)

    if not pruning:
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
    else:
        # pruning mode
        self.trainer.pruning = True
        self.trainer.model = self.model
        # self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)

        # replace some functions to disable half precision saving
        self.trainer.__setattr__("save_model", save_model_v2.__get__(self.trainer))
        self.trainer.__setattr__("final_eval", final_eval_v2.__get__(self.trainer))
        # self.trainer.save_model = save_model_v2.__get__(self.trainer)
        # self.trainer.final_eval = final_eval_v2.__get__(self.trainer)
        # self.trainer.

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()

    # Update model and cfg after training
    if RANK in (-1, 0):
        # self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        # self.overrides = self.model.args
        # self.metrics = getattr(self.trainer.validator, 'metrics', None)

        ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
        self.model, self.ckpt = attempt_load_one_weight(ckpt)
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP

    return self.metrics


def update_model_yaml(model_dict):
    width_multiple = model_dict['width_multiple']
    for section in ['backbone', 'head']:
        for layer in model_dict[section]:
            # 检查每层的参数列表是否包含32倍数的值
            args = layer[-1]
            for i in range(len(args)):
                if isinstance(args[i], int) and args[i] % 32 == 0:
                    # 乘以 width_multiple 并四舍五入为最接近的 32 倍数
                    args[i] = round(args[i] * width_multiple / 32) * 32
    # 将 width_multiple 更新为 1.0
    model_dict['width_multiple'] = 1.0
    return model_dict


def update_pruned_model_yaml(original_yaml, pruned_model):
    """
    根据剪枝后的模型，更新原始模型的 model.model.yaml 字典。

    Args:
        original_yaml (dict): 原始模型的 model.model.yaml 字典。
        pruned_model (nn.Module): 剪枝后的模型。

    Returns:
        dict: 更新后的 model.model.yaml 字典。
    """
    # 遍历剪枝后的模型层
    pruned_layers = []
    for idx, layer in enumerate(pruned_model.model):
        pruned_layers.append(layer)

    # 遍历原始模型的 yaml，更新通道数
    updated_yaml = original_yaml.copy()
    backbone = updated_yaml["backbone"]
    head = updated_yaml["head"]

    # 更新 backbone 的通道数
    for i, layer_info in enumerate(backbone):
        module_type = layer_info[2]  # 模块类型 (如 'Conv', 'C2f_v2', 等)
        if module_type == "Conv":  # 更新 Conv 的通道数
            pruned_out_channels = pruned_layers[i].conv.out_channels
            backbone[i][3][0] = pruned_out_channels
        elif module_type.startswith("C2f"):  # 更新 C2f_v2 的通道数
            pruned_out_channels = pruned_layers[i].cv2.conv.out_channels
            backbone[i][3][0] = pruned_out_channels

    # 更新 head 的通道数
    for i, layer_info in enumerate(head):
        module_type = layer_info[2]
        if module_type == "Conv":  # 更新 Conv 的通道数
            pruned_out_channels = pruned_layers[len(backbone) + i].conv.out_channels
            head[i][3][0] = pruned_out_channels
        elif module_type.startswith("C2f"):  # 更新 C2f_v2 的通道数
            pruned_out_channels = pruned_layers[len(backbone) + i].cv2.conv.out_channels
            head[i][3][0] = pruned_out_channels

    # 返回更新后的 yaml
    updated_yaml["backbone"] = backbone
    updated_yaml["head"] = head
    return updated_yaml


def prune(prune_param: dict):
    # 加载yolo模型（以剪枝的也可以），并绑定自定义训练方法train_v2
    model = YOLO(prune_param["model"])
    model_name = os.path.basename(prune_param["model"])
    model.__setattr__("train_v2", train_v2.__get__(model))
    # model.train_v2 = train_v2.__get__(model, type(model))

    # 加载剪枝参数
    pruning_cfg = yaml_load(check_yaml(prune_param["cfg_path"]))

    # 加载模型，并把其中的C2f模块替换成C2f_v2
    model.to(pruning_cfg["device"])

    print(model)
    replace_c2f_with_c2f_v2(model)
    print(model)
    return
    # replace_silu_with_relu(model)
    model.model.yaml = replace_module_in_dict(model.model.yaml, "C2f", "C2f_v2")
    model.model.yaml = update_model_yaml(model.model.yaml)
    # model.model.yaml = replace_module_in_dict(model.model.yaml, "SiLU", "ReLU")

    print(model.model.yaml)
    model.save("best_c2f_v2_w1.pt")

    # replace_silu_with_relu(model)
    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace

    for name, param in model.model.named_parameters():
        param.requires_grad = True

    # 初始化Torch-Pruning剪枝的相关参数
    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(pruning_cfg["device"])
    macs_list, nparams_list, map_list, pruned_map_list = [], [], [], []
    model.model.cuda(pruning_cfg["device"])
    base_macs, base_nparams = tp.utils.count_ops_and_params(model.model, example_inputs)

    # do validation before pruning model
    pruning_cfg['name'] = f"baseline_val"
    pruning_cfg['batch'] = 4
    validation_model = deepcopy(model)
    metric = validation_model.val(**pruning_cfg)
    init_map = metric.box.map
    macs_list.append(base_macs)
    nparams_list.append(100)
    map_list.append(init_map)
    pruned_map_list.append(init_map)
    print(f"Before Pruning: MACs={base_macs / 1e9: .5f} G, #Params={base_nparams / 1e6: .5f} M, mAP={init_map: .5f}")

    # prune same ratio of filter based on initial size
    pruning_ratio = 1 - math.pow((1 - prune_param["target_prune_rate"]), 1 / prune_param["iterative_steps"])

    for i in range(prune_param["iterative_steps"]):
        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        ignored_layers = []
        # unwrapped_parameters = []

        # 忽略含 “model.0.conv”，不剪枝最浅层卷积
        for name, module in model.named_modules():
            if "model.0.conv" in name or "dfl" in name:
                ignored_layers.append(module)

        # 不剪枝检测头
        for m in model.model.modules():
            if isinstance(m, (Detect,)):
                ignored_layers.append(m)

        example_inputs = example_inputs.to(model.device)
        pruner = tp.pruner.GroupNormPruner(
            model.model,
            example_inputs,
            importance=tp.importance.GroupNormImportance(),  # L2 norm pruning,
            iterative_steps=1,
            pruning_ratio=pruning_ratio,
            ignored_layers=ignored_layers,
            round_to=32,
            # unwrapped_parameters=unwrapped_parameters
        )

        # Test regularization
        # output = model.model(example_inputs)
        # (output[0].sum() + sum([o.sum() for o in output[1]])).backward()
        # pruner.regularize(model.model)

        pruner.step()
        # pre fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_pre_val"
        pruning_cfg['batch'] = 1
        validation_model.model = deepcopy(model.model)
        metric = validation_model.val(**pruning_cfg)
        pruned_map = metric.box.map
        pruned_macs, pruned_nparams = tp.utils.count_ops_and_params(pruner.model, example_inputs.to(model.device))
        current_speed_up = float(macs_list[0]) / pruned_macs
        print(f"After pruning iter {i + 1}: MACs={pruned_macs / 1e9} G, #Params={pruned_nparams / 1e6} M, "
              f"mAP={pruned_map}, speed up={current_speed_up}")

        # 保存剪枝后的模型
        # print(model.ckpt['model'])
        # ckpt = {
        #     'epoch': -1,
        #     'best_fitness': None,
        #     'model': model.ckpt['model'],
        #     'ema': None,
        #     'updates': None,
        #     'optimizer': None,
        #     'train_args': model.ckpt['train_args'],
        #     'date': '2025-01-29',
        #     'version': '8.3.67'
        # }
        # torch.save(ckpt, os.path.join(pruning_cfg['name'], model_name))
        model.model.yaml = update_pruned_model_yaml(model.model.yaml, model.model)
        print(model.model.yaml)

        model.save(os.path.join(pruning_cfg['name'], model_name))
        return

        # fine-tuning
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        pruning_cfg['name'] = f"step_{i}_finetune"
        pruning_cfg['batch'] = 1  # restore batch size
        model.train_v2(pruning=True, **pruning_cfg)

        # post fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_post_val"
        pruning_cfg['batch'] = 1
        validation_model = YOLO(model.trainer.best)
        metric = validation_model.val(**pruning_cfg)
        current_map = metric.box.map
        print(f"After fine tuning mAP={current_map}")

        macs_list.append(pruned_macs)
        nparams_list.append(pruned_nparams / base_nparams * 100)
        pruned_map_list.append(pruned_map)
        map_list.append(current_map)

        # remove pruner after single iteration
        del pruner

        save_pruning_performance_graph(nparams_list, map_list, macs_list, pruned_map_list)

        if init_map - current_map > prune_param["max_map_drop"]:
            print("Pruning early stop")
            break

    model.export(format='onnx')


if __name__ == "__main__":
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    prune_param = {
        "model": r'../trains_det/kx_s_241118/train/weights/best.pt',
        # "model": r'G:\code\yolo_v8_v11\torch_pruning/best.pt',
        "cfg_path": 'setting.yaml',     # Pruning config file. Having same format with ultralytics/cfg/default.yaml
        "iterative_steps": 16,          # Total pruning iteration step
        "target_prune_rate": 0.2,       # Target pruning rate
        "max_map_drop": 1.0,            # Allowed maximum map drop after fine-tuning
    }

    prune(prune_param)
