"""梯度和参数监控工具 - 修复维度不匹配问题"""
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional

class GradientMonitor:
    """监控网络梯度和参数变化的工具类"""

    def __init__(self, writer: SummaryWriter, monitor_frequency: int = 10):
        self.writer = writer
        self.monitor_frequency = monitor_frequency
        self.update_count = 0
        self.prev_params = {}
        self.dimension_mismatch_params = set()

    def should_monitor(self) -> bool:
        """保留"""
        return self.update_count % self.monitor_frequency == 0

    def _is_healthy_tensor(self, tensor: torch.Tensor) -> bool:
        """保留并简化"""
        return not (torch.isnan(tensor).any() or torch.isinf(tensor).any())

    def _safe_compute(self, func, *args, default=0.0):
        """保留并简化"""
        try:
            result = func(*args)
            if isinstance(result, torch.Tensor):
                return result.item() if result.numel() == 1 else result
            return result if not (np.isnan(result) or np.isinf(result)) else default
        except Exception:
            return default

    def monitor_gradients(self, model: torch.nn.Module, model_name: str, step: int):
        """保留并合并改进功能"""
        if not self.should_monitor():
            return

        grad_norms = []
        healthy_grads = 0
        total_grads = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                total_grads += 1
                grad = param.grad.detach()

                if not self._is_healthy_tensor(grad):
                    continue

                healthy_grads += 1
                grad_norm = self._safe_compute(torch.norm, grad)

                if grad_norm != 0.0:
                    grad_norms.append(grad_norm)

                layer_name = name.replace('.', '/')
                self.writer.add_scalar(f'{model_name}/gradients/{layer_name}/norm', grad_norm, step)
                self.writer.add_scalar(f'{model_name}/gradients/{layer_name}/mean',
                                     self._safe_compute(torch.mean, grad), step)

                # 添加直方图记录（合并改进功能）
                if step % (self.monitor_frequency * 2) == 0:
                    self.writer.add_histogram(f'{model_name}/gradients/{layer_name}/distribution', grad, step)

        # 全局统计
        if grad_norms:
            self.writer.add_scalar(f'{model_name}/gradients/global/mean_norm', np.mean(grad_norms), step)
            self.writer.add_scalar(f'{model_name}/gradients/global/max_norm', np.max(grad_norms), step)
            self.writer.add_scalar(f'{model_name}/gradients/global/std_norm', np.std(grad_norms), step)

        # 健康状况
        health_ratio = healthy_grads / max(total_grads, 1)
        self.writer.add_scalar(f'{model_name}/gradients/health_ratio', health_ratio, step)

    def monitor_parameters(self, model: torch.nn.Module, model_name: str, step: int):
        """保留并简化"""
        if not self.should_monitor():
            return

        current_params = {}
        param_norms = []
        param_changes = []

        for name, param in model.named_parameters():
            param_data = param.detach().clone()

            if not self._is_healthy_tensor(param_data):
                continue

            param_norm = self._safe_compute(torch.norm, param_data)
            if param_norm != 0.0:
                param_norms.append(param_norm)

            layer_name = name.replace('.', '/')
            self.writer.add_scalar(f'{model_name}/parameters/{layer_name}/norm', param_norm, step)

            # 参数变化计算（简化版本）
            if name in self.prev_params:
                prev_param = self.prev_params[name]
                if param_data.shape == prev_param.shape and self._is_healthy_tensor(prev_param):
                    param_change = self._safe_compute(
                        lambda x, y: torch.norm(x - y), param_data, prev_param
                    )
                    if param_change != 0.0:
                        param_changes.append(param_change)
                        self.writer.add_scalar(f'{model_name}/parameters/{layer_name}/change', param_change, step)

            current_params[name] = param_data

        # 全局统计
        if param_norms:
            self.writer.add_scalar(f'{model_name}/parameters/global/mean_norm', np.mean(param_norms), step)
        if param_changes:
            self.writer.add_scalar(f'{model_name}/parameters/global/mean_change', np.mean(param_changes), step)

        self.prev_params.update(current_params)

    def monitor_learning_rate(self, optimizer: torch.optim.Optimizer, model_name: str, step: int):
        """保留"""
        if not self.should_monitor():
            return

        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            if not (np.isnan(lr) or np.isinf(lr)):
                self.writer.add_scalar(f'{model_name}/learning_rate/group_{i}', lr, step)

    def increment_update_count(self):
        """保留"""
        self.update_count += 1

    def get_dimension_mismatch_report(self):
        """保留"""
        return {
            'mismatch_params': list(self.dimension_mismatch_params),
            'mismatch_count': len(self.dimension_mismatch_params)
        }


class NetworkAnalyzer:
    """网络结构分析工具"""

    @staticmethod
    def analyze_network(model: torch.nn.Module, writer: SummaryWriter, model_name: str):
        """
        分析网络结构并记录
        Args:
            model: 要分析的模型
            writer: tensorboard writer
            model_name: 模型名称
        """
        total_params = 0
        trainable_params = 0
        param_shapes = {}

        for name, param in model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            param_shapes[name] = list(param.shape)

            if param.requires_grad:
                trainable_params += param_count

        # 记录参数数量
        writer.add_text(f'{model_name}/model_info/total_parameters', str(total_params))
        writer.add_text(f'{model_name}/model_info/trainable_parameters', str(trainable_params))
        writer.add_text(f'{model_name}/model_info/non_trainable_parameters', str(total_params - trainable_params))

        # 记录参数形状信息
        shapes_text = "\n".join([f"{name}: {shape}" for name, shape in param_shapes.items()])
        writer.add_text(f'{model_name}/model_info/parameter_shapes', shapes_text)

        # 记录模型结构
        model_structure = str(model)
        writer.add_text(f'{model_name}/model_info/structure', model_structure)

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'parameter_shapes': param_shapes,
            'structure': model_structure
        }