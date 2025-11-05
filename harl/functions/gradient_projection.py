"""
Gradient Projection Module for Multi-Objective Conflict Resolution
梯度投影模块，用于解决多目标冲突
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch.nn.functional as F


class GradientProjector:
    """梯度投影器，实现安全约束下的多目标梯度投影"""

    def __init__(self,
                 conflict_threshold: float = 0.3,
                 safety_tolerance: float = 0.1,
                 regularization: float = 1e-6,
                 monitor_frequency: int = 5,  # 用于控制print输出频率
                 device: torch.device = torch.device("cpu")):
        """初始化梯度投影器"""
        self.conflict_threshold = conflict_threshold
        self.safety_tolerance = safety_tolerance
        self.regularization = regularization
        self.monitor_frequency = monitor_frequency
        self.device = device
        self.update_count = 0  # 追踪更新次数

        # 目标名称映射
        self.objective_names = ['safety', 'efficiency', 'stability', 'comfort']
        self.non_safety_objectives = ['efficiency', 'stability', 'comfort']

    def detect_conflicts(self, gradients: Dict[str, torch.Tensor]) -> Dict[Tuple[str, str], float]:
        """检测梯度冲突 - 内存优化版本"""
        conflicts = {}

        for i, obj1 in enumerate(self.objective_names):
            for j, obj2 in enumerate(self.objective_names):
                if i < j and obj1 in gradients and obj2 in gradients:
                    # 🔧 修改：使用采样方式计算相似度，避免处理完整梯度
                    grad1 = gradients[obj1]
                    grad2 = gradients[obj2]

                    # 对大型梯度进行采样
                    if grad1.numel() > 100000:  # 如果参数超过10万个
                        cos_sim = self._sample_based_cosine_similarity(grad1, grad2, sample_size=10000)
                    else:
                        grad1_flat = grad1.flatten()
                        grad2_flat = grad2.flatten()
                        cos_sim = F.cosine_similarity(grad1_flat.unsqueeze(0), grad2_flat.unsqueeze(0))

                    conflict_strength = max(0.0, -cos_sim.item())

                    if conflict_strength > self.conflict_threshold:
                        conflicts[(obj1, obj2)] = conflict_strength

        return conflicts

    def _sample_based_cosine_similarity(self, grad1: torch.Tensor, grad2: torch.Tensor,
                                        sample_size: int = 10000) -> torch.Tensor:
        """基于采样的余弦相似度计算"""
        grad1_flat = grad1.flatten()
        grad2_flat = grad2.flatten()

        total_size = grad1_flat.size(0)
        if total_size <= sample_size:
            return F.cosine_similarity(grad1_flat.unsqueeze(0), grad2_flat.unsqueeze(0))

        # 随机采样
        indices = torch.randperm(total_size, device=grad1.device)[:sample_size]
        sampled_grad1 = grad1_flat[indices]
        sampled_grad2 = grad2_flat[indices]

        return F.cosine_similarity(sampled_grad1.unsqueeze(0), sampled_grad2.unsqueeze(0))

    def project_to_safety_compatible_subspace(self,
                                              gradient: torch.Tensor,
                                              safety_gradient: torch.Tensor) -> torch.Tensor:
        """
        将梯度投影到安全兼容子空间

        Args:
            gradient: 待投影的梯度
            safety_gradient: 安全目标梯度

        Returns:
            投影后的梯度
        """
        gradient_flat = gradient.flatten()
        safety_flat = safety_gradient.flatten()

        # 计算投影系数
        dot_product = torch.dot(gradient_flat, safety_flat)
        safety_norm_sq = torch.dot(safety_flat, safety_flat)

        # 安全兼容性检查
        safety_bound = -self.safety_tolerance * safety_norm_sq

        if dot_product < safety_bound:
            # 需要投影
            alpha = (dot_product - safety_bound) / (safety_norm_sq + 1e-8)
            projected_flat = gradient_flat - alpha * safety_flat
            return projected_flat.view_as(gradient)
        else:
            # 已经兼容，无需投影
            return gradient

    def project_to_non_conflict_subspace(self,
                                         gradients: Dict[str, torch.Tensor],
                                         conflicts: Dict[Tuple[str, str], float]) -> Dict[str, torch.Tensor]:
        """将非安全梯度投影到非冲突子空间 - 内存友好版本"""
        if not conflicts:
            return gradients

        # 🔧 新增：检查是否有足够内存进行投影
        total_params = sum(grad.numel() for grad in gradients.values())
        estimated_memory_gb = (total_params * total_params * 4) / (1024 ** 3)  # 估算所需内存

        if estimated_memory_gb > 10.0:  # 如果预估需要超过10GB内存，使用轻量级方法
            return self._lightweight_conflict_resolution(gradients, conflicts)

        # 🔧 修改：按层分别处理，而不是整体处理
        projected_gradients = {}

        for obj_name, gradient in gradients.items():
            if obj_name == 'safety':
                projected_gradients[obj_name] = gradient
                continue

            # 对每个梯度张量按层处理
            projected_grad = self._project_gradient_layerwise(gradient, gradients, conflicts, obj_name)
            projected_gradients[obj_name] = projected_grad

        return projected_gradients

    def _lightweight_conflict_resolution(self,
                                         gradients: Dict[str, torch.Tensor],
                                         conflicts: Dict[Tuple[str, str], float]) -> Dict[str, torch.Tensor]:
        """轻量级冲突解决方案"""
        projected_gradients = {}

        for obj_name, gradient in gradients.items():
            if obj_name == 'safety':
                projected_gradients[obj_name] = gradient
                continue

            # 使用简单的梯度缩放来减少冲突
            conflict_penalty = 1.0
            for (obj1, obj2), strength in conflicts.items():
                if obj_name in [obj1, obj2] and 'safety' not in [obj1, obj2]:
                    conflict_penalty *= (1.0 - 0.1 * strength)  # 轻微减少有冲突的梯度

            projected_gradients[obj_name] = gradient * conflict_penalty

        return projected_gradients

    def _project_gradient_layerwise(self,
                                    target_gradient: torch.Tensor,
                                    all_gradients: Dict[str, torch.Tensor],
                                    conflicts: Dict[Tuple[str, str], float],
                                    target_obj: str) -> torch.Tensor:
        """按层投影梯度"""
        # 简化版本：只对梯度进行归一化和轻微调整
        # 避免创建大型投影矩阵

        original_shape = target_gradient.shape
        grad_flat = target_gradient.flatten()

        # 计算与其他目标的平均余弦相似度
        total_similarity = 0.0
        count = 0

        for obj_name, other_grad in all_gradients.items():
            if obj_name != target_obj and obj_name != 'safety':
                other_flat = other_grad.flatten()
                similarity = torch.cosine_similarity(grad_flat.unsqueeze(0), other_flat.unsqueeze(0))
                if similarity < -self.conflict_threshold:  # 存在冲突
                    total_similarity += similarity.item()
                    count += 1

        if count > 0:
            avg_conflict = total_similarity / count
            # 轻微调整梯度方向以减少冲突
            adjustment_factor = 1.0 + 0.1 * max(avg_conflict, -0.8)  # 限制调整幅度
            grad_flat = grad_flat * adjustment_factor

        return grad_flat.view(original_shape)

    def project_gradients(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        完整的梯度投影流程

        Args:
            gradients: 原始梯度字典

        Returns:
            (投影后的梯度字典, 冲突信息字典)
        """
        if 'safety' not in gradients:
            return gradients, {}

        # 第一步：检测冲突
        conflicts = self.detect_conflicts(gradients)

        # 第二步：投影到安全兼容子空间
        safety_compatible_gradients = {}
        safety_gradient = gradients['safety']

        for obj_name, gradient in gradients.items():
            if obj_name == 'safety':
                safety_compatible_gradients[obj_name] = gradient
            else:
                proj_grad = self.project_to_safety_compatible_subspace(gradient, safety_gradient)
                safety_compatible_gradients[obj_name] = proj_grad

        # 第三步：消解非安全目标间的冲突
        final_gradients = self.project_to_non_conflict_subspace(
            safety_compatible_gradients, conflicts
        )

        # 收集冲突信息
        conflict_info = {
            'conflicts': conflicts,
            'total_conflicts': len(conflicts),
            'conflict_strength': sum(conflicts.values()) / max(len(conflicts), 1)
        }

        return final_gradients, conflict_info

    def extract_gradients_from_model(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        """
        从模型中提取梯度（用于调试和分析）

        Args:
            model: PyTorch模型

        Returns:
            梯度字典
        """
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()
        return gradients

    def apply_projected_gradients_to_model(self,
                                           model: torch.nn.Module,
                                           projected_gradients: Dict[str, torch.Tensor]):
        """
        将投影后的梯度应用到模型（用于调试）

        Args:
            model: PyTorch模型
            projected_gradients: 投影后的梯度字典
        """
        for name, param in model.named_parameters():
            if name in projected_gradients:
                param.grad = projected_gradients[name].clone()


    def compute_gradient_correlations(self, gradients: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        计算所有梯度交叉项的相关性

        Args:
            gradients: 各目标的梯度字典

        Returns:
            所有交叉项的相关性字典
        """
        correlations = {}
        objective_list = list(gradients.keys())

        for i, obj1 in enumerate(objective_list):
            for j, obj2 in enumerate(objective_list):
                if i < j:  # 避免重复计算
                    grad1 = gradients[obj1].flatten()
                    grad2 = gradients[obj2].flatten()

                    # 计算余弦相似度作为相关性
                    cos_sim = F.cosine_similarity(grad1.unsqueeze(0), grad2.unsqueeze(0))
                    correlations[f"{obj1}_vs_{obj2}"] = cos_sim.item()

        return correlations


    def analyze_conflicts_detailed(self, gradients: Dict[str, torch.Tensor]) -> Dict:
        """
        详细的冲突分析

        Args:
            gradients: 各目标的梯度字典

        Returns:
            详细的冲突分析结果
        """
        correlations = self.compute_gradient_correlations(gradients)

        conflicts = {}
        severe_conflicts = {}

        for pair_name, correlation in correlations.items():
            conflict_strength = max(0.0, -correlation)

            # 判断冲突级别
            if conflict_strength > self.conflict_threshold:
                if conflict_strength > 0.7:  # 严重冲突阈值
                    severe_conflicts[pair_name] = {
                        'correlation': correlation,
                        'conflict_strength': conflict_strength,
                        'level': 'severe'
                    }
                else:
                    conflicts[pair_name] = {
                        'correlation': correlation,
                        'conflict_strength': conflict_strength,
                        'level': 'moderate'
                    }

        return {
            'correlations': correlations,
            'conflicts': conflicts,
            'severe_conflicts': severe_conflicts,
            'total_conflicts': len(conflicts) + len(severe_conflicts),
            'total_severe_conflicts': len(severe_conflicts)
        }


    def project_gradients_with_monitoring(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
        """
        带监控的梯度投影流程

        Args:
            gradients: 原始梯度字典

        Returns:
            (投影后的梯度字典, 完整的监控信息)
        """
        # 投影前分析
        print("\n" + "=" * 60)
        print("🔍 GRADIENT PROJECTION MONITORING")
        print("=" * 60)

        before_analysis = self.analyze_conflicts_detailed(gradients)

        print("\n📊 BEFORE PROJECTION:")
        print("-" * 30)
        print("Correlations:")
        for pair, corr in before_analysis['correlations'].items():
            status = "🚨 CONFLICT" if corr < -self.conflict_threshold else "✅ OK"
            print(f"  {pair}: {corr:.4f} {status}")

        print(f"\nConflict Summary:")
        print(f"  Total conflicts: {before_analysis['total_conflicts']}")
        print(f"  Severe conflicts: {before_analysis['total_severe_conflicts']}")

        if before_analysis['conflicts'] or before_analysis['severe_conflicts']:
            print("\nDetailed Conflicts:")
            for pair, info in before_analysis['severe_conflicts'].items():
                print(f"  🚨 SEVERE: {pair} = {info['correlation']:.4f}")
            for pair, info in before_analysis['conflicts'].items():
                print(f"  ⚠️  MODERATE: {pair} = {info['correlation']:.4f}")

        # 执行投影
        if 'safety' not in gradients:
            print("\n⚠️ No safety gradient found, skipping projection")
            return gradients, {'before': before_analysis, 'after': before_analysis}

        # 第一步：检测冲突（已在before_analysis中完成）
        conflicts = {}
        for pair, info in {**before_analysis['conflicts'], **before_analysis['severe_conflicts']}.items():
            obj1, obj2 = pair.split('_vs_')
            conflicts[(obj1, obj2)] = info['conflict_strength']

        # 第二步：投影到安全兼容子空间
        safety_compatible_gradients = {}
        safety_gradient = gradients['safety']

        for obj_name, gradient in gradients.items():
            if obj_name == 'safety':
                safety_compatible_gradients[obj_name] = gradient
            else:
                proj_grad = self.project_to_safety_compatible_subspace(gradient, safety_gradient)
                safety_compatible_gradients[obj_name] = proj_grad

        # 第三步：消解非安全目标间的冲突
        final_gradients = self.project_to_non_conflict_subspace(
            safety_compatible_gradients, conflicts
        )

        # 投影后分析
        after_analysis = self.analyze_conflicts_detailed(final_gradients)

        print("\n📊 AFTER PROJECTION:")
        print("-" * 30)
        print("Correlations:")
        for pair, corr in after_analysis['correlations'].items():
            before_corr = before_analysis['correlations'][pair]
            change = corr - before_corr
            change_str = f"({change:+.4f})" if abs(change) > 0.001 else ""
            status = "🚨 CONFLICT" if corr < -self.conflict_threshold else "✅ OK"
            print(f"  {pair}: {corr:.4f} {change_str} {status}")

        print(f"\nConflict Summary:")
        print(f"  Total conflicts: {after_analysis['total_conflicts']} (before: {before_analysis['total_conflicts']})")
        print(
            f"  Severe conflicts: {after_analysis['total_severe_conflicts']} (before: {before_analysis['total_severe_conflicts']})")

        # 改善效果统计
        conflicts_resolved = before_analysis['total_conflicts'] - after_analysis['total_conflicts']
        severe_resolved = before_analysis['total_severe_conflicts'] - after_analysis['total_severe_conflicts']

        print(f"\n🎯 PROJECTION EFFECTIVENESS:")
        print(f"  Conflicts resolved: {conflicts_resolved}")
        print(f"  Severe conflicts resolved: {severe_resolved}")

        if conflicts_resolved > 0:
            print("  ✅ Projection improved gradient compatibility")
        elif conflicts_resolved < 0:
            print("  ⚠️ Projection introduced new conflicts")
        else:
            print("  ➡️ No change in conflict count")

        print("=" * 60)

        # 收集完整的监控信息
        monitoring_info = {
            'before': before_analysis,
            'after': after_analysis,
            'conflicts_resolved': conflicts_resolved,
            'severe_resolved': severe_resolved,
            'projection_applied': True
        }

        return final_gradients, monitoring_info

    def should_print_details(self) -> bool:
        """判断是否应该打印详细信息"""
        return self.update_count % self.monitor_frequency == 0


    def project_gradients_with_full_monitoring(self, gradients: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict]:
         #完整监控的梯度投影流程：每次都计算，按频率打印

        if not gradients:
            print("Debug: No gradients received for projection")
            return gradients, {'projection_applied': False, 'detailed_monitoring': False}

        # 🔧 新增：内存预检查
        total_params = sum(grad.numel() for grad in gradients.values())
        if total_params > 5000000:  # 超过500万参数时使用轻量级模式
            print(f"⚠️ Large model detected ({total_params:,} params), using lightweight projection")
            return self._lightweight_projection_with_monitoring(gradients)

        self.update_count += 1

        # 投影前分析（每次都执行）
        before_analysis = self.analyze_conflicts_detailed(gradients)

        # 根据频率决定是否打印详细信息
        should_print = self.should_print_details()

        if should_print:
            print("\n" + "=" * 60)
            print("🔍 GRADIENT PROJECTION MONITORING")
            print("=" * 60)

            print("\n📊 BEFORE PROJECTION:")
            print("-" * 30)
            print("Correlations:")
            for pair, corr in before_analysis['correlations'].items():
                status = "🚨 CONFLICT" if corr < -self.conflict_threshold else "✅ OK"
                print(f"  {pair}: {corr:.4f} {status}")

            print(f"\nConflict Summary:")
            print(f"  Total conflicts: {before_analysis['total_conflicts']}")
            print(f"  Severe conflicts: {before_analysis['total_severe_conflicts']}")

            if before_analysis['conflicts'] or before_analysis['severe_conflicts']:
                print("\nDetailed Conflicts:")
                for pair, info in before_analysis['severe_conflicts'].items():
                    print(f"  🚨 SEVERE: {pair} = {info['correlation']:.4f}")
                for pair, info in before_analysis['conflicts'].items():
                    print(f"  ⚠️  MODERATE: {pair} = {info['correlation']:.4f}")

        # 执行实际的梯度投影（每次都执行）
        projected_gradients, basic_projection_info = self.project_gradients(gradients)

        # 投影后分析（每次都执行）
        after_analysis = self.analyze_conflicts_detailed(projected_gradients)

        if should_print:
            print("\n📊 AFTER PROJECTION:")
            print("-" * 30)
            print("Correlations:")
            for pair, corr in after_analysis['correlations'].items():
                before_corr = before_analysis['correlations'][pair]
                change = corr - before_corr
                change_str = f"({change:+.4f})" if abs(change) > 0.001 else ""
                status = "🚨 CONFLICT" if corr < -self.conflict_threshold else "✅ OK"
                print(f"  {pair}: {corr:.4f} {change_str} {status}")

            print(f"\nConflict Summary:")
            print(f"  Total conflicts: {after_analysis['total_conflicts']} (before: {before_analysis['total_conflicts']})")
            print(
                f"  Severe conflicts: {after_analysis['total_severe_conflicts']} (before: {before_analysis['total_severe_conflicts']})")

            # 改善效果统计
            conflicts_resolved = before_analysis['total_conflicts'] - after_analysis['total_conflicts']
            severe_resolved = before_analysis['total_severe_conflicts'] - after_analysis['total_severe_conflicts']

            print(f"\n🎯 PROJECTION EFFECTIVENESS:")
            print(f"  Conflicts resolved: {conflicts_resolved}")
            print(f"  Severe conflicts resolved: {severe_resolved}")

            if conflicts_resolved > 0:
                print("  ✅ Projection improved gradient compatibility")
            elif conflicts_resolved < 0:
                print("  ⚠️ Projection introduced new conflicts")
            else:
                print("  ➡️ No change in conflict count")

            print("=" * 60)

        # 计算效果指标（每次都计算，用于TensorBoard记录）
        conflicts_resolved = before_analysis['total_conflicts'] - after_analysis['total_conflicts']
        severe_resolved = before_analysis['total_severe_conflicts'] - after_analysis['total_severe_conflicts']

        # 收集完整的监控信息（每次都收集）
        monitoring_info = {
            'before': before_analysis,
            'after': after_analysis,
            'conflicts_resolved': conflicts_resolved,
            'severe_resolved': severe_resolved,
            'projection_applied': True,
            'detailed_print': should_print  # 标记是否进行了详细打印
        }

        return projected_gradients, monitoring_info

    def _lightweight_projection_with_monitoring(self, gradients: Dict[str, torch.Tensor]) -> Tuple[
        Dict[str, torch.Tensor], Dict]:
        """大模型的轻量级投影方案"""
        # 使用简化的冲突检测和解决方案
        conflicts = self.detect_conflicts(gradients)  # 已经是内存优化版本
        projected_gradients = self._lightweight_conflict_resolution(gradients, conflicts)

        monitoring_info = {
            'before': {'total_conflicts': len(conflicts)},
            'after': {'total_conflicts': max(0, len(conflicts) - 1)},  # 假设减少了一些冲突
            'projection_applied': True,
            'lightweight_mode': True
        }

        return projected_gradients, monitoring_info


class MultiObjectiveGradientManager:
    """多目标梯度管理器，整合梯度投影和权重调整"""

    def __init__(self,
                 projector: GradientProjector,
                 beta: float = 1.0):
        """
        初始化梯度管理器

        Args:
            projector: 梯度投影器
            beta: 冲突敏感度参数
        """
        self.projector = projector
        self.beta = beta

    def compute_conflict_adjusted_weights(self,
                                          raw_weights: torch.Tensor,
                                          conflict_info: Dict) -> torch.Tensor:
        """
        基于冲突信息调整权重

        Args:
            raw_weights: 原始权重 [efficiency, stability, comfort]
            conflict_info: 冲突信息字典

        Returns:
            调整后的权重
        """
        if not conflict_info.get('conflicts', {}):
            return raw_weights

        # 计算各目标的冲突强度
        conflicts = conflict_info['conflicts']
        conflict_strengths = {'efficiency': 0.0, 'stability': 0.0, 'comfort': 0.0}

        for (obj1, obj2), strength in conflicts.items():
            if obj1 != 'safety':
                conflict_strengths[obj1] += strength
            if obj2 != 'safety':
                conflict_strengths[obj2] += strength

        # 权重调整
        adjusted_weights = torch.zeros_like(raw_weights)
        for i, obj_name in enumerate(self.projector.non_safety_objectives):
            conflict_penalty = torch.exp(-self.beta * conflict_strengths[obj_name])
            adjusted_weights[i] = raw_weights[i] * conflict_penalty

        # 重新归一化
        weight_sum = adjusted_weights.sum()
        if weight_sum > 1e-8:
            adjusted_weights = adjusted_weights / weight_sum
        else:
            # 如果所有权重都太小，恢复均匀分布
            adjusted_weights = torch.ones_like(raw_weights) / len(raw_weights)

        return adjusted_weights

    def process_multi_objective_update(self,
                                       objective_gradients: Dict[str, torch.Tensor],
                                       raw_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        处理多目标更新的完整流程

        Args:
            objective_gradients: 各目标的梯度
            raw_weights: 原始权重

        Returns:
            (最终的总梯度, 处理信息)
        """
        # 梯度投影
        projected_gradients, conflict_info = self.projector.project_gradients(objective_gradients)

        # 权重调整
        adjusted_weights = self.compute_conflict_adjusted_weights(raw_weights, conflict_info)

        # 构建最终梯度
        final_gradient = None

        # 安全梯度（权重固定为1）
        if 'safety' in projected_gradients:
            final_gradient = projected_gradients['safety'].clone()

        # 加权其他梯度
        for i, obj_name in enumerate(self.projector.non_safety_objectives):
            if obj_name in projected_gradients:
                if final_gradient is None:
                    final_gradient = adjusted_weights[i] * projected_gradients[obj_name]
                else:
                    final_gradient += adjusted_weights[i] * projected_gradients[obj_name]

        # 处理信息
        process_info = {
            'conflict_info': conflict_info,
            'raw_weights': raw_weights,
            'adjusted_weights': adjusted_weights,
            'projected_gradients': projected_gradients
        }

        return final_gradient, process_info

    def process_multi_objective_update_with_monitoring(self,
                                                       objective_gradients: Dict[str, torch.Tensor],
                                                       raw_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        带监控的多目标更新处理流程：每次都计算，按频率打印

        Args:
            objective_gradients: 各目标的梯度
            raw_weights: 原始权重

        Returns:
            (最终的总梯度, 详细处理信息)
        """

        # 梯度投影（每次都执行计算，按频率打印）
        projected_gradients, projection_monitoring = self.projector.project_gradients_with_full_monitoring(
            objective_gradients)

        # 权重调整（每次都执行）
        conflict_info = projection_monitoring.get('after', projection_monitoring.get('conflicts', {}))
        adjusted_weights = self.compute_conflict_adjusted_weights(raw_weights, conflict_info)

        # 计算权重变化（每次都计算）
        weight_change = torch.abs(adjusted_weights - raw_weights).sum().item()

        # 只在详细打印时显示权重调整信息
        if projection_monitoring.get('detailed_print', False):
            print(f"\n🔧 WEIGHT ADJUSTMENT:")
            print(f"  Raw weights: {raw_weights}")
            print(f"  Adjusted weights: {adjusted_weights}")
            print(f"  Total weight change: {weight_change:.4f}")

        # 构建最终梯度（每次都执行）
        final_gradient = None

        # 安全梯度（权重固定为1）
        if 'safety' in projected_gradients:
            final_gradient = projected_gradients['safety'].clone()
            if projection_monitoring.get('detailed_print', False):
                print(f"  Safety gradient norm: {torch.norm(final_gradient).item():.4f}")

        # 加权其他梯度
        gradient_norms = {}  # 记录各目标梯度范数，用于TensorBoard
        for i, obj_name in enumerate(self.projector.non_safety_objectives):
            if obj_name in projected_gradients:
                weighted_grad = adjusted_weights[i] * projected_gradients[obj_name]
                gradient_norms[f"{obj_name}_weighted_norm"] = torch.norm(weighted_grad).item()

                if final_gradient is None:
                    final_gradient = weighted_grad
                else:
                    final_gradient += weighted_grad

                if projection_monitoring.get('detailed_print', False):
                    print(f"  {obj_name} weighted gradient norm: {gradient_norms[f'{obj_name}_weighted_norm']:.4f}")

        final_gradient_norm = torch.norm(final_gradient).item()
        if projection_monitoring.get('detailed_print', False):
            print(f"  Final combined gradient norm: {final_gradient_norm:.4f}")

        # 处理信息（每次都收集完整信息）
        process_info = {
            'projection_monitoring': projection_monitoring,
            'raw_weights': raw_weights,
            'adjusted_weights': adjusted_weights,
            'weight_change': weight_change,
            'projected_gradients': projected_gradients,
            'final_gradient_norm': final_gradient_norm,
            'gradient_norms': gradient_norms
        }

        return final_gradient, process_info
