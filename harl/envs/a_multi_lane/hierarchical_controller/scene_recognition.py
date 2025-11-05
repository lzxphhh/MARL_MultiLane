# scene_recognition.py
"""
场景识别模块实现
Scene Recognition Module Implementation
基于马氏距离进行场景分类，输出场景概率和权重配置
"""

import numpy as np
from typing import Dict, Tuple, Optional
from harl.envs.a_multi_lane.project_structure import (
    ScenarioType, WeightConfig, TrafficScenarioParams, SystemParameters,
    SCENARIO_PARAMS, DEFAULT_WEIGHTS, NORMALIZATION_COEFFICIENTS
)

class SceneRecognitionModule:
    """场景识别模块"""
    
    def __init__(self, system_params: SystemParameters):
        """
        初始化场景识别模块
        
        Args:
            system_params: 系统参数配置
        """
        self.params = system_params
        self.scenario_params = SCENARIO_PARAMS
        self.default_weights = DEFAULT_WEIGHTS
        
        # 历史权重用于平滑更新
        self.previous_weights: Optional[WeightConfig] = None
        
    def extract_features(self, traffic_states: Dict[str, float]) -> np.ndarray:
        """
        构建特征向量
        
        Args:
            traffic_states: 实时交通状态数据
                - 'density': 当前时刻多车道交通密度 [veh/km]
                - 'density_std': 单车道交通密度标准差 [veh/km]
                - 'velocity_mean': 当前时刻多车道平均速度 [m/s]
                - 'velocity_std': 单车道平均速度标准差 [m/s]
        
        Returns:
            4维特征向量 [κ(t), σκ(t), v̄(t), σv̄(t)]
        """
        feature_vector = np.array([
            traffic_states['density_mean'],
            traffic_states['density_std'],
            traffic_states['velocity_mean'],
            traffic_states['velocity_std']
        ])
        # 应用归一化系数
        # normalized_features = feature_vector * NORMALIZATION_COEFFICIENTS
        
        return feature_vector
    
    def compute_mahalanobis_distance(self, 
                                   feature_vector: np.ndarray, 
                                   scenario_params: TrafficScenarioParams) -> float:
        """
        计算马氏距离的平方
        
        Args:
            feature_vector: 当前特征向量
            scenario_params: 场景参数（均值和协方差矩阵）
            
        Returns:
            马氏距离的平方
        """
        diff = feature_vector - scenario_params.mean_vector
        try:
            inv_cov = np.linalg.inv(scenario_params.covariance_matrix)
            mahalanobis_distance_squared = np.dot(np.dot(diff.T, inv_cov), diff)
        except np.linalg.LinAlgError:
            # 如果协方差矩阵不可逆，使用伪逆
            pinv_cov = np.linalg.pinv(scenario_params.covariance_matrix)
            mahalanobis_distance_squared = np.dot(np.dot(diff.T, pinv_cov), diff)
            
        return mahalanobis_distance_squared
    
    def compute_scenario_probabilities(self, 
                                     feature_vector: np.ndarray) -> Dict[ScenarioType, float]:
        """
        计算各场景的后验概率
        
        Args:
            feature_vector: 当前特征向量
            
        Returns:
            各场景的概率分布
        """
        distances = {}
        
        # 计算所有场景的马氏距离
        for scenario_type, params in self.scenario_params.items():
            distances[scenario_type] = self.compute_mahalanobis_distance(
                feature_vector, params
            )
        
        # 使用Softmax计算概率分布
        beta = self.params.temperature_param
        exp_values = {}
        sum_exp = 0.0
        
        for scenario_type, distance in distances.items():
            exp_value = np.exp(-beta * distance)
            exp_values[scenario_type] = exp_value
            sum_exp += exp_value
        
        # 归一化得到概率
        probabilities = {}
        for scenario_type, exp_value in exp_values.items():
            probabilities[scenario_type] = exp_value / sum_exp
            
        return probabilities
    
    def determine_dominant_scenario(self, 
                                  probabilities: Dict[ScenarioType, float]) -> Tuple[ScenarioType, float]:
        """
        确定主导场景及其置信度
        
        Args:
            probabilities: 各场景概率分布
            
        Returns:
            (主导场景类型, 置信度)
        """
        dominant_scenario = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_scenario]
        
        return dominant_scenario, confidence
    
    def determine_weight_configuration(self, 
                                     probabilities: Dict[ScenarioType, float], 
                                     confidence: float) -> WeightConfig:
        """
        确定权重配置策略
        
        Args:
            probabilities: 各场景概率分布
            confidence: 主导场景置信度
            
        Returns:
            权重配置
        """
        if confidence >= self.params.confidence_threshold:
            # 高置信度：直接采用主导场景对应权重
            dominant_scenario = max(probabilities, key=probabilities.get)
            weight_config = self.default_weights[dominant_scenario]
        else:
            # 低置信度：采用概率加权混合策略
            # 初始化权重配置
            mixed_weights = {
                'efficiency': 0.0, 'equilibrium': 0.0,
                'ws': 0.0, 'we': 0.0, 'wc': 0.0, 'wt': 0.0, 'wcoop': 0.0
            }
            
            # 概率加权混合
            for scenario_type, prob in probabilities.items():
                scenario_weights = self.default_weights[scenario_type]
                mixed_weights['efficiency'] += prob * scenario_weights.efficiency
                mixed_weights['equilibrium'] += prob * scenario_weights.equilibrium
                mixed_weights['ws'] += prob * scenario_weights.ws
                mixed_weights['we'] += prob * scenario_weights.we
                mixed_weights['wc'] += prob * scenario_weights.wc
                mixed_weights['wt'] += prob * scenario_weights.wt
                mixed_weights['wcoop'] += prob * scenario_weights.wcoop
            
            weight_config = WeightConfig(**mixed_weights)
            
        return weight_config
    
    def smooth_weight_update(self, new_weights: WeightConfig) -> WeightConfig:
        """
        平滑更新权重以避免剧烈波动
        
        Args:
            new_weights: 新的权重配置
            
        Returns:
            平滑后的权重配置
        """
        if self.previous_weights is None:
            self.previous_weights = new_weights
            return new_weights
        
        alpha = self.params.smooth_factor
        
        # 平滑更新每个权重
        smoothed_weights = WeightConfig(
            efficiency=alpha * self.previous_weights.efficiency + (1 - alpha) * new_weights.efficiency,
            equilibrium=alpha * self.previous_weights.equilibrium + (1 - alpha) * new_weights.equilibrium,
            ws=alpha * self.previous_weights.ws + (1 - alpha) * new_weights.ws,
            we=alpha * self.previous_weights.we + (1 - alpha) * new_weights.we,
            wc=alpha * self.previous_weights.wc + (1 - alpha) * new_weights.wc,
            wt=alpha * self.previous_weights.wt + (1 - alpha) * new_weights.wt,
            wcoop=alpha * self.previous_weights.wcoop + (1 - alpha) * new_weights.wcoop
        )
        
        self.previous_weights = smoothed_weights
        return smoothed_weights
    
    def recognize_scenario(self, 
                          traffic_states: Dict[str, float]) -> Dict:
        """
        场景识别主函数
        
        Args:
            traffic_states: 实时交通状态数据
            
        Returns:
            场景识别结果字典，包含：
            - 'probabilities': 各场景概率分布
            - 'dominant_scenario': 主导场景类型
            - 'confidence': 置信度
            - 'weight_config': 权重配置
        """
        # 1. 构建特征向量
        feature_vector = self.extract_features(traffic_states)
        
        # 2. 计算场景概率分布
        probabilities = self.compute_scenario_probabilities(feature_vector)
        
        # 3. 确定主导场景和置信度
        dominant_scenario, confidence = self.determine_dominant_scenario(probabilities)
        
        # 4. 确定权重配置
        raw_weight_config = self.determine_weight_configuration(probabilities, confidence)
        
        # 5. 平滑权重更新
        smoothed_weight_config = self.smooth_weight_update(raw_weight_config)
        
        return {
            'probabilities': probabilities,
            'dominant_scenario': dominant_scenario,
            'confidence': confidence,
            'weight_config': smoothed_weight_config,
            'feature_vector': feature_vector
        }