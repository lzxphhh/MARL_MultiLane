"""
批量训练结果分析工具
分析训练结果，生成对比报告和可视化图表
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json
import yaml
from datetime import datetime
import argparse

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False


class ResultsAnalyzer:
    """结果分析器"""

    def __init__(self, results_dir: str):
        """初始化分析器

        Args:
            results_dir: 批量训练结果目录
        """
        self.results_dir = Path(results_dir)
        self.successful_results = None
        self.failed_results = None
        self.case_configs = {}

        self.load_results()
        self.load_case_configs()

    def load_results(self):
        """加载训练结果"""
        # 加载成功的训练结果
        success_file = self.results_dir / "training_results.csv"
        if success_file.exists():
            self.successful_results = pd.read_csv(success_file)
            print(f"加载成功结果: {len(self.successful_results)} 条")
        else:
            print("未找到成功结果文件")
            self.successful_results = pd.DataFrame()

        # 加载失败的训练结果
        failed_file = self.results_dir / "failed_trainings.csv"
        if failed_file.exists():
            self.failed_results = pd.read_csv(failed_file)
            print(f"加载失败结果: {len(self.failed_results)} 条")
        else:
            print("未找到失败结果文件")
            self.failed_results = pd.DataFrame()

    def load_case_configs(self):
        """加载案例配置"""
        configs_dir = self.results_dir / "configs"
        if not configs_dir.exists():
            print("未找到配置文件目录")
            return

        for config_file in configs_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.load(f, Loader=yaml.FullLoader)

                # 从文件名提取案例ID
                case_id = int(config_file.stem.split('_')[1])
                self.case_configs[case_id] = config

            except Exception as e:
                print(f"加载配置文件 {config_file} 失败: {e}")

    def generate_summary_report(self) -> dict:
        """生成汇总报告"""
        total_cases = len(self.successful_results) + len(self.failed_results)

        report = {
            'analysis_time': datetime.now().isoformat(),
            'results_directory': str(self.results_dir),
            'summary': {
                'total_cases': total_cases,
                'successful_cases': len(self.successful_results),
                'failed_cases': len(self.failed_results),
                'success_rate': len(self.successful_results) / total_cases * 100 if total_cases > 0 else 0
            },
            'performance_analysis': {},
            'parameter_analysis': {},
            'recommendations': []
        }

        if not self.successful_results.empty:
            # 性能分析
            report['performance_analysis'] = {
                'best_performing_case': self._find_best_case(),
                'metric_statistics': self._calculate_metric_statistics(),
                'training_time_analysis': self._analyze_training_time()
            }

            # 参数分析
            report['parameter_analysis'] = self._analyze_parameters()

            # 生成建议
            report['recommendations'] = self._generate_recommendations()

        return report

    def _find_best_case(self) -> dict:
        """找到表现最好的案例"""
        if self.successful_results.empty:
            return {}

        # 以最终奖励为主要指标
        best_idx = self.successful_results['final_reward'].idxmax()
        best_case = self.successful_results.loc[best_idx]

        return {
            'case_id': best_case['case_id'],
            'test_desc': best_case['test_desc'],
            'final_reward': best_case['final_reward'],
            'collision_rate': best_case['collision_rate'],
            'mean_speed': best_case['mean_speed'],
            'training_steps': best_case['training_steps']
        }

    def _calculate_metric_statistics(self) -> dict:
        """计算指标统计"""
        metrics = ['final_reward', 'collision_rate', 'mean_speed', 'training_steps']
        stats = {}

        for metric in metrics:
            if metric in self.successful_results.columns:
                data = self.successful_results[metric].dropna()
                stats[metric] = {
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'median': float(data.median())
                }

        return stats

    def _analyze_training_time(self) -> dict:
        """分析训练时间"""
        if 'start_time' not in self.successful_results.columns or 'end_time' not in self.successful_results.columns:
            return {}

        try:
            start_times = pd.to_datetime(self.successful_results['start_time'])
            end_times = pd.to_datetime(self.successful_results['end_time'])
            durations = (end_times - start_times).dt.total_seconds() / 3600  # 小时

            return {
                'mean_duration_hours': float(durations.mean()),
                'std_duration_hours': float(durations.std()),
                'min_duration_hours': float(durations.min()),
                'max_duration_hours': float(durations.max())
            }
        except Exception as e:
            print(f"分析训练时间失败: {e}")
            return {}

    def _analyze_parameters(self) -> dict:
        """分析参数影响"""
        if self.successful_results.empty or not self.case_configs:
            return {}

        analysis = {}

        # 分析奖励权重的影响
        reward_weights_analysis = self._analyze_reward_weights()
        if reward_weights_analysis:
            analysis['reward_weights'] = reward_weights_analysis

        # 分析其他关键参数
        other_params_analysis = self._analyze_other_parameters()
        if other_params_analysis:
            analysis['other_parameters'] = other_params_analysis

        return analysis

    def _analyze_reward_weights(self) -> dict:
        """分析奖励权重影响"""
        weights_data = []

        for _, row in self.successful_results.iterrows():
            case_id = row['case_id']
            if case_id in self.case_configs:
                config = self.case_configs[case_id]
                reward_weights = config.get('reward_weights', {})

                weights_data.append({
                    'case_id': case_id,
                    'test_desc': row['test_desc'],
                    'final_reward': row['final_reward'],
                    'collision_rate': row['collision_rate'],
                    'mean_speed': row['mean_speed'],
                    'safety_weight': reward_weights.get('safety', 0),
                    'efficiency_weight': reward_weights.get('efficiency', 0),
                    'stability_weight': reward_weights.get('stability', 0),
                    'comfort_weight': reward_weights.get('comfort', 0)
                })

        if not weights_data:
            return {}

        df = pd.DataFrame(weights_data)

        # 计算相关性
        weight_cols = ['safety_weight', 'efficiency_weight', 'stability_weight', 'comfort_weight']
        metric_cols = ['final_reward', 'collision_rate', 'mean_speed']

        correlations = {}
        for metric in metric_cols:
            correlations[metric] = {}
            for weight in weight_cols:
                corr = df[weight].corr(df[metric])
                correlations[metric][weight] = float(corr) if not pd.isna(corr) else 0

        return {
            'correlations': correlations,
            'data': weights_data
        }

    def _analyze_other_parameters(self) -> dict:
        """分析其他参数影响"""
        params_analysis = {}

        # 分析穿透率影响
        penetration_data = []
        for _, row in self.successful_results.iterrows():
            case_id = row['case_id']
            if case_id in self.case_configs:
                config = self.case_configs[case_id]
                penetration = config.get('penetration_CAV', 0.4)
                penetration_data.append({
                    'penetration_CAV': penetration,
                    'final_reward': row['final_reward'],
                    'collision_rate': row['collision_rate']
                })

        if penetration_data:
            df = pd.DataFrame(penetration_data)
            params_analysis['penetration_CAV'] = {
                'reward_correlation': float(df['penetration_CAV'].corr(df['final_reward'])),
                'collision_correlation': float(df['penetration_CAV'].corr(df['collision_rate']))
            }

        return params_analysis

    def _generate_recommendations(self) -> list:
        """生成优化建议"""
        recommendations = []

        if self.successful_results.empty:
            return ["无成功案例，建议检查训练配置和环境设置"]

        # 基于最佳表现给出建议
        best_case = self._find_best_case()
        if best_case:
            recommendations.append(
                f"最佳性能案例: {best_case['test_desc']} "
                f"(奖励: {best_case['final_reward']:.2f}, "
                f"碰撞率: {best_case['collision_rate']:.3f})"
            )

        # 基于参数分析给出建议
        param_analysis = self._analyze_reward_weights()
        if param_analysis and 'correlations' in param_analysis:
            reward_corr = param_analysis['correlations'].get('final_reward', {})

            # 找到与奖励相关性最高的权重
            best_weight = max(reward_corr.items(), key=lambda x: x[1])
            recommendations.append(
                f"建议重点关注 {best_weight[0]} (与最终奖励相关性: {best_weight[1]:.3f})"
            )

        # 基于失败案例给出建议
        if not self.failed_results.empty:
            common_failures = self.failed_results['failure_reason'].value_counts()
            most_common = common_failures.index[0]
            recommendations.append(f"主要失败原因: {most_common}，建议针对性优化")

        return recommendations

    def plot_performance_comparison(self, save_path: str = None):
        """绘制性能对比图"""
        if self.successful_results.empty:
            print("没有成功结果可以绘制")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('训练结果性能对比', fontsize=16)

        # 最终奖励对比
        axes[0, 0].bar(range(len(self.successful_results)),
                       self.successful_results['final_reward'])
        axes[0, 0].set_title('最终奖励对比')
        axes[0, 0].set_xlabel('案例编号')
        axes[0, 0].set_ylabel('最终奖励')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # 碰撞率对比
        axes[0, 1].bar(range(len(self.successful_results)),
                       self.successful_results['collision_rate'])
        axes[0, 1].set_title('碰撞率对比')
        axes[0, 1].set_xlabel('案例编号')
        axes[0, 1].set_ylabel('碰撞率')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # 平均速度对比
        axes[1, 0].bar(range(len(self.successful_results)),
                       self.successful_results['mean_speed'])
        axes[1, 0].set_title('平均速度对比')
        axes[1, 0].set_xlabel('案例编号')
        axes[1, 0].set_ylabel('平均速度')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # 训练步数对比
        axes[1, 1].bar(range(len(self.successful_results)),
                       self.successful_results['training_steps'])
        axes[1, 1].set_title('训练步数对比')
        axes[1, 1].set_xlabel('案例编号')
        axes[1, 1].set_ylabel('训练步数')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图已保存到: {save_path}")
        else:
            plt.show()

    def plot_parameter_analysis(self, save_path: str = None):
        """绘制参数分析图"""
        param_analysis = self._analyze_reward_weights()
        if not param_analysis or 'data' not in param_analysis:
            print("没有足够的参数数据进行分析")
            return

        df = pd.DataFrame(param_analysis['data'])

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('奖励权重参数分析', fontsize=16)

        weight_cols = ['safety_weight', 'efficiency_weight', 'stability_weight', 'comfort_weight']
        colors = ['red', 'blue', 'green', 'orange']

        for i, (weight_col, color) in enumerate(zip(weight_cols, colors)):
            row = i // 2
            col = i % 2

            # 散点图显示权重与最终奖励的关系
            axes[row, col].scatter(df[weight_col], df['final_reward'],
                                   alpha=0.7, color=color, s=50)
            axes[row, col].set_xlabel(weight_col.replace('_', ' ').title())
            axes[row, col].set_ylabel('最终奖励')
            axes[row, col].set_title(f'{weight_col.replace("_", " ").title()} vs 最终奖励')
            axes[row, col].grid(True, alpha=0.3)

            # 添加趋势线
            z = np.polyfit(df[weight_col], df['final_reward'], 1)
            p = np.poly1d(z)
            axes[row, col].plot(df[weight_col], p(df[weight_col]),
                                color=color, linestyle='--', alpha=0.8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"参数分析图已保存到: {save_path}")
        else:
            plt.show()

    def plot_correlation_heatmap(self, save_path: str = None):
        """绘制相关性热力图"""
        param_analysis = self._analyze_reward_weights()
        if not param_analysis or 'correlations' not in param_analysis:
            print("没有相关性数据可以绘制")
            return

        correlations = param_analysis['correlations']

        # 构建相关性矩阵
        weights = ['safety_weight', 'efficiency_weight', 'stability_weight', 'comfort_weight']
        metrics = ['final_reward', 'collision_rate', 'mean_speed']

        corr_matrix = []
        for metric in metrics:
            row = []
            for weight in weights:
                row.append(correlations[metric].get(weight, 0))
            corr_matrix.append(row)

        corr_df = pd.DataFrame(corr_matrix, index=metrics, columns=weights)

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr_df, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('奖励权重与性能指标相关性热力图', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"相关性热力图已保存到: {save_path}")
        else:
            plt.show()

    def export_analysis_report(self, output_dir: str):
        """导出完整分析报告"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 生成汇总报告
        summary_report = self.generate_summary_report()

        # 保存JSON报告
        with open(output_dir / "analysis_report.json", 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)

        # 生成图表
        self.plot_performance_comparison(str(output_dir / "performance_comparison.png"))
        self.plot_parameter_analysis(str(output_dir / "parameter_analysis.png"))
        self.plot_correlation_heatmap(str(output_dir / "correlation_heatmap.png"))

        # 生成详细的Excel报告
        self._export_excel_report(output_dir / "detailed_analysis.xlsx", summary_report)

        print(f"完整分析报告已导出到: {output_dir}")

    def _export_excel_report(self, file_path: str, summary_report: dict):
        """导出Excel详细报告"""
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # 成功结果
            if not self.successful_results.empty:
                self.successful_results.to_excel(writer, sheet_name='成功结果', index=False)

            # 失败结果
            if not self.failed_results.empty:
                self.failed_results.to_excel(writer, sheet_name='失败结果', index=False)

            # 汇总统计
            if 'performance_analysis' in summary_report:
                stats = summary_report['performance_analysis'].get('metric_statistics', {})
                if stats:
                    stats_df = pd.DataFrame(stats).T
                    stats_df.to_excel(writer, sheet_name='统计汇总')

            # 参数分析
            param_analysis = self._analyze_reward_weights()
            if param_analysis and 'data' in param_analysis:
                param_df = pd.DataFrame(param_analysis['data'])
                param_df.to_excel(writer, sheet_name='参数分析', index=False)

        print(f"Excel详细报告已保存到: {file_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分析批量训练结果")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="批量训练结果目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_output",
        help="分析结果输出目录"
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="显示图表而不是保存"
    )

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"错误: 结果目录 {args.results_dir} 不存在")
        return 1

    try:
        # 创建分析器
        analyzer = ResultsAnalyzer(args.results_dir)

        # 生成分析报告
        print("正在生成分析报告...")
        analyzer.export_analysis_report(args.output_dir)

        # 显示汇总信息
        summary = analyzer.generate_summary_report()
        print("\n" + "=" * 60)
        print("批量训练结果汇总")
        print("=" * 60)
        print(f"总案例数: {summary['summary']['total_cases']}")
        print(f"成功案例: {summary['summary']['successful_cases']}")
        print(f"失败案例: {summary['summary']['failed_cases']}")
        print(f"成功率: {summary['summary']['success_rate']:.1f}%")

        if summary['recommendations']:
            print("\n优化建议:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"{i}. {rec}")

        print(f"\n详细分析报告已保存到: {args.output_dir}")

    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())