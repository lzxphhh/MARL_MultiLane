"""
训练监控工具
实时监控训练日志，解析训练指标，判断训练状态
"""

import os
import re
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd


class TrainingMonitor:
    """训练监控器"""

    def __init__(self, log_file: str, monitor_config: Dict = None):
        """初始化监控器

        Args:
            log_file: 日志文件路径
            monitor_config: 监控配置
        """
        self.log_file = Path(log_file)
        self.monitor_config = monitor_config or {}

        # 监控状态
        self.is_monitoring = False
        self.metrics_history = []
        self.last_position = 0
        self.last_update_time = datetime.now()

        # 训练状态
        self.training_started = False
        self.training_completed = False
        self.training_failed = False
        self.failure_reason = ""

        # 当前指标
        self.current_metrics = {}
        self.episode_count = 0
        self.training_steps = 0

        # 监控线程
        self.monitor_thread = None
        self.stop_monitoring = False

    def start_monitoring(self):
        """开始监控"""
        if self.is_monitoring:
            return

        print(f"开始监控日志文件: {self.log_file}")
        self.is_monitoring = True
        self.stop_monitoring = False

        # 启动监控线程
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring_thread(self):
        """停止监控"""
        self.stop_monitoring = True
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """监控主循环"""
        check_interval = self.monitor_config.get('check_interval', 5)

        while not self.stop_monitoring:
            try:
                if self.log_file.exists():
                    self._check_log_updates()
                    self._check_training_status()
                else:
                    print(f"等待日志文件创建: {self.log_file}")

                time.sleep(check_interval)

            except Exception as e:
                print(f"监控过程中出现异常: {e}")
                time.sleep(check_interval)

    def _check_log_updates(self):
        """检查日志文件更新"""
        try:
            current_size = self.log_file.stat().st_size
            if current_size > self.last_position:
                # 读取新内容
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self.last_position)
                    new_lines = f.readlines()
                    self.last_position = f.tell()

                if new_lines:
                    self._parse_new_lines(new_lines)
                    self.last_update_time = datetime.now()

        except Exception as e:
            print(f"检查日志更新失败: {e}")

    def _parse_new_lines(self, lines: List[str]):
        """解析新的日志行"""
        for line in lines:
            self._parse_training_progress(line)
            self._parse_metrics(line)
            self._check_error_patterns(line)

    def _parse_training_progress(self, line: str):
        """解析训练进度"""
        # 检查训练开始
        if not self.training_started:
            if any(keyword in line.lower() for keyword in ['start training', '开始训练', 'warming up']):
                self.training_started = True
                print("检测到训练开始")

        # 解析episode数
        episode_match = re.search(r'episode[:\s]+(\d+)', line, re.IGNORECASE)
        if episode_match:
            self.episode_count = int(episode_match.group(1))

        # 解析训练步数
        step_matches = re.findall(r'step[s]?[:\s]+(\d+)', line, re.IGNORECASE)
        if step_matches:
            self.training_steps = max(int(step) for step in step_matches)

        # 检查训练完成
        if any(keyword in line.lower() for keyword in ['training completed', '训练完成', 'training finished']):
            self.training_completed = True
            print("检测到训练完成")

    def _parse_metrics(self, line: str):
        """解析训练指标"""
        # 奖励相关
        reward_patterns = {
            'total_reward': r'total reward[:\s]+([-]?\d+\.?\d*)',
            'episode_reward': r'episode reward[:\s]+([-]?\d+\.?\d*)',
            'mean_reward': r'mean reward[:\s]+([-]?\d+\.?\d*)',
        }

        # 性能指标
        performance_patterns = {
            'collision_rate': r'collision[:\s]+([-]?\d+\.?\d*)',
            'mean_speed': r'mean speed[:\s]+([-]?\d+\.?\d*)',
            'mean_acceleration': r'mean acceleration[:\s]+([-]?\d+\.?\d*)',
        }

        # 安全指标
        safety_patterns = {
            'safety_SI': r'safety_SI[:\s]+([-]?\d+\.?\d*)',
            'TTC': r'TTC[:\s]+([-]?\d+\.?\d*)',
            'ACT': r'ACT[:\s]+([-]?\d+\.?\d*)',
        }

        # 效率指标
        efficiency_patterns = {
            'efficiency_ASR': r'efficiency_ASR[:\s]+([-]?\d+\.?\d*)',
            'efficiency_TFR': r'efficiency_TFR[:\s]+([-]?\d+\.?\d*)',
        }

        all_patterns = {**reward_patterns, **performance_patterns, **safety_patterns, **efficiency_patterns}

        for metric, pattern in all_patterns.items():
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    self.current_metrics[metric] = value

                    # 添加到历史记录
                    metric_record = {
                        'timestamp': datetime.now().isoformat(),
                        'episode': self.episode_count,
                        'steps': self.training_steps,
                        'metric': metric,
                        'value': value
                    }
                    self.metrics_history.append(metric_record)

                except ValueError:
                    continue

    def _check_error_patterns(self, line: str):
        """检查错误模式"""
        error_keywords = [
            'error', 'exception', 'failed', 'traceback',
            'cuda out of memory', 'segmentation fault',
            '错误', '异常', '失败'
        ]

        for keyword in error_keywords:
            if keyword in line.lower():
                if not self.training_failed:
                    self.training_failed = True
                    self.failure_reason = f"检测到错误: {line.strip()}"
                    print(f"检测到训练错误: {line.strip()}")
                break

    def _check_training_status(self):
        """检查训练状态"""
        now = datetime.now()
        time_since_update = (now - self.last_update_time).total_seconds()

        # 检查是否长时间无更新
        max_idle_time = self.monitor_config.get('max_idle_seconds', 300)  # 5分钟
        if self.training_started and not self.training_completed and time_since_update > max_idle_time:
            if not self.training_failed:
                self.training_failed = True
                self.failure_reason = f"训练长时间无更新 ({time_since_update:.0f}秒)"
                print(f"检测到训练可能卡死: {time_since_update:.0f}秒无更新")

    def get_current_status(self) -> Dict:
        """获取当前监控状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'training_started': self.training_started,
            'training_completed': self.training_completed,
            'training_failed': self.training_failed,
            'failure_reason': self.failure_reason,
            'episode_count': self.episode_count,
            'training_steps': self.training_steps,
            'current_metrics': self.current_metrics.copy(),
            'last_update_time': self.last_update_time.isoformat(),
        }

    def get_metrics_summary(self) -> Dict:
        """获取指标摘要"""
        if not self.metrics_history:
            return {}

        df = pd.DataFrame(self.metrics_history)

        summary = {}
        for metric in df['metric'].unique():
            metric_data = df[df['metric'] == metric]['value']
            summary[metric] = {
                'latest': metric_data.iloc[-1] if len(metric_data) > 0 else None,
                'mean': metric_data.mean(),
                'std': metric_data.std(),
                'min': metric_data.min(),
                'max': metric_data.max(),
                'count': len(metric_data)
            }

        return summary

    def export_metrics(self, output_dir: str):
        """导出指标数据"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # 导出原始数据
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            csv_path = output_dir / "training_metrics.csv"
            df.to_csv(csv_path, index=False)
            print(f"指标数据已导出到: {csv_path}")

        # 导出摘要
        summary = self.get_metrics_summary()
        if summary:
            summary_path = output_dir / "metrics_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"指标摘要已导出到: {summary_path}")

        # 生成可视化图表
        self.plot_metrics(output_dir)

    def plot_metrics(self, output_dir: str):
        """生成指标可视化图表"""
        if not self.metrics_history:
            return

        output_dir = Path(output_dir)
        df = pd.DataFrame(self.metrics_history)

        # 按指标类型分组绘图
        metric_groups = {
            'rewards': ['total_reward', 'episode_reward', 'mean_reward'],
            'performance': ['collision_rate', 'mean_speed', 'mean_acceleration'],
            'safety': ['safety_SI', 'TTC', 'ACT'],
            'efficiency': ['efficiency_ASR', 'efficiency_TFR']
        }

        for group_name, metrics in metric_groups.items():
            available_metrics = [m for m in metrics if m in df['metric'].values]
            if not available_metrics:
                continue

            plt.figure(figsize=(12, 8))

            for i, metric in enumerate(available_metrics):
                metric_data = df[df['metric'] == metric]
                if len(metric_data) > 0:
                    plt.subplot(len(available_metrics), 1, i + 1)
                    plt.plot(metric_data['steps'], metric_data['value'], 'o-', alpha=0.7)
                    plt.title(f'{metric} over Training Steps')
                    plt.xlabel('Training Steps')
                    plt.ylabel(metric)
                    plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plot_path = output_dir / f"{group_name}_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{group_name}指标图表已保存到: {plot_path}")

    def check_success_criteria(self, criteria: Dict) -> Tuple[bool, Dict]:
        """检查是否满足成功标准"""
        if not self.current_metrics:
            return False, {"reason": "没有可用的指标数据"}

        results = {}
        overall_success = True

        for metric, threshold in criteria.items():
            if metric not in self.current_metrics:
                results[metric] = {"status": "missing", "message": "指标数据不可用"}
                overall_success = False
                continue

            current_value = self.current_metrics[metric]

            if isinstance(threshold, dict):
                # 范围检查
                success = True
                if 'min' in threshold:
                    success &= current_value >= threshold['min']
                if 'max' in threshold:
                    success &= current_value <= threshold['max']

                results[metric] = {
                    "status": "pass" if success else "fail",
                    "current_value": current_value,
                    "threshold": threshold,
                    "message": f"当前值: {current_value}, 要求: {threshold}"
                }
            else:
                # 简单阈值检查
                success = current_value >= threshold
                results[metric] = {
                    "status": "pass" if success else "fail",
                    "current_value": current_value,
                    "threshold": threshold,
                    "message": f"当前值: {current_value}, 最小要求: {threshold}"
                }

            overall_success &= success

        return overall_success, results


class BatchMonitorManager:
    """批量训练监控管理器"""

    def __init__(self):
        self.monitors = {}  # case_id -> TrainingMonitor

    def add_monitor(self, case_id: str, log_file: str, monitor_config: Dict = None):
        """添加监控器"""
        monitor = TrainingMonitor(log_file, monitor_config)
        self.monitors[case_id] = monitor
        monitor.start_monitoring()
        print(f"为案例 {case_id} 添加监控器")

    def remove_monitor(self, case_id: str):
        """移除监控器"""
        if case_id in self.monitors:
            self.monitors[case_id].stop_monitoring_thread()
            del self.monitors[case_id]
            print(f"移除案例 {case_id} 的监控器")

    def get_all_status(self) -> Dict:
        """获取所有监控状态"""
        status = {}
        for case_id, monitor in self.monitors.items():
            status[case_id] = monitor.get_current_status()
        return status

    def get_summary_report(self) -> Dict:
        """获取汇总报告"""
        report = {
            'total_cases': len(self.monitors),
            'training_started': 0,
            'training_completed': 0,
            'training_failed': 0,
            'case_details': {}
        }

        for case_id, monitor in self.monitors.items():
            status = monitor.get_current_status()

            if status['training_started']:
                report['training_started'] += 1
            if status['training_completed']:
                report['training_completed'] += 1
            if status['training_failed']:
                report['training_failed'] += 1

            report['case_details'][case_id] = {
                'status': status,
                'metrics_summary': monitor.get_metrics_summary()
            }

        return report

    def export_all_results(self, output_dir: str):
        """导出所有结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        for case_id, monitor in self.monitors.items():
            case_dir = output_dir / f"case_{case_id}"
            monitor.export_metrics(str(case_dir))

        # 导出汇总报告
        summary_report = self.get_summary_report()
        report_path = output_dir / "batch_summary_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(summary_report, f, indent=2, ensure_ascii=False)
        print(f"批量训练汇总报告已保存到: {report_path}")

    def cleanup(self):
        """清理所有监控器"""
        for case_id in list(self.monitors.keys()):
            self.remove_monitor(case_id)


if __name__ == "__main__":
    # 测试单个监控器
    import sys

    if len(sys.argv) > 1:
        log_file = sys.argv[1]
        monitor = TrainingMonitor(log_file)
        monitor.start_monitoring()

        try:
            while True:
                time.sleep(10)
                status = monitor.get_current_status()
                print(f"状态更新: {status}")

                if status['training_completed'] or status['training_failed']:
                    break

        except KeyboardInterrupt:
            print("停止监控")
        finally:
            monitor.stop_monitoring_thread()
            monitor.export_metrics("./monitor_results")
    else:
        print("用法: python training_monitor.py <log_file_path>")