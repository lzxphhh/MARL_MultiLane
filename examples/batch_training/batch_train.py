"""
批量训练控制器
自动进行多组参数配置的训练，监控训练进度，记录结果
"""

import os
import sys
import time
import json
import yaml
import shutil
import pandas as pd
import subprocess
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from email.header import Header
from email.utils import formataddr
from pathlib import Path
import threading
import queue
import signal


class BatchTrainer:
    def __init__(self, config_file="batch_config.yaml"):
        """初始化批量训练器"""
        self.config_file = config_file
        self.load_batch_config()
        self.setup_directories()
        self.init_result_tracking()
        self.current_training_process = None
        self.stop_signal = False

    def load_batch_config(self):
        """加载批量训练配置"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.batch_config = yaml.load(f, Loader=yaml.FullLoader)

        # 基础配置
        self.base_yaml_path = self.batch_config['base_yaml_path']
        self.training_cases = self.batch_config['training_cases']
        self.exp_name = self.batch_config['exp_name']
        self.email_config = self.batch_config.get('email_config', {})
        self.monitoring_config = self.batch_config.get('monitoring_config', {})
        self.success_criteria = self.batch_config.get('success_criteria', {})

    def setup_directories(self):
        """设置目录结构"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.batch_results_dir = Path(f"batch_results_{timestamp}")
        self.batch_results_dir.mkdir(exist_ok=True)

        # 创建子目录
        self.logs_dir = self.batch_results_dir / "logs"
        self.configs_dir = self.batch_results_dir / "configs"
        self.results_dir = self.batch_results_dir / "results"
        self.failed_dir = self.batch_results_dir / "failed"

        for dir_path in [self.logs_dir, self.configs_dir, self.results_dir, self.failed_dir]:
            dir_path.mkdir(exist_ok=True)

    def init_result_tracking(self):
        """初始化结果跟踪"""
        self.results_csv_path = self.batch_results_dir / "training_results.csv"
        self.failed_csv_path = self.batch_results_dir / "failed_trainings.csv"

        # 创建结果CSV文件
        results_columns = [
            'case_id', 'test_desc', 'model_name', 'start_time', 'end_time',
            'training_steps', 'final_reward', 'collision_rate', 'mean_speed',
            'success_metrics', 'config_path', 'model_path', 'log_path'
        ]

        failed_columns = [
            'case_id', 'test_desc', 'model_name', 'start_time', 'end_time',
            'training_steps', 'failure_reason', 'error_log', 'config_path'
        ]

        pd.DataFrame(columns=results_columns).to_csv(self.results_csv_path, index=False)
        pd.DataFrame(columns=failed_columns).to_csv(self.failed_csv_path, index=False)

    def create_training_config(self, case_config, case_id):
        """为每个训练案例创建配置文件"""
        # 加载基础配置
        with open(self.base_yaml_path, 'r', encoding='utf-8') as f:
            base_config = yaml.load(f, Loader=yaml.FullLoader)

        # 更新配置参数
        for key, value in case_config.get('params', {}).items():
            if '.' in key:  # 支持嵌套参数，如 reward_weights.safety
                keys = key.split('.')
                config_section = base_config
                for k in keys[:-1]:
                    config_section = config_section[k]
                config_section[keys[-1]] = value
            else:
                base_config[key] = value

        # 保存新配置文件
        config_filename = f"case_{case_id:03d}_{case_config['test_desc']}.yaml"
        config_path = self.configs_dir / config_filename

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)

        return config_path, base_config

    def start_training(self, case_config, case_id, config_path, exp_name):
        """启动训练进程"""
        test_desc = case_config['test_desc']
        model_name = f"seed-{case_id:03d}"

        # 构建训练命令
        cmd = [
            'python', 'train.py',
            '--algo', case_config.get('algo', 'mappo'),
            '--env', case_config.get('env', 'a_multi_lane'),
            '--exp_name', exp_name,
            '--test_desc', test_desc
        ]

        # 添加其他参数
        if 'additional_args' in case_config:
            for key, value in case_config['additional_args'].items():
                cmd.extend([f'--{key}', str(value)])

        print(f"[Case {case_id:03d}] 启动训练: {test_desc}")
        print(f"[Case {case_id:03d}] 命令: {' '.join(cmd)}")

        # 设置环境变量，指定配置文件路径
        env = os.environ.copy()
        env['HARL_CONFIG_PATH'] = str(config_path)

        # 启动训练进程
        log_file = self.logs_dir / f"case_{case_id:03d}_{test_desc}.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.getcwd()
            )

        return process, log_file

    def monitor_training(self, process, case_config, case_id, log_file):
        """监控训练进程"""
        start_time = datetime.now()
        test_desc = case_config['test_desc']
        model_name = f"seed-{case_id:03d}"

        print(f"[Case {case_id:03d}] 开始监控训练进程 PID: {process.pid}")

        # 监控配置
        max_training_time = self.monitoring_config.get('max_training_hours', 24) * 3600
        check_interval = self.monitoring_config.get('check_interval_seconds', 30)

        training_steps = 0
        last_log_size = 0
        no_progress_count = 0
        max_no_progress = self.monitoring_config.get('max_no_progress_checks', 20)

        while process.poll() is None and not self.stop_signal:
            time.sleep(check_interval)

            # 检查训练时间
            elapsed_time = (datetime.now() - start_time).total_seconds()
            if elapsed_time > max_training_time:
                print(f"[Case {case_id:03d}] 训练超时，终止进程")
                process.terminate()
                self.record_failed_training(
                    case_id, test_desc, model_name, start_time,
                    datetime.now(), training_steps, "训练超时", str(log_file)
                )
                return False

            # 检查日志文件大小变化（判断是否有进度）
            try:
                current_log_size = log_file.stat().st_size
                if current_log_size > last_log_size:
                    last_log_size = current_log_size
                    no_progress_count = 0

                    # 解析训练步数
                    training_steps = self.parse_training_progress(log_file)
                    if training_steps > 0:
                        print(f"[Case {case_id:03d}] 训练进度: {training_steps} steps")
                else:
                    no_progress_count += 1

                if no_progress_count >= max_no_progress:
                    print(f"[Case {case_id:03d}] 训练无进度，可能卡死，终止进程")
                    process.terminate()
                    self.record_failed_training(
                        case_id, test_desc, model_name, start_time,
                        datetime.now(), training_steps, "训练无进度", str(log_file)
                    )
                    return False

            except Exception as e:
                print(f"[Case {case_id:03d}] 监控异常: {e}")

        # 检查训练结果
        if process.poll() == 0:  # 正常结束
            end_time = datetime.now()
            print(f"[Case {case_id:03d}] 训练完成")
            return self.check_training_success(
                case_id, test_desc, model_name, start_time, end_time, log_file
            )
        else:  # 异常结束
            end_time = datetime.now()
            print(f"[Case {case_id:03d}] 训练异常结束，返回码: {process.poll()}")
            self.record_failed_training(
                case_id, test_desc, model_name, start_time, end_time,
                training_steps, f"进程异常结束，返回码: {process.poll()}", str(log_file)
            )
            return False

    def parse_training_progress(self, log_file):
        """从日志文件解析训练进度"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # 寻找最新的训练步数信息
            for line in reversed(lines):
                if "episode" in line.lower() and "steps" in line.lower():
                    # 尝试提取步数信息（根据实际日志格式调整）
                    import re
                    match = re.search(r'step[s]?[:\s]+(\d+)', line, re.IGNORECASE)
                    if match:
                        return int(match.group(1))

        except Exception as e:
            print(f"解析训练进度失败: {e}")

        return 0

    def check_training_success(self, case_id, test_desc, model_name, start_time, end_time, log_file):
        """检查训练是否成功并记录结果"""
        try:
            # 解析训练结果
            metrics = self.parse_training_metrics(log_file)

            # 检查成功标准
            success = True
            success_details = {}

            for metric, threshold in self.success_criteria.items():
                if metric in metrics:
                    if isinstance(threshold, dict):
                        # 范围检查
                        if 'min' in threshold:
                            success &= metrics[metric] >= threshold['min']
                        if 'max' in threshold:
                            success &= metrics[metric] <= threshold['max']
                    else:
                        # 简单阈值检查
                        success &= metrics[metric] >= threshold
                    success_details[metric] = metrics[metric]

            if success:
                print(f"[Case {case_id:03d}] 训练成功，指标达标")
                self.record_successful_training(
                    case_id, test_desc, model_name, start_time, end_time, metrics
                )
                self.send_success_notification(case_id, test_desc, metrics)
            else:
                print(f"[Case {case_id:03d}] 训练完成但指标未达标")
                self.record_failed_training(
                    case_id, test_desc, model_name, start_time, end_time,
                    metrics.get('training_steps', 0), "指标未达标", str(log_file)
                )

            return success

        except Exception as e:
            print(f"[Case {case_id:03d}] 检查训练结果失败: {e}")
            self.record_failed_training(
                case_id, test_desc, model_name, start_time, end_time,
                0, f"结果检查失败: {e}", str(log_file)
            )
            return False

    def parse_training_metrics(self, log_file):
        """从日志文件解析训练指标"""
        metrics = {}

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 根据实际日志格式解析指标（需要根据具体日志调整）
            import re

            # 示例解析规则
            patterns = {
                'final_reward': r'total reward.*?(\d+\.?\d*)',
                'collision_rate': r'collision.*?(\d+\.?\d*)',
                'mean_speed': r'mean speed.*?(\d+\.?\d*)',
                'training_steps': r'step[s]?[:\s]+(\d+)'
            }

            for metric, pattern in patterns.items():
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    metrics[metric] = float(matches[-1])  # 取最后一个匹配

        except Exception as e:
            print(f"解析训练指标失败: {e}")

        return metrics

    def record_successful_training(self, case_id, test_desc, model_name, start_time, end_time, metrics):
        """记录成功的训练结果"""
        result_data = {
            'case_id': case_id,
            'test_desc': test_desc,
            'model_name': model_name,
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_steps': metrics.get('training_steps', 0),
            'final_reward': metrics.get('final_reward', 0),
            'collision_rate': metrics.get('collision_rate', 0),
            'mean_speed': metrics.get('mean_speed', 0),
            'success_metrics': json.dumps(metrics),
            'config_path': str(self.configs_dir / f"case_{case_id:03d}_{test_desc}.yaml"),
            'model_path': '',  # 需要根据实际模型保存路径填写
            'log_path': str(self.logs_dir / f"case_{case_id:03d}_{test_desc}.log")
        }

        # 追加到CSV文件
        df = pd.DataFrame([result_data])
        df.to_csv(self.results_csv_path, mode='a', header=False, index=False)

    def record_failed_training(self, case_id, test_desc, model_name, start_time, end_time, training_steps,
                               failure_reason, error_log):
        """记录失败的训练"""
        failed_data = {
            'case_id': case_id,
            'test_desc': test_desc,
            'model_name': model_name,
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'training_steps': training_steps,
            'failure_reason': failure_reason,
            'error_log': error_log,
            'config_path': str(self.configs_dir / f"case_{case_id:03d}_{test_desc}.yaml")
        }

        # 追加到失败记录CSV
        df = pd.DataFrame([failed_data])
        df.to_csv(self.failed_csv_path, mode='a', header=False, index=False)

        # 保存错误日志到txt文件
        error_file = self.failed_dir / f"case_{case_id:03d}_{test_desc}_error.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(f"案例ID: {case_id}\n")
            f.write(f"测试描述: {test_desc}\n")
            f.write(f"模型名称: {model_name}\n")
            f.write(f"开始时间: {start_time}\n")
            f.write(f"结束时间: {end_time}\n")
            f.write(f"训练步数: {training_steps}\n")
            f.write(f"失败原因: {failure_reason}\n")
            f.write(f"错误日志: {error_log}\n")

    def send_success_notification(self, case_id, test_desc, metrics):
        """发送成功通知邮件"""
        if not self.email_config:
            return

        subject = f"✅ 训练成功 - Case {case_id:03d}: {test_desc}"

        body = f"""
训练任务成功完成！

📊 训练信息:
- 案例ID: {case_id:03d}
- 测试描述: {test_desc}
- 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 性能指标:
"""

        for metric, value in metrics.items():
            body += f"- {metric}: {value}\n"

        body += f"\n📁 结果路径: {self.batch_results_dir}"

        self.send_email(subject, body)

    def send_email(self, subject, body):
        """发送邮件"""
        try:
            msg = MIMEText(body, 'plain', 'utf-8')
            msg['From'] = formataddr(("BatchTrainer", self.email_config['sender']))
            msg['To'] = formataddr(("User", self.email_config['receiver']))
            msg['Subject'] = Header(subject, 'utf-8')

            server = smtplib.SMTP_SSL(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            )
            server.login(self.email_config['sender'], self.email_config['password'])
            server.sendmail(
                self.email_config['sender'],
                [self.email_config['receiver']],
                msg.as_string()
            )
            server.quit()
            print("邮件发送成功")

        except Exception as e:
            print(f"邮件发送失败: {e}")

    def signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n收到中断信号，正在优雅退出...")
        self.stop_signal = True
        if self.current_training_process:
            self.current_training_process.terminate()

    def run_batch_training(self):
        """运行批量训练"""
        # 注册信号处理器
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        print(f"开始批量训练，共 {len(self.training_cases)} 个案例")
        print(f"结果保存目录: {self.batch_results_dir}")

        successful_cases = 0
        failed_cases = 0
        exp_name = self.exp_name

        for case_id, case_config in enumerate(self.training_cases, 1):
            if self.stop_signal:
                print("收到停止信号，退出批量训练")
                break

            print(f"\n{'=' * 60}")
            print(f"开始训练案例 {case_id}/{len(self.training_cases)}: {case_config['test_desc']}")
            print(f"{'=' * 60}")

            try:
                # 创建训练配置
                config_path, _ = self.create_training_config(case_config, case_id)
                print(f"[Case {case_id:03d}] 配置文件: {config_path}")

                # 启动训练
                process, log_file = self.start_training(case_config, case_id, config_path, exp_name)
                self.current_training_process = process

                # 监控训练
                success = self.monitor_training(process, case_config, case_id, log_file)

                if success:
                    successful_cases += 1
                else:
                    failed_cases += 1

            except Exception as e:
                print(f"[Case {case_id:03d}] 训练过程异常: {e}")
                failed_cases += 1
                self.record_failed_training(
                    case_id, case_config['test_desc'], f"seed-{case_id:03d}",
                    datetime.now(), datetime.now(), 0, f"启动异常: {e}", ""
                )

            finally:
                self.current_training_process = None

        # 打印最终统计
        print(f"\n{'=' * 60}")
        print(f"批量训练完成!")
        print(f"成功: {successful_cases} 个案例")
        print(f"失败: {failed_cases} 个案例")
        print(f"总计: {successful_cases + failed_cases} 个案例")
        print(f"结果保存在: {self.batch_results_dir}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    trainer = BatchTrainer()
    trainer.run_batch_training()