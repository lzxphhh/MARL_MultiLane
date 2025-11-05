#!/usr/bin/env python3
"""
批量训练启动脚本
使用方法: python run_batch_training.py [config_file]
"""

import sys
import os
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="启动批量训练")
    parser.add_argument(
        "--config",
        type=str,
        default="batch_config.yaml",
        help="批量训练配置文件路径"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="从指定的结果目录恢复训练"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示训练计划，不实际执行"
    )

    args = parser.parse_args()

    # 检查配置文件
    if not os.path.exists(args.config):
        print(f"错误: 配置文件 {args.config} 不存在")
        print("请创建配置文件或使用 --config 指定正确的路径")
        return 1

    # 导入批量训练器
    try:
        from batch_train import BatchTrainer
    except ImportError as e:
        print(f"错误: 无法导入批量训练模块: {e}")
        print("请确保 batch_train.py 在当前目录下")
        return 1

    try:
        # 创建批量训练器
        trainer = BatchTrainer(args.config)

        if args.dry_run:
            # 干运行模式，只显示训练计划
            print("=" * 60)
            print("批量训练计划 (干运行模式)")
            print("=" * 60)
            print(f"配置文件: {args.config}")
            print(f"训练案例数量: {len(trainer.training_cases)}")
            print(f"结果保存目录: {trainer.batch_results_dir}")
            print()

            for i, case in enumerate(trainer.training_cases, 1):
                print(f"案例 {i:03d}: {case['test_desc']}")
                print(f"  算法: {case.get('algo', 'mappo')}")
                print(f"  环境: {case.get('env', 'a_single_lane')}")
                if 'params' in case:
                    print("  参数:")
                    for key, value in case['params'].items():
                        print(f"    {key}: {value}")
                print()

            print("使用 --dry-run=false 或去掉 --dry-run 参数来实际执行训练")

        else:
            # 实际执行训练
            trainer.run_batch_training()

    except KeyboardInterrupt:
        print("\n收到中断信号，正在退出...")
        return 1
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())