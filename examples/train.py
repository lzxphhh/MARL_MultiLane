import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

"""Train an algorithm."""
import argparse
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args

"""
copy-paste 
# referred experiment
--load_config xxx.json
"""

"""
--algo <ALGO> --env <ENV> --exp_name <EXPERIMENT NAME>
--algo mappo --env a_multi_lane --exp_name 0712_debug --test_desc MLP_fusion
or
--load_config <TUNED CONFIG PATH> --exp_name <EXPERIMENT NAME>
"""
def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # 使用什么算法
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=[
            "mappo",
            "mpmappo"
        ],
        help="Algorithm name. Choose from: mappo, mpmappo.",
    )
    # 使用什么环境
    parser.add_argument(
        "--env",
        type=str,
        default="a_multi_lane",
        choices=[
            "a_multi_lane",
        ],
        help="Environment name. Choose from: a_multi_lane.",
    )
    # 实验名称
    parser.add_argument(
        "--exp_name", type=str, default="exp01", help="Experiment name."
    )
    # 测试项说明
    parser.add_argument(
        "--test_desc", type=str, default="test01", help="Test description."
    )
    # 是否使用config file
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    # 读取命令行参数
    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    # env-specific的参数
    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    runner.run()
    runner.close()


if __name__ == "__main__":
    main()
