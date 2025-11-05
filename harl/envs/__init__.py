from absl import flags
from harl.envs.a_multi_lane.multilane_logger import MultiLaneLogger

FLAGS = flags.FLAGS
FLAGS(["train_sc.py"])

LOGGER_REGISTRY = {
    "a_multi_lane": MultiLaneLogger,
}
