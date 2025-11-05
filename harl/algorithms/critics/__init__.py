"""Critic registry."""
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.mp_mh_v_critic import MHVCritic

CRITIC_REGISTRY = {
    "mappo": VCritic,
    "mpmappo": MHVCritic,
}
