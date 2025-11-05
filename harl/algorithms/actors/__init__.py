"""Algorithm registry."""
from harl.algorithms.actors.mappo import MAPPO
from harl.algorithms.actors.multi_policy_mappo import MPMAPPO

ALGO_REGISTRY = {
    "mappo": MAPPO,
    "mpmappo": MPMAPPO,
}
