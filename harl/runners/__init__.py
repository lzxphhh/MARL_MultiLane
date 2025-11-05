"""Runner registry."""
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.multi_policy_mhc_ma_runner import MultiPolicyMARunner

RUNNER_REGISTRY = {
    "mappo": OnPolicyMARunner,
    "mpmappo": MultiPolicyMARunner,
}
