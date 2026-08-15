"""Production runtime supervision primitives for Freqtrade Hedge."""

from .config import DeploymentMode, HedgeDeploymentConfig, RestartPolicy
from .single_instance import InstanceLockError, SingleInstanceLock
from .state import RuntimePhase, RuntimeState, RuntimeStateStore
from .supervisor import HedgeProcessSupervisor, SupervisorResult

__all__ = [
    "DeploymentMode",
    "HedgeDeploymentConfig",
    "HedgeProcessSupervisor",
    "InstanceLockError",
    "RestartPolicy",
    "RuntimePhase",
    "RuntimeState",
    "RuntimeStateStore",
    "SingleInstanceLock",
    "SupervisorResult",
]
