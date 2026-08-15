"""Freqtrade native-feature convergence adapters for the Hedge runtime."""

from .analysis import HedgeLookaheadAnalyzer, HedgeRecursiveAnalyzer
from .audit import AuditReport, AuditStatus, NativeAuditRunner
from .backtest import (
    HedgeBacktestArtifact,
    HedgeBacktestResultAdapter,
    PortfolioBacktestComparison,
)
from .exchange_capabilities import (
    CapabilityLevel,
    HedgeExchangeCapabilities,
    HedgeExchangeCapabilityRegistry,
    default_exchange_registry,
)
from .exit_overlay import NativeExitOverlay, policies_from_config
from .freqai import (
    HedgeFreqAISignalAdapter,
    HedgeFreqAITarget,
    HedgeModelManifest,
    HedgeModelReadinessGate,
    HedgeSignalEnvelope,
)
from .hyperopt import HedgeHyperoptRunner, HedgeHyperoptSpace, HedgeMultiObjectiveLoss
from .notifications import HedgeNotificationFormatter, HedgeRpcEventBridge
from .multipair import MultiPairCycleResult, MultiPairPaperHedgeRuntime
from .producer import HedgeProducerConsumerGate, ProducerIdentity
from .rl import (
    HedgeRLAction,
    HedgeRLActionMask,
    HedgeRLRewardFunction,
    HedgeRLState,
    VectorHedgeRLEnvironment,
)
from .rpc import HedgeAccountProjection, HedgePositionProjection, HedgeRpcProjectionService
from .universe import HedgeUniverseManager
from .admission import (
    CompositeAdmissionPolicy,
    apply_planning_admission_gate,
    planner_intent_to_native,
)
from .callbacks import (
    CallbackCompatibilityMode,
    HedgeCallbackContext,
    HedgeOrderView,
    HedgeStrategyCallbackAdapter,
    HedgeTradeView,
)
from .capital import FreqtradeCapitalPolicyAdapter
from .coordinator import NativeConvergenceCoordinator, build_native_convergence
from .exits import HedgeExitPolicy, HedgeExitPolicyEngine
from .models import (
    AdmissionCode,
    AdmissionDecision,
    BotStateSnapshot,
    CapitalSnapshot,
    ExitDecision,
    HedgeAction,
    HedgeBucket,
    HedgeEvent,
    HedgeSide,
    LegSnapshot,
    ModelReadinessSnapshot,
    NativeBotMode,
    NativeOrderIntent,
    ProtectionSnapshot,
)
from .protections import HedgeProtectionAdapter
from .state import HedgeBotStateAdapter

__all__ = [
    "AdmissionCode",
    "AdmissionDecision",
    "BotStateSnapshot",
    "CallbackCompatibilityMode",
    "CapitalSnapshot",
    "CompositeAdmissionPolicy",
    "ExitDecision",
    "FreqtradeCapitalPolicyAdapter",
    "HedgeAction",
    "HedgeBotStateAdapter",
    "HedgeBucket",
    "HedgeCallbackContext",
    "HedgeEvent",
    "HedgeExitPolicy",
    "HedgeExitPolicyEngine",
    "HedgeOrderView",
    "HedgeProtectionAdapter",
    "HedgeSide",
    "HedgeStrategyCallbackAdapter",
    "HedgeTradeView",
    "LegSnapshot",
    "ModelReadinessSnapshot",
    "MultiPairPaperHedgeRuntime",
    "MultiPairCycleResult",
    "NativeBotMode",
    "NativeConvergenceCoordinator",
    "NativeOrderIntent",
    "ProtectionSnapshot",
    "apply_planning_admission_gate",
    "build_native_convergence",
    "planner_intent_to_native",
    "AuditReport",
    "AuditStatus",
    "CapabilityLevel",
    "HedgeAccountProjection",
    "HedgeBacktestArtifact",
    "HedgeBacktestResultAdapter",
    "HedgeExchangeCapabilities",
    "HedgeExchangeCapabilityRegistry",
    "HedgeFreqAISignalAdapter",
    "HedgeFreqAITarget",
    "HedgeHyperoptRunner",
    "HedgeHyperoptSpace",
    "HedgeLookaheadAnalyzer",
    "HedgeModelManifest",
    "HedgeModelReadinessGate",
    "HedgeMultiObjectiveLoss",
    "HedgeNotificationFormatter",
    "HedgePositionProjection",
    "HedgeProducerConsumerGate",
    "HedgeRecursiveAnalyzer",
    "HedgeRLAction",
    "HedgeRLActionMask",
    "HedgeRLRewardFunction",
    "HedgeRLState",
    "HedgeRpcEventBridge",
    "HedgeRpcProjectionService",
    "HedgeSignalEnvelope",
    "HedgeUniverseManager",
    "NativeAuditRunner",
    "NativeExitOverlay",
    "PortfolioBacktestComparison",
    "ProducerIdentity",
    "VectorHedgeRLEnvironment",
    "default_exchange_registry",
    "policies_from_config",
]
