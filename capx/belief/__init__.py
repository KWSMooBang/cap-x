"""CaP-BG Belief Graph: graph structure, oracle estimator, prompt serialization, refinement signals,
and the per-trial memory conditions (none / vdm_history / bg_oracle).

The estimated graph (bg_est) adds goal parsing (goal), perception (perception), API-call hooks (hooks),
the verification VDM (verify) and the fusion estimator (estimator); these read no simulator state."""

from .estimator import BeliefEstimator, EstConfig, EstTurn
from .goal import GoalResult, parse_goal_json, parse_goal_llm
from .graph import P_EXPECTED, P_OBS_FALSE, P_OBS_TRUE, P_PRIOR, BeliefGraph, Edge, Event, Node
from .hooks import ApiCallRecorder, CallRecord
from .memory import (
    MEMORY_MODES,
    BgOracleMemory,
    MemoryTurn,
    NoMemory,
    VdmHistoryMemory,
    make_memory,
)
from .oracle import GroundTruthSource, LiberoMemSource, OracleEstimator
from .perception import Detection, PerceptionSource, Sam3PerceptionSource, SimPerceptionSource
from .progress import Progress, goal_branches
from .serialize import serialize
from .signals import FinishGate, Signal, SignalContext, SignalDetector, Uncertain, Violation
from .verify import VdmVerifier

__all__ = [
    "P_EXPECTED", "P_OBS_FALSE", "P_OBS_TRUE", "P_PRIOR",
    "BeliefGraph", "Edge", "Event", "Node",
    "MEMORY_MODES", "BgOracleMemory", "MemoryTurn", "NoMemory", "VdmHistoryMemory", "make_memory",
    "GroundTruthSource", "LiberoMemSource", "OracleEstimator",
    "Progress", "goal_branches", "serialize",
    "FinishGate", "Signal", "SignalContext", "SignalDetector", "Uncertain", "Violation",
    "BeliefEstimator", "EstConfig", "EstTurn", "GoalResult", "parse_goal_json", "parse_goal_llm",
    "ApiCallRecorder", "CallRecord", "Detection", "PerceptionSource", "Sam3PerceptionSource",
    "SimPerceptionSource", "VdmVerifier",
]
