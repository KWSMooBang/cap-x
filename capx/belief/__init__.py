"""CaP-BG Belief Graph: graph structure, oracle estimator, prompt serialization, refinement signals,
and the per-trial memory conditions (none / vdm_history / bg_oracle)."""

from .graph import P_EXPECTED, P_OBS_FALSE, P_OBS_TRUE, P_PRIOR, BeliefGraph, Edge, Event, Node
from .memory import (
    MEMORY_MODES,
    BgOracleMemory,
    MemoryTurn,
    NoMemory,
    VdmHistoryMemory,
    make_memory,
)
from .oracle import GroundTruthSource, LiberoMemSource, OracleEstimator
from .progress import Progress, goal_branches
from .serialize import serialize
from .signals import FinishGate, Signal, SignalContext, SignalDetector

__all__ = [
    "P_EXPECTED", "P_OBS_FALSE", "P_OBS_TRUE", "P_PRIOR",
    "BeliefGraph", "Edge", "Event", "Node",
    "MEMORY_MODES", "BgOracleMemory", "MemoryTurn", "NoMemory", "VdmHistoryMemory", "make_memory",
    "GroundTruthSource", "LiberoMemSource", "OracleEstimator",
    "Progress", "goal_branches", "serialize",
    "FinishGate", "Signal", "SignalContext", "SignalDetector",
]
