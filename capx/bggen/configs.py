"""Benchmark presets. Thresholds marked ASSUMPTION are guesses until the first real episode is dumped
(``scripts/bggen/prep_*.py`` prints the gripper width distribution and event candidates)."""

from __future__ import annotations

from dataclasses import replace

from .events import EventConfig
from .generator import GenConfig
from .tracks import TrackConfig
from .verifier import VerifierConfig
from .vocab import VER1_B1K, VER1_LIBERO


def libero_mem(mode: str = "A") -> GenConfig:
    return GenConfig(
        mode=mode, camera="agentview", view="front", look_views={"robot0_eye_in_hand": "wrist"},
        relations=("on", "in", "at"),
        # LIBERO gripper_states: two finger joints; width = |q0| + |q1| (sign convention independent).
        # kubectl 1a (task 06 demo_1): range 0.0021-0.04 and every bowl grasp stopped near the bottom, so the
        # fully-closed width is unknown: closes are "close" and co-motion decides (closed_width from baccus later).
        events=EventConfig(close_delta=0.004, open_delta=0.004, stop_speed=0.0004, stop_frames=4, closed_width=None,
                           empty_margin=0.002, keyframe_s=1.5),
        tracks=TrackConfig(assoc_radius=0.08, merge_radius=0.10, grasp_radius=0.05, fixed_categories=("plate",)),
        verifier=VerifierConfig(),
    )


def b1k(mode: str = "A") -> GenConfig:
    return GenConfig(
        mode=mode, camera="head", view="head", look_views={"left_wrist": "wrist", "right_wrist": "wrist"},
        relations=("on", "in", "open", "toggled_on", "cooked"), extra_categories=("countertop",),
        max_obs=12, max_init_obs=16,
        # R1Pro gripper_<arm>_qpos: two fingers 0..0.05 m each; width = sum. ASSUMPTION: a hot dog stops > 5 mm.
        events=EventConfig(close_delta=0.006, open_delta=0.006, stop_speed=0.0006, stop_frames=6, closed_width=0.0, empty_margin=0.005,
                           keyframe_s=5.0, base_move_speed=0.05, base_stop_speed=0.02),
        tracks=TrackConfig(assoc_radius=0.15, assoc_px=60.0, merge_radius=0.15, grasp_radius=0.10, fixed_categories=()),
        verifier=VerifierConfig(flat_dz={}),  # LIBERO table does not apply; first accepted Observe 1 per pair
        cook_s=10.0, navigation_subtask=True,
    )


PRESETS = {"libero_mem": libero_mem, "b1k": b1k}
RENAMES = {"libero_mem": VER1_LIBERO, "b1k": VER1_B1K}


def preset(name: str, mode: str = "A", **overrides) -> GenConfig:
    cfg = PRESETS[name](mode)
    return replace(cfg, **overrides) if overrides else cfg
