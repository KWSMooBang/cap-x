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
        # identity (kubectl run 1: bowl_A/B became bowl_C/D after the grasps, plate_D/E appeared): fixed camera, every
        # object visible at t=0 -> closed world (a new bowl needs 3 bowl detections in one frame), plates never new,
        # an unmatched bowl detection within 12 cm of a known unmatched bowl (held / unseen included) is that bowl
        tracks=TrackConfig(assoc_radius=0.08, merge_radius=0.10, grasp_radius=0.05, fixed_categories=("plate",),
                           reacquire_radius=0.12, reacquire_px=60.0, closed_world=True, fixed_closed=True),
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
        # identity (kubectl run 1: countertop_A..U, microwave_A..G, fridge_A..C on the moving head camera): world-frame
        # association; one countertop / fridge / microwave per scene (ASSUMPTION for task 45's kitchen: further
        # detections are that object); large objects (>= 0.4 m) associate by box gap; an unmatched hot dog within
        # 25 cm of a known unmatched one is that one. Not closed-world: objects enter the view later.
        # bggen-fix3 B3 (batch1 00450030: hotdog_A..F for two hot dogs): a category the init's goal names is capped at
        # the number of names it has after init (hotdog_A / hotdog_B expected -> 2). reacquire_radius stays 0.25 m (the
        # two hot dogs lie close together; a larger radius would join them).
        # bggen-fix4 C (batch2 040: microwave_A took detections 4.1 m away 4 times between t1229-3905, hotdog_B ones 1.08
        # and 2.55 m away; the boundaries 5771 / 6002 vanished): a full category's detection joins an existing track only
        # within 1.0 m (large objects: box gap). Why 1.0 m: 4x reacquire_radius (0.25 m, the same unmoved hot dog seen
        # again) for depth / base-pose drift of the moving head camera at kitchen range; an object the robot moved is
        # measured from the hand / release place. ASSUMPTION: re-measure from summary ``capped`` / ``capped_dropped``.
        # The countertop is exempt: the one countertop of the scene is several slabs (an L-shaped counter), a far slab is
        # still it.
        # bggen-fix5 (batch3 d401b65 kubectl traces):
        # A4 held_reacquire_radius 0.15 m from the EEF (030: the hot dog in the left hand was named hotdog_A, then
        # hotdog_B by the expected-name binding): a detection this close to the hand holding a hot dog is it, unless
        # another not-held hot dog track is nearer to it; held tracks are then not re-acquired by the 0.25 m reacquire
        # radius (lead review of 15d2f65: a second hot dog lying 15-20 cm beside the held one was taken). Why 0.15:
        # batch3 EEF-to-held-hot-dog centres 0.07 / 0.03 m + half a hot dog (0.075 m) = grasp_radius level (0.10) plus
        # head-camera depth noise; = assoc_radius. ASSUMPTION: re-measure from 'reacquired at the hand' distances.
        # B bind_confirm: microwave_A was bound to the built-in oven door (010 t746, 020 t1135; 4.8 m from the microwave)
        # and the cap of 1 dropped the real microwave 38 / 49 times: the first binding of a capped-at-1 name (fridge,
        # microwave) asks the VDM. The countertop is exempt (never an expected name: extra category; and its far slabs
        # are still it, as for cap_assign_exempt).
        # At most 5 identity calls per category and episode (a false detector that wanders defeats the place cache;
        # batch3 010/020 had one oven place): beyond -> not bound, recorded (identity_limit_skipped).
        # bggen-fix6 B (batch4 010 t1083 / 020 t3361: the real microwave's one VDM 'no' rejected its place for good, the
        # 38 / 39 later detections there were rejected without a call; 3 / 4 of the 5 calls used): a place is rejected
        # for good after 2 'no' answers from different events (bind_confirm_reject_n); a first 'no' holds back its own
        # event only. Calls 5 -> 8: with 2 per rejected place, batch4's oven places (2 per episode: 3 rejections) and the
        # microwave's first 'no' take up to 2 x 2 + 2 = 6 calls before the real microwave is asked a third time; +2 margin.
        # One call is ~1 VDM image query (~$0.002 at Qwen3.8-27B rates).
        # C relocate 3 events within 0.3 m (spec): the track moves to a place it keeps being dropped at; evidence
        # accumulates per place and a place must beat the current one; at most 2 moves per track (the first + one
        # correction; two false appliances winning in turn cannot ping-pong).
        tracks=TrackConfig(assoc_radius=0.15, assoc_px=60.0, merge_radius=0.15, grasp_radius=0.10, fixed_categories=(),
                           reacquire_radius=0.25, max_instances={"countertop": 1, "fridge": 1, "microwave": 1},
                           large_extent=0.4, cap_from_init=True, cap_assign_radius=1.0,
                           cap_assign_exempt=("countertop",),
                           held_reacquire_radius=0.15, bind_confirm=True, bind_confirm_exempt=("countertop",),
                           bind_confirm_max_calls=8, bind_confirm_reject_n=2,
                           relocate_min_events=3, relocate_radius=0.3, relocate_max=2),
        verifier=VerifierConfig(flat_dz={}),  # LIBERO table does not apply; first accepted Observe 1 per pair
        cook_s=10.0, navigation_subtask=True,
        # bggen-fix5 A (030: both hot dogs came out of the closed fridge never detected; the grasps at t2057 / t2720 had no
        # candidate and the hot dogs stayed behind): the arm's wrist camera at the grasp names the category; a detection
        # box within 0.10 m of the EEF (= grasp_radius; batch3 EEF-to-hot-dog centres 0.07 / 0.03 m) counts. Mode A also
        # looks up to 0.5 s later (the fingers / arm may cover the object at the closing frame).
        unseen_grasp=True, arm_cameras={"left": "left_wrist", "right": "right_wrist"}, wrist_grasp_radius=0.10,
        unseen_grasp_settle_s=0.5,
        # bggen-fix6 A (batch4 05e842e: open 0.81 -> 0.57, (open fridge_A) mem 0.9 vs truth False 17-24 rows in 010-050):
        # a grasp of the fridge / microwave door handle (or the counter) operates the object, it is never held: the
        # track stays in place (no hand following, no held reacquire, normal association) and (open X) is observed at
        # the grasp and the let-go. The scene's fixed / articulated categories = the capped-at-1 ones above.
        operate_categories=("fridge", "microwave", "countertop"),
    )


PRESETS = {"libero_mem": libero_mem, "b1k": b1k}
RENAMES = {"libero_mem": VER1_LIBERO, "b1k": VER1_B1K}


def preset(name: str, mode: str = "A", **overrides) -> GenConfig:
    cfg = PRESETS[name](mode)
    return replace(cfg, **overrides) if overrides else cfg
