"""Regression tests for bggen-fix5 (batch3 d401b65, kubectl traces):

A. B1K 030: two hot dogs come out of the closed fridge never detected; the right / left grasps had no candidate with a
   position, so nothing was held and the later detections of the hot dog in the left hand were named hotdog_A, then
   hotdog_B (expected-name binding). Now the grasp binds a position-less expected name to the hand (wrist camera
   detection; without one the only / first pointed expected name, 'unverified'), the track follows the hand and a
   detection near a held track is that track before any expected name is bound.
B. B1K 010 / 020: microwave_A was bound to the built-in oven door (cap 1 dropped the real microwave). The first binding
   of a capped-at-1 name asks the VDM; 0 -> not bound (the place is remembered), ? -> bound and marked.
C. A capped-at-1 track whose far dropped detections repeat at one place is moved there.
D. LIBERO: the output and every prompt of the LIBERO preset are the d401b65 ones (golden digests).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from capx.bggen.configs import b1k as b1k_cfg
from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.events import Event
from capx.bggen.generator import Generator
from capx.bggen.perception import Detection
from capx.bggen.prompts import parse_identity
from capx.bggen.records import ArmState, CameraFrame, FrameRecord
from capx.bggen.tracks import Tracker
from capx.bggen.vocab import category
from tests import bggen_golden

HOT = (0.15, 0.05, 0.05)
MW = (0.5, 0.35, 0.3)


def _det(cat, x, y, z, size, score=0.9, px=(10.0, 10.0)):
    return Detection(cat, score, px, (px[0] - 5, px[1] - 5, px[0] + 5, px[1] + 5), np.array([x, y, z], float),
                     np.array(size, float))


# --- fakes ---------------------------------------------------------------------------------------------------------

class _Client:
    def __init__(self, fn):
        self.fn = fn
        self.calls: list = []

    def query(self, messages):
        self.calls.append(messages)
        return {"content": self.fn(messages), "finish_reason": "stop"}


def _texts(messages):
    return "\n".join(p.get("text", "") for m in messages for p in (m["content"] if isinstance(m["content"], list)
                                                                  else [{"text": m["content"]}]))


class _Models:
    """Scripted models: the agent asks nothing; the VDM answers the identity check with ``identity(t_image)``."""

    def __init__(self, identity=lambda text: "?"):
        self.agent = _Client(lambda m: '{"calls": [], "why": "test"}')
        self.vdm = _Client(lambda m: f"Answer: {identity(_texts(m))}" if "red box" in _texts(m) else "Observe: none")
        self.init = _Client(lambda m: "{}")


class _Ep:
    """Two arms (left / right), a head camera and two wrist cameras with an image at every frame; fixed base."""

    fps, mobile, arms = 20.0, False, ["left", "right"]
    cameras = ["head", "left_wrist", "right_wrist"]
    instruction = "cook the two hotdogs in the microwave"

    def __init__(self, n, eef):
        self.n_frames, self._eef = n, eef
        self.image_reads = 0

    def robot(self, t):
        return FrameRecord(t, t / self.fps, {a: ArmState(0.02, np.asarray(self._eef(a, t), float)) for a in self.arms})

    def frame(self, t, cams=(), causal=True):
        fr = self.robot(t)
        fr.cams = {c: CameraFrame(c, t, np.zeros((60, 80, 3), np.uint8)) for c in cams if c in self.cameras}
        return fr


class _Det:
    """Scripted detector: ``at(cam, t)`` -> detections (filtered by the asked categories)."""

    name = "scripted"

    def __init__(self, at):
        self.at, self.calls = at, 0

    def detect(self, cam, cats):
        self.calls += 1
        keys = {c.key for c in cats}
        return [d for d in self.at(cam.name, cam.t) if d.category in keys]


# --- A: 030 shape ----------------------------------------------------------------------------------------------------

R0, R1, R2 = np.array([0.8, -0.2, 0.9]), np.array([0.4, -0.6, 1.0]), np.array([0.5, 0.9, 0.95])
L0, L1 = np.array([0.85, -0.1, 0.95]), np.array([0.4, 0.6, 1.2])
FRIDGE = _det("fridge", 1.6, -0.15, 0.9, (0.8, 0.8, 1.8))  # box 0.4 m beyond the hands


def _eef(arm, t):
    if arm == "right":
        return R0 if t < 150 else R1 if t < 400 else R2
    return np.array([0.2, 0.4, 1.0]) if t < 200 else L0 if t < 250 else L1


def _030(wrist: bool, goal_hot=True):
    def at(cam, t):
        if cam == "head":
            out = [FRIDGE]
            if 300 <= t < 400:  # both hot dogs seen in the hands after the carry
                out += [_det("hotdog", *(R1 + [0.05, 0.0, -0.03]), HOT), _det("hotdog", *(L1 + [0.0, 0.04, -0.02]), HOT)]
            if t >= 450:  # hotdog in the right hand released at R2
                out += [_det("hotdog", *(R2 + [0.02, 0.0, -0.04]), HOT)]
            return out
        if not wrist:
            return []
        if cam == "right_wrist" and t == 100:
            return [_det("hotdog", *(R0 + [0.03, 0.0, -0.02]), HOT)]
        if cam == "left_wrist" and t == 200:
            return [_det("hotdog", *(L0 + [0.0, 0.02, -0.01]), HOT)]
        return []

    ep = _Ep(600, _eef)
    gen = Generator(ep, _Det(at), _Models(), b1k_cfg("A"))
    gen.categories = [category("hotdog"), category("fridge"), category("microwave")]
    gen.tracker.initialize(0, "head", [FRIDGE])
    for a in ("hotdog_A", "hotdog_B", "microwave_A"):
        gen.tracker.expect(a.split("_")[0], a)
    gen.tracker.set_caps({"hotdog": 2, "microwave": 1})
    gen.geo.observe_sizes()
    return gen


def _run_030(gen):
    gen.step(Event(100, "grasp", "right"), 200)
    gen.step(Event(160, "keyframe"), 200)
    gen.step(Event(200, "grasp", "left"), 400)
    gen.step(Event(300, "keyframe"), 400)
    gen.step(Event(400, "release", "right"), 600)
    gen.step(Event(450, "keyframe"), 600)


def test_030_wrist_detection_binds_expected_names_to_the_hands_in_letter_order():
    gen = _030(wrist=True)
    tr = gen.tracker
    gen.step(Event(100, "grasp", "right"), 200)
    rec = gen.trace[-1]
    assert rec["grasp"]["target"] is None and rec["grasp_unseen"]["result"] == "bound (wrist)"
    assert rec["grasp_unseen"]["wrist"]["category"] == "hotdog" and rec["grasp_unseen"]["wrist"]["distance"] <= 0.10
    assert tr.held("right") == "hotdog_A" and not tr.tracks["hotdog_A"].expected
    assert gen.rows[-1].obj == "hotdog_A" and gen.graph.p(("inhand", "hotdog_A", "right")) >= 0.9
    gen.step(Event(160, "keyframe"), 200)
    assert np.allclose(tr.tracks["hotdog_A"].pos, R1)  # follows the hand during the carry
    gen.step(Event(200, "grasp", "left"), 400)
    assert tr.held("left") == "hotdog_B" and tr.grasp_binds[-1]["how"] == "wrist"
    assert tr.counts["grasp_bound_wrist"] == 2 and tr.expected() == ["microwave_A"]
    # both hot dogs seen in the hands: each is its held track; nothing new, nothing dropped
    gen.step(Event(300, "keyframe"), 400)
    rep = gen.trace[-1]["track"]
    assert set(rep["matched"]) >= {"hotdog_A", "hotdog_B"} and rep["new"] == [] and rep["dropped"] == 0
    assert tr.counts["capped_dropped"] == 0
    # release: re-acquired at the release place
    gen.step(Event(400, "release", "right"), 600)
    assert gen.rows[-1].obj == "hotdog_A" and tr.held("right") is None
    gen.step(Event(450, "keyframe"), 600)
    assert tr.tracks["hotdog_A"].seen_now and np.linalg.norm(tr.tracks["hotdog_A"].pos - R2) < 0.1
    assert tr.held("left") == "hotdog_B" and sorted(a for a in tr.tracks if a.startswith("hotdog")) == ["hotdog_A", "hotdog_B"]


def test_030_without_wrist_detection_binds_first_letter_unverified_then_the_only_name():
    gen = _030(wrist=False)
    tr = gen.tracker
    _run_030(gen)
    binds = tr.grasp_binds
    assert [(b["alias"], b["how"]) for b in binds] == [("hotdog_A", "unverified"), ("hotdog_B", "only")]
    assert binds[0]["candidates"] == ["hotdog_A", "hotdog_B"]
    assert tr.counts["grasp_bound_unverified"] == 1 and tr.counts["grasp_bound_only"] == 1
    assert tr.tracks["hotdog_A"].bind_note == "grasp unverified"
    assert tr.held("left") == "hotdog_B" and tr.counts["capped_dropped"] == 0
    s = gen.summary()["track_identity"]
    assert len(s["grasp_binds"]) == 2 and s["expected_unbound"] == ["microwave_A"]


def test_wrist_seeing_a_support_at_the_hand_binds_nothing():
    """A door-handle grasp: the wrist camera sees the fridge at the fingers, no hot dog -> no expected name is taken."""
    def at(cam, t):
        if cam == "right_wrist":
            return [_det("fridge", *(R0 + [0.3, 0.0, 0.0]), (0.8, 0.8, 1.8))]
        return []

    gen = _030(wrist=False)
    gen.det = _Det(at)
    gen.step(Event(100, "grasp", "right"), 200)
    rec = gen.trace[-1]["grasp_unseen"]
    assert rec["result"].startswith("closed on a fridge") and gen.tracker.held("right") is None
    assert gen.tracker.unplaced("hotdog") == ["hotdog_A", "hotdog_B"]


def test_countertop_at_the_hand_does_not_block_and_a_known_stale_track_wins():
    # the wrist camera sees only the countertop under the hand: ignored -> the pointed expected name (unverified)
    gen = _030(wrist=False)
    gen.det = _Det(lambda cam, t: [_det("countertop", *(R0 + [0.0, 0.0, -0.1]), (2.0, 0.6, 0.1))] if cam == "right_wrist" else [])
    gen.categories.append(category("countertop"))
    gen.step(Event(100, "grasp", "right"), 200)
    assert gen.trace[-1]["grasp_unseen"]["result"] == "bound (unverified)" and gen.tracker.held("right") == "hotdog_A"
    # a hot dog known at 0.2 m from the hand (beyond grasp_radius 0.10, within reacquire 0.25): it, not hotdog_B
    gen = _030(wrist=True)
    gen.tracker.update(50, "head", [FRIDGE, _det("hotdog", *(R0 + [0.27, 0.0, 0.0]), HOT)])
    assert gen.tracker.expected() == ["hotdog_B", "microwave_A"]  # hotdog_A bound at t=50
    gen.step(Event(100, "grasp", "right"), 200)
    rec = gen.trace[-1]
    assert rec["grasp"]["target"] is None and rec["grasp_unseen"]["result"].startswith("known track")
    assert gen.tracker.held("right") == "hotdog_A" and gen.tracker.unplaced("hotdog") == ["hotdog_B"]


def test_a_detection_near_a_held_track_is_it_before_an_expected_name():
    """A4: hotdog_A held with its track 0.25 m off the fingers (grasped with a stale offset), hotdog_B expected; a
    detection 0.12 m from the hand (0.31 m from hotdog_A's track: beyond the 0.25 m reacquire radius)."""
    for on in (True, False):
        cfg = b1k_cfg().tracks
        if not on:
            cfg.held_reacquire_radius = 0.0
        tr = _held_a(cfg, offset=np.array([-0.25, 0.0, 0.0]))
        rep = tr.update(300, "head", [FRIDGE, _det("hotdog", *(L1 + [0.12, 0.0, 0.0]), HOT)])
        ident = rep.get("identified", [])
        if on:
            assert ident and ident[0]["into"] == "hotdog_A" and "at the hand" in ident[0]["why"]
            assert tr.expected() == ["hotdog_B"] and tr.counts["reacquired_held"] == 1
        else:  # the fix4 behaviour: the expected name took it
            assert ident and ident[0]["into"] == "hotdog_B"


def _held_a(cfg=None, offset=np.zeros(3), b_at=None):
    """hotdog_A in the left hand at L1 (track at L1 + offset); hotdog_B expected, or known at ``b_at``."""
    tr = Tracker(cfg or b1k_cfg().tracks)
    tr.initialize(0, "head", [FRIDGE])
    tr.expect("hotdog", "hotdog_A")
    tr.expect("hotdog", "hotdog_B")
    tr.set_caps({"hotdog": 2})
    tr.bind_at_grasp("hotdog_A", 100, "unverified")
    tr.tracks["hotdog_A"].pos = L0 + offset
    tr.hold("hotdog_A", "left", L0)
    if b_at is not None:  # hotdog_B seen (bound) while the hand is still at L0, far from it
        tr.update(150, "head", [FRIDGE, _det("hotdog", *b_at, HOT)])
        assert tr.expected() == []
    tr.follow("left", L1)
    for a in tr.tracks:
        tr.tracks[a].seen_now = False
    return tr


def test_held_matching_comes_before_the_reacquire_of_not_held_tracks():
    """Lead review of a3c6215: the held track's own detection (0.02 m from the EEF) is not taken by a not-held track
    0.20 m away (within its 0.25 m reacquire radius); a detection 0.12 m from the EEF and 0.05 m from a not-held track is
    that track. The held track's position is 0.25 m off the fingers so the frame's first association pass misses it."""
    off = np.array([-0.25, 0.0, 0.0])
    tr = _held_a(offset=off, b_at=L1 + [0.02, 0.20, 0.0])
    rep = tr.update(300, "head", [FRIDGE, _det("hotdog", *(L1 + [0.02, 0.0, 0.0]), HOT)])
    ident = rep.get("identified", [])
    assert [r["into"] for r in ident] == ["hotdog_A"] and "at the hand" in ident[0]["why"]
    tr = _held_a(offset=off, b_at=L1 + [0.0, 0.17, 0.0])
    rep = tr.update(300, "head", [FRIDGE, _det("hotdog", *(L1 + [0.0, 0.12, 0.0]), HOT)])
    into = [r["into"] for r in rep.get("identified", [])] + [a for a in rep["matched"] if a.startswith("hotdog")]
    assert into == ["hotdog_B"] and tr.counts.get("reacquired_held", 0) == 0


def test_a_second_hot_dog_beside_the_held_one_is_not_taken_by_the_hand():
    """Lead review P1: a second hot dog 15-20 cm beside the held one goes to its own track (known or expected)."""
    side = np.array([0.0, 0.17, 0.0])
    # hotdog_B known at 17 cm from the hand: its detection is hotdog_B
    tr = _held_a(b_at=L1 + side)
    rep = tr.update(300, "head", [FRIDGE, _det("hotdog", *(L1 + side + [0.01, 0.0, 0.0]), HOT)])
    into = {r["into"] for r in rep.get("identified", [])} | set(rep["matched"])
    assert "hotdog_B" in into and "hotdog_A" not in into and tr.held("left") == "hotdog_A"
    # a detection 0.12 m from the hand but nearer to hotdog_B (5 cm): hotdog_B, not the held one
    tr = _held_a(b_at=L1 + side)
    rep = tr.update(310, "head", [FRIDGE, _det("hotdog", *(L1 + [0.0, 0.12, 0.0]), HOT)])
    into = {r["into"] for r in rep.get("identified", [])} | set(rep["matched"])
    assert into == {"hotdog_B"} | ({"fridge_A"} & into) and tr.counts.get("reacquired_held", 0) == 0
    # hotdog_B expected: a detection 18 cm beside the hand binds hotdog_B (fix5 first version: taken by the hand)
    tr = _held_a()
    rep = tr.update(300, "head", [FRIDGE, _det("hotdog", *(L1 + [0.0, 0.18, 0.0]), HOT)])
    assert [r["into"] for r in rep.get("identified", [])] == ["hotdog_B"] and tr.expected() == []
    # the held hot dog itself, 6 cm off the fingers: still the held track
    tr = _held_a()
    rep = tr.update(300, "head", [FRIDGE, _det("hotdog", *(L1 + [0.06, 0.0, 0.0]), HOT)])
    assert "hotdog_A" in rep["matched"] and tr.expected() == ["hotdog_B"]


def test_retracted_provisional_grasp_unbinds_the_name():
    gen = _030(wrist=True)
    gen.step(Event(100, "close", "right"), 200)
    assert gen.tracker.held("right") == "hotdog_A" and gen.pending_grasp["right"]["bound_at_grasp"]
    gen._retract("right", gen.pending_grasp.pop("right"), 120, "test")
    assert gen.tracker.unplaced("hotdog") == ["hotdog_A", "hotdog_B"] and gen.tracker.held("right") is None


# --- B: identity check of the first binding ------------------------------------------------------------------------

OVEN = _det("microwave", 1.0, 2.0, 0.6, (0.6, 0.1, 0.6), px=(20.0, 30.0))
MICRO = _det("microwave", 1.0, -2.8, 1.2, MW, px=(60.0, 20.0))


def _mw_tracker(answers):
    tr = Tracker(b1k_cfg().tracks)
    tr.initialize(0, "head", [_det("countertop", 1.0, 0.0, 0.9, (2.0, 0.6, 0.9))])
    tr.expect("microwave", "microwave_A")
    tr.set_caps({"microwave": 1})
    asked = []

    def confirm(alias, d, t):
        asked.append((alias, t))
        return answers(d)

    return tr, confirm, asked


def test_oven_detection_is_not_bound_as_the_microwave():
    tr, confirm, asked = _mw_tracker(lambda d: "0" if d.center[1] > 0 else "1")
    rep = tr.update(746, "head", [OVEN], confirm=confirm)
    assert rep["dropped"] == 1 and tr.expected() == ["microwave_A"] and asked == [("microwave_A", 746)]
    assert tr.counts["rejected_identity"] == 1 and tr.identity_checks[-1]["answer"] == "0"
    # the oven again (a little moved) at another event: asked again (fix6 B: one 'no' holds back its event only);
    # the second 'no' rejects the place for good
    tr.update(900, "head", [_det("microwave", 1.05, 2.02, 0.6, (0.6, 0.1, 0.6))], confirm=confirm)
    assert len(asked) == 2 and tr.identity_checks[-1]["asked_again"] and tr.identity_checks[-1]["no_count"] == 2
    tr.update(950, "head", [_det("microwave", 1.02, 2.01, 0.6, (0.6, 0.1, 0.6))], confirm=confirm)
    assert len(asked) == 2 and tr.counts["rejected_identity_cached"] == 1 and tr.expected() == ["microwave_A"]
    # the real microwave (with the oven in the same frame): asked, bound
    rep = tr.update(1500, "head", [OVEN, MICRO], confirm=confirm)
    assert tr.expected() == [] and np.allclose(tr.tracks["microwave_A"].pos, MICRO.center)
    assert len(asked) == 3 and tr.tracks["microwave_A"].bind_note is None and tr.counts["bound_expected"] == 1


def test_identity_unknown_binds_and_marks():
    tr, confirm, asked = _mw_tracker(lambda d: "?")
    tr.update(746, "head", [OVEN], confirm=confirm)
    assert tr.expected() == [] and tr.tracks["microwave_A"].bind_note == "identity ?"
    assert tr.tracks["microwave_A"].to_dict()["bind_note"] == "identity ?" and tr.counts["bound_identity_unknown"] == 1
    # bound: later detections are association, no more calls
    tr.tracks["microwave_A"].seen_now = False
    tr.update(800, "head", [OVEN], confirm=confirm)
    assert len(asked) == 1


def test_rejection_without_3d_is_cached_by_camera_pixel_and_calls_are_limited():
    """Lead review P2: a detection without a world point is cached by its pixel in that camera; at most 8 (fix6 B; fix5
    5) identity calls per category and episode, then not bound (recorded)."""
    tr, confirm, asked = _mw_tracker(lambda d: "0")
    flat = Detection("microwave", 0.9, (20.0, 30.0), (10, 20, 30, 40))  # no depth
    tr.update(100, "head", [flat], confirm=confirm)
    tr.update(200, "head", [Detection("microwave", 0.9, (25.0, 32.0), (15, 22, 35, 42))], confirm=confirm)  # 2nd 'no'
    tr.update(205, "head", [Detection("microwave", 0.9, (24.0, 31.0), (15, 22, 35, 42))], confirm=confirm)
    assert len(asked) == 2 and tr.counts["rejected_identity_cached"] == 1
    assert tr.identity_checks[-1]["cached"] and tr.identity_checks[-1]["unit"] == "px"
    # another camera: not the cached place
    tr.update(210, "left_wrist", [flat], confirm=confirm)
    assert len(asked) == 3
    # eight calls per category: beyond, a distinct place is not asked and not bound
    for k in range(8):
        tr.update(300 + k, "head", [_det("microwave", 1.0, 3.0 + k, 0.6, MW)], confirm=confirm)
    assert tr.cfg.bind_confirm_max_calls == 8 and len(asked) == 8
    assert tr.counts["identity_limit_skipped"] == 3 and tr.identity_checks[-1]["limit"] == 8
    assert tr.expected() == ["microwave_A"]


def test_hotdog_binding_is_not_checked_and_countertop_is_exempt():
    tr = Tracker(b1k_cfg().tracks)
    tr.expect("hotdog", "hotdog_A")
    tr.expect("countertop", "countertop_A")
    tr.set_caps({"hotdog": 1, "countertop": 1})

    def confirm(alias, d, t):
        raise AssertionError("no identity call for these")

    tr.update(10, "head", [_det("hotdog", 1, 0, 1, HOT), _det("countertop", 1, 0, 0.9, (2.0, 0.6, 0.9))], confirm=confirm)
    assert tr.expected() == []


def test_generator_asks_the_vdm_with_a_boxed_image_once_per_binding():
    def at(cam, t):
        return [OVEN] if cam == "head" and t < 500 else [MICRO] if cam == "head" else []

    ep = _Ep(800, lambda a, t: np.array([0.3, 0.3 if a == "left" else -0.3, 1.0]))
    models = _Models(identity=lambda text: "0")
    gen = Generator(ep, _Det(at), models, b1k_cfg("B"))
    gen.categories = [category("microwave")]
    gen.tracker.initialize(0, "head", [])
    gen.tracker.expect("microwave", "microwave_A")
    gen.tracker.set_caps({"microwave": 1})
    gen.step(Event(100, "keyframe"), 800)
    gen.step(Event(200, "keyframe"), 800)  # fix6 B: asked again at another event (2nd 'no': the place is rejected)
    gen.step(Event(300, "keyframe"), 800)
    asked = [m for m in models.vdm.calls if "red box" in _texts(m)]
    assert len(asked) == 2 and "Is the object in the red box (mark 1) a microwave" in _texts(asked[0])
    assert "small box-shaped appliance" in _texts(asked[0])  # fix6 B: the category cue
    assert gen.trace[-3]["identity"][0]["answer"] == "0" and gen.trace[-2]["identity"][0]["answer"] == "0"
    assert "identity" not in gen.trace[-1]
    assert gen.tracker.expected() == ["microwave_A"]
    models.vdm.fn = lambda m: "Answer: 1" if "red box" in _texts(m) else "Observe: none"
    gen.step(Event(600, "keyframe"), 800)
    assert gen.tracker.expected() == [] and np.allclose(gen.tracker.tracks["microwave_A"].pos, MICRO.center)
    assert gen.summary()["track_identity"]["identity_calls"] == 3


def test_parse_identity():
    assert parse_identity("Answer: 0") == "0" and parse_identity("**Answer:** 1") == "1"
    assert parse_identity("I think it is an oven.\nAnswer: ?") == "?" and parse_identity("") == "?"
    assert parse_identity("thinking 1 ... Answer: 0") == "0"


# --- C: relocation ----------------------------------------------------------------------------------------------------

def _bound_at_oven():
    tr, confirm, _ = _mw_tracker(lambda d: "1")
    tr.update(746, "head", [OVEN], confirm=confirm)  # wrongly confirmed
    assert tr.tracks["microwave_A"].bound_t == 746
    return tr, confirm


def _micro_at(dx):
    return _det("microwave", 1.0 + dx, -2.8, 1.2, MW)


def test_capped_track_moves_to_a_repeated_far_place():
    tr, confirm = _bound_at_oven()
    for k, t in enumerate((1000, 1100)):
        tr.tracks["microwave_A"].seen_now = False
        rep = tr.update(t, "head", [_micro_at(0.05 * k)], confirm=confirm)
        assert rep["dropped"] == 1 and tr.counts["capped_dropped"] == k + 1
    tr.tracks["microwave_A"].seen_now = False
    rep = tr.update(1200, "head", [_micro_at(0.1)], confirm=confirm)
    assert rep["dropped"] == 0 and tr.counts["relocated"] == 1 and tr.counts["capped_dropped"] == 2
    r = tr.relocations[-1]
    assert r["events"] == [1000, 1100, 1200] and r["redetections_since_bound"] == 0 and r["bound_t"] == 746
    m = tr.tracks["microwave_A"]
    assert np.allclose(m.pos, _micro_at(0.1).center) and m.bound_t == 1200 and "relocated @1200" in m.bind_note
    # now the microwave is associated normally
    m.seen_now = False
    rep = tr.update(1300, "head", [_micro_at(0.0)], confirm=confirm)
    assert "microwave_A" in rep["matched"]


def test_no_relocation_while_the_track_is_redetected_as_often():
    tr, confirm = _bound_at_oven()
    t = 800
    for _ in range(3):  # the oven re-detected 3 times since the binding
        tr.tracks["microwave_A"].seen_now = False
        tr.update(t, "head", [OVEN], confirm=confirm)
        t += 10
    for k in range(3):
        tr.tracks["microwave_A"].seen_now = False
        tr.update(1000 + 100 * k, "head", [_micro_at(0.05 * k)], confirm=confirm)
    assert tr.counts.get("relocated", 0) == 0 and tr.counts["capped_dropped"] == 3
    tr.tracks["microwave_A"].seen_now = False
    tr.update(1300, "head", [_micro_at(0.0)], confirm=confirm)  # 4 events > 3 re-detections
    assert tr.counts["relocated"] == 1


def _far(tr, confirm, t, x, y):
    tr.tracks["microwave_A"].seen_now = False
    return tr.update(t, "head", [_det("microwave", x, y, 1.2, MW)], confirm=confirm)


def test_relocation_keeps_per_place_evidence_and_is_limited():
    """Lead review P2: two false places cannot win in turn -- a place must beat the current one's accumulated
    evidence, and a track moves at most twice."""
    tr, confirm = _bound_at_oven()  # microwave_A at the oven (1.0, 2.0)
    for k in range(3):  # place X (1.0, -2.8): 3 events > 0 at the oven -> moved to X
        _far(tr, confirm, 1000 + k, 1.0, -2.8)
    assert tr.counts["relocated"] == 1
    for k in range(3):  # place Y (4.0, 0.0): 3 events, not more than X's 3 -> stays
        _far(tr, confirm, 1100 + k, 4.0, 0.0)
    assert tr.counts["relocated"] == 1 and np.allclose(tr.tracks["microwave_A"].pos[:2], [1.0, -2.8])
    _far(tr, confirm, 1103, 4.0, 0.0)  # 4 > 3: moved to Y (second move)
    assert tr.counts["relocated"] == 2 and tr.relocations[-1]["current_place_events"] == 3
    for k in range(6):  # X again, 3 + 6 events > Y's 4, but the track moved twice already
        _far(tr, confirm, 1200 + k, 1.0, -2.8)
    assert tr.counts["relocated"] == 2 and tr.counts["relocate_blocked"] >= 1
    assert np.allclose(tr.tracks["microwave_A"].pos[:2], [4.0, 0.0])


def test_scattered_far_detections_do_not_relocate():
    tr, confirm = _bound_at_oven()
    for k in range(4):  # 0.5 m apart: no place repeats
        tr.tracks["microwave_A"].seen_now = False
        tr.update(1000 + 100 * k, "head", [_micro_at(0.5 * k)], confirm=confirm)
    assert tr.counts.get("relocated", 0) == 0 and tr.counts["capped_dropped"] == 4


# --- D: LIBERO ----------------------------------------------------------------------------------------------------------

def test_libero_preset_has_every_fix5_rule_off():
    cfg = lm_cfg()
    t = cfg.tracks
    assert not cfg.unseen_grasp and not cfg.arm_cameras
    assert t.held_reacquire_radius == 0 and not t.bind_confirm and t.relocate_min_events == 0


@pytest.mark.parametrize("name", list(bggen_golden.VARIANTS))
def test_libero_outputs_and_prompts_equal_fix4(tmp_path: Path, name: str):
    golden = json.loads(bggen_golden.GOLDEN.read_text())[name]
    assert bggen_golden.digest(tmp_path, name) == golden
