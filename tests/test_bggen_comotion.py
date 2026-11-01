"""Regression tests for the review of 88f4ad4 (lead_cap-bg): grasp co-motion check.
1 a release with the grasp still unverified is resolved with the release image, or left UNVERIFIED
2 a detection at the hand does not confirm while the same object is still seen at its old place
3 no detection of the category is "cannot tell", not "the old place is empty"
4 tracks without 3D use image co-motion (2D path); no reference at all -> unverified at the release
"""

from __future__ import annotations

import json

import numpy as np

from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.fake_models import FakeServer
from capx.bggen.generator import Generator
from capx.bggen.models import Models
from capx.bggen.perception import Detection
from capx.bggen.pipeline import run_episode
from capx.bggen.records import load_episode
from tests.bggen_fakes import FakeDetector, SwapWorld, TruthBrain, make_world, project, write_swap_episode


def _first_move(carried: bool = True) -> SwapWorld:
    w = make_world()
    cut = w.grasp_t[1] - 5
    if not carried:
        start = w.bowls[0]["bowl_1"].copy()
        for t in range(w.n):
            if w.held[t] == "bowl_1":
                w.held[t] = None
            w.bowls[t]["bowl_1"] = start.copy()
    for name in ("eef", "width", "bowls", "held"):
        setattr(w, name, getattr(w, name)[:cut])
    w.grasp_t, w.release_t = w.grasp_t[:1], w.release_t[:1]
    return w


def _run(tmp_path, w, hide):
    write_swap_episode(tmp_path, w)
    with FakeServer(TruthBrain(w)) as srv:
        s = run_episode(tmp_path, tmp_path / "out", lm_cfg("A"), Models.from_env(env=srv.env()), FakeDetector(w, hide=hide),
                        "libero_mem")
    trace = [json.loads(x) for x in (tmp_path / "out" / "trace.jsonl").read_text().splitlines()]
    return s, trace


# 1 ---------------------------------------------------------------------------------------------------
def test_release_confirms_a_grasp_hidden_during_the_hold(tmp_path):
    w = _first_move(carried=True)
    g, r = w.grasp_t[0], w.release_t[0]
    s, trace = _run(tmp_path, w, lambda t, n: g - 8 <= t < r - 3)  # no bowl detected while the hand is closed
    assert [(c["result"], c["at"]) for c in s["grasp_checks"]] == [("confirmed", "release")]
    rel = next(t for t in trace if t["event"]["kind"] == "release")
    assert rel["event"]["obj"] == "bowl_A" and not rel["event"]["info"].get("place_unverified")


def test_release_retracts_when_the_object_never_left(tmp_path):
    w = _first_move(carried=False)
    g, r = w.grasp_t[0], w.release_t[0]
    s, trace = _run(tmp_path, w, lambda t, n: g - 8 <= t < r - 3)
    assert [(c["result"], c["at"]) for c in s["grasp_checks"]] == [("retracted", "release")]
    rel = next(t for t in trace if t["event"]["kind"] == "release")
    assert rel["event"]["obj"] is None  # nothing was carried: nothing placed
    assert rel["graph"]["(on bowl_A plate_B)"][0] == 0.9  # its relation before the close, remembered
    log = json.loads((tmp_path / "out" / "backward_log.json").read_text())
    assert [x["kind"] for x in log["segments"]] == ["articulated"]


def test_release_without_evidence_leaves_the_place_unverified(tmp_path):
    w = _first_move(carried=True)
    g, r = w.grasp_t[0], w.release_t[0]
    s, trace = _run(tmp_path, w, lambda t, n: g - 8 <= t < r + 40)  # not seen at the release either
    assert [(c["result"], c["at"], c["evidence"]) for c in s["grasp_checks"]] == [("unverified", "release", "no_detection")]
    rel = next(t for t in trace if t["event"]["kind"] == "release")
    assert rel["event"]["info"]["place_unverified"] is True
    log = json.loads((tmp_path / "out" / "backward_log.json").read_text())
    assert log["segments"][0]["uncertain"] is True and log["segments"][0]["effect"] == []
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows_backward.jsonl").read_text().splitlines()]
    labels = {r_["subtask"] for r_ in rows}
    assert "maybe_pick(bowl_A, plate_B)" in labels
    assert not any(x.startswith(("pick(bowl_A", "place(bowl_A")) for x in labels)  # (later rows: the causal next goal)
    assert not any(r_["effect"] for r_ in rows)


# 2, 3, 4: the evidence function ---------------------------------------------------------------------
def _gen(tmp_path):
    w = write_swap_episode(tmp_path)
    with FakeServer(TruthBrain(w)) as srv:
        gen = Generator(load_episode(tmp_path), FakeDetector(w), Models.from_env(env=srv.env()), lm_cfg("A"))
        gen.initialize()
    return gen, w


def _det(p, cat="bowl"):
    p = np.asarray(p, float)
    u, v = project(p)
    return Detection(cat, 0.9, (u, v), (u - 9, v - 9, u + 9, v + 9), p, np.array([0.13, 0.13, 0.05]))


def _pend(gen, t, hand_px=None):
    a = gen.tracker.tracks["bowl_A"]
    eef0 = np.array([0.6, 0.0, 0.038])
    return {"obj": "bowl_A", "t": t, "row": 0, "eef0": eef0, "pos0": a.pos.copy(), "px0": a.pixel.get("agentview"),
            "eef_px0": project(eef0) if hand_px is None else hand_px, "loc0": {}}


def test_detection_at_the_hand_does_not_confirm_while_the_object_is_still_at_its_place(tmp_path):
    gen, w = _gen(tmp_path)
    t = w.grasp_t[0] + 50  # hand lifted and travelling
    fr = gen.ep.frame(t, ["agentview"])
    cam = fr.cams["agentview"]
    pend = _pend(gen, w.grasp_t[0])
    hand = fr.eef_world("main") + (pend["pos0"] - pend["eef0"])
    old = pend["pos0"]
    v, info = gen._grasp_evidence(pend, "main", fr, cam, [_det(hand), _det(old)], hand)
    assert v == "ambiguous" and info["at_hand"] == 1 and info["at_old"] == 1  # two same-looking bowls: no confirmation
    v, _ = gen._grasp_evidence(pend, "main", fr, cam, [_det(old)], hand)
    assert v == "left"
    v, _ = gen._grasp_evidence(pend, "main", fr, cam, [_det(hand)], hand)
    assert v == "with_hand"
    # a detection at the hand that the OTHER bowl's known position explains is no evidence
    gen.tracker.tracks["bowl_B"].pos = hand.copy()
    v, info = gen._grasp_evidence(pend, "main", fr, cam, [_det(hand)], hand)
    assert v != "with_hand" and info["at_hand"] == 0


def test_no_detection_is_cannot_tell_even_after_lifting(tmp_path):
    gen, w = _gen(tmp_path)
    t = w.grasp_t[0] + 40  # lifted 0.15 m
    fr = gen.ep.frame(t, ["agentview"])
    pend = _pend(gen, w.grasp_t[0])
    hand = fr.eef_world("main") + (pend["pos0"] - pend["eef0"])
    v, _ = gen._grasp_evidence(pend, "main", fr, fr.cams["agentview"], [], hand)
    assert v == "no_detection"
    gen.pending_grasp["main"] = pend
    assert gen._verify_grasps(fr, fr.cams["agentview"], [], t) == [] and "main" in gen.pending_grasp  # still pending
    # the category IS seen (the other bowl), nothing at either place, hand lifted: while the hand is still above
    # the old place it may hide it -> unclear; once the hand has travelled away -> gone_lifted
    other = gen.tracker.tracks["bowl_B"].pos
    v, info = gen._grasp_evidence(pend, "main", fr, fr.cams["agentview"], [_det(other)], hand)
    assert v == "unclear" and info["old_under_hand"]
    t2 = w.grasp_t[0] + 60
    fr2 = gen.ep.frame(t2, ["agentview"])
    hand2 = fr2.eef_world("main") + (pend["pos0"] - pend["eef0"])
    v, _ = gen._grasp_evidence(pend, "main", fr2, fr2.cams["agentview"], [_det(other)], hand2)
    assert v == "gone_lifted"


def test_two_d_path_uses_image_co_motion(tmp_path):
    gen, w = _gen(tmp_path)
    t = w.grasp_t[0] + 60
    fr = gen.ep.frame(t, ["agentview"])
    cam = fr.cams["agentview"]
    pend = _pend(gen, w.grasp_t[0])
    pend["pos0"] = None  # no 3D: the 2D path
    eef_px = gen._eef_pixel(cam, fr, "main")
    hand_px = np.asarray(pend["px0"]) + (np.asarray(eef_px) - np.asarray(pend["eef_px0"]))

    def det_px(px):
        return Detection("bowl", 0.9, (float(px[0]), float(px[1])), (px[0] - 9, px[1] - 9, px[0] + 9, px[1] + 9))

    v, info = gen._grasp_evidence(pend, "main", fr, cam, [det_px(hand_px)], None)
    assert v == "with_hand" and info["mode"] == "2d"
    v, _ = gen._grasp_evidence(pend, "main", fr, cam, [det_px(pend["px0"])], None)
    assert v == "left"
    pend["px0"] = None  # no reference at all: never decided during the hold
    v, _ = gen._grasp_evidence(pend, "main", fr, cam, [det_px(hand_px)], None)
    assert v == "no_reference"
