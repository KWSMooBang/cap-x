"""Regression tests for bggen-fix6 (batch4 05e842e, kubectl traces):

A. B1K open 0.81 -> 0.57: a fridge door-handle grasp held the fridge (hold(): the track followed the hand, its detections
   were then dropped far from the hand) and the closed door was never observed ((open fridge_A) mem 0.9 vs truth
   False). A grasp of a fixed / articulated category operates it: no hold, the track stays, (open X) is asked at the
   grasp and at the let-go.
B. B1K 010 / 020: the real microwave's one VDM 'no' rejected its place for good. A place is rejected for good after
   2 'no' answers from different events; one 'no' holds back its own event only; one 'yes' binds.
C. B1K backward Subtask erased forward boundaries (020: 2933 inside the place span 2030-6432; 7537 / 7783 after the
   last effect). The backward labels keep every forward boundary.
D. t06 demo_3: init JSON with goal_branches' brackets closed early -> re-asked once with the decoder's error.
"""

from __future__ import annotations

import numpy as np

from capx.bggen import backward as bw
from capx.bggen.agent import json_error
from capx.bggen.configs import b1k as b1k_cfg
from capx.bggen.events import Event
from capx.bggen.generator import Generator, Row
from capx.bggen.perception import Detection
from capx.bggen.tracks import Tracker
from capx.bggen.vocab import category
from tests.test_bggen_fix5 import _Client, _Det, _Ep, _Models, _det, _texts

FRIDGE = _det("fridge", 1.6, -0.15, 0.9, (0.8, 0.8, 1.8))  # front face at x = 1.2
HANDLE = np.array([1.18, -0.15, 1.0])  # 2 cm in front of the door
FAR = np.array([0.2, -1.6, 1.0])  # the hand 1.8 m away while the door swings / the base backs up


# --- A: operating a fridge -----------------------------------------------------------------------------------------

def _fridge_gen(cfg=None):
    def eef(arm, t):
        if arm == "left":
            return np.array([0.2, 0.5, 1.0])
        return HANDLE if t < 150 or t >= 280 else FAR

    state = {"open": "1"}

    def vdm(m):
        txt = _texts(m)
        if "red box" in txt:
            return "Answer: 1"
        return f"Observe: (open fridge_A) {state['open']}" if "(open fridge_A)" in txt else "Observe: none"

    models = _Models()
    models.vdm = _Client(vdm)
    gen = Generator(_Ep(400, eef), _Det(lambda cam, t: [FRIDGE] if cam == "head" else []), models, cfg or b1k_cfg("A"))
    gen.categories = [category("hotdog"), category("fridge")]
    gen.tracker.initialize(0, "head", [FRIDGE])
    gen.tracker.expect("hotdog", "hotdog_A")
    gen.tracker.set_caps({"fridge": 1, "hotdog": 1})
    gen.geo.observe_sizes()
    return gen, state


def test_fridge_handle_grasp_operates_the_fridge_and_the_closed_door_is_observed():
    gen, state = _fridge_gen()
    tr = gen.tracker
    gen.step(Event(100, "grasp", "right"), 300)
    rec = gen.trace[-1]
    assert rec["grasp"]["target"] == "fridge_A" and tr.held("right") is None and gen.operating == {"right": "fridge_A"}
    assert rec["operate_observe"] == "(open fridge_A)" and "(open fridge_A)" in rec["plan"]["forced"]
    assert gen.rows[-1].obj == "fridge_A" and gen.graph.p(("open", "fridge_A", None)) >= 0.9
    assert gen.graph.p(("handempty", "right", None)) >= 0.9 and ("inhand", "fridge_A", "right") not in gen.graph.edges
    # the hand swings the door / backs up 1.8 m: the fridge track stays where the fridge is, still detected
    gen.step(Event(200, "keyframe"), 300)
    rec = gen.trace[-1]
    assert "fridge_A" in rec["track"]["matched"] and np.allclose(tr.tracks["fridge_A"].pos, FRIDGE.center)
    assert tr.counts["capped_dropped"] == 0 and "fridge_A" in gen.rows[-1].touched
    assert gen.rows[-1].subtask == "manipulate(fridge_A)"
    # the door is closed and the hand lets go: (open fridge_A) asked whatever the agent chose -> 0
    state["open"] = "0"
    gen.step(Event(300, "release", "right"), 400)
    rec = gen.trace[-1]
    assert gen.rows[-1].obj == "fridge_A" and gen.operating == {} and rec["operate"] == {"let_go": "fridge_A"}
    assert "(open fridge_A)" in rec["plan"]["forced"] and gen.graph.p(("open", "fridge_A", None)) <= 0.1
    assert "settle" not in rec["tools"] and "geometry" not in rec["tools"]  # a let-go places nothing
    ops = gen.summary()["track_identity"]["operations"]
    assert [(o["what"], o["alias"]) for o in ops] == [("operate", "fridge_A"), ("let go", "fridge_A")]
    # backward: one articulated segment, close(fridge_A) from the effect open 1 -> 0
    segs = bw.segments(gen.rows, {"fridge_A": "fridge"})
    assert [(s.kind, s.obj, s.start_t, s.end_t) for s in segs] == [("articulated", "fridge_A", 100, 300)]


def test_without_operate_categories_the_fridge_is_held_as_in_fix5():
    cfg = b1k_cfg("A")
    cfg.operate_categories = ()
    gen, _ = _fridge_gen(cfg)
    gen.step(Event(100, "grasp", "right"), 300)
    assert gen.tracker.held("right") == "fridge_A" and "operate_observe" not in gen.trace[-1]
    gen.step(Event(200, "keyframe"), 300)
    assert not np.allclose(gen.tracker.tracks["fridge_A"].pos, FRIDGE.center)  # it followed the hand


def test_wrist_sees_the_fridge_at_the_hand_operates_the_known_fridge():
    """An unseen grasp (no track within grasp_radius) whose wrist camera sees the fridge at the fingers: the known
    fridge track is operated (not held), no expected name is bound."""
    gen, _ = _fridge_gen()
    gen.det = _Det(lambda cam, t: [_det("fridge", *(np.array([0.5, 0.0, 1.0]) + [0.3, 0, 0]), (0.8, 0.8, 1.8))]
                   if cam == "right_wrist" else [FRIDGE] if cam == "head" else [])
    gen.ep._eef = lambda arm, t: np.array([0.5, 0.0, 1.0]) if arm == "right" else np.array([0.2, 0.5, 1.0])
    gen.step(Event(100, "grasp", "right"), 300)
    rec = gen.trace[-1]
    assert rec["grasp_unseen"]["result"].endswith("operated") and gen.operating == {"right": "fridge_A"}
    assert gen.tracker.held("right") is None and gen.tracker.unplaced("hotdog") == ["hotdog_A"]


def test_hot_dog_taken_inside_the_fridge_box_is_held_not_the_fridge_operated():
    """The EEF inside the fridge's box (box distance 0): the fridge is the grasp target, but the wrist camera sees a
    hot dog at the fingers -> the expected hot dog is bound and held, the fridge is not operated."""
    gen, _ = _fridge_gen()
    inside = np.array([1.5, -0.15, 1.0])
    gen.ep._eef = lambda arm, t: inside if arm == "right" else np.array([0.2, 0.5, 1.0])
    gen.det = _Det(lambda cam, t: [_det("hotdog", *(inside + [0.03, 0, -0.02]), (0.15, 0.05, 0.05))]
                   if cam == "right_wrist" else [FRIDGE] if cam == "head" else [])
    gen.step(Event(100, "grasp", "right"), 300)
    rec = gen.trace[-1]
    assert rec["grasp"]["target"] == "fridge_A" and rec["grasp_wrist"]["result"] == "bound (wrist)"
    assert gen.tracker.held("right") == "hotdog_A" and gen.operating == {}
    # without a wrist detection of a movable: the fridge is operated
    gen, _ = _fridge_gen()
    gen.ep._eef = lambda arm, t: inside if arm == "right" else np.array([0.2, 0.5, 1.0])
    gen.step(Event(100, "grasp", "right"), 300)
    assert gen.operating == {"right": "fridge_A"} and gen.tracker.unplaced("hotdog") == ["hotdog_A"]


def test_fridge_first_detected_at_the_grasp_frame_does_not_hide_the_hot_dog():
    """Review of 07baa92: nothing tracked before the grasp frame; its detection binds the fridge (whose box contains the
    EEF) and a hot dog 0.2 m off the fingers (beyond grasp_radius); the retry after detection targets the fridge, the
    wrist camera sees the hot dog at the fingers -> the hot dog is held (known track), the fridge is not operated."""
    gen, _ = _fridge_gen()
    gen.tracker = Tracker(gen.cfg.tracks)
    gen.geo.tracker = gen.tracker
    gen.tracker.initialize(0, "head", [])
    gen.tracker.expect("fridge", "fridge_A")
    gen.tracker.expect("hotdog", "hotdog_A")
    gen.tracker.set_caps({"fridge": 1, "hotdog": 1})
    inside = np.array([1.5, -0.15, 1.0])
    hot_head = _det("hotdog", *(inside + [0.2, 0, 0]), (0.15, 0.05, 0.05))
    gen.ep._eef = lambda arm, t: inside if arm == "right" else np.array([0.2, 0.5, 1.0])
    gen.det = _Det(lambda cam, t: [_det("hotdog", *(inside + [0.03, 0, -0.02]), (0.15, 0.05, 0.05))]
                   if cam == "right_wrist" else [FRIDGE, hot_head] if cam == "head" else [])
    gen.step(Event(100, "grasp", "right"), 300)
    rec = gen.trace[-1]
    assert rec["grasp"]["target"] is None and rec["grasp_retry"]["target"] == "fridge_A"
    assert rec["grasp_wrist"]["result"].startswith("known track") and gen.tracker.held("right") == "hotdog_A"
    assert gen.operating == {} and gen.rows[-1].obj == "hotdog_A"


# --- B: one 'no' is not final -----------------------------------------------------------------------------------------

MICRO = _det("microwave", 1.0, -2.8, 1.2, (0.5, 0.35, 0.3), px=(60.0, 20.0))


def _mw(answers):
    tr = Tracker(b1k_cfg().tracks)
    tr.initialize(0, "head", [_det("countertop", 1.0, 0.0, 0.9, (2.0, 0.6, 0.9))])
    tr.expect("microwave", "microwave_A")
    tr.set_caps({"microwave": 1})
    asked = []

    def confirm(alias, d, t):
        asked.append(t)
        return answers(t)

    return tr, confirm, asked


def test_first_no_holds_back_its_event_and_a_later_yes_binds():
    tr, confirm, asked = _mw(lambda t: "0" if t < 2000 else "1")
    tr.update(1083, "head", [MICRO], confirm=confirm)
    assert tr.expected() == ["microwave_A"] and asked == [1083] and tr.counts["rejected_identity_pending"] == 1
    # the same event again (the agent's detect tool): held back without a second call
    tr.update(1083, "head", [MICRO], confirm=confirm, keep_seen=True)
    assert asked == [1083] and tr.identity_checks[-1]["held_event"] and tr.counts["rejected_identity_held"] == 1
    # another event (after the base moved): asked again, 'yes' -> bound
    tr.update(2400, "head", [_det("microwave", 1.03, -2.78, 1.2, (0.5, 0.35, 0.3))], confirm=confirm)
    assert asked == [1083, 2400] and tr.expected() == [] and tr.identity_checks[-1]["asked_again"]
    assert tr.counts.get("rejected_identity_cached", 0) == 0


def test_two_no_answers_at_different_events_reject_the_place_for_good():
    tr, confirm, asked = _mw(lambda t: "0")
    for t in (746, 900, 1100, 1300):
        tr.update(t, "head", [_det("microwave", 1.0 + 0.01 * (t % 7), -2.8, 1.2, (0.5, 0.35, 0.3))], confirm=confirm)
    assert asked == [746, 900] and tr.counts["rejected_identity_cached"] == 2 and tr.expected() == ["microwave_A"]
    assert [c.get("no_count") for c in tr.identity_checks[:2]] == [1, 2]
    assert len(tr.rejected_spots) == 1 and tr.rejected_spots[0].no_ts == [746, 900]


def test_reject_n_1_keeps_the_fix5_rule():
    cfg = b1k_cfg().tracks
    cfg.bind_confirm_reject_n = 1
    tr = Tracker(cfg)
    tr.expect("microwave", "microwave_A")
    tr.set_caps({"microwave": 1})
    asked = []
    tr.update(10, "head", [MICRO], confirm=lambda a, d, t: asked.append(t) or "0")
    tr.update(20, "head", [MICRO], confirm=lambda a, d, t: asked.append(t) or "0")
    assert asked == [10] and tr.counts["rejected_identity_cached"] == 1


def test_identity_cue_only_for_b1k_categories():
    from capx.bggen.prompts import IDENTITY_CUES, identity_prompt

    assert set(IDENTITY_CUES) == {"microwave", "fridge"}
    txt = _texts(identity_prompt("microwave", "data:image/png;base64,", "head", IDENTITY_CUES["microwave"]))
    assert "small box-shaped appliance" in txt and "built-in oven" in txt
    assert "small box" not in _texts(identity_prompt("bowl", "data:image/png;base64,", "front"))


# --- C: backward keeps the forward boundaries ------------------------------------------------------------------------

def _row(t, fwd, kind="keyframe", obj=None, belief=None):
    return Row(t, kind, "left" if kind in ("grasp", "release") else None, obj, [], belief or {}, [], "", 0, 2, fwd,
               [obj] if obj else [], {"left": None})


ON = ("on", "hotdog_B", "countertop_A")


def _rows_020():
    """020 shape: hotdog_B held 30-150 (base moves inside), its effect confirmed at the release; forward boundaries at
    90 (inside the hold) and 180 / 210 (after the last effect)."""
    pb = "place(hotdog_B, ?)"
    return [_row(0, "pick(hotdog_B, ?)", "start", belief={ON: (0.1, "obs")}),
            _row(30, pb, "grasp", "hotdog_B", {ON: (0.03, "robot")}), _row(60, pb), _row(90, "move_to(countertop_A)"),
            _row(120, "move_to(countertop_A)"),
            _row(150, "pick(hotdog_A, ?)", "release", "hotdog_B", {ON: (0.9, "obs")}),
            _row(180, "pick(hotdog_A, countertop_A)", belief={ON: (0.9, "obs")}), _row(210, "open(microwave_A)", belief={ON: (0.9, "obs")}),
            _row(240, "open(microwave_A)", belief={ON: (0.9, "obs")})]


def _bounds(rows, key=lambda r: r.subtask):
    return [rows[i].t for i in range(1, len(rows)) if key(rows[i]) != key(rows[i - 1])]


def test_backward_keeps_forward_boundaries_inside_a_long_hold_and_after_the_last_effect():
    rows = _rows_020()
    fwd_b = _bounds(rows)
    cats = {"hotdog_B": "hotdog", "countertop_A": "countertop", "hotdog_A": "hotdog", "microwave_A": "microwave"}
    res = bw.backward(rows, [], cats)
    lab = {r.t: r.subtask for r in res.rows}
    assert set(fwd_b) <= set(_bounds(res.rows)), (fwd_b, lab)
    # the hold: the forward sub-spans before the last one keep their labels; the end of the hold gets the B4 label
    assert lab[60] == "place(hotdog_B, ?)" and lab[90] == lab[120] == "place(hotdog_B, countertop_A)"
    seg = res.segments[0]
    assert seg.effect and seg.label_place == "place(hotdog_B, countertop_A)"
    # after the last effect: the forward labels throughout
    assert lab[150] == "pick(hotdog_A, ?)" and lab[180] == "pick(hotdog_A, countertop_A)" and lab[210] == "open(microwave_A)" and lab[240] == "open(microwave_A)"
    keep = next(x for x in res.log if x["rule"] == "B4-keep")
    assert any(u["forward_boundaries"] == [90] for u in keep["units"])
    # the old rule (keep_forward off) erased them
    rows2 = [r.copy() for r in _rows_020()]
    segs = bw.segments(rows2, cats)
    bw.effects(rows2, segs, [])
    bw.subtasks(rows2, segs, cats, [], False, False, [], keep_forward=False)
    assert 90 not in _bounds(rows2) and 210 not in _bounds(rows2)


def test_backward_without_forward_boundaries_inside_is_unchanged():
    """The forward labels change only where B4 changes them: the B4 labels as before (the LIBERO swap golden)."""
    rows = [_row(0, "pick(bowl_A, ?)", "start"), _row(30, "place(bowl_A, plate_A)", "grasp", "bowl_A"),
            _row(60, "place(bowl_A, plate_A)"), _row(90, "place(bowl_A, plate_A)", "release", "bowl_A", {("on", "bowl_A", "plate_A"): (0.9, "obs")})]
    for r in rows[:3]:
        r.belief = {("on", "bowl_A", "plate_A"): (0.1, "obs")}
    res = bw.backward(rows, [], {"bowl_A": "bowl", "plate_A": "plate"})
    assert [r.subtask for r in res.rows] == ["pick(bowl_A, ?)", "place(bowl_A, plate_A)", "place(bowl_A, plate_A)",
                                             "place(bowl_A, plate_A)"]
    assert not any(x["rule"] == "B4-forward-unit" for x in res.log)


def test_a_forward_boundary_at_the_last_release_keeps_the_tail_forward():
    """B4 continues the last place label after the last release (no boundary there); the forward rows changed at the
    release: the tail takes its forward label, the hold keeps the B4 label."""
    rows = [_row(0, "pick(bowl_A, ?)", "start"), _row(30, "place(bowl_A, ?)", "grasp", "bowl_A"), _row(60, "place(bowl_A, ?)"),
            _row(90, "pick(bowl_B, ?)", "release", "bowl_A", {("on", "bowl_A", "plate_A"): (0.9, "obs")})]
    for r in rows[:3]:
        r.belief = {("on", "bowl_A", "plate_A"): (0.1, "obs")}
    res = bw.backward(rows, [], {"bowl_A": "bowl", "plate_A": "plate"})
    assert [r.subtask for r in res.rows] == ["pick(bowl_A, ?)", "place(bowl_A, plate_A)", "place(bowl_A, plate_A)",
                                             "pick(bowl_B, ?)"]
    assert not any(x["rule"] == "B4-merged" for x in res.log)


def test_a_tail_with_a_constant_forward_label_keeps_it():
    """Review of 07baa92: a tail without forward boundaries inside keeps its forward label too (not the last place)."""
    rows = _rows_020()
    for r in rows[5:]:
        r.subtask = "pick(hotdog_A, ?)"
    res = bw.backward(rows, [], {"hotdog_B": "hotdog", "countertop_A": "countertop"})
    assert [r.subtask for r in res.rows][5:] == ["pick(hotdog_A, ?)"] * 4


def test_a_forward_boundary_at_a_unit_edge_is_kept_by_forward_labels():
    """The hold's end would carry the B4 label the forward rows name right after the release: the hold takes the
    forward labels throughout (the boundary at the release stays)."""
    rows = _rows_020()
    rows[6].subtask = rows[5].subtask = "place(hotdog_B, countertop_A)"  # forward: the place label from the release on
    rows[7].subtask = rows[8].subtask = "pick(hotdog_A, ?)"
    fwd_b = _bounds(rows)
    res = bw.backward(rows, [], {"hotdog_B": "hotdog", "countertop_A": "countertop"})
    assert set(fwd_b) <= set(_bounds(res.rows)) and not any(x["rule"] == "B4-merged" for x in res.log)
    assert next(x for x in res.log if x["rule"] == "B4-forward-unit")["t"] == [30]
    assert [r.subtask for r in res.rows][1:5] == ["place(hotdog_B, ?)", "place(hotdog_B, ?)", "move_to(countertop_A)",
                                                  "move_to(countertop_A)"]


# --- D: init JSON error -------------------------------------------------------------------------------------------------

BAD = ('{"goal_branches": [[["on","bowl_A","plate_A"]], [["on","bowl_B","plate_B"]]], [["on","bowl_B","plate_A"]]], '
       '"observe": [], "remember": []}')  # goal_branches closed after its 2nd step: "[[" where a key should be
GOOD = '{"goal_branches": [[[["on","bowl_A","plate_A"]], [["on","bowl_B","plate_B"]]]], "observe": [], "remember": []}'


def test_json_error_points_at_the_break():
    err = json_error("Here it is:\n```json\n" + BAD + "\n```")
    assert err is not None and "line 1" in err and "<HERE>" in err
    assert json_error(GOOD) is None and json_error("no json") is None


def _init_gen(replies):
    from capx.bggen.configs import libero_mem as lm_cfg

    it = iter(replies)
    models = _Models()
    models.init = _Client(lambda m: next(it))
    det = _Det(lambda cam, t: [Detection("bowl", 0.9, (10.0, 10.0), (5, 5, 15, 15), np.array([0.0, 0.0, 0.9]), np.array([0.1, 0.1, 0.05])),
                               Detection("bowl", 0.9, (40.0, 10.0), (35, 5, 45, 15), np.array([0.0, 0.3, 0.9]), np.array([0.1, 0.1, 0.05])),
                               Detection("plate", 0.9, (10.0, 40.0), (5, 35, 15, 45), np.array([0.3, 0.0, 0.88]), np.array([0.2, 0.2, 0.02])),
                               Detection("plate", 0.9, (40.0, 40.0), (35, 35, 45, 45), np.array([0.3, 0.3, 0.88]), np.array([0.2, 0.2, 0.02]))])

    class Ep(_Ep):
        arms = ["main"]
        cameras = ["agentview"]
        instruction = "put the bowls on the plates"

    gen = Generator(Ep(100, lambda a, t: np.array([0.0, 0.0, 1.2])), det, models, lm_cfg("A"))
    return gen, models


def test_init_bracket_error_is_reasked_with_the_error_then_succeeds():
    gen, models = _init_gen([BAD, BAD, GOOD])  # init_retry 1 + the JSON fix query
    gen.initialize()
    info = gen.init_info
    assert "failed" not in info and len(info["replies"]) == 3 and info["json_fix"]["attempt"] == 1
    assert len(gen.goal.branches) == 1 and len(gen.goal.branches[0]) == 2
    second = _texts(models.init.calls[1])
    assert "not valid JSON" in second and "SAME content" in second and "<HERE>" in second
    third = _texts(models.init.calls[2])
    assert "could not be used" in third.split("SAME content")[-1]  # the 2nd failure: the usual re-query


def test_init_cut_json_without_closing_brace_is_reasked_with_the_error():
    cut = '{"goal_branches": [[[["on","bowl_A","plate_A"]], [["on","bowl_B"'
    err = json_error("```json\n" + cut)
    assert err is not None and "<HERE>" in err
    gen, models = _init_gen([cut, GOOD])
    gen.initialize()
    assert "failed" not in gen.init_info and gen.init_info["json_fix"]["attempt"] == 1
    assert "not valid JSON" in _texts(models.init.calls[1])


def test_init_json_fix_is_used_at_most_once():
    gen, models = _init_gen([BAD, BAD, BAD, GOOD])
    gen.initialize()
    assert gen.init_info["failed"].startswith("init failed") and len(models.init.calls) == 3
