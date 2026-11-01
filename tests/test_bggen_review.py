"""Regression tests for the review of 278cc4e (lead_cap-bg, 10 major items)."""

from __future__ import annotations

import json

import numpy as np

from capx.bggen import backward as bw
from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.events import Event
from capx.bggen.fake_models import FakeServer
from capx.bggen.generator import Generator, Row
from capx.bggen.graph import BeliefGraph, GoalTracker
from capx.bggen.models import Models
from capx.bggen.perception import Detection
from capx.bggen.pipeline import run_episode
from capx.bggen.records import Episode, load_episode
from capx.bggen.tracks import Tracker
from capx.bggen.verifier import Verifier
from tests.bggen_fakes import FakeDetector, TruthBrain, write_swap_episode


def _row(t, observed, belief, kind="keyframe", obj=None, arm=None, done=0, total=1):
    return Row(t, kind, arm, obj, observed, belief, [], f"{done}/{total}", done, total, "?", [], {"main": None})


# 1 -------------------------------------------------------------------------------------------------
def test_noise_restore_skips_rows_answered_unknown():
    k = ("on", "bowl_B", "plate_B")
    rows = [_row(0, [(k, "1")], {k: (0.9, "obs")}), _row(10, [(k, "0")], {k: (0.1, "obs")}),
            _row(20, [(k, "?")], {k: (0.1, "mem")}), _row(30, [(k, "1")], {k: (0.9, "obs")})]
    res = bw.backward(rows, [], {})
    assert dict(res.rows[1].observed)[k] == "1" and res.rows[1].belief[k][0] == 0.9  # the single wrong answer
    assert dict(res.rows[2].observed)[k] == "?" and res.rows[2].belief[k] == (0.1, "mem")  # untouched
    assert bw.check_forbidden(rows, res.rows) == []


# 2 -------------------------------------------------------------------------------------------------
def test_done_label_only_from_confirmed_row():
    rows = [_row(0, [], {}), _row(10, [], {}, kind="grasp", obj="bowl_A", arm=None),
            _row(20, [], {}, kind="release", obj="bowl_A", arm=None), _row(30, [], {}), _row(40, [], {}, done=1)]
    res = bw.backward(rows, [], {"bowl_A": "bowl"}, goal_done=True)
    assert res.rows[3].subtask != "done" and res.rows[4].subtask == "done"


# 3 -------------------------------------------------------------------------------------------------
def test_hindsight_image_before_next_gripper_event(tmp_path):
    w = write_swap_episode(tmp_path, image_stride=40)
    with FakeServer(TruthBrain(w)) as srv:
        gen = Generator(load_episode(tmp_path), FakeDetector(w), Models.from_env(env=srv.env()), lm_cfg("A"))
        gen.initialize()
        e = Event(41, "keyframe")
        h = gen._hindsight(e, [("on", "bowl_A", "plate_B")], 1.5, next_gripper_t=75)
    # requested 71; the nearest stored image is 80 (after the grasp at 75): not allowed; the latest <= 71 is 40 <= e.t
    assert h["t"] is None and h["values"] == {}
    with FakeServer(TruthBrain(w)) as srv:
        gen.models = Models.from_env(env=srv.env())
        h = gen._hindsight(Event(30, "keyframe"), [("on", "bowl_A", "plate_B")], 1.0, next_gripper_t=75)
    assert h["t"] == 40 and h["requested_t"] == 50 and h["values"]


# 4, 5 ------------------------------------------------------------------------------------------------
def test_lifted_is_robot_owned_and_held_relations_stay_off():
    g = BeliefGraph()
    v = Verifier(g)
    lk = ("lifted", "bowl_A", None)
    v.robot(lk, True, 5)
    v.observe({lk: "0"}, 6, touched={"bowl_A"})  # touched: R7 would not hold it, only robot ownership protects it
    v.observe({lk: "0"}, 7, touched={"bowl_A"})
    assert g.get(lk).p == 0.97 and g.get(lk).source == "robot"
    on = ("on", "bowl_A", "plate_A")  # never seen before the grasp: a new edge while held
    v.observe({on: "1"}, 7, held={"bowl_A"})
    assert g.get(on).p == 0.03 and g.get(on).source == "robot"
    v.expect(lk, False, 8, set())
    assert g.get(lk).p == 0.97


# 6 -------------------------------------------------------------------------------------------------
def test_cooking_timer_restarts_when_taken_out(tmp_path):
    w = write_swap_episode(tmp_path)
    cfg = lm_cfg("A")
    cfg.cook_s = 10.0
    gen = Generator(load_episode(tmp_path), FakeDetector(w), None, cfg)
    fps = gen.ep.fps
    ink, on, ck = ("in", "hotdog_A", "microwave_A"), ("toggled_on", "microwave_A", None), ("cooked", "hotdog_A", None)
    gen.graph.set(on, 0.9, "obs", 0)
    gen.graph.set(ink, 0.9, "obs", 0)
    gen._process_model(0)
    gen.graph.set(ink, 0.1, "obs", int(5 * fps))  # taken out after 5 s
    gen._process_model(int(5 * fps))
    gen.graph.set(ink, 0.9, "obs", int(20 * fps))  # put back in at 20 s
    gen._process_model(int(20 * fps))
    assert gen.graph.get(ck) is None  # not cooked at once
    gen._process_model(int(25 * fps))
    assert gen.graph.get(ck) is None
    gen._process_model(int(30 * fps))
    assert gen.graph.get(ck).p == 0.7 and gen.graph.get(ck).source == "mem"


# 7 -------------------------------------------------------------------------------------------------
def test_detections_use_the_image_time_base_pose():
    n = 20
    base = np.tile(np.eye(4), (n, 1, 1))
    base[:, 0, 3] = np.linspace(0.0, 1.9, n)  # the base drives 0.1 m per frame
    proprio = {"t": np.arange(n), "eef_pos_main": np.tile([0.5, 0, 0.3], (n, 1)), "gripper_main": np.full(n, 0.08),
               "base_pose": base}
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [0.5, 0.0, 1.0]
    meta = {"instruction": "put the bowl on the plate", "fps": 10.0, "n_frames": n, "arms": ["main"],
            "cameras": {"cam": {"depth": True}}}
    ep = Episode(meta, proprio, {"cam": {"t": np.array([0]), "K": np.eye(3) * 100, "T": T[None]}},
                 lambda c, t: (np.zeros((40, 40, 3), np.uint8), np.ones((40, 40), np.float32)))

    class Det:
        name, calls = "fixed", 0

        def detect(self, cam, cats):
            return [Detection(c.key, 0.9, (20.0, 20.0), (15, 15, 25, 25), np.array([0.6, 0.0, 0.04]), np.array([0.1, 0.1, 0.05]))
                    for c in cats]

    cfg = lm_cfg("B")
    cfg.camera = "cam"
    gen = Generator(ep, Det(), None, cfg)
    gen.categories = [c for c in __import__("capx.bggen.vocab", fromlist=["x"]).categories_for("bowl")]
    seen = {}
    real = gen.tracker.update

    def spy(t, cam, dets, base_pose=None, cats=None):
        seen["base"] = base_pose
        return real(t, cam, dets, base_pose, cats)

    gen.tracker.update = spy
    gen.step(Event(15, "base_start"), n)
    assert np.allclose(seen["base"], base[0])  # the image is from frame 0, not the event frame 15
    assert np.allclose(gen.tracker.tracks["bowl_A"].pos, [0.6, 0.0, 0.04])


# 8 -------------------------------------------------------------------------------------------------
def test_box_extent_rotated_into_world():
    tr = Tracker()
    R = np.eye(4)
    R[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # base yaw 90 deg
    d = Detection("hotdog", 0.9, (1, 1), (0, 0, 2, 2), np.array([0.5, 0.0, 0.9]), np.array([0.4, 0.1, 0.1]))
    tr.initialize(0, "head", [d], R)
    t = tr.tracks["hotdog_A"]
    assert np.allclose(t.pos, [0.0, 0.5, 0.9]) and np.allclose(t.size, [0.1, 0.4, 0.1])
    a, _ = tr.grasp_target(np.array([0.0, 0.68, 0.9]))  # 0.18 m along world y: inside the rotated box + 3 cm
    assert a == "hotdog_A"


# 9 -------------------------------------------------------------------------------------------------
def test_expectation_does_not_override_a_key_answered_unknown(tmp_path):
    w = write_swap_episode(tmp_path)

    class UnknownLater(TruthBrain):
        def answer(self, stmt, t):
            return super().answer(stmt, t) if t == 0 else "?"

    reply = json.dumps({"calls": [{"tool": "observe", "statements": ["(on bowl_A plate_B)"]}],
                        "expect": [{"statement": "(on bowl_A plate_B)", "value": 0}]})
    with FakeServer(UnknownLater(w, agent_reply=reply)) as srv:
        run_episode(tmp_path, tmp_path / "out", lm_cfg("B"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    rows = [json.loads(x) for x in (tmp_path / "out" / "rows_forward.jsonl").read_text().splitlines()]
    first_kf = next(r for r in rows if r["kind"] == "keyframe")
    assert ["(on bowl_A plate_B)", "?"] in first_kf["observed"]
    assert first_kf["belief"]["(on bowl_A plate_B)"] == [0.9, "mem"]


# 10 ------------------------------------------------------------------------------------------------
def test_goal_text_with_branches_of_different_length():
    x, y = ("on", "bowl_A", "plate_A"), ("on", "bowl_B", "plate_B")
    gt = GoalTracker([[(x,), (y,)], [(x,)]])
    g = BeliefGraph()
    g.set(x, 0.9, "obs", 1)
    gt.update(g, 1)
    assert gt.done == [1, 1]
    assert gt.text(g) == "1/2 · next (on bowl_B plate_B)"
