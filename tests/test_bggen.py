"""Tests of capx.bggen (Agent as BG Generator): synthetic fixtures, fake detector, fake model server."""

from __future__ import annotations

import ast
import builtins
import io
import json
from pathlib import Path

import numpy as np
import pytest

from capx.bggen import backward as bw
from capx.bggen.adapters import b1k, libero_mem
from capx.bggen.agent import PlanError, default_plan, parse_plan
from capx.bggen.configs import b1k as b1k_cfg
from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.events import EventConfig, Event, add_keyframes, base_events, gripper_events
from capx.bggen.evaluate import evaluate
from capx.bggen.fake_models import FakeServer
from capx.bggen.generator import Row
from capx.bggen.graph import BeliefGraph, GoalTracker
from capx.bggen.models import ChatClient, Models, RoleConfig
from capx.bggen.perception import Detection
from capx.bggen.pipeline import run_episode
from capx.bggen.records import load_episode
from capx.bggen.serialize import cot_rows, row_fields
from capx.bggen.tracks import Tracker, TrackConfig
from capx.bggen.verifier import Verifier, VerifierConfig
from capx.bggen.vocab import VER1_B1K, parse_statement, parse_values, statement_text
from tests.bggen_fakes import FakeDetector, TruthBrain, make_world, write_swap_episode

GEN_MODULES = ("records", "events", "perception", "tracks", "geometry", "graph", "verifier", "agent", "prompts",
               "models", "generator", "backward", "serialize", "pipeline", "vocab", "configs")


# --- fixtures ------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def swap_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("swap")
    w = write_swap_episode(root)
    flip_from = w.grasp_t[2] + 5
    brain = TruthBrain(w, flip=[("(on bowl_B plate_B)", flip_from, flip_from + 35)])
    with FakeServer(brain) as srv:
        s = run_episode(root, root / "out", lm_cfg("A"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    return root, root / "out", w, s, brain


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# --- adapters ------------------------------------------------------------------------------------------

B1K_TASK_OBS_KEYS = ['agent.n.01_1_real', 'agent.n.01_1_pos', 'agent.n.01_1_ori_cos', 'agent.n.01_1_ori_sin'] + [
    f"{o}_{f}" for o in ("microwave.n.02_1", "hotdog.n.02_1", "hotdog.n.02_2", "electric_refrigerator.n.01_1", "floor.n.01_1")
    for f in ("real", "pos", "ori_cos", "ori_sin", "in_gripper_left", "in_gripper_right")]
B1K_SKILLS = [  # episode_00450010 skill_annotation (subset of fields)
    ("move to", ["fridge_dszchb_0"], [0, 363]), ("open door", ["fridge_dszchb_0"], [363, 1240]),
    ("pick up from", ["hotdog_207", "fridge_dszchb_0"], [1240, 1761]), ("pick up from", ["hotdog_208", "fridge_dszchb_0"], [1761, 2226]),
    ("close door", ["fridge_dszchb_0"], [2226, 3468]), ("move to", ["microwave_abzvij_0"], [3468, 4071]),
    ("place on", ["hotdog_207", "countertop_kelker_0"], [4071, 4299]), ("open door", ["microwave_abzvij_0"], [4299, 5062]),
    ("place in", ["hotdog_208", "microwave_abzvij_0"], [5062, 5484]), ("pick up from", ["hotdog_207", "countertop_kelker_0"], [5484, 5891]),
    ("place in", ["hotdog_207", "microwave_abzvij_0"], [5895, 6205]), ("close door", ["microwave_abzvij_0"], [6205, 6415]),
    ("turn on switch", ["microwave_abzvij_0"], [6415, 6645])]


def test_b1k_depth_roundtrip():
    d = np.array([0.0, 0.3, 1.0, 2.5, 7.0, 10.0])
    back = b1k.dequantize_depth(b1k.quantize_depth(d))
    assert np.allclose(back, d, atol=2e-3 * (1 + d))


def test_b1k_cam_pose_flips_to_opencv():
    T = b1k.cam_pose([1.0, 2.0, 3.0, 0, 0, 0, 1])  # identity rotation (xyzw)
    assert np.allclose(T[:3, 3], [1, 2, 3])
    assert np.allclose(T[:3, :3], np.diag([1, -1, -1]))  # optical axis = -z of the OpenGL camera


def test_b1k_parse_state_layout():
    s = np.zeros((2, 256))
    s[:, 186:189] = [0.5, 0.1, 0.9]
    s[:, 189:193] = [0, 0, 0, 1]
    s[:, 193:195] = [0.02, 0.015]
    s[:, 140:143] = [6.4, -0.8, 0.0]
    s[1, 149] = np.pi / 2
    p = b1k.parse_state(s)
    assert np.allclose(p["eef_pos_left"][0], [0.5, 0.1, 0.9])
    assert np.allclose(p["eef_quat_left"][0], [1, 0, 0, 0])  # wxyz
    assert np.isclose(p["gripper_left"][0], 0.035)
    assert np.allclose(p["base_pose"][1][:3, 3], [6.4, -0.8, 0.0])
    assert np.allclose(p["base_pose"][1][:2, 0], [0, 1], atol=1e-9)


def test_b1k_task_info_layout_and_names():
    lay = b1k.task_info_layout(B1K_TASK_OBS_KEYS)
    assert max(e for _, e in lay.values()) == 10 + 5 * 12
    assert lay["hotdog.n.02_1_pos"] == (10 + 12 + 1, 10 + 12 + 4)
    assert b1k.category_of("hotdog.n.02_2") == "hotdog" and b1k.category_of("fridge_dszchb_0") == "fridge"
    assert b1k.category_of("electric_refrigerator.n.01_1") == "fridge" and b1k.category_of("floor.n.01_1") is None


def test_b1k_annotation_timeline():
    skills = [{"skill_description": [d], "object_id": [o], "frame_duration": f} for d, o, f in B1K_SKILLS]
    segs, tl = b1k.annotation_timeline(skills, 6805)
    assert len(segs) == 13 and segs[2]["start"] == 1240
    r0 = b1k.relations_at(tl, 0)
    assert ("in", "hotdog_207", "fridge_dszchb_0") in r0 and ("open", "fridge_dszchb_0", None) not in r0
    assert ("open", "fridge_dszchb_0", None) in b1k.relations_at(tl, 1300)
    r = b1k.relations_at(tl, 6000)
    assert ("in", "hotdog_207", "fridge_dszchb_0") not in r and ("in", "hotdog_208", "microwave_abzvij_0") in r
    assert ("toggled_on", "microwave_abzvij_0", None) in b1k.relations_at(tl, 6645)


def test_b1k_files_only_one_episode():
    f = b1k.files_for()
    assert "data/task-0045/episode_00450010.parquet" in f
    assert not any("seg_instance_id" in x for x in f)
    assert sum("videos/" in x for x in f) == 2


def test_libero_dump_structure_and_width(tmp_path):
    import h5py

    p = tmp_path / "demo.hdf5"
    with h5py.File(p, "w") as f:
        g = f.create_group("data")
        g.attrs["problem_info"] = json.dumps({"language_instruction": "swap the 2 bowls on their plates using the empty plate"})
        d = g.create_group("demo_0")
        d.create_dataset("actions", data=np.zeros((5, 7)))
        d.create_dataset("states", data=np.zeros((5, 40)))
        d.create_dataset("obs/gripper_states", data=np.array([[0.04, -0.04]] * 4 + [[0.01, -0.01]]))
        d.create_dataset("obs/ee_pos", data=np.zeros((5, 3)))
    s = libero_mem.dump_structure(p)
    assert s["demos"] == 1 and s["lengths"]["demo_0"] == 5
    assert s["datasets"]["data/demo_0/states"]["shape"] == [5, 40]
    assert s["gripper_first_demo"]["width_abs_sum"]["max"] == pytest.approx(0.08)
    assert s["gripper_source"] == "obs/gripper_states"
    assert libero_mem.instruction_of(p).startswith("swap the 2 bowls")
    rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    assert np.allclose(libero_mem.world_to_base(np.array([1.0, 1.0, 1.0]), np.array([1.0, 0, 0]), rot), rot.T @ [0, 1, 1])


def test_libero_dump_raw_layout_demo1_states_only(tmp_path):
    """kubectl 1a: LIBERO-Mem-Raw demos start at demo_1 and hold only actions / init_state / states; the dump
    used to list no dataset at all. Fingers are states[:, 8:10] (q1 negative): width = q0 - q1."""
    import h5py

    p = tmp_path / "raw.hdf5"
    st = np.zeros((418, 90))
    st[:, 8] = np.linspace(0.04, 0.0021, 418)
    st[:, 9] = -st[:, 8]
    with h5py.File(p, "w") as f:
        for d in ("demo_1", "demo_2", "demo_10"):
            g = f.create_group(f"data/{d}")
            g.create_dataset("actions", data=np.zeros((418, 7)))
            g.create_dataset("init_state", data=st[0])
            g.create_dataset("states", data=st)
    s = libero_mem.dump_structure(p)
    assert s["first_demo"] == "demo_1" and s["demos"] == 3 and libero_mem.first_demo(p) == "demo_1"
    assert set(s["datasets"]) == {"data/demo_1/actions", "data/demo_1/init_state", "data/demo_1/states"}
    assert s["gripper_source"] == "states[:, 8:10]"
    w = s["gripper_first_demo"]["width_abs_sum"]
    assert w["max"] == pytest.approx(0.08) and w["min"] == pytest.approx(0.0042)
    assert np.allclose(libero_mem.gripper_width(np.array([[0.02, -0.02], [0.02, 0.02]])), [0.04, 0.04])


def test_final_goal_relations_of_a_sequence():
    swap = [[(("on", "A", "pA"),), (("on", "B", "pB"),), (("on", "A", "pC"),)]]
    assert libero_mem.final_goal_preds(swap) == [[("on", "A", "pC"), ("on", "B", "pB")]]

    class Src:
        def goal_state(self):
            return [["sequence", ["on", "A", "pA"], ["on", "B", "pB"], ["on", "A", "pC"]]]

        def eval_predicate(self, p):
            return tuple(p) in {("on", "A", "pC"), ("on", "B", "pB")}

    g = libero_mem.goal_success(Src())
    assert g["success"] and g["branches"] == [{"on A pC": True, "on B pB": True}]


def test_episode_roundtrip_causal_and_no_truth(tmp_path):
    w = write_swap_episode(tmp_path)
    ep = load_episode(tmp_path)
    assert ep.n_frames == w.n and ep.instruction.startswith("swap")
    assert ep.image_t("agentview", 5) == 4  # images every 2 frames: the latest at or before t
    cam = ep.camera("agentview", 5)
    assert cam.has_geometry and cam.t == 4 and cam.rgb[0, 0, 2] == 7
    for name in ("oracle", "truth", "task_info", "oracle_states", "annotations"):
        with pytest.raises(AttributeError):
            getattr(ep, name)


# --- events ----------------------------------------------------------------------------------------------

def test_gripper_events_swap():
    w = make_world()
    assert [e.kind for e in gripper_events(np.array(w.width), EventConfig())] == ["close", "release"] * 3  # width alone: unknown
    ev = gripper_events(np.array(w.width), EventConfig(closed_width=0.0), None, np.stack(w.eef))
    kinds = [e.kind for e in ev]
    assert kinds == ["grasp", "release"] * 3
    for e, g in zip([e for e in ev if e.kind == "grasp"], w.grasp_t):
        assert 0 <= e.t - g <= 8
    for e, r in zip([e for e in ev if e.kind == "release"], w.release_t):
        assert abs(e.t - r) <= 2


def test_gripper_close_empty_and_open():
    width = np.r_[np.full(10, 0.08), np.linspace(0.08, 0.0, 10), np.zeros(10), np.linspace(0, 0.08, 10), np.full(10, 0.08)]
    kinds = [e.kind for e in gripper_events(width, EventConfig(closed_width=0.0))]
    assert kinds == ["close_empty", "open"]
    # LIBERO-Mem (kubectl 1a): a bowl-rim grasp stops at 4.2-5.6 mm, the same as an empty close (4.2 mm):
    # with closed_width unknown both are "close" (decided later by co-motion), never "close_empty"
    rim = np.r_[np.full(10, 0.08), np.linspace(0.08, 0.0042, 10), np.full(20, 0.0042), np.linspace(0.0042, 0.08, 10)]
    assert [e.kind for e in gripper_events(rim, EventConfig())] == ["close", "release"]
    assert [e.kind for e in gripper_events(rim, EventConfig(closed_width=0.0042))] == ["close_empty", "open"]


def test_base_debounce_merges_short_stops_and_drops_short_moves():
    fps = 30.0
    # 1.0 m drive with two 0.5 s pauses (one move), a 2 s stop, then a 3 cm shuffle (dropped)
    x = np.r_[np.zeros(30), np.linspace(0, 0.4, 40), np.full(15, 0.4), np.linspace(0.4, 0.7, 30), np.full(15, 0.7),
              np.linspace(0.7, 1.0, 30), np.full(60, 1.0), np.linspace(1.0, 1.03, 20), np.full(60, 1.03)]
    T = np.tile(np.eye(4), (len(x), 1, 1))
    T[:, 0, 3] = x
    raw = base_events(T, fps, EventConfig(base_min_stop_s=0.0, base_min_move_m=0.0))
    ev = base_events(T, fps, EventConfig())
    assert len(raw) > len(ev) and [e.kind for e in ev] == ["base_start", "base_stop"]
    assert ev[0].info["disp_m"] == pytest.approx(1.0, abs=0.05)


def test_base_events_and_keyframes():
    xs = np.r_[np.zeros(30), np.linspace(0, 1.0, 60), np.full(60, 1.0)]
    T = np.tile(np.eye(4), (len(xs), 1, 1))
    T[:, 0, 3] = xs
    ev = base_events(T, 30.0, EventConfig())
    assert [e.kind for e in ev] == ["base_start", "base_stop"]
    assert 25 <= ev[0].t <= 40 and 85 <= ev[1].t <= 100
    kf = add_keyframes([Event(100, "grasp")], 200, 20.0, 1.5)
    assert [e.t for e in kf if e.kind == "keyframe"] == [30, 60, 130, 160]


# --- verifier -------------------------------------------------------------------------------------------

def _ver(**kw):
    g = BeliefGraph()
    return g, Verifier(g, VerifierConfig(**kw))


K_ON = ("on", "bowl_B", "plate_B")


def test_verifier_obs_robot_expect_unknown():
    g, v = _ver()
    v.observe({K_ON: "1"}, 10)
    assert g.get(K_ON).p == 0.9 and g.get(K_ON).source == "obs"
    v.observe({K_ON: "?"}, 20)
    assert g.get(K_ON).p == 0.9 and g.get(K_ON).source == "mem"  # R2
    v.robot(("inhand", "bowl_A", None), True, 30)
    v.observe({("inhand", "bowl_A", None): "0"}, 30)
    assert g.get(("inhand", "bowl_A", None)).p == 0.97  # R3 wins
    assert any(d.rule == "R3-conflict" for d in v.decisions)
    v.expect(("on", "bowl_A", "plate_A"), True, 40, set())
    assert g.get(("on", "bowl_A", "plate_A")).p == 0.7 and g.get(("on", "bowl_A", "plate_A")).source == "exp"
    v.expect(K_ON, False, 40, {K_ON})  # observed now: no expectation
    assert g.get(K_ON).p == 0.9


def test_verifier_geometric_veto():
    g, v = _ver()
    cats = {"bowl_B": "bowl", "plate_B": "plate"}
    rim = {K_ON: {"mode": "depth", "d_xy": 0.066, "dz": 0.031, "r": 0.06}}
    v.observe({K_ON: "1"}, 10, set(), rim, (), cats)
    assert g.get(K_ON).p == 0.5 and g.get(K_ON).source == "conflict" and v.s4
    flat = {K_ON: {"mode": "depth", "d_xy": 0.018, "dz": 0.028, "r": 0.06}}
    v.begin()
    v.observe({K_ON: "1"}, 20, set(), flat, (), cats)
    assert g.get(K_ON).p == 0.9 and not v.s4
    g2, v2 = _ver()
    v2.observe({K_ON: "1"}, 10, set(), {K_ON: {"mode": "2d", "overlap": 0.2}}, (), cats)  # 2D path: never a veto
    assert g2.get(K_ON).p == 0.9


def test_verifier_touch_rule_recheck():
    g, v = _ver()
    v.observe({K_ON: "1"}, 500)
    v.observe({K_ON: "0"}, 620, touched=set())  # no robot contact: held
    assert g.get(K_ON).p == 0.9 and v.recheck_keys() == [K_ON]
    assert v.decisions[-1].rule == "R7-hold"
    v.observe({K_ON: "1"}, 650)  # the re-check agrees with the belief
    assert g.get(K_ON).p == 0.9 and not v.recheck_keys()
    v.observe({K_ON: "0"}, 700)
    v.observe({K_ON: "0"}, 730)  # two agreeing answers: applied
    assert g.get(K_ON).p == 0.1 and [d.rule for d in v.decisions[-2:]] == ["R7-confirm", "R1"]
    g, v = _ver()
    v.observe({K_ON: "1"}, 500)
    v.observe({K_ON: "0"}, 620, touched={"bowl_B"})  # touched: applied at once
    assert g.get(K_ON).p == 0.1
    g, v = _ver(touch_rule=False)
    v.observe({K_ON: "1"}, 500)
    v.observe({K_ON: "0"}, 620)
    assert g.get(K_ON).p == 0.1


def test_verifier_hidden_state_is_not_observed():
    g, v = _ver()
    ck = ("cooked", "hotdog_A", None)
    v.observe({ck: "1"}, 10, hidden={"hotdog_A"})
    assert g.get(ck) is None and v.decisions[-1].rule == "R2"


# --- tracker -------------------------------------------------------------------------------------------

def _det(cat, x, y, u):
    return Detection(cat, 0.9, (u, 50.0), (u - 5, 45, u + 5, 55), np.array([x, y, 0.04]), np.array([0.1, 0.1, 0.05]))


def test_tracker_aliases_left_to_right_and_event_merge():
    tr = Tracker(TrackConfig())
    tr.initialize(0, "cam", [_det("bowl", 0.6, -0.2, 120), _det("bowl", 0.6, 0.0, 80), _det("plate", 0.6, 0.2, 40)])
    assert tr.tracks["bowl_A"].pos[1] == pytest.approx(0.0) and tr.tracks["bowl_B"].pos[1] == pytest.approx(-0.2)
    a, _ = tr.grasp_target(np.array([0.6, 0.0, 0.04]))
    assert a == "bowl_A"
    tr.hold("bowl_A", "main", np.array([0.6, 0.0, 0.04]))
    tr.follow("main", np.array([0.6, 0.3, 0.2]))
    tr.release("main", 100, np.array([0.6, 0.3, 0.04]))
    # bowl_A shows up 9 cm off its release place, unmatched by distance -> named by the event
    rep = tr.update(110, "cam", [_det("bowl", 0.6, 0.39, 20), _det("bowl", 0.6, -0.2, 120), _det("plate", 0.6, 0.2, 40)])
    assert rep["new"] == [] and rep["merged"][0]["into"] == "bowl_A"
    assert sorted(tr.tracks) == ["bowl_A", "bowl_B", "plate_A"]


def test_tracker_late_merge_records_alias():
    tr = Tracker(TrackConfig(merge_radius=0.05))
    tr.initialize(0, "cam", [_det("bowl", 0.6, 0.0, 80)])
    tr.hold("bowl_A", "main", np.array([0.6, 0.0, 0.04]))
    tr.release("main", 100, np.array([0.6, 0.3, 0.04]))
    tr.update(110, "cam", [_det("bowl", 0.6, 0.38, 20)])  # too far: a new track bowl_B
    assert "bowl_B" in tr.tracks
    tr.tracks["bowl_B"].pos = np.array([0.6, 0.31, 0.04])  # it settles at the release place
    tr.update(120, "cam", [_det("bowl", 0.6, 0.31, 22)])
    assert tr.merges and tr.merges[0]["from"] == "bowl_B" and tr.merges[0]["into"] == "bowl_A"
    assert "bowl_B" not in tr.tracks


# --- agent / statements ---------------------------------------------------------------------------------

NAMES = ["bowl_A", "bowl_B", "plate_A", "plate_B", "fridge_A", "hotdog_A"]


def test_statements_and_vocab():
    assert parse_statement("(on BOWL_a plate_B)", NAMES) == ("on", "bowl_A", "plate_B")
    assert parse_statement("(ontop bowl_A table)", NAMES) == ("on", "bowl_A", "table")
    assert parse_statement("(inhand_left hotdog_A)", NAMES, ("left", "right")) == ("inhand", "hotdog_A", "left")
    assert parse_statement("(on bowl_Z plate_B)", NAMES) is None
    assert statement_text(("in", "hotdog_A", "fridge_A"), VER1_B1K) == "(inside hotdog_A fridge_A)"
    assert statement_text(("handempty", "right", None)) == "(handempty_right)"
    vals, notes, bad = parse_values("Observe: (on bowl_A plate_B) 1 | (on bowl_B plate_B) ? | (on x y) 1\nNotes: bowl_B tilted",
                                    [("on", "bowl_A", "plate_B"), ("on", "bowl_B", "plate_B")], NAMES)
    assert vals == {("on", "bowl_A", "plate_B"): "1", ("on", "bowl_B", "plate_B"): "?"} and notes == "bowl_B tilted"


def test_parse_plan_and_fallback():
    ok = json.dumps({"calls": [{"tool": "observe", "statements": ["(on bowl_A plate_A)", "(on nothing here)"]},
                               {"tool": "geometry", "pairs": [["bowl_A", "plate_A"]]}, {"tool": "look", "camera": "wrist"}],
                     "expect": [{"statement": "(on bowl_A plate_A)", "value": 1}], "why": "x"})
    p = parse_plan("```json\n" + ok + "\n```", NAMES, cameras=["agentview", "wrist"])
    assert p.observe == [("on", "bowl_A", "plate_A")] and p.geometry == [("bowl_A", "plate_A")] and p.camera == "wrist"
    assert p.dropped == ["(on nothing here)"] and p.expect == [(("on", "bowl_A", "plate_A"), True)]
    for bad in ("I would check the bowl.", '{"calls": [{"tool": "fly"}]}', '{"plan": []}',
                '{"calls": [{"tool": "observe", "statements": ["(on x y)"]}]}'):
        with pytest.raises(PlanError):
            parse_plan(bad, NAMES)
    with pytest.raises(PlanError):  # hindsight is mode A only
        parse_plan('{"calls": [{"tool": "hindsight", "statements": ["(on bowl_A plate_A)"]}]}', NAMES, offline=False)
    d = default_plan([("on", "bowl_A", "plate_A")] * 2, [], "bad")
    assert d.fallback and d.observe == [("on", "bowl_A", "plate_A")]


# --- models ---------------------------------------------------------------------------------------------

def test_role_config_from_env_only():
    env = {"NEBIUS_SERVER_URL": "https://x/v1/chat/completions", "NEBIUS_API_KEY": "k", "BGGEN_VDM_URL": "http://h:8000/v1",
           "BGGEN_VDM_MODEL": "local-qwen"}
    init, vdm = RoleConfig.from_env("init", env=env), RoleConfig.from_env("vdm", env=env)
    assert init.url == "https://x/v1/chat/completions" and init.model == "moonshotai/Kimi-K3" and init.api_key == "k"
    assert vdm.url == "http://h:8000/v1/chat/completions" and vdm.model == "local-qwen"
    assert RoleConfig.from_env("agent", env={}).url == ""


def test_chat_client_retries_and_counts(monkeypatch):
    monkeypatch.setattr("capx.bggen.models.time.sleep", lambda s: None)

    class R:
        def __init__(self, code, body=None):
            self.status_code, self._b = code, body

        def json(self):
            return self._b

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    replies = [R(429), R(200, {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                               "usage": {"prompt_tokens": 7, "completion_tokens": 2}})]
    c = ChatClient(RoleConfig("vdm", "m", "http://x/v1/chat/completions"), post=lambda *a, **k: replies.pop(0))
    assert c.query([{"role": "user", "content": "x"}])["content"] == "hi"
    assert c.usage.calls == 1 and c.usage.prompt_tokens == 7


# --- end to end (synthetic swap, fake HTTP models) --------------------------------------------------------

def test_swap_forward_matches_design(swap_run):
    root, out, w, s, brain = swap_run
    trace = _jsonl(out / "trace.jsonl")
    assert s["goal"]["final_satisfied"] and max(s["goal"]["done"]) == 3
    # LIBERO preset: width cannot tell a grasp, so closes are provisional and co-motion confirms all three
    assert s["events"]["close"] == 3 and s["events"]["release"] == 3
    assert [c["result"] for c in s["grasp_checks"]] == ["confirmed"] * 3
    # names stay with the objects (bowl_A was first on plate_B)
    assert s["init"]["notes"]["orig(bowl_A)"] == "plate_B" and s["init"]["notes"]["orig(bowl_B)"] == "plate_C"
    rel = [t for t in trace if t["event"]["kind"] == "release"]
    assert [t["event"]["obj"] for t in rel] == ["bowl_A", "bowl_B", "bowl_A"]
    # the 2nd release lands on the rim: geometric veto -> 0.5 conflict + S4, confirmed at a later keyframe
    assert any(d["rule"] == "R5" and d["key"] == "(on bowl_B plate_B)" for d in rel[1]["decisions"]) and rel[1]["s4"]
    assert rel[1]["graph"]["(on bowl_B plate_B)"] == [0.5, "conflict"]
    after = [t for t in trace if rel[1]["t"] < t["t"] < rel[2]["t"]]
    assert any(t["graph"].get("(on bowl_B plate_B)", [0])[0] == 0.9 for t in after)
    # the VDM's wrong 0 for the untouched bowl_B is held and re-checked (R7), never applied
    holds = [(t["t"], d) for t in trace for d in t.get("decisions", []) if d["rule"] == "R7-hold"]
    assert holds and all(d["key"] == "(on bowl_B plate_B)" and d["after"] == 0.9 for _, d in holds)
    nxt = next(t for t in trace if t["t"] > holds[0][0])
    assert "(on bowl_B plate_B)" in nxt["plan"]["observe"]
    # raw replies and prompts are kept
    assert all("replies" in t["agent"] for t in trace[1:] if t.get("agent"))
    assert (out / "keyframes").exists() and len(list((out / "keyframes").glob("*.png"))) == len(trace)


def test_swap_backward_and_labels(swap_run):
    root, out, w, s, brain = swap_run
    log = json.loads((out / "backward_log.json").read_text())
    assert log["forbidden_violations"] == []
    b2 = [x for x in log["log"] if x["rule"] == "B2"]
    assert b2 and all(x["key"] == "(on bowl_B plate_B)" and x["was"] == "0" and x["now"] == "1" for x in b2)
    effs = [x["effect"] for x in log["log"] if x["rule"] == "B3"]
    assert effs[0] == ["(on bowl_A plate_A) 0>1", "(on bowl_A plate_B) 1>0"]
    assert effs[1] == ["(on bowl_B plate_B) 0>1", "(on bowl_B plate_C) 1>0"]
    b4 = [(x["pick"], x["place"]) for x in log["log"] if x["rule"] == "B4"]
    assert b4 == [("pick(bowl_A, plate_B)", "place(bowl_A, plate_A)"), ("pick(bowl_B, plate_C)", "place(bowl_B, plate_B)"),
                  ("pick(bowl_A, plate_A)", "place(bowl_A, plate_C)")]
    rows = _jsonl(out / "rows_backward.jsonl")
    fwd = _jsonl(out / "rows_forward.jsonl")
    assert [r["delta"] for r in rows] == [r["delta"] for r in fwd]  # Delta never changed
    eff_row = next(r for r in rows if r["effect"] and "bowl_B" in r["effect"][0][0])
    assert eff_row["t"] > w.release_t[1]  # the rim release: Effect only once the end state is confirmed
    assert rows[0]["subtask"] == "pick(bowl_A, plate_B)"
    cot = _jsonl(out / "cot_rows.jsonl")
    ev = [c for c in cot if c["kind"] == "event"]
    assert ev[0]["text"].splitlines()[0].startswith("Subtask: pick(bowl_A, plate_B)")
    assert all(c["text"].count("\n") == 4 for c in cot)
    hz = [c for c in cot if c["kind"] == "hz"]
    assert len(hz) == int((w.n - 1) // 20.0) + 1 and hz[1]["sec"] == 1
    assert (out / "report.md").read_text().startswith("# BG generator report")


def test_swap_eval_two_lines(swap_run):
    root, out, w, s, brain = swap_run
    res = evaluate(out, root)
    assert res["alias_map"] == {"bowl_A": "bowl_1", "bowl_B": "bowl_2", "plate_A": "plate_1", "plate_B": "plate_2",
                                "plate_C": "plate_3"}
    for tag in ("forward", "forward+backward"):
        assert res[tag]["relations"]["accuracy"] >= 0.98
    assert res["name_consistency"]["rate"] == 1.0
    assert {k: res["goal_consistency"][k] for k in ("final_graph_satisfies_goal", "oracle_success", "goal_known")} == \
        {"final_graph_satisfies_goal": True, "oracle_success": True, "goal_known": True}
    assert not res["goal_consistency"]["region_goal_missing"]
    md = (out / "eval.md").read_text()
    assert "| forward |" in md and "| forward+backward |" in md


def test_agent_format_failure_falls_back(tmp_path):
    w = write_swap_episode(tmp_path)
    brain = TruthBrain(w, bad_every=3)
    with FakeServer(brain) as srv:
        s = run_episode(tmp_path, tmp_path / "out", lm_cfg("B"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    trace = _jsonl(tmp_path / "out" / "trace.jsonl")
    fb = [t for t in trace if t.get("plan", {}).get("fallback")]
    assert s["agent_fallbacks"] == len(fb) > 0
    assert all(t["plan"]["fallback_reason"] == "no JSON object in the reply" and t["plan"]["observe"] for t in fb)
    assert not (tmp_path / "out" / "rows_backward.jsonl").exists()  # mode B: forward only
    assert s["goal"]["final_satisfied"]


def test_mode_a_hindsight_used_only_for_noise(tmp_path):
    w = write_swap_episode(tmp_path)
    flip_from = w.grasp_t[2] + 5

    def agent(messages):
        text = "\n".join(p.get("text", "") for m in messages if isinstance(m["content"], list) for p in m["content"]
                         if p.get("type") == "text")
        calls = [{"tool": "observe", "statements": ["(on bowl_B plate_B)", "(on bowl_A plate_C)"]}]
        if "hindsight" in text and "keyframe" in text:
            calls.append({"tool": "hindsight", "statements": ["(on bowl_B plate_B)"], "after_s": 0.5})
        return json.dumps({"calls": calls})

    brain = TruthBrain(w, flip=[("(on bowl_B plate_B)", flip_from, flip_from + 5)], agent_reply=agent)
    cfg = lm_cfg("A")
    cfg.verifier.touch_rule = False  # the wrong answer is applied in the forward pass
    with FakeServer(brain) as srv:
        run_episode(tmp_path, tmp_path / "out", cfg, Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    fwd = _jsonl(tmp_path / "out" / "rows_forward.jsonl")
    bwd = _jsonl(tmp_path / "out" / "rows_backward.jsonl")
    wrong = [r for r in fwd if ["(on bowl_B plate_B)", "0"] in r["observed"] and r["t"] >= flip_from]
    assert wrong and wrong[0]["belief"]["(on bowl_B plate_B)"][0] == 0.1  # forward applied it (rule off)
    fixed = next(r for r in bwd if r["t"] == wrong[0]["t"])
    assert ["(on bowl_B plate_B)", "1"] in fixed["observed"] and fixed["belief"]["(on bowl_B plate_B)"][0] == 0.9
    assert any(r["hindsight"] for r in fwd)


def test_two_arm_statements_and_b1k_vocabulary(tmp_path):
    w = write_swap_episode(tmp_path)
    ep_dir = tmp_path / "episode"
    meta = json.loads((ep_dir / "meta.json").read_text())
    meta["arms"] = ["left", "right"]
    (ep_dir / "meta.json").write_text(json.dumps(meta))
    z = dict(np.load(ep_dir / "proprio.npz"))
    z["eef_pos_left"], z["gripper_left"] = z.pop("eef_pos_main"), z.pop("gripper_main")
    z["eef_pos_right"], z["gripper_right"] = np.tile([0.3, -0.4, 0.3], (w.n, 1)), np.full(w.n, 0.08)
    np.savez(ep_dir / "proprio.npz", **z)
    with FakeServer(TruthBrain(w)) as srv:
        run_episode(tmp_path, tmp_path / "out", lm_cfg("A"), Models.from_env(env=srv.env()), FakeDetector(w), "b1k")
    cot = _jsonl(tmp_path / "out" / "cot_rows.jsonl")
    text = "\n".join(c["text"] for c in cot)
    assert "(inhand_left bowl_A) 0.97 robot" in text and "(handempty_right) 0.97 robot" in text
    assert "(ontop bowl_A plate_A)" in text and "(on bowl_A" not in text


# --- backward forbidden list ------------------------------------------------------------------------------

def _row(t, observed, belief, kind="keyframe", touched=(), obj=None, arm=None, delta="0/1 · next (on a b)"):
    return Row(t, kind, arm, obj, observed, belief, [], delta, 0, 1, "?", list(touched), {"main": None})


def test_backward_never_fills_unknown_or_changes_delta():
    k = ("on", "bowl_A", "plate_A")
    rows = [_row(0, [(k, "1")], {k: (0.9, "obs")}), _row(10, [(k, "?")], {k: (0.9, "mem")}),
            _row(20, [(k, "0")], {k: (0.1, "obs")}, touched=["bowl_A"]), _row(30, [(k, "?")], {k: (0.1, "mem")})]
    res = bw.backward(rows, [], {"bowl_A": "bowl", "plate_A": "plate"})
    assert [dict(r.observed)[k] for r in res.rows] == ["1", "?", "0", "?"]
    assert [r.belief[k][0] for r in res.rows] == [0.9, 0.9, 0.1, 0.1]
    assert bw.check_forbidden(rows, res.rows) == []
    tampered = [r.copy() for r in res.rows]
    tampered[1].observed = [(k, "1")]
    tampered[3].delta = "1/1 · done"
    bad = bw.check_forbidden(rows, tampered)
    assert any("? filled" in b for b in bad) and any("Delta changed" in b for b in bad)
    hid = [r.copy() for r in res.rows]
    hid[3].belief[("cooked", "hotdog_A", None)] = (0.9, "obs")
    assert bw.check_forbidden(rows, hid, [(30, ("cooked", "hotdog_A", None))])


def test_backward_noise_needs_no_contact():
    k = ("on", "bowl_B", "plate_B")
    rows = [_row(0, [(k, "1")], {k: (0.9, "obs")}), _row(10, [(k, "0")], {k: (0.1, "obs")}),
            _row(20, [(k, "1")], {k: (0.9, "obs")})]
    res = bw.backward(rows, [], {})
    assert dict(res.rows[1].observed)[k] == "1" and res.rows[1].belief[k][0] == 0.9
    rows[1].touched = ["bowl_B"]  # the robot touched it: a real change, kept
    res = bw.backward(rows, [], {})
    assert dict(res.rows[1].observed)[k] == "0"


def test_backward_unifies_merged_ids():
    rows = [_row(0, [], {("on", "bowl_C", "plate_B"): (0.9, "obs")}), _row(10, [(("on", "bowl_C", "plate_B"), "1")], {})]
    res = bw.backward(rows, [{"t": 12, "from": "bowl_C", "into": "bowl_B"}], {})
    assert ("on", "bowl_B", "plate_B") in res.rows[0].belief and res.rows[1].observed[0][0][1] == "bowl_B"


def test_goal_tracker_branches_and_pending():
    g = BeliefGraph()
    a1, b1 = ("on", "bowl_A", "plate_A"), ("on", "bowl_B", "plate_A")
    gt = GoalTracker([[(a1,), (("on", "bowl_B", "plate_B"),)], [(b1,), (("on", "bowl_A", "plate_C"),)]])
    assert gt.text(g) == "0/2 · next (on bowl_A plate_A) or (on bowl_B plate_A)"
    g.set(a1, 0.7, "exp", 5)
    assert gt.update(g, 5) == []  # an expectation is not a confirmation
    g.set(a1, 0.9, "obs", 6)
    assert gt.update(g, 6) == [(0, 0)] and gt.text(g) == "1/2 · next (on bowl_B plate_B)"


# --- no ground truth in the generator ---------------------------------------------------------------------

def test_generator_modules_do_not_import_oracle():
    pkg = Path(__file__).resolve().parents[1] / "capx" / "bggen"
    for m in GEN_MODULES:
        tree = ast.parse((pkg / f"{m}.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = (node.module or "")
                assert "oracle" not in mod.split(".")[-1:] or mod.startswith("capx.belief"), (m, mod)
                assert not mod.endswith(("bggen.oracle", "bggen.evaluate")) and mod not in ("oracle", "evaluate"), (m, mod)
                assert not mod.endswith("bggen.adapters") and "adapters" not in mod, (m, mod)


def test_generator_never_reads_the_oracle_dir(tmp_path, monkeypatch):
    w = write_swap_episode(tmp_path)
    opened: list[str] = []
    real_open, real_load = builtins.open, np.load

    def guard(p):
        s = str(p)
        if "/oracle" in s:
            raise AssertionError(f"generator opened {s}")
        opened.append(s)

    def g_open(file, *a, **k):
        guard(file)
        return real_open(file, *a, **k)

    def g_load(file, *a, **k):
        guard(file)
        return real_load(file, *a, **k)

    monkeypatch.setattr(builtins, "open", g_open)
    monkeypatch.setattr(np, "load", g_load)
    monkeypatch.setattr(io, "open", g_open)
    with FakeServer(TruthBrain(w)) as srv:
        run_episode(tmp_path, tmp_path / "out", lm_cfg("A"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    assert any("proprio.npz" in s for s in opened) and (tmp_path / "oracle" / "states.jsonl").exists()


def test_b1k_preset_vocabulary():
    cfg = b1k_cfg()
    assert {"open", "toggled_on", "cooked"} <= set(cfg.relations) and cfg.camera == "head" and cfg.cook_s
    from capx.bggen.vocab import categories_for

    cats = [c.key for c in categories_for("Take the two hot dogs out of the refrigerator in the kitchen and cook them in "
                                          "the microwave until both are cooked.", ("countertop",))]
    assert cats == ["hotdog", "fridge", "microwave", "countertop"]


def test_serialize_one_hz_effect_window():
    k = ("on", "bowl_A", "plate_A")
    r0 = _row(0, [(k, "0")], {k: (0.1, "obs")})
    r1 = _row(25, [(k, "1")], {k: (0.9, "obs")})
    r1.effect = [(k, 0, 1)]
    r1.subtask = "place(bowl_A, plate_A)"
    rows = cot_rows([r0, r1], 20.0, n_frames=60)
    hz = [c for c in rows if c["kind"] == "hz"]
    assert [c["sec"] for c in hz] == [0, 1, 2]
    assert hz[1]["Effect"] == "none" and hz[2]["Effect"] == "(on bowl_A plate_A) 0>1"
    # a row's Subtask holds from its frame on (bggen-fix2: the backward pass puts each boundary on its event row)
    assert hz[1]["Subtask"] == r0.subtask and hz[2]["Subtask"] == "place(bowl_A, plate_A)"
    assert row_fields(r1)["Belief"].startswith("(on bowl_A plate_A) 0.90 obs")
