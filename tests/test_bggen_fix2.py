"""Regression tests for bggen-fix2 (findings of the first real-model kubectl run, exp/agent-bg-gen f6eb61e):

A. output limit: empty / cut replies are re-queried with a larger limit, finish_reason in the trace, init failure loud
B. track identity: reacquire / closed world / one-instance large objects instead of new names
C. expected objects: names of not-yet-visible objects are bound on first detection (the goal parses)
D. backward boundaries: pick -> place at the close event, the next pick from the release on
E. evaluation: oracle <-> track map over the episode, event tolerance, articulated grasps apart, two boundary refs
F. detect only the task's categories
7. runbook exit codes, SAM3 client without capx.envs
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from capx.bggen import backward as bw
from capx.bggen.agent import parse_plan
from capx.bggen.configs import b1k as b1k_cfg
from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.evaluate import TrackMap, boundary_error, contact_boundaries, name_consistency, relation_accuracy
from capx.bggen.fake_models import FakeServer
from capx.bggen.generator import Row
from capx.bggen.models import ChatClient, Models, RoleConfig
from capx.bggen.oracle import load_oracle, write_oracle
from capx.bggen.perception import Detection
from capx.bggen.pipeline import run_episode
from capx.bggen.prompts import agent_prompt
from capx.bggen.tracks import TrackConfig, Tracker
from capx.bggen.vocab import categories_for
from tests.bggen_fakes import FakeDetector, TruthBrain, write_swap_episode

REPO = Path(__file__).resolve().parents[1]


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# --- A: output limit ------------------------------------------------------------------------------------

class _R:
    def __init__(self, body):
        self.status_code, self._b = 200, body

    def json(self):
        return self._b

    def raise_for_status(self):
        pass


def _body(content, finish="stop", reasoning=0):
    return {"choices": [{"message": {"content": content, "reasoning_content": "r" * reasoning}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": reasoning + 3,
                      "completion_tokens_details": {"reasoning_tokens": reasoning}}}


def test_default_limit_is_harness_baseline_and_env_override():
    assert RoleConfig("agent", "m").max_tokens == 32768
    c = RoleConfig.from_env("init", env={"BGGEN_INIT_MAX_TOKENS": "40000", "NEBIUS_SERVER_URL": "http://x/v1"})
    assert c.max_tokens == 40000 and c.retry_max_tokens == 80000
    m = Models.from_env(env={"NEBIUS_SERVER_URL": "http://x/v1"}, max_tokens={"vdm": 1000})
    assert m.vdm.cfg.max_tokens == 1000 and m.vdm.cfg.retry_max_tokens == 65536 and m.agent.cfg.max_tokens == 32768


def test_empty_content_is_requeried_with_a_larger_limit():
    sent = []
    replies = [_R(_body("", "length", reasoning=4000)), _R(_body('{"calls": []}', "stop", reasoning=900))]

    def post(url, headers, data, timeout):
        sent.append(json.loads(data)["max_tokens"])
        return replies.pop(0)

    c = ChatClient(RoleConfig("agent", "m", "http://x/v1/chat/completions"), post=post)
    out = c.query([{"role": "user", "content": "x"}])
    assert sent == [32768, 65536]
    assert out["content"] == '{"calls": []}' and out["incomplete"] is None and out["finish_reason"] == "stop"
    assert [a["finish_reason"] for a in out["attempts"]] == ["length", "stop"]
    assert out["attempts"][0]["incomplete"] == "length" and out["attempts"][0]["reasoning_tokens"] == 4000
    assert out["attempts"][0]["reasoning_chars"] == 4000
    u = c.usage
    assert (u.calls, u.cut, u.requeries, u.incomplete, u.reasoning_tokens) == (2, 1, 1, 0, 4900)


def test_still_empty_after_requery_is_marked_incomplete():
    c = ChatClient(RoleConfig("init", "m", "http://x/v1/chat/completions"),
                   post=lambda *a, **k: _R(_body("", "stop", reasoning=10)))
    out = c.query([{"role": "user", "content": "x"}])
    assert out["incomplete"] == "empty" and len(out["attempts"]) == 2
    assert c.usage.empty == 2 and c.usage.incomplete == 1


class _CutInitBrain(TruthBrain):
    """Kimi-K3 at the old 4096 limit: init reasons until the limit and returns no content; the agent's first
    reply is empty at the default limit and fine with the larger one."""

    def init(self, messages):
        return {"content": "", "finish_reason": "length", "reasoning_content": "thinking " * 50}

    def agent(self, messages):
        if (self.max_tokens or 0) <= 32768:
            return {"content": "", "finish_reason": "length", "reasoning_content": "hmm " * 20}
        return super().agent(messages)


def test_init_failure_is_loud_and_finish_reason_is_traced(tmp_path):
    w = write_swap_episode(tmp_path)
    brain = _CutInitBrain(w)
    with FakeServer(brain) as srv:
        s = run_episode(tmp_path, tmp_path / "out", lm_cfg("B"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    assert s["init_failed"] and "init failed: no goal" in s["init_failed"] and "empty reply" in s["init_failed"]
    assert (tmp_path / "out" / "report.md").read_text().split("\n")[2].startswith("> **!!! INIT FAILED")
    u = s["models"]["init"]
    assert u["requeries"] == 2 and u["incomplete"] == 2 and u["cut"] == 4 and u["reasoning_tokens"] > 0
    trace = _jsonl(tmp_path / "out" / "trace.jsonl")
    att = trace[0]["init"]["attempts"]
    assert [a["finish_reason"] for a in att] == ["length", "length"] and len(att[0]["attempts"]) == 2
    # agent: cut at 32768, fine at the re-query -> no fallback, both attempts in the trace
    rep = next(t for t in trace[1:] if t.get("agent"))["agent"]["replies"][0]
    assert [a["finish_reason"] for a in rep["attempts"]] == ["length", "stop"] and rep["finish_reason"] == "stop"
    assert s["agent_fallbacks"] == 0


# --- B: track identity ------------------------------------------------------------------------------------

def _det(cat, x, y, z=0.04, size=(0.12, 0.12, 0.05), score=0.9):
    return Detection(cat, score, (100 - 200 * y, 100 * x), (0, 0, 1, 1), np.array([x, y, z]), np.array(size))


def _libero_tracker(cfg: TrackConfig) -> Tracker:
    tr = Tracker(cfg)
    tr.initialize(0, "cam", [_det("bowl", 0.6, 0.0), _det("bowl", 0.6, -0.2), _det("plate", 0.6, 0.2, 0.01),
                             _det("plate", 0.6, 0.0, 0.01), _det("plate", 0.6, -0.2, 0.01)])
    return tr


def test_held_bowl_far_from_prediction_keeps_its_name_in_a_closed_world():
    def run(cfg):
        tr = _libero_tracker(cfg)
        held = next(a for a, t in tr.tracks.items() if t.category == "bowl" and abs(t.pos[1]) < 0.05)
        other = next(a for a, t in tr.tracks.items() if t.category == "bowl" and a != held)
        eef = tr.tracks[held].pos + np.array([0, 0, 0.02])
        tr.hold(held, "main", eef)
        tr.follow("main", eef + np.array([0.0, 0.1, 0.15]))  # lifted and moved
        # SAM3's center of the held bowl is 0.16 m from the EEF-following prediction (rim grasp, occlusion)
        rep = tr.update(30, "cam", [_det("bowl", 0.6, 0.1 + 0.16, 0.19), _det("bowl", 0.6, -0.2)])
        return tr, rep, held, other

    tr, rep, held, other = run(lm_cfg().tracks)
    assert rep["new"] == [] and sorted(rep["matched"]) == [other] and tr.counts["closed_assigned"] == 1
    assert sorted(a for a, t in tr.tracks.items() if t.category == "bowl") == ["bowl_A", "bowl_B"]
    # the first run's behaviour (no identity rules): a new name
    tr0, rep0, *_ = run(TrackConfig(assoc_radius=0.08, merge_radius=0.10, grasp_radius=0.05))
    assert rep0["new"] == ["bowl_C"]


def test_reacquire_unseen_track_and_drop_spurious_plate():
    tr = _libero_tracker(lm_cfg().tracks)
    a = next(x for x, t in tr.tracks.items() if t.category == "bowl" and abs(t.pos[1]) < 0.05)
    # one bowl visible, 0.10 m from its track (> assoc 0.08, < reacquire 0.12), plus a plate where no plate is
    rep = tr.update(30, "cam", [_det("bowl", 0.6, 0.10), _det("plate", 0.3, 0.3, 0.01)], detected_categories=["bowl", "plate"])
    assert rep["new"] == [] and tr.counts["reacquired"] == 1 and tr.counts["closed_dropped"] == 1
    assert np.allclose(tr.tracks[a].pos[:2], [0.6, 0.10])
    # a third bowl detection in one frame with both known bowls matched is evidence of a new one
    rep = tr.update(40, "cam", [_det("bowl", 0.6, 0.10), _det("bowl", 0.6, -0.2), _det("bowl", 0.2, 0.3)])
    assert rep["new"] == ["bowl_C"] and tr.counts["new_after_init"] == 1


def test_b1k_large_fixed_objects_stay_one_track_on_a_moving_camera():
    tr = Tracker(b1k_cfg().tracks)
    big = (2.0, 0.6, 0.9)
    tr.initialize(0, "head", [_det("countertop", 1.0, 0.0, 0.9, big), _det("countertop", 1.2, 0.5, 0.9, big)])
    assert [a for a in tr.tracks] == ["countertop_A"]  # one countertop per scene
    fridge = tr.expect("fridge")
    # the head turns: another part of the counter (center 0.8 m away, inside its box), and the fridge from two sides
    rep = tr.update(10, "head", [_det("countertop", 1.6, 0.2, 0.9, big), _det("fridge", 3.0, 1.0, 0.9, (0.8, 0.7, 1.8))])
    assert rep["new"] == [] and "countertop_A" in rep["matched"] and tr.counts["bound_expected"] == 1
    rep = tr.update(20, "head", [_det("countertop", 3.5, 3.0, 0.9, big), _det("fridge", 3.9, 1.8, 0.9, (0.8, 0.7, 1.8))])
    assert rep["new"] == []
    assert sorted(tr.tracks) == ["countertop_A", fridge] and tr.counts["closed_assigned"] >= 1


# --- C: expected objects --------------------------------------------------------------------------------

def test_not_yet_visible_objects_are_expected_then_bound(tmp_path):
    """No bowl is detected in the first image (B1K: fridge / hot dogs out of view at t=0). The init goal names
    bowl_A and bowl_B anyway: they become expected objects instead of a rejected (empty) goal, and the first bowl
    detections are bound to them."""
    w = write_swap_episode(tmp_path)
    det = FakeDetector(w, hide=lambda t, name: t < 10 and name.startswith("bowl"))
    brain = TruthBrain(w)
    with FakeServer(brain) as srv:
        s = run_episode(tmp_path, tmp_path / "out", lm_cfg("A"), Models.from_env(env=srv.env()), det, "libero_mem")
    assert not s["init_failed"] and len(s["goal"]["branches"]) == 2
    trace = _jsonl(tmp_path / "out" / "trace.jsonl")
    init = trace[0]["init"]
    assert init["expected"] == ["bowl_A", "bowl_B"] and not init["errors"]
    assert "NOT visible yet" in init["prompt"] and "bowl_A (black bowl)" in init["prompt"]
    ti = s["track_identity"]
    assert ti["bound_expected"] == 2 and ti["expected_unbound"] == [] and ti["new_after_init"] == 0
    assert ti["tracks_per_category"] == {"bowl": 2, "plate": 3}


# --- D: backward boundaries -------------------------------------------------------------------------------

def _r(t, kind="keyframe", obj=None, where=None):
    belief = {("on", obj or "bowl_A", where): (0.9, "obs")} if where else {}
    return Row(t, kind, None, obj, [], belief, [], "", 0, 2, "?", [obj] if obj else [], {"main": None})


def test_backward_boundaries_at_close_and_after_release():
    rows = [_r(0, "start"), _r(30, obj=None), _r(60, "close", "bowl_A"), _r(90, "keyframe"), _r(120, "release", "bowl_A"),
            _r(150), _r(180, "close", "bowl_B"), _r(210, "release", "bowl_B"), _r(240)]
    res = bw.backward(rows, [], {"bowl_A": "bowl", "bowl_B": "bowl", "plate_A": "plate"})
    lab = {r.t: r.subtask.split("(")[0] for r in res.rows}
    assert lab == {0: "pick", 30: "pick", 60: "place", 90: "place", 120: "pick", 150: "pick", 180: "place", 210: "place",
                   240: "place"}
    assert [r.subtask for r in res.rows][120 // 30].startswith("pick(bowl_B")
    gen = [res.rows[i].t for i in range(1, len(res.rows)) if res.rows[i].subtask != res.rows[i - 1].subtask]
    assert gen == [60, 120, 180]  # the close events and the release; not the confirming keyframe (90)
    b4 = [x for x in res.log if x["rule"] == "B4"]
    assert b4[0]["pick_to_place_t"] == 60 and b4[0]["next_from_t"] == 120


# --- E: evaluation ---------------------------------------------------------------------------------------

def _oracle(root: Path):
    states = []
    for t in range(0, 101, 2):
        b1 = [0.0, 0.0, 0.0] if t < 50 else [1.0, 0.0, 0.0]
        states.append({"t": t, "pos": {"bowl_1": b1, "bowl_2": [0.5, 0.0, 0.0], "plate_1": [0.0, 0.0, -0.02]},
                       "true": [["on", "bowl_1", "plate_1"]] if t < 50 else [],
                       "inhand": {"left": "bowl_1" if 40 <= t < 50 else None, "right": None}})
    segs = [{"start": 0, "end": 40, "label": "pick up from"}, {"start": 20, "end": 30, "label": "open door"},
            {"start": 40, "end": 50, "label": "place on"}]
    write_oracle(root, {"objects": {"bowl_1": "bowl", "bowl_2": "bowl", "plate_1": "plate", "fridge_1": "fridge"},
                        "relations": ["on", "inhand", "handempty"], "arms": ["left", "right"]}, states, segs, {"success": True})
    return load_oracle(root)


def _trk(a, cat, pos, seen=True):
    return {"alias": a, "category": cat, "pos": pos, "seen_now": seen, "held_by": None}


def test_eval_maps_tracks_over_the_episode_with_tolerance_and_articulated(tmp_path):
    o = _oracle(tmp_path)
    trace = [{"t": 0, "tracks": [_trk("bowl_A", "bowl", [0, 0, 0]), _trk("bowl_B", "bowl", [0.5, 0, 0]),
                                 _trk("plate_A", "plate", [0, 0, -0.02]), _trk("fridge_A", "fridge", None, False)]},
             {"t": 60, "tracks": [_trk("bowl_A", "bowl", [0, 0, 0], False), _trk("bowl_C", "bowl", [1.0, 0.0, 0.0]),
                                  _trk("bowl_B", "bowl", [0.5, 0, 0]), _trk("plate_A", "plate", [0, 0, -0.02])]}]
    tm = TrackMap(trace, o, 0.15)
    assert tm.first() == {"bowl_A": "bowl_1", "bowl_B": "bowl_2", "plate_A": "plate_1", "fridge_A": "fridge_1"}
    assert tm.at(60)["bowl_C"] == "bowl_1"  # a renamed track is still evaluated (frame-0 map: skipped)
    nc = name_consistency(tm)
    assert nc["objects_with_several_aliases"] == {"bowl_1": ["bowl_A", "bowl_C"]}
    names = ["bowl_A", "bowl_B", "bowl_C", "plate_A", "fridge_A"]
    arms = ["left", "right"]
    rows = [{"t": 48, "belief": {"(on bowl_A plate_A)": [0.1, "obs"]}},  # 2 frames before the oracle's change
            {"t": 60, "belief": {"(on bowl_C plate_A)": [0.1, "obs"]}},
            {"t": 25, "belief": {"(inhand_left fridge_A)": [0.97, "robot"], "(handempty_left)": [0.03, "robot"]}}]
    r5 = relation_accuracy(rows, o, tm, names, arms, tol=5)
    r0 = relation_accuracy(rows, o, tm, names, arms, tol=0)
    assert r5["decided"] == 2 and r5["accuracy"] == 1.0 and r5["right_only_within_tolerance"] == 1
    assert r0["accuracy"] == 0.5
    assert r5["articulated"]["excluded_robot_keys"] == {"inhand=1": 1, "handempty=0": 1}
    assert r5["per_pred"]["on"]["right"] == 2 and "handempty" not in r5["per_pred"]
    assert contact_boundaries(o) == [40, 50]
    b = boundary_error([{"t": 0, "subtask": "a"}, {"t": 42, "subtask": "b"}, {"t": 55, "subtask": "c"}], o)
    assert b["intent"]["reference"] == [20, 40] and b["contact"]["signed"] == [2, 5]


# --- F: detect categories ---------------------------------------------------------------------------------

def test_detect_limited_to_task_categories_in_prompt_and_parser():
    cats = [c.key for c in categories_for("Take the two hot dogs out of the refrigerator and cook them in the microwave",
                                          ("countertop",))]
    reply = json.dumps({"calls": [{"tool": "detect", "categories": ["bowl", "cup", "hotdog"]},
                                  {"tool": "observe", "statements": ["(open fridge_A)"]}]})
    plan = parse_plan(reply, ["fridge_A", "hotdog_A"], categories=cats)
    assert plan.detect == ["hotdog"] and "detect 'bowl'" in plan.dropped and "detect 'cup'" in plan.dropped
    msgs = agent_prompt("e", "g", "goal", [], [], "m", [], None, ["head"], True, 10, categories=cats)
    text = msgs[1]["content"][-1]["text"]
    assert "only these categories: hotdog, fridge, microwave, countertop" in text and '"categories": ["hotdog"]' in text


# --- 7: runbook / imports ---------------------------------------------------------------------------------

def test_sam3_detector_does_not_import_capx_envs():
    code = ("import sys\nfrom capx.bggen.perception import Sam3Detector\nSam3Detector()\n"
            "import capx.bggen.pipeline, capx.bggen.cli\n"
            "bad = [m for m in sys.modules if m.startswith(('capx.envs', 'capx.integrations'))]\n"
            "assert not bad, bad\n")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_runbook_propagates_failures():
    sh = (REPO / "scripts/bggen/kubectl_runbook.sh").read_text()
    assert "set -Eeuo pipefail" in sh and "trap " in sh
    body = [ln for ln in sh.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("| head" in ln or "| tail" in ln for ln in body)  # SIGPIPE under pipefail
    assert "SAM3 server not up" in sh and "exit 3" in sh
    subprocess.run(["bash", "-n", str(REPO / "scripts/bggen/kubectl_runbook.sh")], check=True)


# --- review of 19453b8 (lead_cap-bg, 4 major) -----------------------------------------------------------

def test_articulated_exclusion_covers_only_the_arm_on_the_door(tmp_path):
    o = _oracle(tmp_path)
    trace = [{"t": 0, "tracks": [_trk("bowl_A", "bowl", [0, 0, 0]), _trk("bowl_B", "bowl", [0.5, 0, 0]),
                                 _trk("plate_A", "plate", [0, 0, -0.02]), _trk("fridge_A", "fridge", None, False)]}]
    tm = TrackMap(trace, o, 0.15)
    rows = [{"t": 25, "belief": {"(inhand_left fridge_A)": [0.97, "robot"], "(handempty_left)": [0.03, "robot"],
                                 # the right arm is wrong (it holds nothing in the oracle): still scored
                                 "(handempty_right)": [0.03, "robot"], "(inhand_right bowl_A)": [0.97, "robot"]}}]
    r = relation_accuracy(rows, o, tm, ["bowl_A", "bowl_B", "plate_A", "fridge_A"], ["left", "right"], tol=5)
    assert r["articulated"]["excluded_robot_keys"] == {"inhand=1": 1, "handempty=0": 1}
    assert r["per_pred"]["handempty"]["wrong"] == 1 and r["per_pred"]["inhand"]["wrong"] == 1 and r["accuracy"] == 0.0


def test_relation_accuracy_uses_one_map_per_episode(tmp_path):
    states = [{"t": t, "pos": {"bowl_1": [0, 0, 0], "bowl_2": [0.5, 0, 0], "plate_1": [0, 0, -0.02]},
               "true": [["on", "bowl_1", "plate_1"]], "inhand": {"main": None}} for t in range(0, 101, 2)]
    write_oracle(tmp_path, {"objects": {"bowl_1": "bowl", "bowl_2": "bowl", "plate_1": "plate"}, "relations": ["on"],
                            "arms": ["main"]}, states, [], {"success": True})
    o = load_oracle(tmp_path)
    plate = _trk("plate_A", "plate", [0, 0, -0.02])
    trace = [{"t": t, "tracks": [_trk("bowl_A", "bowl", [0, 0, 0]), _trk("bowl_B", "bowl", [0.5, 0, 0]), plate]}
             for t in (0, 30, 60)]
    # the tracker swapped the names at t=90
    trace.append({"t": 90, "tracks": [_trk("bowl_A", "bowl", [0.5, 0, 0]), _trk("bowl_B", "bowl", [0, 0, 0]), plate]})
    tm = TrackMap(trace, o, 0.15)
    assert tm.fixed() == {"bowl_A": "bowl_1", "bowl_B": "bowl_2", "plate_A": "plate_1"}
    rows = [{"t": 90, "belief": {"(on bowl_A plate_A)": [0.1, "obs"], "(on bowl_B plate_A)": [0.9, "obs"]}}]
    names = ["bowl_A", "bowl_B", "plate_A"]
    assert relation_accuracy(rows, o, tm, names, [None], tol=0)["accuracy"] == 0.0  # main: the swap is wrong
    assert relation_accuracy(rows, o, tm, names, [None], tol=0, remap=True)["accuracy"] == 1.0  # auxiliary
    nc = name_consistency(tm)
    assert nc["map_changes"] == 2 and nc["rate"] < 1.0


def test_zero_distance_duplicate_mask_is_dropped_not_given_to_another_object():
    big = (2.0, 0.6, 0.9)
    tr = Tracker(TrackConfig(assoc_radius=0.15, large_extent=0.4, closed_world=True))
    tr.initialize(0, "head", [_det("countertop", 0.0, 0.0, 0.9, big), _det("countertop", 5.0, 0.0, 0.9, big)])
    far = next(a for a, t in tr.tracks.items() if t.pos[0] > 1)
    pos = tr.tracks[far].pos.copy()
    # two masks of the near counter, both inside its box (box gap 0)
    rep = tr.update(10, "head", [_det("countertop", 0.3, 0.0, 0.9, big),
                                 _det("countertop", -0.3, 0.0, 0.9, big, score=0.8)])
    assert rep["new"] == [] and tr.counts["duplicates_dropped"] == 1 and tr.counts["closed_assigned"] == 0
    assert np.allclose(tr.tracks[far].pos, pos) and not tr.tracks[far].seen_now


def test_reacquire_is_a_joint_assignment_not_score_greedy():
    tr = _libero_tracker(lm_cfg().tracks)
    a = next(x for x, t in tr.tracks.items() if t.category == "bowl" and abs(t.pos[1]) < 0.05)  # y 0
    b = next(x for x, t in tr.tracks.items() if t.category == "bowl" and x != a)  # y -0.2
    # d1 (high score) is ambiguous: 0.105 from A, 0.095 from B; d2 is 0.09 from B only. Greedy by score gave B to
    # d1 and pushed d2 onto A (0.29 away) through the closed world; jointly d1 -> A, d2 -> B.
    rep = tr.update(30, "cam", [_det("bowl", 0.6, -0.105, score=0.95), _det("bowl", 0.6, -0.29, score=0.6)],
                    detected_categories=["bowl"])
    assert rep["new"] == [] and tr.counts["reacquired"] == 2 and tr.counts["closed_assigned"] == 0
    assert np.isclose(tr.tracks[a].pos[1], -0.105) and np.isclose(tr.tracks[b].pos[1], -0.29)
