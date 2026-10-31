"""Estimated Belief Graph (bg_est) without a simulator: fake detections (noise, misses, two same-shape
bowls), fake hook records from a fake gripper world, scripted goal LLM and verification VDM."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from capx.belief import ApiCallRecorder, BeliefEstimator, EstConfig, FinishGate, SimPerceptionSource
from capx.belief.goal import GoalParseError, goal_from_bddl, parse_goal_json, parse_goal_llm
from capx.belief.perception import Sam3PerceptionSource, categories_from_instruction, mask_points
from capx.belief.signals import SignalContext, est_detectors
from capx.belief.verify import VdmVerifier, parse_verify_reply
from tests.belief_fakes import (
    REPEAT_GOAL_JSON,
    SWAP_GOAL_JSON,
    FakeDetector,
    FakeWorld,
    ScriptedModel,
    repeat_world,
    swap_world,
)

SWAP = "swap the 2 bowls on their plates using the empty plate"
REPEAT = "lift the bowl and place it back on the plate 3 times"
SIM_NAMES = ("akita_black_bowl", "plate_1", "plate_2", "plate_3")


def setup(world: FakeWorld, goal_json: str, instruction: str, *, verifier=None, noise: float = 0.0, **cfg):
    det = FakeDetector(world, noise=noise)
    rec = ApiCallRecorder(world.proprio, lambda: world.steps)
    world.api = rec.wrap_functions(world.functions())
    est = BeliefEstimator(det, EstConfig(**cfg), goal_query_fn=ScriptedModel(goal_json), verifier=verifier)
    est.start(world.get_observation(), instruction)
    return est, det, rec


def end(est, world, rec, turn):
    return est.end_turn(turn, world.get_observation(), rec.drain())


def identity(est, world) -> dict[str, str]:
    """alias -> world object it currently sits on (test-only ground truth check)."""
    out = {}
    for a, t in est.tracks.items():
        same = [n for n in world.objects if world.objects[n]["cat"] == t.category.key]
        out[a] = min(same, key=lambda n: np.linalg.norm(world.objects[n]["pos"] - t.pos))
    return out


# --- perception / categories -------------------------------------------------------------------

def test_categories_from_instruction():
    keys = lambda s: [c.key for c in categories_from_instruction(s)]  # noqa: E731
    assert keys(SWAP) == ["bowl", "plate"]
    assert keys("put the cream cheese in the nearest basket and place that basket in the center") == ["cream_cheese", "basket"]
    assert keys("pick up the bottle and put it down the plate 3 times") == ["bottle", "plate"]
    assert keys(SWAP + " and the bottle", ) == ["bowl", "plate", "bottle"]
    assert len(categories_from_instruction("bowl plate bottle basket mug cream cheese", max_categories=3)) == 3


def test_sam3_source_lifts_masks_to_3d():
    obs = FakeWorld({}).get_observation()
    obs["agentview"]["images"]["depth"] = np.full((8, 8), 1.0, np.float32)
    mask = np.zeros((8, 8), bool)
    mask[3:5, 3:5] = True
    calls = []

    def seg(rgb, text_prompt):
        calls.append(text_prompt)
        return [{"mask": mask, "score": 0.9}, {"mask": mask, "score": 0.8}, {"mask": ~mask, "score": 0.1}]

    src = Sam3PerceptionSource(segment_fn=seg, min_points=1)
    dets = src.detect(obs, categories_from_instruction(SWAP))
    assert calls == ["black bowl", "plate"] and src.calls == 2
    assert len(dets) == 2  # duplicate mask suppressed, low score dropped
    pts = mask_points(mask, obs["agentview"]["images"]["depth"], obs["agentview"]["intrinsics"], obs["agentview"]["pose_mat"])
    assert np.allclose(dets[0].center, np.median(pts, axis=0))
    assert dets[0].center[2] == pytest.approx(0.5)  # camera 1.5 m above, looking down, depth 1 m


def test_sim_fake_detections_carry_no_sim_names(capsys):
    truth = SimpleNamespace(
        movable_objects=lambda: ["akita_black_bowl_1", "plate_1", "wooden_cabinet_1"],
        positions=lambda: {"akita_black_bowl_1": (0.5, 0, 0.02), "plate_1": (0.5, 0, 0), "wooden_cabinet_1": (0, 0, 0)},
    )
    src = SimPerceptionSource(truth, dropout=0.0, occlusion=0.0)
    assert "perception=sim_fake" in capsys.readouterr().err
    dets = src.detect({}, categories_from_instruction(SWAP))
    assert sorted(d.category for d in dets) == ["bowl", "plate"]
    assert not any("akita" in repr(vars(d)) or "plate_1" in repr(vars(d)) for d in dets)


# --- hooks -------------------------------------------------------------------------------------

def test_hooks_record_and_do_not_change_calls():
    state = {"g": 1.0, "step": 0}

    def close_gripper():
        state["g"], state["step"] = 0.3, state["step"] + 30
        return "ret"

    def goto_pose(position, quaternion_wxyz=None):
        raise ValueError("ik failed")

    rec = ApiCallRecorder(lambda: (np.zeros(3), state["g"]), lambda: state["step"])
    fns = rec.wrap_functions({"close_gripper": close_gripper, "goto_pose": goto_pose, "solve_ik": len})
    assert fns["solve_ik"] is len  # not hooked
    assert fns["close_gripper"]() == "ret"
    with pytest.raises(ValueError, match="ik failed"):
        fns["goto_pose"](np.array([0.5, 0.0, 0.3]))
    r1, r2 = rec.drain()
    assert (r1.fn, r1.gripper_before, r1.gripper_after, r1.step_before, r1.step_after) == ("close_gripper", 1.0, 0.3, 0, 30)
    assert r2.error == "ValueError" and "0.5" in r2.args
    assert rec.drain() == []
    assert rec.wrap("close_gripper", fns["close_gripper"]) is fns["close_gripper"]  # no double wrapping


def test_hook_proprio_failure_does_not_break_call():
    rec = ApiCallRecorder(lambda: 1 / 0, lambda: None)
    assert rec.wrap("open_gripper", lambda: 7)() == 7
    assert rec.drain()[0].eef_after is None


# --- goal parsing -------------------------------------------------------------------------------

NAMES = ["bowl_A", "bowl_B", "plate_A", "plate_B", "plate_C"]


def test_goal_json_or_and_repeat():
    branches = parse_goal_json(SWAP_GOAL_JSON, NAMES)
    assert len(branches) == 2 and branches[0][0] == (("on", "bowl_A", "plate_C"),)
    rep = parse_goal_json("```json\n" + REPEAT_GOAL_JSON + "\n```", NAMES)
    assert rep == [[(("on", "bowl_A", "plate_A"),)] * 3]
    table = parse_goal_json('{"goal": {"steps": [[["on", "bowl_A", "table"]], [["inhand", "bowl_B"]]]}}', NAMES)
    assert table == [[(("on", "bowl_A", "table"),), (("inhand", "bowl_B"),)]]
    for bad in ('{"goal": {"steps": [[["on", "bowl_Z", "plate_A"]]]}}', '{"goal": {"steps": []}}',
                '{"goal": {"steps": [{"preds": [["near", "bowl_A", "plate_A"]]}]}}', "no json here",
                '{"goal": {"steps": [{"preds": [["on", "bowl_A", "plate_A"]], "repeat": 99}]}}'):
        with pytest.raises(GoalParseError):
            parse_goal_json(bad, NAMES)


def test_goal_llm_retries_once_then_unknown():
    model = ScriptedModel("not json", REPEAT_GOAL_JSON)
    res = parse_goal_llm(model, REPEAT, "- bowl_A ...", NAMES)
    assert res.branches is not None and res.attempts == 2 and len(model.prompts) == 2
    assert "could not be used" in model.prompts[1][-1]["content"][0]["text"]
    bad = parse_goal_llm(ScriptedModel("nope"), REPEAT, "-", NAMES)
    assert bad.branches is None and bad.attempts == 2 and len(bad.errors) == 2


def test_goal_parse_failure_gives_unknown_progress_and_no_s5():
    world = repeat_world()
    est, _, rec = setup(world, "garbage", REPEAT)
    assert est.goal.branches is None
    world.pick_place("akita_black_bowl_1", "plate_1")
    r = end(est, world, rec, 1)
    assert not r.progress.known and r.suppress_completion
    assert "progress unknown (the goal could not be parsed" in est.render()
    assert FinishGate().check(r.progress, suppress=r.suppress_completion) is None


def test_goal_from_bddl_maps_names():
    res = goal_from_bddl([["sequence", ["and", ["in", "cream_cheese_1", "basket_1_contain_region"]],
                           ["and", ["on", "basket_1", "kitchen_table_the_center"]]]],
                         {"cream_cheese_1": "cream_cheese_A", "basket_1": "basket_B"})
    assert res.source == "bddl"
    assert res.branches == [[(("in", "cream_cheese_A", "basket_B"),), (("on", "basket_B", "kitchen_table_the_center"),)]]


# --- data association / identity ----------------------------------------------------------------

def test_aliases_follow_y_order_and_swap_keeps_names():
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP, noise=0.005)
    assert identity(est, world) == {"bowl_A": "akita_black_bowl_1", "bowl_B": "akita_black_bowl_2",
                                    "plate_A": "plate_1", "plate_B": "plate_2", "plate_C": "plate_3"}
    first = dict(identity(est, world))
    plan = [("akita_black_bowl_1", "plate_3"), ("akita_black_bowl_2", "plate_1"), ("akita_black_bowl_1", "plate_2")]
    for turn, (bowl, plate) in enumerate(plan, 1):
        world.pick_place(bowl, plate)
        r = end(est, world, rec, turn)
        assert identity(est, world) == first  # no identity swap between the two same-shape bowls
        assert len(r.progress.done) == turn and not r.new_tracks
    assert r.progress.complete and not r.suppress_completion
    assert [e.text for e in est.graph.events][:2] == [
        "t1 bowl_A on plate_A -> in gripper (exp)", "t1 bowl_A in gripper -> on plate_C (exp+obs)"]


def test_both_bowls_moved_without_hooks_rematch_by_distance():
    """Code that moves objects without hooked calls: pass-2 re-matching (no new nodes) still keeps
    the names when each bowl moved to the nearest free spot."""
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    world.objects["akita_black_bowl_2"]["pos"] = np.array([0.5, 0.2, 0.02])  # bowl_B -> plate_C, no hooks
    r = end(est, world, rec, 1)
    assert not r.new_tracks and r.rematched == ["bowl_B"]
    assert identity(est, world)["bowl_B"] == "akita_black_bowl_2"
    assert [e.text for e in r.events] == ["t1 bowl_B on plate_B -> on plate_C (obs)"]


def test_missed_detection_keeps_memory_then_stale_s4():
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    det.miss_always = {"akita_black_bowl_1"}
    r1 = end(est, world, rec, 1)
    assert r1.unmatched_tracks == ["bowl_A"] and est.graph.edges[("on", "bowl_A", "plate_A")].source == "mem"
    assert "bowl_A (black bowl) at (0.50, -0.20, 0.02) on plate_A 0.97 mem, not seen since t0" in est.render()
    assert not any(u.pred.startswith("bowl_A") for u in r1.uncertain)  # 1 turn: not stale yet
    r2 = end(est, world, rec, 2)
    stale = [u for u in r2.uncertain if u.reason == "stale"]
    assert stale and stale[0].pred == "bowl_A on plate_C" and r2.suppress_completion
    sig = [s for s in (d.detect(_ctx(est, r2)) for d in est_detectors()) if s]
    assert [s.id for s in sig] == ["S2", "S4"] and "goto_home_joint_position" in sig[1].text


def _ctx(est, r, prev=None):
    return SignalContext(r.turn, est.graph, r.progress, prev or r.progress, r.events,
                         violations=est.violations, uncertain=r.uncertain, suppress_completion=r.suppress_completion)


# --- repeat counter -------------------------------------------------------------------------------

def test_repeat_counter_counts_pick_and_place_back_within_a_turn():
    world = repeat_world()
    est, det, rec = setup(world, REPEAT_GOAL_JSON, REPEAT)
    assert est.progress_text().startswith("Progress: 0/3 repetitions done (bowl_A on plate_A)")
    for turn in (1, 2):
        world.pick_place("akita_black_bowl_1", "plate_1")
        r = end(est, world, rec, turn)
        assert len(r.progress.done) == turn
    # Turn 3 picks and places back twice: counts as two (the end state alone never changes).
    world.pick_place("akita_black_bowl_1", "plate_1")
    world.pick_place("akita_black_bowl_1", "plate_1")
    r = end(est, world, rec, 3)
    assert len(r.progress.done) == 3 and r.progress.complete


def test_repeat_needs_a_lift():
    world = repeat_world()
    est, det, rec = setup(world, REPEAT_GOAL_JSON, REPEAT)
    p = world.objects["akita_black_bowl_1"]["pos"].copy()
    world.api["goto_pose"](p + [0, 0, 0.02])
    world.api["close_gripper"]()
    world.api["open_gripper"]()  # grasped and released without lifting
    r = end(est, world, rec, 1)
    assert r.progress.done == [] and r.events == []


def test_state_only_turn_does_not_count_repeat():
    """No hooks (moved and put back outside the API): the state alone cannot show a repetition."""
    world = repeat_world()
    est, det, rec = setup(world, REPEAT_GOAL_JSON, REPEAT)
    r = end(est, world, rec, 1)
    assert r.progress.done == []


# --- S1 / S4 / S5 -------------------------------------------------------------------------------

def test_s1_drop_while_holding():
    world = repeat_world()
    est, det, rec = setup(world, REPEAT_GOAL_JSON, REPEAT)
    world.slip_next_lift = True
    world.pick_place("akita_black_bowl_1", "plate_2")
    r = end(est, world, rec, 1)
    assert len(r.violations) == 1 and "object dropped" in r.violations[0].observed
    assert r.violations[0].call.startswith("goto_pose (sim step")
    sig = [s for s in (d.detect(_ctx(est, r)) for d in est_detectors()) if s]
    assert sig[0].id == "S1" and "expected bowl_A in gripper" in sig[0].text
    assert "Violations: t1 goto_pose" in est.render()


def test_s1_expected_place_contradicted_by_geometry():
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    world.pick_place("akita_black_bowl_1", "plate_3")
    world.objects["akita_black_bowl_1"]["pos"] = np.array([0.3, 0.2, 0.0])  # rolled off the plate
    r = end(est, world, rec, 1)
    assert [v.expected for v in r.violations] == ["bowl_A on plate_C"]
    assert "bowl_A on table" in r.violations[0].observed
    assert r.progress.done == [] and r.rolled_back == ["bowl_A on plate_C"]  # hook count withdrawn
    assert len(r.violations) == 1  # the withdrawal reuses the place violation


def test_s5_suppressed_when_goal_predicate_uncertain():
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    for turn, (b, p) in enumerate([("akita_black_bowl_1", "plate_3"), ("akita_black_bowl_2", "plate_1")], 1):
        world.pick_place(b, p)
        end(est, world, rec, turn)
    world.pick_place("akita_black_bowl_1", "plate_2")
    det.miss = {"akita_black_bowl_1"}  # the last placed bowl is not seen: only the expectation (0.7)
    r = end(est, world, rec, 3)
    assert r.progress.complete
    assert r.pending == [2]  # counted from the hooks, the bowl not seen at the end of the turn
    assert [u.reason for u in r.uncertain] == ["unconfirmed"]  # the same predicate is not listed twice
    assert r.uncertain[0].pred.startswith("bowl_A on plate_B (step 3") and r.uncertain[0].p == pytest.approx(0.7)
    assert "1 counted step(s) not yet confirmed" in est.progress_text()
    assert r.suppress_completion
    assert "S5b" not in [s.id for s in (d.detect(_ctx(est, r)) for d in est_detectors()) if s]
    assert FinishGate().check(r.progress, suppress=True) is None
    assert "estimate uncertain (see S4)" in est.progress_text(uncertain=True)
    # Seen again next turn: certain -> S5b.
    r4 = end(est, world, rec, 4)
    assert not r4.suppress_completion
    assert "S5b" in [s.id for s in (d.detect(_ctx(est, r4)) for d in est_detectors()) if s]


def test_s5a_rejects_when_certain_and_steps_remain():
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    world.pick_place("akita_black_bowl_1", "plate_3")
    r = end(est, world, rec, 1)
    assert not r.suppress_completion
    sig = FinishGate().check(r.progress, suppress=r.suppress_completion)
    assert sig is not None and "2 step(s) remain: bowl_B on plate_A; then bowl_A on plate_B" in sig.text


# --- verification VDM --------------------------------------------------------------------------

def test_verify_fills_unseen_and_conflict_rule():
    world = swap_world()
    vdm = ScriptedModel(json.dumps({"answers": [{"id": 1, "answer": "true"}, {"id": 2, "answer": "false"}]}))
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP, verifier=VdmVerifier(vdm))
    world.pick_place("akita_black_bowl_1", "plate_3")
    det.miss = {"akita_black_bowl_1"}
    r = end(est, world, rec, 1)
    assert [h["edge"] for h in r.hypotheses][:1] == ["bowl_A on plate_C"]
    assert len(vdm.prompts) == 1 and r.verify["ok"]
    content = vdm.prompts[0][1]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 2  # before and after
    assert "bowl_A: a black bowl, near pixel" in content[-1]["text"]
    edge = est.graph.edges[("on", "bowl_A", "plate_C")]
    assert (edge.p, edge.source, est.origin[edge.key]) == (pytest.approx(0.9), "obs", "vdm")
    # Conflict: geometry (seen, P 0.97) beats a VDM 'false' (0.9); recorded.
    world.pick_place("akita_black_bowl_2", "plate_1")
    vdm.replies = [json.dumps({"answers": [{"id": 1, "answer": "false"}]})]
    r2 = end(est, world, rec, 2)
    assert r2.conflicts and r2.conflicts[0]["chosen"] == "geo"
    assert est.graph.edges[("on", "bowl_B", "plate_A")].p == pytest.approx(0.97)


def test_verify_bad_reply_is_ignored():
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP, verifier=VdmVerifier(ScriptedModel("I think so")))
    world.pick_place("akita_black_bowl_1", "plate_3")
    r = end(est, world, rec, 1)
    assert r.verify["ok"] is False and r.verify["error"]
    assert est.graph.edges[("on", "bowl_A", "plate_C")].p == pytest.approx(0.97)
    assert parse_verify_reply('{"answers": [{"id": 1, "answer": "Not visible"}]}', [1]) == {1: "not_visible"}


# --- serialization ------------------------------------------------------------------------------

def test_render_format_and_budget():
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    world.pick_place("akita_black_bowl_1", "plate_3")
    end(est, world, rec, 1)
    text = est.render()
    lines = text.split("\n")
    assert lines[0].startswith("Belief graph (turn 1), ESTIMATED")
    assert lines[1].startswith("Progress: step 1/3 done (bowl_A on plate_C)")
    assert lines[2].startswith("Objects (estimated xyz in robot base frame, m): bowl_A (black bowl) at (0.50, 0.20, 0.02) on plate_C 0.97 obs")
    assert lines[3].startswith("Events: t1 bowl_A on plate_A -> in gripper (exp)")
    assert lines[4:] == ["Violations: none", "Signals: none"]
    small = est.render(max_chars=len(text) - 40)
    assert "earlier omitted)" in small and "Objects" in small and len(small) <= len(text) - 40
    tiny = est.render(max_chars=100)
    assert "Objects" not in tiny and tiny.split("\n")[-2:] == ["Violations: none", "Signals: none"]
    assert not any(n in text for n in SIM_NAMES)


# --- no simulator access ----------------------------------------------------------------------

class SimForbidden:
    """A low-level env stand-in: robot observation and step counter only; anything else fails."""

    def __init__(self, world: FakeWorld) -> None:
        object.__setattr__(self, "_world", world)

    def get_observation(self):
        return self._world.get_observation()

    @property
    def _sim_step_count(self):
        return self._world.steps

    def __getattr__(self, name):
        raise AssertionError(f"estimated graph read simulator state: {name}")


def test_estimation_path_reads_no_sim_state():
    world = swap_world()
    env = SimForbidden(world)
    from capx.belief.perception import proprio_from_obs

    rec = ApiCallRecorder(lambda: proprio_from_obs(env.get_observation()), lambda: env._sim_step_count)
    world.api = rec.wrap_functions(world.functions())
    vdm = ScriptedModel('{"answers": []}')
    est = BeliefEstimator(FakeDetector(world), goal_query_fn=ScriptedModel(SWAP_GOAL_JSON), verifier=VdmVerifier(vdm))
    est.start(env.get_observation(), SWAP)
    for turn, (b, p) in enumerate([("akita_black_bowl_1", "plate_3"), ("akita_black_bowl_2", "plate_1"),
                                   ("akita_black_bowl_1", "plate_2")], 1):
        world.pick_place(b, p)
        r = est.end_turn(turn, env.get_observation(), rec.drain())
    assert r.progress.complete
    with pytest.raises(AssertionError):
        env.sim  # noqa: B018 - the guard itself works


# --- review fixes (L1 Phase A review) -------------------------------------------------------------

def test_both_bowls_moved_together_and_back_keep_names():
    """Swap scene: both bowls moved in one turn (hooked), one of them missed by the detector, then
    both moved back; the global assignment never hands a moved bowl's detection to the other."""
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP, noise=0.01)
    first = dict(identity(est, world))
    world.pick_place("akita_black_bowl_1", "plate_3")
    world.pick_place("akita_black_bowl_2", "plate_1")
    det.miss = {"akita_black_bowl_2"}
    r1 = end(est, world, rec, 1)
    assert identity(est, world) == first and not r1.new_tracks
    world.pick_place("akita_black_bowl_2", "plate_2")
    world.pick_place("akita_black_bowl_1", "plate_1")
    r2 = end(est, world, rec, 2)
    assert identity(est, world) == first and not r2.new_tracks
    r3 = end(est, world, rec, 3)  # no motion, both seen
    assert identity(est, world) == first and not r3.rematched


def test_global_assignment_prefers_overall_nearest():
    """Two same-shape bowls both shift, one by almost the gate distance: the per-category assignment
    keeps the names (the pair choice is made over all tracks and detections at once)."""
    world = FakeWorld({"plate_1": ("plate", (0.5, -0.3, 0.0)), "b1": ("bowl", (0.5, 0.0, 0.02)),
                       "b2": ("bowl", (0.5, 0.1, 0.02))})
    est, det, rec = setup(world, SWAP_GOAL_JSON, "move the bowl to the plate")
    world.objects["b1"]["pos"] = np.array([0.5, 0.08, 0.02])
    world.objects["b2"]["pos"] = np.array([0.5, 0.16, 0.02])
    r = end(est, world, rec, 1)
    assert identity(est, world) == {"bowl_A": "b1", "bowl_B": "b2", "plate_A": "plate_1"}
    assert not r.new_tracks and not r.rematched


def test_step_counter_consecutive_steps():
    from capx.belief.estimator import StepCounter

    s1, s2 = (("on", "bowl_A", "plate_C"),), (("on", "bowl_B", "plate_A"),)
    none = {"bowl_A": frozenset({("on", "plate_A")}), "bowl_B": frozenset({("on", "plate_B")})}
    first = {"bowl_A": frozenset({("on", "plate_C")}), "bowl_B": frozenset({("on", "plate_B")})}
    both = {"bowl_A": frozenset({("on", "plate_C")}), "bowl_B": frozenset({("on", "plate_A")})}
    c = StepCounter([[s1, s2]])
    c.feed(none)
    assert c.feed(first) == [s1] and c.feed(both) == [s2]  # the next step right after a count
    c2 = StepCounter([[s1, s2]])
    c2.feed(none)
    assert c2.feed(both) == [s1, s2]  # two steps completed between snapshots
    rep = StepCounter([[s1, s1]])
    rep.feed(none)
    assert rep.feed(first) == [s1] and rep.feed(first) == []  # a repeat must go false again
    assert rep.feed(none) == [] and rep.feed(first) == [s1]


def test_repeat_place_that_rolls_off_is_withdrawn():
    world = repeat_world()
    est, det, rec = setup(world, REPEAT_GOAL_JSON, REPEAT)
    world.pick_place("akita_black_bowl_1", "plate_1")
    r1 = end(est, world, rec, 1)
    assert len(r1.progress.done) == 1 and r1.pending == []
    world.pick_place("akita_black_bowl_1", "plate_1")
    world.objects["akita_black_bowl_1"]["pos"] = np.array([0.35, -0.2, 0.0])  # rolled off onto the table
    r2 = end(est, world, rec, 2)
    assert len(r2.progress.done) == 1 and r2.rolled_back == ["bowl_A on plate_A"]
    assert [v.expected for v in r2.violations] == ["bowl_A on plate_A"]
    sig = [s for s in (d.detect(_ctx(est, r2)) for d in est_detectors()) if s]
    assert sig[0].id == "S1"
    # Put back properly next turn: counts (the bowl was off the plate in between).
    world.pick_place("akita_black_bowl_1", "plate_1")
    r3 = end(est, world, rec, 3)
    assert len(r3.progress.done) == 2 and r3.pending == []


def test_two_repeats_in_one_turn_second_rolls_off():
    world = repeat_world()
    est, det, rec = setup(world, REPEAT_GOAL_JSON, REPEAT)
    world.pick_place("akita_black_bowl_1", "plate_1")
    world.pick_place("akita_black_bowl_1", "plate_1")
    world.objects["akita_black_bowl_1"]["pos"] = np.array([0.35, -0.2, 0.0])
    r = end(est, world, rec, 1)
    # The first place-back still counts but has no end-of-turn evidence: pending, S5 suppressed.
    assert len(r.progress.done) == 1 and r.rolled_back == ["bowl_A on plate_A"]
    assert r.pending == [0] and r.suppress_completion


def test_s5_blocked_by_any_unconfirmed_counted_step():
    """Step 1 counted from the hooks but never seen: it blocks S5 until a later confirmed count
    vouches for it."""
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    world.pick_place("akita_black_bowl_1", "plate_3")
    det.miss = {"akita_black_bowl_1"}
    r1 = end(est, world, rec, 1)
    assert r1.pending == [0] and r1.suppress_completion
    assert FinishGate().check(r1.progress, suppress=r1.suppress_completion) is None
    world.pick_place("akita_black_bowl_2", "plate_1")
    r2 = end(est, world, rec, 2)
    assert r2.pending == [] and len(r2.progress.done) == 2  # bowl_A seen again: step 1 confirmed
    assert r2.confirmed_by == ["obs", "obs"]


def test_grasp_missed_near_object_is_s1():
    world = repeat_world()
    est, det, rec = setup(world, REPEAT_GOAL_JSON, REPEAT)
    p = world.objects["akita_black_bowl_1"]["pos"]
    world.api["goto_pose"](p + [0.07, 0, 0.0])  # 7 cm off: the fake gripper closes on air
    world.api["close_gripper"]()
    r = end(est, world, rec, 1)
    assert len(r.violations) == 1
    v = r.violations[0]
    assert v.call.startswith("close_gripper") and v.expected.startswith("bowl_A in gripper")
    assert "nothing grasped" in v.observed
    world.api["goto_pose"](np.array([0.2, 0.4, 0.3]))  # nothing nearby: no signal
    world.api["close_gripper"]()
    assert end(est, world, rec, 2).violations == []


def test_table_only_from_table_height():
    world = FakeWorld({"plate_1": ("plate", (0.5, -0.2, 0.0)), "b1": ("bowl", (0.5, 0.1, 0.0))})
    est, det, rec = setup(world, SWAP_GOAL_JSON, "put the bowl on the plate")
    assert est.table_z == pytest.approx(-0.025)  # lowest bottom at turn 0
    assert est.pred_belief(("on", "bowl_A", "table"))[0] == pytest.approx(0.97)
    assert est.pred_belief(("on", "plate_A", "table"))[0] > 0.7  # bottom 1.5 cm above: still likely
    world.objects["b1"]["pos"] = np.array([0.5, 0.1, 0.03])  # bottom 3 cm above the table: not sure
    r = end(est, world, rec, 1)
    assert 0.3 < est.pred_belief(("on", "bowl_A", "table"))[0] < 0.7
    world.objects["b1"]["pos"] = np.array([0.5, 0.1, 0.25])  # floating (e.g. on something untracked)
    end(est, world, rec, 2)
    assert est.pred_belief(("on", "bowl_A", "table"))[0] == pytest.approx(0.03)
    assert "bowl_A (black bowl) at (0.50, 0.10, 0.25) location unknown" in est.render()
    assert r is not None
    est2 = BeliefEstimator(FakeDetector(FakeWorld({})))  # nothing seen at turn 0: no table height
    est2.start(FakeWorld({}).get_observation(), "put the bowl on the plate")
    assert est2.table_z is None and est2.pred_belief(("on", "bowl_A", "table")) == (0.5, "prior", None)


def test_hook_recording_failures_never_change_the_call():
    class BadRepr:
        def __repr__(self):
            raise RuntimeError("no repr")

    eef = np.array([0.1, 0.2, 0.3])
    rec = ApiCallRecorder(lambda: (eef, 0.5), lambda: 1 / 0)
    assert rec.wrap("goto_pose", lambda x: "ok")(BadRepr()) == "ok"
    r = rec.drain()[0]
    assert r.args == "<unprintable>" and r.step_after is None
    eef[0] = 9.0
    assert r.eef_after[0] == pytest.approx(0.1)  # recorded copy
    rec2 = ApiCallRecorder(lambda: ("not an array", "nan?"), None)
    with pytest.raises(KeyError):
        rec2.wrap("open_gripper", lambda: {}["x"])()


def test_state_reports_perception_source():
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    assert est.state()["perception"] == "fake"
    world.pick_place("akita_black_bowl_1", "plate_3")
    assert end(est, world, rec, 1).to_log()["perception"] == "fake"


# --- review round 2 ---------------------------------------------------------------------------

def test_gate_match_not_pushed_out_by_long_crossing_pair():
    """bowl_A (y=0) has a detection 0.08 m away (inside the gate); bowl_B (y=0.5) has none near.
    Pass 1 must keep bowl_A's gated match instead of a cheaper-overall crossing assignment
    (bowl_A -> y=-0.3, bowl_B -> y=0.08); the far detection becomes a new node."""
    world = FakeWorld({"b1": ("bowl", (0.5, 0.0, 0.02)), "b2": ("bowl", (0.5, 0.5, 0.02))})
    est, det, rec = setup(world, SWAP_GOAL_JSON, "move the bowl")
    world.objects["b1"]["pos"] = np.array([0.5, 0.08, 0.02])
    world.objects["b2"]["pos"] = np.array([0.5, -0.3, 0.02])
    r = end(est, world, rec, 1)
    assert est.tracks["bowl_A"].pos[1] == pytest.approx(0.08)  # kept its gated match
    assert r.rematched == [] and r.new_tracks == ["bowl_C"] and r.unmatched_tracks == ["bowl_B"]


def test_final_snapshot_count_contradicted_is_withdrawn(monkeypatch):
    """A count on the end-of-turn snapshot that the observation contradicts takes the same path as a
    hook-snapshot one: withdrawn, S1, not left pending. (With one fused graph this needs an
    observation check that disagrees with the snapshot; forced here.)"""
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    world.objects["akita_black_bowl_1"]["pos"] = np.array([0.5, 0.2, 0.02])  # moved without hooks
    real = est._observed
    monkeypatch.setattr(est, "_observed", lambda p: False if p == ("on", "bowl_A", "plate_C") else real(p))
    r = end(est, world, rec, 1)
    assert r.progress.done == [] and r.pending == []
    assert r.rolled_back == ["bowl_A on plate_C"]
    assert [v.expected for v in r.violations] == ["bowl_A on plate_C at the end of the turn"]
    assert "count withdrawn" in r.violations[0].observed


def test_confirmed_by_records_later_step_confirmations():
    world = swap_world()
    est, det, rec = setup(world, SWAP_GOAL_JSON, SWAP)
    world.pick_place("akita_black_bowl_1", "plate_3")
    det.miss = {"akita_black_bowl_1"}
    r1 = end(est, world, rec, 1)
    assert r1.confirmed_by == [None]
    world.pick_place("akita_black_bowl_2", "plate_1")
    det.miss_always = {"akita_black_bowl_1"}  # bowl_A still unseen: only step 2 is observed
    r2 = end(est, world, rec, 2)
    det.miss_always = set()
    assert r2.confirmed_by == ["later_step", "obs"] and r2.to_log()["confirmed_by"] == ["later_step", "obs"]
    assert est.state()["later_step_confirmations"] == 1
    assert "1 step(s) confirmed only by a later step, not seen" in est.progress_text()
    world.pick_place("akita_black_bowl_1", "plate_2")
    r3 = end(est, world, rec, 3)
    assert r3.confirmed_by == ["later_step", "obs", "obs"]


def test_later_repeat_does_not_confirm_earlier_unseen_repeat():
    """Seen in LIBERO-Mem task_02 on CPU: a tilted bowl's borderline observation confirmed the 3rd
    repeat and, through it, two unseen earlier ones -> wrong S5b. Repeats confirm only themselves."""
    world = repeat_world()
    est, det, rec = setup(world, REPEAT_GOAL_JSON, REPEAT)
    world.pick_place("akita_black_bowl_1", "plate_1")
    det.miss = {"akita_black_bowl_1"}
    r1 = end(est, world, rec, 1)
    assert r1.pending == [0]
    world.pick_place("akita_black_bowl_1", "plate_1")
    r2 = end(est, world, rec, 2)
    assert r2.confirmed_by == [None, "obs"] and r2.pending == [0] and r2.suppress_completion
    assert est.state()["later_step_confirmations"] == 0
    world.pick_place("akita_black_bowl_1", "plate_1")
    r3 = end(est, world, rec, 3)
    assert r3.progress.complete and r3.suppress_completion
    text = est.progress_text(uncertain=True)
    assert "all steps counted, but NOT all confirmed" in text and "estimated complete" not in text
