"""bg_est2 (estimated Belief Graph v2) without a simulator: Observe parser, marks, declarations, subtask
expansion, goal normalization, fusion / confirm / pending / withdraw, signals S1-S5, serialization,
belief_trace, and the no-simulator-access rule."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from capx.belief import ApiCallRecorder
from capx.belief.declare import RecordFacts, check_records, parse_declaration
from capx.belief.estimator2 import BeliefEstimatorV2, Est2Config
from capx.belief.goal import (
    GoalParseError,
    horizontal_word,
    normalize_steps,
    parse_goal_json_v2,
)
from capx.belief.memory import make_memory
from capx.belief.memory2 import BgEst2Memory
from capx.belief.observe import VdmObserver, note_flags, parse_observe, statement_key, statement_text
from capx.belief.prompts_v2 import M4_RULES, marks_now_text
from capx.belief.signals import SignalContextV2, est2_detectors
from capx.belief.som import MarkTable, draw_marks
from capx.belief.subtasks import expand_branch, expand_step
from tests.belief_fakes import SHORT_IDS, ScriptedModel, WorldEnv, WorldTruth, pick_place_code, repeat_world, swap_world
from tests.bg_est2_fakes import (
    BASKET_GOAL,
    REPEAT2_GOAL,
    SWAP2_GOAL,
    SWAP2_GOAL_CUMULATIVE,
    AnyDetector,
    FakeObserveVdm,
    basket_world,
    declared,
)

SWAP = "swap the 2 bowls on their plates using the empty plate"
REPEAT = "lift the bowl and place it back on the plate 3 times"
BASKET = "put the cream cheese in the nearest basket and place that basket in the center"
A, B = "akita_black_bowl_1", "akita_black_bowl_2"
SIM_NAMES = ("akita_black_bowl", "plate_1", "plate_2", "plate_3", "basket_1", "basket_2", "cream_cheese_1")


# --- Observe parser ---------------------------------------------------------------------------------

K_AB = ("on", "bowl_A", "plate_B")
K_AT = ("on", "bowl_A", "table")
K_IH = ("inhand", "bowl_B", None)
NAMES = ["bowl_A", "bowl_B", "plate_A", "plate_B"]


def test_observe_parser_lenient():
    asked = [K_AB, K_AT, K_IH, ("handempty", "gripper", None)]
    r = parse_observe("**Observe:** (ON  bowl_a plate_b) 1 |(inhand BOWL_B)? | (on bowl_A table): 0 | (handempty) true",
                      asked, NAMES)
    assert r.ok and r.values == {K_AB: "1", K_IH: "?", K_AT: "0", ("handempty", "gripper", None): "1"}
    # Order does not matter; a statement that was not asked or names an unknown object is ignored.
    r = parse_observe("Observe: (on bowl_B plate_A) 1 | (on cup_Z plate_A) 0 | (on bowl_A plate_B) 0", asked, NAMES)
    assert r.values == {K_AB: "0"} and len(r.unknown) == 2
    # No "Observe:" prefix: the statements anywhere in the reply are read.
    assert parse_observe("I see (on bowl_A plate_B) 1.", asked, NAMES).values == {K_AB: "1"}


def test_observe_parser_notes_and_failure():
    r = parse_observe("Observe: (on bowl_A plate_B) ? | (inhand bowl_B) 0\nNotes: the robot arm covers mark 2; bowl_A is tilted.",
                      [K_AB, K_IH], NAMES)
    assert r.values[K_AB] == "?" and r.notes.startswith("the robot arm covers")
    bad = parse_observe("Sorry, I cannot see the image.", [K_AB], NAMES)
    assert not bad.ok and bad.error
    assert not parse_observe("", [K_AB], NAMES).ok


def test_statement_text_roundtrip():
    for k in [K_AB, K_IH, ("handempty", "gripper", None), ("at", "basket_A", "center"), ("above", "gripper", "plate_B")]:
        toks = statement_text(k).strip("()").split()
        assert statement_key(toks[0], toks[1:]) == k


def test_note_flags():
    marks = {"bowl_A": "A", "bowl_B": "B", "plate_B": "2"}
    names = list(marks)
    assert note_flags("bowl_B is tilted against the rim of plate_B.", names, marks) == {"bowl_B": "tilted", "plate_B": "tilted"}
    assert note_flags("Mark A is on the wrong object.", names, marks) == {"bowl_A": "wrong mark"}
    assert note_flags("A robot arm covers mark 2.", names, marks) == {}
    assert note_flags("bowl_A fell off the plate", names, marks) == {"bowl_A": "fallen"}
    assert note_flags(None, names, marks) == {}


# --- marks ------------------------------------------------------------------------------------------

def test_overlapping_marks_are_spread():
    from capx.belief.som import mark_radius, spread_marks

    out = spread_marks([("A", (100.0, 100.0)), ("1", (100.0, 100.0)), ("2", (300.0, 100.0))], (512, 800, 3))
    r = mark_radius((512, 800, 3))
    assert out[0] == ("A", (100.0, 100.0)) and out[2] == ("2", (300.0, 100.0))
    assert out[1][0] == "1" and out[1][1][1] - 100.0 >= 2 * r


def test_mark_table_and_draw():
    mt = MarkTable()
    mt.assign_all([("plate_B", "plate"), ("bowl_A", "bowl"), ("plate_A", "plate"), ("bowl_B", "bowl"), ("basket_A", "basket")])
    assert (mt.mark("bowl_A"), mt.mark("bowl_B"), mt.mark("basket_A"), mt.mark("plate_A"), mt.mark("plate_B")) == ("A", "B", "1", "2", "3")
    assert mt.ordered()[0] == ("A", "bowl_A") and mt.alias("2") == "plate_A"
    assert mt.assign("cream_cheese_A", "cream_cheese") == "C"
    img = np.zeros((64, 64, 3), np.uint8)
    out = draw_marks(img, [("A", (20.0, 20.0)), ("1", (100.0, 5.0))])  # the second is outside the image
    assert out.shape == img.shape and out[20, 20].any() and not out[5, 60].any()
    assert marks_now_text([("A", "bowl_A", (250, 205)), ("1", "plate_A", (250, 214))], ["B", "2"]) == \
        "A = bowl_A (pixel 250, 205) | 1 = plate_A (250, 214) | B, 2: not detected"


# --- declarations -----------------------------------------------------------------------------------

def test_declaration_parse_ok_and_fence():
    code = "```python\n# Subtask: place(bowl_B, plate_B)\n# Effect: (on bowl_B plate_B) 0>1 | (inhand bowl_B) 1>0\nx = 1\n```"
    d = parse_declaration(code, NAMES)
    assert d.valid and d.op == "place" and d.args == ["bowl_B", "plate_B"]
    assert [(e.key, e.before, e.after) for e in d.effects] == [(("on", "bowl_B", "plate_B"), "0", True), (K_IH, "1", False)]
    assert d.subtask_text == "place(bowl_B, plate_B)" and d.effect_text == "(on bowl_B plate_B) 0>1 | (inhand bowl_B) 1>0"


def test_declaration_missing_or_malformed():
    assert not parse_declaration("x = 1\ny = 2\n# Subtask: pick(bowl_A)", NAMES).declared  # not in the first two lines
    assert not parse_declaration(None, NAMES).declared
    d = parse_declaration("# Subtask: grab(bowl_A)\n# Effect: (inhand bowl_A) 0>1", NAMES)
    assert not d.declared and "unknown op" in d.errors[0]
    d = parse_declaration("# Subtask: place(bowl_Z, plate_A)", NAMES)
    assert not d.declared and "unknown object" in d.errors[0]
    d = parse_declaration("# Subtask: pick(bowl_A, plate_B)\n# Effect: (inhand bowl_A) 0>1 | bowl_A lifted | (on bowl_A plate_Q) 1>0", NAMES)
    assert d.valid and len(d.effects) == 1 and len(d.errors) == 2
    d = parse_declaration("# Subtask: place_region(basket_A, center)\n# Effect: (at basket_A center) 0>1", ["basket_A"], ["center"])
    assert d.valid and d.effects[0].key == ("at", "basket_A", "center")


def test_declaration_vs_api_records():
    d = check_records(parse_declaration("# Subtask: place(bowl_B, plate_B)", NAMES),
                      RecordFacts(opened=True, released=[("bowl_B", "plate_A")]))
    assert not d.valid and "above plate_A" in d.dropped
    d = check_records(parse_declaration("# Subtask: pick(bowl_A, plate_A)", NAMES), RecordFacts(closed=True, grasped=["bowl_B"]))
    assert not d.valid and "closed on bowl_B" in d.dropped
    d = check_records(parse_declaration("# Subtask: place(bowl_B, plate_B)", NAMES), RecordFacts(opened=True, released=[("bowl_B", None)]))
    assert d.valid and d.applied  # released over no tracked support: a possible failed place, not a mismatch
    d = check_records(parse_declaration("# Subtask: place(bowl_B, plate_B)", NAMES), RecordFacts(closed=True))
    assert d.valid and not d.applied  # open_gripper never ran: Effects not applied
    # Grasp missed, open_gripper ran with nothing in hand (LIBERO-Mem task 9 CPU check): not applied.
    d = check_records(parse_declaration("# Subtask: put_in(bowl_A, plate_B)", NAMES),
                      RecordFacts(closed=True, opened=True, grasped=[], released=[]))
    assert d.valid and not d.applied
    d = check_records(parse_declaration("# Subtask: pick(bowl_A, plate_A)", NAMES), RecordFacts(closed=True, grasped=[]))
    assert d.valid and not d.applied


def test_missed_grasp_counts_nothing():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    world.api["open_gripper"]()
    world.api["goto_pose"](np.array([0.6, 0.3, 0.02]))  # nothing here
    world.api["close_gripper"]()
    world.api["goto_pose"](np.array([0.5, 0.2, 0.15]))
    world.api["open_gripper"]()
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    assert not r.declaration["applied"] and len(r.progress.done) == 0 and not est.pending
    assert est.graph.edges[("on", "bowl_A", "plate_A")].holds


# --- subtasks / goal --------------------------------------------------------------------------------

LOC0 = {"bowl_A": frozenset({("on", "plate_A")}), "bowl_B": frozenset({("on", "plate_B")}),
        "plate_A": frozenset({("on", "table")}), "plate_B": frozenset({("on", "table")}), "plate_C": frozenset({("on", "table")})}


def test_expand_swap_repeat_basket():
    swap = [(("on", "bowl_A", "plate_C"),), (("on", "bowl_B", "plate_A"),), (("on", "bowl_A", "plate_B"),)]
    ops = expand_branch(swap, LOC0)
    assert [[s.text for s in st] for st in ops] == [
        ["pick(bowl_A, plate_A)", "place(bowl_A, plate_C)"], ["pick(bowl_B, plate_B)", "place(bowl_B, plate_A)"],
        ["pick(bowl_A, plate_C)", "place(bowl_A, plate_B)"]]
    rep = [(("on", "bowl_A", "plate_A"),)] * 3
    assert [[s.text for s in st] for st in expand_branch(rep, LOC0)] == [["pick(bowl_A, plate_A)", "place(bowl_A, plate_A)"]] * 3
    loc = {"cream_cheese_A": frozenset({("on", "table")}), "basket_A": frozenset({("on", "table")})}
    bk = [(("in", "cream_cheese_A", "basket_A"),), (("at", "basket_A", "center"),)]
    assert [[s.text for s in st] for st in expand_branch(bk, loc, {"basket_A"})] == [
        ["pick(cream_cheese_A, table)", "put_in(cream_cheese_A, basket_A)"], ["pick(basket_A, table)", "place_region(basket_A, center)"]]
    # Holding the object already: no pick.
    assert [s.text for s in expand_step(swap[0], LOC0, held="bowl_A")] == ["place(bowl_A, plate_C)"]
    pre = expand_step(swap[0], LOC0)[0].preconditions()
    assert (("handempty", "gripper", None), True) in pre and (("on", "bowl_A", "plate_A"), True) in pre


def test_goal_v2_normalizes_cumulative_and_keeps_repeats():
    names = ["bowl_A", "bowl_B", "plate_A", "plate_B", "plate_C"]
    init = {"bowl_A": frozenset({("on", "plate_A")}), "bowl_B": frozenset({("on", "plate_B")})}
    cum = normalize_steps(parse_goal_json_v2(SWAP2_GOAL_CUMULATIVE, names), init)
    assert cum == normalize_steps(parse_goal_json_v2(SWAP2_GOAL, names), init)
    assert cum[0] == [(("on", "bowl_A", "plate_C"),), (("on", "bowl_B", "plate_A"),), (("on", "bowl_A", "plate_B"),)]
    rep = normalize_steps(parse_goal_json_v2(REPEAT2_GOAL, names), init)
    assert rep == [[(("on", "bowl_A", "plate_A"),)] * 3]
    bk = parse_goal_json_v2(BASKET_GOAL, ["cream_cheese_A", "basket_A", "basket_B"], ["center"])
    assert bk[0] == [(("in", "cream_cheese_A", "basket_A"),), (("at", "basket_A", "center"),)]
    with pytest.raises(GoalParseError):
        parse_goal_json_v2(BASKET_GOAL, ["cream_cheese_A", "basket_A", "basket_B"], [])  # no region in the scene


def test_scene_words_follow_the_image():
    ys = [-0.2, 0.0, 0.2]
    # The robot's right (-y) is the image left.
    assert horizontal_word(-0.2, ys) == "left side" and horizontal_word(0.2, ys) == "right side"
    assert horizontal_word(0.0, ys) == "center"


# --- estimator with a fake world --------------------------------------------------------------------

def setup(world, goal_json, instruction, detector=None, **cfg):
    det = detector or AnyDetector(world)
    rec = ApiCallRecorder(world.proprio, lambda: world.steps)
    world.api = rec.wrap_functions(world.functions())
    holder = {}
    vdm = FakeObserveVdm(world, lambda: holder.get("est"))
    goal_model = ScriptedModel(goal_json)
    est = BeliefEstimatorV2(det, Est2Config(**cfg), goal_query_fn=goal_model, observer=VdmObserver(vdm))
    holder["est"] = est
    first = est.start(world.get_observation(), instruction)
    return est, rec, vdm, goal_model, first


def turn(est, world, rec, n, code=None):
    return est.end_turn(n, world.get_observation(), rec.drain(), code=code)


def place_code(o, t):
    return declared(f"place({o}, {t})", f"(on {o} {t}) 0>1 | (inhand {o}) 1>0", "...")


def test_start_m2_m3_prompts_and_initial_belief():
    world = swap_world()
    est, rec, vdm, goal_model, first = setup(world, SWAP2_GOAL_CUMULATIVE, SWAP)
    m2 = vdm.prompts[0]
    text = "\n".join(c.get("text", "") for c in m2[1]["content"])
    assert m2[0]["content"].startswith("You are the visual observer for a robot's belief graph.")
    assert "The robot stands at the top of the image, so the robot's left is the image's right." in text
    assert "- A = bowl_A (black bowl), mark at pixel" in text and "- 3 = plate_C (plate), mark at pixel" in text
    assert "(on bowl_A plate_A)\n" in text and "(on bowl_B table)" in text
    assert sum(1 for c in m2[1]["content"] if c["type"] == "image_url") == 1
    m3 = goal_model.prompts[0][1]["content"][0]["text"]
    assert "- bowl_A (black bowl): on plate_A; left side" in m3 and "- plate_C (plate): empty; right side" in m3
    assert "Each step lists only the predicates that BECOME true at that step" in m3
    assert est.goal.branches[0][1] == (("on", "bowl_B", "plate_A"),)  # normalized
    assert first.fallback is False and est.origin[("on", "bowl_A", "plate_A")] == "vdm"
    text0 = est.render()
    assert text0.splitlines()[0].startswith("Belief graph (turn 0, ESTIMATED; names match the marks in the camera image;")
    assert "Delta: step 0/3 | next: one of (on bowl_A plate_C) OR (on bowl_B plate_C) | remaining 3" in text0
    assert "Subtasks (next): pick(bowl_A, plate_A) -> place(bowl_A, plate_C)   [or pick(bowl_B, plate_B) -> place(bowl_B, plate_C)]" in text0
    assert "(on bowl_A plate_A) 0.90 vdm" in text0 and "(handempty) 0.97 robot" in text0
    assert "plate_C at (0.50, 0.20, 0.00) empty" in text0
    assert not any(n in text0 for n in SIM_NAMES)


def test_declared_place_confirmed_by_observe():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    world.pick_place(A, "plate_3")
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    m5 = "\n".join(c.get("text", "") for c in vdm.prompts[-1][1]["content"])
    assert "Image 1: before the last code block. Image 2: now." in m5
    assert "The last code block declared:\nSubtask: place(bowl_A, plate_C)\nEffect: (on bowl_A plate_C) 0>1 | (inhand bowl_A) 1>0" in m5
    assert m5.index("(on bowl_A plate_C)") < m5.index("(inhand bowl_A)") < m5.index("(on bowl_A plate_A)")
    assert sum(1 for c in vdm.prompts[-1][1]["content"] if c["type"] == "image_url") == 2
    assert len(r.statements) <= 10
    assert r.declaration["valid"] and r.declaration["applied"]
    assert [s["status"] for s in r.judged] == ["done", "done"]  # pick (robot) + place (vdm)
    assert len(r.progress.done) == 1 and not est.pending and est.confirmed_by[0] == "vdm"
    text = est.render()
    assert "Subtask (declared): place(bowl_A, plate_C)\nEffect (declared): (on bowl_A plate_C) 0>1 | (inhand bowl_A) 1>0" in text
    assert "Observe (t1): (on bowl_A plate_C) 1 | (inhand bowl_A) 0" in text
    assert "Delta: step 1/3 done | next: (on bowl_B plate_A) | remaining 2" in text
    assert "Subtasks: pick(bowl_A, plate_A) done -> place(bowl_A, plate_C) done | next: pick(bowl_B, plate_B) -> place(bowl_B, plate_A)" in text
    assert "S1" not in text


def test_place_failed_withdrawn_s1():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    world.pick_place(A, "plate_3")
    vdm.sticky = {"(on bowl_A plate_C)": "0", "(on bowl_A table)": "1"}
    vdm.notes = "bowl_A is tilted against the rim of mark 3."
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    assert [s["status"] for s in r.judged] == ["done", "failed"]
    assert len(r.progress.done) == 0 and r.rolled_back
    assert any("place(bowl_A, plate_C) failed: bowl_A is on table (tilted, per Notes). Retry the place, not the whole task" in s
               for s in r.s1_items)
    fired = signals(est, r)
    assert "S1" in fired and "S5b" not in fired and "S2" not in fired  # the pick was progress
    text = est.render([])
    assert "Subtask (declared): place(bowl_A, plate_C) — FAILED" in text
    assert "place(bowl_A, plate_C) FAILED" in text and "bowl_A at" in text and "(tilted, per Notes)" in text


def test_question_mark_keeps_memory_pending_then_s4():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    world.pick_place(A, "plate_3")
    vdm.sticky = {"(on bowl_A plate_C)": "?"}
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    e = est.graph.edges[("on", "bowl_A", "plate_C")]
    assert e.p == pytest.approx(0.7) and est.origin[e.key] == "exp"  # declared Effect + API ran: expected
    assert est.pending == {0} and r.suppress_completion
    assert [s["status"] for s in r.judged][-1] == "pending"
    assert "S4" not in signals(est, r)  # one ? only
    r = turn(est, world, rec, 2, declared("look(bowl_A)", "", "..."))
    assert "S4" in signals(est, r)
    assert any("not visible" in t or "not seen by the camera" in t for t in r.s4_items)
    vdm.sticky = {}
    r = turn(est, world, rec, 3, declared("look(bowl_A)", "", "..."))
    assert not est.pending and est.confirmed_by[0] == "vdm"  # confirmed later
    assert r.judged and r.judged[0]["status"] == "done"  # the pending place resolved


def test_observe_garbage_falls_back_to_geometry():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    world.pick_place(A, "plate_3")
    vdm.garbage = True
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    assert r.fallback and est.origin[("on", "bowl_A", "plate_C")] == "geo"
    assert len(r.progress.done) == 1 and "Observe (t1): failed" in est.render()


def test_geometric_anomaly_and_robot_priority():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    vdm.sticky = {"(on bowl_A plate_C)": "1", "(inhand bowl_A)": "1"}  # bowl_A is still on plate_A
    r = turn(est, world, rec, 1, declared("place(bowl_A, plate_C)", "(on bowl_A plate_C) 0>1 | (inhand bowl_A) 1>0", "..."))
    assert est.graph.edges[("on", "bowl_A", "plate_C")].p == 0.5 and r.vetoes and r.vetoes[0]["kind"] == "apart"
    assert est.graph.edges[("inhand", "bowl_A", None)].p == pytest.approx(0.03) and r.robot_conflicts
    assert "S4" in signals(est, r) and any("mark may be on the wrong object" in t for t in r.s4_items)
    assert len(r.progress.done) == 0


def test_s2_only_without_subtask_progress_and_s3_holding():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    r = turn(est, world, rec, 1, declared("look(bowl_A)", "", "pass"))
    assert "S2" in signals(est, r)
    # Grasp and lift bowl_B (the second branch would start with it; the primary branch wants bowl_A).
    p = world.objects[B]["pos"].copy()
    world.api["goto_pose"](p + [0, 0, 0.02])
    world.api["close_gripper"]()
    world.api["goto_pose"](p + [0, 0, 0.15])
    r = turn(est, world, rec, 2, declared("pick(bowl_B, plate_B)", "(inhand bowl_B) 0>1", "..."))
    fired = signals(est, r)
    assert "S2" not in fired and r.subtask_progress
    assert est.subtasks_text().startswith("Subtasks: pick(bowl_B, plate_B) done | next: place(bowl_B, plate_C)")
    assert "S3" not in fired  # holding bowl_B fits the second branch


def test_s3_next_subtask_needs_empty_hand():
    world = swap_world()
    one_branch = json.dumps({"goal": {"type": "sequence", "steps": [{"preds": [["on", "bowl_A", "plate_C"]]}]}})
    est, rec, vdm, _, _ = setup(world, one_branch, SWAP)
    p = world.objects[B]["pos"].copy()
    world.api["goto_pose"](p + [0, 0, 0.02])
    world.api["close_gripper"]()
    world.api["goto_pose"](p + [0, 0, 0.15])
    r = turn(est, world, rec, 1, declared("pick(bowl_B, plate_B)", "(inhand bowl_B) 0>1", "..."))
    assert "S3" in signals(est, r)
    assert r.s3_items == ["next subtask pick(bowl_A, plate_A) needs (handempty), but the gripper holds bowl_B: "
                          "put bowl_B where the goal needs it (or back down) first"]


def test_dropped_declaration_falls_back_to_hooks():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    world.pick_place(A, "plate_3")
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_B"))  # declared plate_B, released above plate_C
    assert not r.declaration["valid"] and "above plate_C" in r.declaration["dropped"]
    assert len(r.progress.done) == 1  # hook inference + Observe
    assert "IGNORED" in est.render()


def test_repeat_counts_each_put_back():
    world = repeat_world()
    est, rec, vdm, _, _ = setup(world, REPEAT2_GOAL, REPEAT)
    for n in range(1, 4):
        world.pick_place(A, "plate_1")
        r = turn(est, world, rec, n, place_code("bowl_A", "plate_A"))
        assert len(r.progress.done) == n
    assert r.progress.complete and not est.pending and "S5b" in signals(est, r)
    # Two put-backs in one block count twice (hook snapshots).
    world = repeat_world()
    est, rec, vdm, _, _ = setup(world, REPEAT2_GOAL, REPEAT)
    world.pick_place(A, "plate_1")
    world.pick_place(A, "plate_1")
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_A"))
    assert len(r.progress.done) == 2


def test_basket_put_in_and_region():
    world = basket_world()
    est, rec, vdm, goal_model, _ = setup(world, BASKET_GOAL, BASKET)
    assert est.regions == ["center"] and np.allclose(est.region_xy["center"], [0.5, 0.0])
    m3 = goal_model.prompts[0][1]["content"][0]["text"]
    assert "- center: a region of the table" in m3 and "- basket_A (basket): on the table; empty;" in m3
    assert "Subtasks (next): pick(cream_cheese_A, table) -> put_in(cream_cheese_A, basket_A)" in est.render()
    world.pick_place("cream_cheese_1", "basket_1")
    r = turn(est, world, rec, 1, declared("put_in(cream_cheese_A, basket_A)", "(in cream_cheese_A basket_A) 0>1", "..."))
    assert [s["status"] for s in r.judged] == ["done", "done"] and len(r.progress.done) == 1
    # Basket to the center (the region is judged by Observe, geometry agrees). The fingers are on the
    # basket rim with the cheese inside within reach: the declaration names the grasped object.
    world.prefer = "basket_1"
    p = world.objects["basket_1"]["pos"].copy()
    world.api["goto_pose"](p + [0, 0, 0.02])
    world.api["close_gripper"]()
    world.api["goto_pose"](p + [0, 0, 0.15])
    world.api["goto_pose"](np.array([0.5, 0.0, 0.15]))
    world.api["goto_pose"](np.array([0.5, 0.0, 0.02]))
    world.api["open_gripper"]()
    world.api["goto_pose"](np.array([0.5, 0.0, 0.3]))
    world.objects["cream_cheese_1"]["pos"] = world.objects["basket_1"]["pos"] + [0, 0, 0.02]
    r = turn(est, world, rec, 2, declared("place_region(basket_A, center)", "(at basket_A center) 0>1", "..."))
    assert r.judged[-1]["op"] == "place_region" and r.judged[-1]["status"] == "done"
    assert r.progress.complete


def signals(est, r):
    ctx = SignalContextV2(r.turn, est.graph, r.progress, est._prev if hasattr(est, "_prev") else None, r.events,
                          violations=r.violations, uncertain=r.uncertain, suppress_completion=r.suppress_completion,
                          s1_items=r.s1_items, subtask_progress=r.subtask_progress, s3_items=r.s3_items,
                          s4_items=r.s4_items)
    if not hasattr(est, "_prev"):
        ctx.prev_progress = r.progress if len(r.progress.done) == 0 else None
    fired = [s for d in est2_detectors() if (s := d.detect(ctx)) is not None]
    est._prev = r.progress
    est.set_signals(fired)
    return [s.id for s in fired]


def test_render_budget_keeps_fixed_lines():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    world.pick_place(A, "plate_3")
    turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    full = est.render()
    short = est.render(max_chars=600)
    assert len(short) < len(full) and "Delta:" in short and "Signals: none" in short and "Subtask (declared)" in short
    assert "Objects:" not in short
    tiny = est.render(max_chars=50)
    assert tiny.startswith("Belief graph (turn 1") and "Delta:" in tiny and "Signals:" in tiny


# --- memory: trace, signals, no sim access ------------------------------------------------------------

def make_mem(tmp_path, shadow=True, low=None):
    low = low or WorldEnv()
    low.reset()
    world = low.world
    mem = BgEst2Memory(perception="sam3", shadow=shadow)
    vdm = FakeObserveVdm(world, lambda: mem.estimator)
    code_env = SimpleNamespace(api_call_wrapper=None)
    mem.reset(low, code_env=code_env, instruction=SWAP, goal_query_fn=ScriptedModel(SWAP2_GOAL),
              vdm_query_fn=vdm, perception_source=AnyDetector(world),
              shadow_source=WorldTruth(low) if shadow else None, trace_path=str(tmp_path / "belief_trace.jsonl"))
    world.api = {n: code_env.api_call_wrapper(n, f) for n, f in world.functions().items()}
    return mem, low, vdm


def test_memory_trace_schema_and_cot(tmp_path):
    mem, low, vdm = make_mem(tmp_path)
    world = low.world
    assert mem.prompt_rules() == M4_RULES and mem.initial_text().startswith("Belief graph (turn 0, ESTIMATED")
    plan = [("b1", "p3", "bowl_A", "plate_C"), ("b2", "p1", "bowl_B", "plate_A"), ("b1", "p2", "bowl_A", "plate_B")]
    for n, (o, t, oa, ta) in enumerate(plan, 1):
        world.pick_place(SHORT_IDS[o], SHORT_IDS[t])
        low.history.append(low.supports())
        out = mem.after_turn(n, "vdm text", executed=True, code=place_code(oa, ta))
    assert [s.id for s in out.signals] == ["S5b"] and mem.check_finish() is None
    lines = [json.loads(x) for x in (tmp_path / "belief_trace.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in lines] == ["start", "turn", "turn", "turn"]
    rec = lines[1]
    for k in ("nodes", "robot", "keys", "goal", "steps", "subtasks", "signals", "cot", "observe", "declaration"):
        assert k in rec, k
    assert rec["nodes"]["bowl_A"]["mark"] == "A" and rec["nodes"]["plate_C"]["mark"] == "3"
    key = rec["keys"]["(on bowl_A plate_C)"]
    assert set(key) >= {"P", "source", "obs", "last_seen_turn", "geom", "shadow"} and key["shadow"] is True
    assert key["P"] == pytest.approx(0.9) and key["source"] == "vdm" and key["obs"] is True
    assert set(rec["robot"]) == {"eef", "gripper", "handempty", "above", "holding", "lifted"}
    assert rec["cot"]["Subtask"] == "place(bowl_A, plate_C)" and rec["cot"]["Observe"].startswith("(on bowl_A plate_C) 1")
    assert rec["observe"]["prompt_text"].count("<image>") == 2
    assert rec["steps"][0]["status"] == "confirmed" and rec["steps"][0]["by"] == "vdm"
    tl = mem.turn_log()
    assert tl["cot_record"]["Signals"] == ["S5b all steps verified complete. If you agree, respond FINISH."]
    tot = mem.state()["shadow_totals"]
    assert tot["declared"] == 3 and tot["effects_applied"] >= 3 and tot["effects_right"] == tot["effects_applied"]
    assert tot["subtasks_judged"] >= 3 and tot["subtasks_right"] == tot["subtasks_judged"]
    assert tot["observe"]["on"]["1"][0] > 0 and tot["observe"]["on"]["1"][0] == tot["observe"]["on"]["1"][1]
    assert lines[0]["shadow"]["goal_match_normalized"] is True
    assert mem.state()["calls"]["observe_vdm"] == 4 and tot["step_error_abs_sum"] == 0


class StrictLow:
    """Robot observation and step counter only; any other attribute (simulator state) fails."""

    def __init__(self, world) -> None:
        object.__setattr__(self, "_world", world)

    def get_observation(self):
        return self._world.get_observation()

    @property
    def _sim_step_count(self):
        return self._world.steps

    def __getattr__(self, name):
        raise AssertionError(f"bg_est2 read simulator state: {name}")


def test_bg_est2_reads_no_sim_state(tmp_path):
    world = swap_world()
    low = StrictLow(world)
    code_env = SimpleNamespace(api_call_wrapper=None)
    mem = BgEst2Memory(perception="sam3", shadow=False)
    holder = {}
    vdm = FakeObserveVdm(world, lambda: holder.get("est"))
    goal = ScriptedModel(SWAP2_GOAL)

    def vdm_fn(prompt):
        holder["est"] = mem.estimator
        return vdm(prompt)

    mem.reset(low, code_env=code_env, instruction=SWAP, goal_query_fn=goal, vdm_query_fn=vdm_fn,
              perception_source=AnyDetector(world), trace_path=str(tmp_path / "t.jsonl"))
    assert mem.shadow is None
    world.api = {n: code_env.api_call_wrapper(n, f) for n, f in world.functions().items()}
    for n, (o, t) in enumerate([("b1", "p3"), ("b2", "p1"), ("b1", "p2")], 1):
        world.pick_place(SHORT_IDS[o], SHORT_IDS[t])
        out = mem.after_turn(n, None, executed=True, code=pick_place_code(o, t))  # no declaration: hooks
    assert [s.id for s in out.signals] == ["S5b"]
    for p in goal.prompts + vdm.prompts:
        text = json.dumps([c for m in p for c in (m["content"] if isinstance(m["content"], list) else [m["content"]])
                           if isinstance(c, str) or c.get("type") == "text"])
        assert not any(n in text for n in SIM_NAMES)


def test_make_memory_bg_est2_options():
    mem = make_memory("bg_est2", perception="sim_fake_for_tests", observe_max_statements=6, event_frames=True,
                      signals=["S1", "S3"])
    assert isinstance(mem, BgEst2Memory) and mem.config.observe_max_statements == 6 and mem.config.event_frames
    assert mem.enabled_signals == ("S1", "S3")
    with pytest.raises(ValueError):
        make_memory("bg_est2", perception="sim")


def test_event_frame_observe_fills_question_marks(tmp_path):
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP, event_frames=True)
    world.pick_place(A, "plate_3")
    frame = {"rgb": np.zeros((8, 8, 3), np.uint8), "after": "open_gripper", "object": "bowl_A", "step": 1000}
    vdm_calls = len(vdm.prompts)
    # The turn-end answer is ? (arm in the way); the event frame right after the open saw it.
    vdm.sticky = {"(on bowl_A plate_C)": "?"}
    r = est.end_turn(1, world.get_observation(), rec.drain(), code=place_code("bowl_A", "plate_C"), event_frames=[])
    assert est.pending
    world2 = swap_world()
    est2, rec2, vdm2, _, _ = setup(world2, SWAP2_GOAL, SWAP, event_frames=True)
    world2.pick_place(A, "plate_3")
    vdm2.sticky = {}

    calls = {"n": 0}
    orig = vdm2.__call__

    def wrapped(prompt):
        calls["n"] += 1
        if calls["n"] == 2:  # the turn-end Observe (after the event one)
            vdm2.sticky = {"(on bowl_A plate_C)": "?"}
        return orig(prompt)

    est2.observer.query_fn = wrapped
    r2 = est2.end_turn(1, world2.get_observation(), rec2.drain(), code=place_code("bowl_A", "plate_C"),
                       event_frames=[frame])
    assert r2.event_observes and r2.event_observes[0]["ok"]
    assert not est2.pending and len(r2.progress.done) == 1
    assert len(vdm.prompts) == vdm_calls + 1
