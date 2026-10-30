"""capx.belief without a simulator: graph updates, event diffs, Delta, S2, S5 and serialization limits."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from capx.belief import (
    P_OBS_FALSE,
    P_OBS_TRUE,
    P_PRIOR,
    BeliefGraph,
    BgOracleMemory,
    FinishGate,
    LiberoMemSource,
    NoMemory,
    Progress,
    VdmHistoryMemory,
    goal_branches,
    make_memory,
    serialize,
)
from capx.belief.progress import progress_line

# LIBERO-Mem task_06 (KITCHEN_SCENE1_7): swap the 2 bowls using the empty plate.
SWAP_GOAL = [[
    "or",
    ["sequence",
     ["and", ["on", "akita_black_bowl_1", "plate_3"]],
     ["and", ["on", "akita_black_bowl_2", "plate_1"]],
     ["and", ["on", "akita_black_bowl_1", "plate_2"]]],
    ["sequence",
     ["and", ["on", "akita_black_bowl_2", "plate_3"]],
     ["and", ["on", "akita_black_bowl_1", "plate_2"]],
     ["and", ["on", "akita_black_bowl_2", "plate_1"]]],
]]
# task_02 (KITCHEN_SCENE1_3): place the bowl back on the plate 3 times.
REPEAT_GOAL = [["sequence"] + [["and", ["on", "akita_black_bowl_1", "plate_1"]]] * 3]

BOWLS = ["akita_black_bowl_1", "akita_black_bowl_2"]
PLATES = ["plate_1", "plate_2", "plate_3"]


class FakeSource:
    """Ground truth for the swap scene; tests move bowls and append completed subgoals."""

    def __init__(self, goal=SWAP_GOAL) -> None:
        self.goal = goal
        self.where = {"akita_black_bowl_1": "plate_1", "akita_black_bowl_2": "plate_2"}
        self.held: str | None = None
        self.done: list = []
        self.is_overshot = False

    def movable_objects(self):
        return BOWLS + PLATES

    def on_targets(self):
        return BOWLS + PLATES + ["kitchen_table"]

    def contain_regions(self):
        return []

    def goal_state(self):
        return self.goal

    def positions(self):
        return {n: (0.1 * i, 0.0, 0.05) for i, n in enumerate(self.movable_objects())}

    def eval_predicate(self, pred):
        name, subj, target = pred
        if subj in PLATES:
            return target == "kitchen_table"
        return self.where.get(subj) == target

    def grasped(self, name):
        return self.held == name

    def completed_steps(self):
        return [tuple(tuple(p) for p in s) for s in self.done]

    def overshot(self):
        return self.is_overshot

    # test helpers
    def move(self, bowl, plate, complete_step=True):
        self.where[bowl] = plate
        if complete_step:
            self.done.append([("on", bowl, plate)])


# --- graph -----------------------------------------------------------------------------------

def test_graph_update_sources_and_events():
    g = BeliefGraph()
    key = ("on", "bowl", "plate_1")
    g.set_prior(key)
    assert g.edges[key].p == P_PRIOR and g.edges[key].source == "prior"

    events = g.update(0, {key: True, ("on", "bowl", "plate_3"): False}, {"bowl": (1.0, 2.0, 3.0)})
    assert events == []  # first observation of the subject: no event
    assert g.edges[key].p == P_OBS_TRUE and g.edges[key].source == "obs"
    assert g.edges[("on", "bowl", "plate_3")].p == P_OBS_FALSE
    assert g.nodes["bowl"].pos == (1.0, 2.0, 3.0) and g.nodes["bowl"].last_seen_turn == 0
    assert g.location("bowl") == "on plate_1"

    events = g.update(1, {key: False, ("on", "bowl", "plate_3"): True})
    assert [e.text for e in events] == ["t1 bowl on plate_1 -> on plate_3"]
    assert g.edges[key].last_seen_turn == 1

    # Not re-observed: P kept (no decay), source becomes mem.
    g.update(2, {("inhand", "bowl", None): True})
    edge = g.edges[("on", "bowl", "plate_3")]
    assert edge.source == "mem" and edge.p == P_OBS_TRUE and edge.last_seen_turn == 1
    assert g.location("bowl") == "in gripper + on plate_3"

    g.set_expected(("on", "bowl", "plate_2"), 3)
    assert g.edges[("on", "bowl", "plate_2")].p == 0.7
    assert g.edges[("on", "bowl", "plate_2")].source == "expected"


# --- progress / Delta ------------------------------------------------------------------------

def test_goal_branches_or_sequence_and_plain():
    br = goal_branches(SWAP_GOAL)
    assert len(br) == 2 and len(br[0]) == 3
    assert br[0][0] == (("on", "akita_black_bowl_1", "plate_3"),)
    assert goal_branches(REPEAT_GOAL)[0] == [(("on", "akita_black_bowl_1", "plate_1"),)] * 3
    plain = goal_branches([["on", "a", "b"], ["on", "c", "d"]])
    assert plain == [[(("on", "a", "b"), ("on", "c", "d"))]]
    # Upper-case BDDL as written in the file is normalized.
    assert goal_branches([["Sequence", ["And", ["On", "a", "b"]]]]) == [[(("on", "a", "b"),)]]


def test_progress_delta_branches():
    br = goal_branches(SWAP_GOAL)
    p0 = Progress.from_completed(br, [])
    assert p0.remaining == 3 and p0.possible == [0, 1] and len(p0.next_steps()) == 2
    assert "next: one of" in progress_line(p0)

    p1 = Progress.from_completed(br, [(("on", "akita_black_bowl_2", "plate_3"),)])
    assert p1.possible == [1] and p1.remaining == 2
    assert p1.next_steps() == [(("on", "akita_black_bowl_1", "plate_2"),)]
    line = progress_line(p1)
    assert line.startswith("Progress: step 1/3 done (akita_black_bowl_2 on plate_3)")
    assert "next: akita_black_bowl_1 on plate_2" in line and "remaining: 2" in line

    p3 = Progress.from_completed(br, list(br[1]))
    assert p3.complete and "all steps verified complete" in progress_line(p3)
    assert not Progress.from_completed(br, list(br[1]), overshot=True).complete


def test_repeat_counter_and_holds_now_note():
    br = goal_branches(REPEAT_GOAL)
    p = Progress.from_completed(br, [br[0][0]])
    assert p.repeat_count() == (1, 3) and p.remaining == 2
    line = progress_line(p, {("on", "akita_black_bowl_1", "plate_1"): True})
    assert line.startswith("Progress: 1/3 repetitions done (akita_black_bowl_1 on plate_1)")
    assert "holds now but not counted yet" in line
    assert Progress.from_completed(goal_branches(SWAP_GOAL), []).repeat_count() is None


# --- signals ---------------------------------------------------------------------------------

def test_bg_oracle_memory_events_s2_s5b():
    src = FakeSource()
    mem = BgOracleMemory()
    mem.reset(source=src)
    init = mem.initial_text()
    assert init.startswith("Belief graph (turn 0):")
    assert "akita_black_bowl_1 at (0.00, 0.00, 0.05) on plate_1" in init
    assert "plate_1 at (0.20, 0.00, 0.05) on kitchen_table" in init
    assert "remaining: 3" in init and "Events: none" in init

    # Turn 1: nothing moved -> S2.
    t1 = mem.after_turn(1)
    assert [s.id for s in t1.signals] == ["S2"]
    assert "Signals: S2 no progress this turn" in t1.text

    # Turn 2: bowl 1 moved to the empty plate -> event, progress, no S2.
    src.move("akita_black_bowl_1", "plate_3")
    t2 = mem.after_turn(2)
    assert t2.signals == []
    assert "Events: t2 akita_black_bowl_1 on plate_1 -> on plate_3" in t2.text
    assert "step 1/3 done (akita_black_bowl_1 on plate_3)" in t2.text
    assert "next: akita_black_bowl_2 on plate_1" in t2.text

    # Turn 3: bowl picked up, still in hand -> event, S2.
    src.where["akita_black_bowl_2"] = None
    src.held = "akita_black_bowl_2"
    t3 = mem.after_turn(3)
    assert "t3 akita_black_bowl_2 on plate_2 -> in gripper" in t3.text
    assert [s.id for s in t3.signals] == ["S2"]

    src.held = None
    src.move("akita_black_bowl_2", "plate_1")
    src.move("akita_black_bowl_1", "plate_2")
    t4 = mem.after_turn(4)
    assert [s.id for s in t4.signals] == ["S5b"]
    assert "all steps verified complete" in t4.text
    assert mem.check_finish() is None  # complete: FINISH accepted


def test_s5a_rejects_once_then_accepts():
    src = FakeSource()
    mem = BgOracleMemory()
    mem.reset(source=src)
    src.move("akita_black_bowl_1", "plate_3")
    mem.after_turn(1)
    rej = mem.check_finish()
    assert rej is not None and rej.id == "S5a"
    assert "2 step(s) remain: akita_black_bowl_2 on plate_1; then akita_black_bowl_1 on plate_2" in rej.text
    assert mem.check_finish() is None  # second FINISH ends the trial


def test_s5a_not_applied_after_overshoot_and_gate_counts():
    br = goal_branches(SWAP_GOAL)
    gate = FinishGate(max_rejections=2)
    p = Progress.from_completed(br, [])
    assert gate.check(p) is not None and gate.check(p) is not None and gate.check(p) is None
    assert FinishGate().check(Progress.from_completed(br, list(br[0]), overshot=True)) is None


def test_signals_can_be_turned_off():
    src = FakeSource()
    mem = BgOracleMemory(signals=())
    mem.reset(source=src)
    assert mem.after_turn(1).signals == []
    assert mem.check_finish() is None
    assert "Signals: none" in mem.render()


# --- serialization limit ---------------------------------------------------------------------

def test_serialize_drops_oldest_events_first():
    src = FakeSource()
    mem = BgOracleMemory(max_chars=10_000)
    mem.reset(source=src)
    for t in range(1, 21):
        src.move("akita_black_bowl_1", "plate_3" if t % 2 else "plate_1", complete_step=False)
        mem.after_turn(t)
    full = mem.render()
    assert "t1 akita_black_bowl_1" in full and "t20 akita_black_bowl_1" in full

    mem.max_chars = len(full) - 50
    short = mem.render()
    assert len(short) <= mem.max_chars
    assert "earlier omitted)" in short and "t1 akita_black_bowl_1" not in short
    assert "t20 akita_black_bowl_1" in short and "Progress:" in short and "Objects:" in short


def test_serialize_hard_limit_keeps_signals():
    src = FakeSource()
    mem = BgOracleMemory(max_chars=500)
    mem.reset(source=src)
    text = mem.after_turn(1).text
    assert len(text) <= 500
    assert "...[truncated]" in text
    assert "Signals: S2" in text  # objects line is cut before signals
    g = BeliefGraph()
    assert len(serialize(g, None, turn=0, max_chars=10)) <= 10


# --- vdm_history / none ----------------------------------------------------------------------

def test_vdm_history_accumulates_and_trims():
    mem = make_memory("vdm_history", max_chars=150)
    assert isinstance(mem, VdmHistoryMemory)
    assert mem.after_turn(1, None).text is None
    t2 = mem.after_turn(2, "The bowl moved\n to plate 3.")
    assert t2.text.endswith("turn 2: The bowl moved to plate 3.")
    mem.after_turn(3, "Nothing changed.")
    t4 = mem.after_turn(4, "The second bowl is now on plate 1.")
    assert len(t4.text) <= 150
    assert "turn 4: The second bowl" in t4.text and "turn 2:" not in t4.text
    assert "earlier turns omitted" in t4.text
    assert t4.signals == [] and mem.check_finish() is None

    long = make_memory("vdm_history", max_chars=100)
    text = long.after_turn(1, "x" * 500).text
    assert len(text) <= 100 and text.endswith("x")

    mem.reset()
    assert mem.entries == []


def test_none_mode_and_unknown_mode():
    mem = make_memory("none")
    assert isinstance(mem, NoMemory)
    mem.reset(object())
    assert mem.initial_text() is None and mem.after_turn(1, "vdm").text is None
    assert mem.check_finish() is None
    with pytest.raises(ValueError):
        make_memory("graph")


def test_visible_variant_not_implemented():
    with pytest.raises(NotImplementedError):
        BgOracleMemory(visibility="visible").reset(source=FakeSource())


# --- LiberoMemSource on a fake LIBERO problem ------------------------------------------------

def test_libero_mem_source_reads_problem():
    calls = []

    def eval_predicate(state):
        calls.append(state)
        if state[0] == "in":
            raise KeyError("no contain geometry")
        return state == ["on", "akita_black_bowl_1", "plate_1"]

    body = {"robot0_base": 0, "akita_black_bowl_1": 1, "plate_1": 2}
    sim = SimpleNamespace(
        model=SimpleNamespace(body_name2id=lambda n: body[n]),
        data=SimpleNamespace(
            xpos=np.array([[1.0, 0.0, 0.0], [0, 0, 0], [0, 0, 0]]),
            xquat=np.array([[0.7071068, 0, 0, 0.7071068], [1, 0, 0, 0], [1, 0, 0, 0]]),  # base yawed 90 deg
            body_xpos=np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.5], [2.0, 0.0, 0.0]]),
        ),
    )
    problem = SimpleNamespace(
        objects_dict={"akita_black_bowl_1": object(), "plate_1": object()},
        fixtures_dict={"kitchen_table": object()},
        object_states_dict={"akita_black_bowl_1": 0, "plate_1": 0, "kitchen_table": 0,
                            "plate_1_contain_region": 0},
        obj_body_id={"akita_black_bowl_1": 1, "plate_1": 2},
        parsed_problem={"goal_state": [["sequence", ["and", ["on", "akita_black_bowl_1", "plate_1"]]]]},
        sim=sim,
        robots=[SimpleNamespace(gripper="g")],
        _eval_predicate=eval_predicate,
        _check_grasp=lambda gripper, object_geoms: False,
        _satisfied_subgoals=[["and", ["on", "akita_black_bowl_1", "plate_1"]]],
        _overshot=False,
    )
    low = SimpleNamespace(handle=SimpleNamespace(env=SimpleNamespace(env=problem)))
    src = LiberoMemSource(low)
    pos = src.positions()
    np.testing.assert_allclose(pos["akita_black_bowl_1"], (1.0, 0.0, 0.5), atol=1e-6)
    np.testing.assert_allclose(pos["plate_1"], (0.0, -1.0, 0.0), atol=1e-6)
    assert src.contain_regions() == ["plate_1_contain_region"]
    assert src.completed_steps() == [(("on", "akita_black_bowl_1", "plate_1"),)]

    mem = BgOracleMemory()
    mem.reset(low)
    text = mem.initial_text()
    assert "akita_black_bowl_1 at (1.00, 0.00, 0.50) on plate_1" in text
    assert "all steps verified complete" in text
    # The In predicate raised once: it is not evaluated again.
    n_in = sum(1 for c in calls if c[0] == "in")
    mem.after_turn(1)
    assert sum(1 for c in calls if c[0] == "in") == n_in
