"""Regression tests for bggen-fix3 (batch1 4350313, kubectl traces):

A. 'in' (container): release geometry always computed and filling '?' (t08 d1), an Observe 0 against inside geometry
   is a conflict re-checked before lowering (t08 d2 / d3), inside = the container's 3D range, not dz to its median
   point (t08 d3), goal steps unobservable until the end counted apart
B. B1K alias duplicates: caps count expected names (fridge_A / fridge_B), one detection never binds AND opens a new
   track, caps from init, a held object follows the hand and is found again at the release place, mode A backward
   same-category merge
C. lifted 4.3 cm and put back on the same plate: a grasp, not a retracted close; a retracted bowl stays pick / place
"""

from __future__ import annotations

import json

import numpy as np

from capx.bggen import aggregate as agg
from capx.bggen import backward as bw
from capx.bggen.configs import b1k as b1k_cfg
from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.fake_models import FakeServer
from capx.bggen.generator import Generator, Row
from capx.bggen.geometry import Geometry
from capx.bggen.graph import BeliefGraph, GoalTracker
from capx.bggen.models import Models
from capx.bggen.perception import Detection
from capx.bggen.pipeline import run_episode
from capx.bggen.records import load_episode
from capx.bggen.tracks import Tracker
from capx.bggen.verifier import Verifier
from tests.bggen_fakes import BOWL_Z, HELD_W, HOME, OPEN_W, FakeDetector, SwapWorld, TruthBrain, make_world, write_swap_episode


def _jsonl(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _det(cat, x, y, z=0.04, size=(0.12, 0.12, 0.05), score=0.9, lo=None, hi=None):
    d = Detection(cat, score, (100 - 200 * y, 100 * x), (0, 0, 1, 1), np.array([x, y, z], float), np.array(size, float))
    d.lo, d.hi = (None if lo is None else np.array(lo, float)), (None if hi is None else np.array(hi, float))
    return d


# --- A1 / R9: t08 d1 -- every answer '?', nobody asked the geometry ----------------------------------------------

class _UnknownAfterStart(TruthBrain):
    """The VDM answers '?' after t=0 (t08 d1: in(cheese, basket_A / B) and on(cheese, table) all '?')."""

    def answer(self, stmt, t):
        return super().answer(stmt, t) if t == 0 else "?"


def test_release_geometry_is_computed_without_the_agent_and_fills_unknown(tmp_path):
    w = write_swap_episode(tmp_path)
    brain = _UnknownAfterStart(w, agent_reply=json.dumps({"calls": [{"tool": "observe", "statements": ["(handempty)"]}],
                                                          "why": "nothing about the bowl"}))
    with FakeServer(brain) as srv:
        run_episode(tmp_path, tmp_path / "out", lm_cfg("B"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    trace = _jsonl(tmp_path / "out" / "trace.jsonl")
    rel = next(t for t in trace if t["event"]["kind"] == "release" and t["event"]["obj"] == "bowl_A")
    assert rel["plan"]["geometry"] == []  # the agent asked no geometry
    assert "(on bowl_A plate_A)" in rel["tools"]["geometry"] and "(on bowl_A plate_A)" in rel["plan"]["forced"]
    r9 = [d for d in rel["decisions"] if d["rule"] == "R9"]
    assert [d["key"] for d in r9] == ["(on bowl_A plate_A)"] and rel["graph"]["(on bowl_A plate_A)"] == [0.7, "exp"]


# --- A3 / R10: t08 d2 (2D geometry) and d3 (depth) ----------------------------------------------------------------

K_IN = ("in", "cream_cheese_A", "basket_B")
CATS = {"cream_cheese_A": "cream_cheese", "basket_B": "basket"}


def test_in_conflict_with_2d_geometry_is_rechecked_before_lowering():
    g = BeliefGraph()
    v = Verifier(g)
    g.set(K_IN, 0.7, "exp", 198)  # d2: R4 at the release (geometry center 0.051 < r 0.074)
    geo2d = {K_IN: {"mode": "2d", "overlap": 0.8, "bottom_contact": True, "h_overlap": 0.9, "positive": True}}
    v.begin()
    v.observe({K_IN: "0"}, 228, {"cream_cheese_A"}, geo2d, categories=CATS)  # d2: next keyframe VDM in=0
    assert g.p(K_IN) == 0.5 and g.get(K_IN).source == "conflict" and v.new_conflicts == [K_IN]
    assert K_IN in v.recheck_keys() and any("re-checking" in s for s in v.s4)
    assert not v.recheck(K_IN, "?", 228, "look robot0_eye_in_hand")  # the wrist camera cannot tell either
    assert g.p(K_IN) == 0.5
    v.begin()
    v.observe({K_IN: "0"}, 258, set(), geo2d, categories=CATS)  # asked again at the next event: 0 again -> lowered
    assert g.p(K_IN) == 0.1 and [d.rule for d in v.decisions] == ["R10-confirm"] and K_IN not in v.recheck_keys()
    # a re-check that sees it inside raises it
    g2 = BeliefGraph()
    v2 = Verifier(g2)
    v2.observe({K_IN: "0"}, 228, {"cream_cheese_A"}, geo2d, categories=CATS)
    assert v2.recheck(K_IN, "1", 228, "hindsight") and g2.p(K_IN) == 0.9 and g2.get(K_IN).source == "obs"
    # no geometry support: a plain Observe 0 (R1), as before
    g3 = BeliefGraph()
    Verifier(g3).observe({K_IN: "0"}, 228, {"cream_cheese_A"}, {K_IN: {"mode": "2d", "overlap": 0.1}}, categories=CATS)
    assert g3.p(K_IN) == 0.1


def _basket_world(top: float):
    """d3: basket median point at z 0.05; the cheese center 0.133 m above it and 0.038 m away horizontally."""
    tr = Tracker(lm_cfg().tracks)
    tr.initialize(0, "cam", [
        _det("basket", 0.60, 0.0, 0.05, (0.25, 0.18, 0.08), lo=(0.475, -0.09, 0.0), hi=(0.725, 0.09, top)),
        _det("cream_cheese", 0.60, 0.038, 0.183, (0.06, 0.04, 0.03))])
    return tr, Geometry(tr)


def test_inside_uses_the_containers_3d_range_not_dz_to_its_median_point():
    tr, geo = _basket_world(top=0.16)  # mask depth points reach 16 cm (rim / handle)
    k = ("in", "cream_cheese_A", "basket_A")
    g = geo.pair(k)
    assert g["mode"] == "depth" and g["dz"] == 0.133 and g["d_xy"] == 0.038
    assert g["inside"] is True and g["positive"] is True and g["z_range"] == [0.0, 0.16]
    # an Observe 0 against it -> R10 conflict, not 0.1
    gr = BeliefGraph()
    v = Verifier(gr)
    v.observe({k: "0"}, 235, {"cream_cheese_A"}, {k: g}, categories={"cream_cheese_A": "cream_cheese", "basket_A": "basket"})
    assert gr.p(k) == 0.5 and v.new_conflicts == [k]
    # a low container (top 8 cm): 18.3 cm is above it -> outside, the Observe 0 stands
    tr2, geo2 = _basket_world(top=0.08)
    g2 = geo2.pair(k)
    assert g2["inside"] is False and g2["inside_xy"] is True and g2["inside_z"] is False


def test_recheck_uses_hindsight_in_mode_a(tmp_path):
    """The generator's R10 re-check: no other camera in the swap episode, mode A -> a later frame of the quiet interval."""
    w = write_swap_episode(tmp_path)
    with FakeServer(TruthBrain(w)) as srv:
        gen = Generator(load_episode(tmp_path), FakeDetector(w), Models.from_env(env=srv.env()), lm_cfg("A"))
        gen.initialize()
        k = ("on", "bowl_A", "plate_B")  # true until the first grasp
        gen.verifier.geo_conflict[k] = 10
        gen.graph.set(k, 0.5, "conflict", 10)
        from capx.bggen.events import Event
        hind: dict = {}
        done, info, adopted = gen._recheck(Event(10, "keyframe"), gen.ep.frame(10, ["agentview"]), "agentview", [k], "check",
                                           w.grasp_t[0], hind)
    assert done == {k} and gen.graph.p(k) == 0.9 and hind == {k: "1"} and info["pending"] == [] and adopted == {k: ("1", "hindsight")}
    assert [d.rule for d in gen.verifier.decisions][-1] == "R10-resolve"


# --- A4: unobservable goal steps -------------------------------------------------------------------------------------

def test_goal_step_unobservable_when_its_object_is_held_until_the_end(tmp_path):
    w = write_swap_episode(tmp_path)
    with FakeServer(TruthBrain(w)) as srv:
        gen = Generator(load_episode(tmp_path), FakeDetector(w), Models.from_env(env=srv.env()), lm_cfg("A"))
        gen.initialize()
    gen.goal = GoalTracker([[(("on", "bowl_B", "plate_A"),), (("on", "bowl_A", "plate_C"),)]])
    gen.rows[-1].held = {"main": "bowl_A"}  # the demonstration ends with bowl_A in the hand
    un = gen.unobservable_steps()
    assert un == [{"branch": 0, "step": 1, "why": {"(on bowl_A plate_C)": "bowl_A held until the end"}}]
    s = gen.summary()
    assert s["goal"]["unobservable"] == un and s["goal"]["final_satisfied"] is False  # the goal decision is unchanged
    # aggregate: counted apart
    d = tmp_path / "runs" / "libero_mem_t08_demo_1"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({"goal": {"branches": [[["a"], ["b"]]], "best": 0, "done": [1],
                                                         "final_satisfied": False, "unobservable": un}, "models": {}}))
    (d / "eval.json").write_text(json.dumps({"goal_consistency": {"final_graph_satisfies_goal": False}}))
    res = agg.aggregate(tmp_path / "runs")
    assert res["all"]["goal_unobservable_steps"] == 1 and res["all"]["goal_unsatisfied_unobservable"] == 1
    assert "1 unobservable" in agg.to_markdown(res)


# --- B: B1K alias duplicates ---------------------------------------------------------------------------------------

FRIDGE = (0.8, 0.7, 1.8)


def test_cap_counts_the_expected_name_one_detection_never_binds_and_opens_a_new_track():
    tr = Tracker(b1k_cfg().tracks)
    tr.initialize(0, "head", [_det("countertop", 1.0, 0.0, 0.9, (2.0, 0.6, 0.9))])
    assert tr.expect("fridge") == "fridge_A"
    # t=386 base_stop: two fridge masks in one frame (door / body, far apart): one binds fridge_A, the other is no
    # second fridge (cap 1 counts fridge_A)
    rep = tr.update(386, "head", [_det("fridge", 3.0, 1.0, 0.9, FRIDGE, 0.9), _det("fridge", 5.0, 3.0, 0.9, FRIDGE, 0.8)])
    assert rep["new"] == [] and sorted(a for a in tr.tracks if a.startswith("fridge")) == ["fridge_A"]
    assert tr.counts["bound_expected"] == 1 and tr.counts["closed_dropped"] == 1
    # the agent's detect tool, same frame: another prompt's mask of the same fridge is a duplicate, not fridge_B
    rep = tr.update(386, "head", [_det("fridge", 3.1, 1.0, 0.9, FRIDGE)], detected_categories=["fridge"], keep_seen=True)
    assert rep["new"] == [] and "fridge_A" not in rep["matched"] and tr.counts["duplicates_dropped"] == 1
    assert tr.tracks["fridge_A"].seen_now and np.allclose(tr.tracks["fridge_A"].pos[:2], [3.0, 1.0])


def test_init_caps_and_a_full_category_goes_to_the_nearest_track_with_distance():
    tr = Tracker(b1k_cfg().tracks)
    tr.initialize(0, "head", [_det("countertop", 1.0, 0.0, 0.9, (2.0, 0.6, 0.9))])
    tr.expect("hotdog", "hotdog_A")
    tr.expect("hotdog", "hotdog_B")
    tr.set_caps({"hotdog": len(tr.named("hotdog"))})
    assert tr.caps["hotdog"] == 2
    tr.update(100, "head", [_det("hotdog", 1.0, 0.0, 0.95, (0.15, 0.05, 0.05))])
    tr.update(200, "head", [_det("hotdog", 1.0, 0.1, 0.95, (0.15, 0.05, 0.05))])  # 0.10 m: reacquired as hotdog_A
    rep = tr.update(300, "head", [_det("hotdog", 2.0, 0.5, 0.95, (0.15, 0.05, 0.05)), _det("hotdog", 1.0, 0.1, 0.95, (0.15, 0.05, 0.05))])
    assert rep["new"] == [] and sorted(a for a in tr.tracks if a.startswith("hotdog")) == ["hotdog_A", "hotdog_B"]
    # t=3364 (1.53 m from hotdog_A's last place): both names taken by now -> the nearest free track, distance logged
    rep = tr.update(3364, "head", [_det("hotdog", 2.5, 1.6, 0.95, (0.15, 0.05, 0.05))])
    assert rep["new"] == [] and tr.counts["capped_assigned"] >= 1
    last = tr.capped[-1]
    assert last["t"] == 3364 and last["why"].startswith("cap full (2 hotdog): the nearest existing track, distance ")
    assert last["why"].endswith(" m")


def test_held_object_follows_the_hand_and_is_found_again_at_the_release_place():
    tr = Tracker(b1k_cfg().tracks)
    tr.initialize(0, "head", [_det("countertop", 1.0, 0.0, 0.9, (2.0, 0.6, 0.9))])
    a = tr.expect("hotdog", "hotdog_A")
    eef0 = np.array([1.0, 0.0, 1.0])
    tr.hold(a, "left", eef0)  # grasped before it was ever located (no 3D position)
    assert tr.tracks[a].held_by == "left" and np.allclose(tr.tracks[a].pos, eef0)
    eef1 = np.array([3.6, 2.6, 1.0])  # carried 3.7 m (B1K t=4982: hotdog_E 2.6-3.8 m from the others)
    tr.follow("left", eef1)
    assert np.allclose(tr.tracks[a].pos, eef1)
    assert tr.release("left", 4900, eef1) == a
    rep = tr.update(4982, "head", [_det("hotdog", 3.65, 2.62, 0.95, (0.15, 0.05, 0.05))])
    assert rep["new"] == [] and (a in rep["matched"] or [m["into"] for m in rep["merged"]] == [a])
    assert tr.tracks[a].seen_now and tr.tracks[a].held_by is None


def _row(t, held=None, kind="keyframe", obj=None, belief=None):
    return Row(t, kind, "left", obj, [], belief or {}, [], "0/1", 0, 1, "?", [], {"left": held})


def test_backward_merges_aliases_never_seen_together_after_a_base_move():
    sight = {"hotdog_A": {"category": "hotdog", "seen": [100, 200], "pos": [[1, 0, 1], [1, 0.1, 1]]},
             "hotdog_B": {"category": "hotdog", "seen": [100, 200, 3400], "pos": [[1, 0.3, 1]] * 3},
             "hotdog_C": {"category": "hotdog", "seen": [3364, 3500], "pos": [[2.5, 1.6, 1], [2.5, 1.6, 1]]}}
    k_old, k_new = ("on", "hotdog_A", "countertop_A"), ("on", "hotdog_C", "countertop_A")
    rows = [_row(100, belief={k_old: (0.9, "obs")}), _row(3364, belief={k_old: (0.9, "mem"), k_new: (0.1, "obs")}),
            _row(3500, belief={k_old: (0.9, "mem"), k_new: (0.1, "obs")})]
    rows[1].s4 = ["hotdog_C may be resting on the rim of countertop_A"]
    base = [(1000, 3000)]
    m = bw.merge_tracks_backward(sight, rows, base, {"hotdog": 2})
    assert [(x["from"], x["into"]) for x in m] == [("hotdog_C", "hotdog_A")]  # B was seen together with C: not B
    assert "base moved" in m[0]["why"]
    res = bw.backward(rows, [], {"hotdog_A": "hotdog", "hotdog_B": "hotdog", "hotdog_C": "hotdog", "countertop_A": "countertop"},
                      base, sight=sight, caps={"hotdog": 2})
    assert res.merges_backward == m and any(x["rule"] == "B1b" for x in res.log)
    assert res.rows[1].belief[k_old] == (0.1, "obs")  # the later alias's (observed) edge wins over the stale one
    # provenance for the evaluation: the statement came from hotdog_C, the survivor's stale entry was displaced
    assert res.rows[1].b1b_origin == {k_old: ["hotdog_C"]} and res.rows[1].b1b_shadowed == [(k_old, (0.9, "mem"), [])]
    assert res.rows[0].b1b_origin == {} and res.rows[0].b1b_shadowed == []
    assert not any("hotdog_C" in str(k) for r in res.rows for k in r.belief)
    assert res.rows[1].s4 == ["hotdog_A may be resting on the rim of countertop_A"]
    assert bw.check_forbidden(rows, res.rows, renamed={"hotdog_C": "hotdog_A"}) == []
    # no base move and nothing carried in the gap: two objects
    assert bw.merge_tracks_backward(sight, rows, [], {"hotdog": 2}) == []
    # the count is known and not exceeded (one hot dog named): no merge needed for B / C either way
    assert bw.merge_tracks_backward({k: v for k, v in sight.items() if k != "hotdog_B"}, rows, base, {"hotdog": 2}) == []


def test_backward_merge_after_a_carry():
    sight = {"hotdog_A": {"category": "hotdog", "seen": [100], "pos": [[1, 0, 1]]},
             "hotdog_E": {"category": "hotdog", "seen": [4982], "pos": [[3.6, 2.6, 1]]}}
    rows = [_row(100), _row(200, held="hotdog_A", kind="grasp", obj="hotdog_A"), _row(4900, kind="release", obj="hotdog_A"),
            _row(4982)]
    m = bw.merge_tracks_backward(sight, rows, [], None)
    assert [(x["from"], x["into"]) for x in m] == [("hotdog_E", "hotdog_A")] and "carried" in m[0]["why"]


def _split_run(tmp_path, into):
    """A swap run whose forward pass "lost" bowl_B near the end and called it bowl_C (trace + forward rows), and whose
    backward rows merged bowl_C into ``into`` (provenance as B1b writes it)."""
    w = write_swap_episode(tmp_path)
    out = tmp_path / "out"
    with FakeServer(TruthBrain(w)) as srv:
        run_episode(tmp_path, out, lm_cfg("A"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    cut = w.release_t[2] + 20  # bowl_B (bowl_2) has most of its frames under its own name: it keeps the map
    trace = _jsonl(out / "trace.jsonl")
    for ln in trace:
        if ln["t"] > cut:
            ln["tracks"] = [{**t, "alias": "bowl_C" if t["alias"] == "bowl_B" else t["alias"]} for t in ln["tracks"]]
    (out / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in trace))
    rows = _jsonl(out / "rows_backward.jsonl")
    for r in rows:  # the statements of bowl_B after the cut came from bowl_C
        if r["t"] > cut:
            r["b1b_origin"] = {s: "bowl_C" for s in r["belief"] if "bowl_B" in s}
            r["b1b_shadowed"] = []
    (out / "rows_backward.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    log = json.loads((out / "backward_log.json").read_text())
    log["merges_backward"] = [{"t": cut + 1, "from": "bowl_C", "into": into}]
    (out / "backward_log.json").write_text(json.dumps(log))
    return out, rows, cut


def test_backward_scoring_uses_the_forward_map_and_a_right_merge_can_score_right(tmp_path):
    from capx.bggen.evaluate import evaluate

    out, rows, cut = _split_run(tmp_path, "bowl_B")
    res = evaluate(out, tmp_path)
    assert res["track_map"]["fixed"]["bowl_B"] == "bowl_2" and "bowl_C" not in res["track_map"]["fixed"]
    [c] = res["merges_backward"]
    assert (c["from"], c["into"], c["identity_from"], c["identity_into"], c["right"]) == ("bowl_C", "bowl_B", "bowl_2", "bowl_2", True)
    assert c["rows_moved"] == sum(1 for r in rows if r["t"] > cut and r["b1b_origin"])
    rb = res["forward+backward"]["relations"]
    assert rb["merge_wrong"] == 0 and rb["accuracy"] >= 0.98
    assert "bowl_C -> bowl_B: RIGHT" in (out / "eval.md").read_text()


def test_wrong_merge_makes_the_moved_statements_wrong(tmp_path):
    from capx.bggen.evaluate import evaluate, relation_accuracy

    out, rows, cut = _split_run(tmp_path, "bowl_A")  # bowl_C is bowl_2, bowl_A maps to bowl_1
    res = evaluate(out, tmp_path)
    [c] = res["merges_backward"]
    assert c["right"] is False and c["identity_from"] == "bowl_2" and c["identity_into"] == "bowl_1"
    rb = res["forward+backward"]["relations"]
    assert rb["merge_wrong"] > 0 and any(str(e["truth"]).startswith("wrong merge of bowl_C") for e in rb["errors"])
    # the population is unchanged: decided + undecided equals a scoring of the same rows without the merge verdict
    from capx.bggen.evaluate import TrackMap
    from capx.bggen.oracle import load_oracle

    oracle = load_oracle(tmp_path)
    tmap = TrackMap(_jsonl(out / "trace.jsonl"), oracle, 0.15, end_t=None)
    names = sorted({t["alias"] for ln in _jsonl(out / "trace.jsonl") for t in ln["tracks"]})
    arms = json.loads((out / "summary.json").read_text()).get("arms") or [None]
    plain = relation_accuracy(rows, oracle, tmap, names, arms, res["tolerance_frames"])
    assert rb["decided"] + rb["undecided"] == plain["decided"] + plain["undecided"]


def test_no_merge_backward_scoring_is_unchanged(tmp_path):
    from capx.bggen.evaluate import TrackMap, evaluate, relation_accuracy
    from capx.bggen.oracle import load_oracle

    w = write_swap_episode(tmp_path)
    out = tmp_path / "out"
    with FakeServer(TruthBrain(w)) as srv:
        run_episode(tmp_path, out, lm_cfg("A"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    res = evaluate(out, tmp_path)
    assert "merges_backward" not in res and "merge_wrong" not in res["forward+backward"]["relations"]
    trace, oracle = _jsonl(out / "trace.jsonl"), load_oracle(tmp_path)
    tmap = TrackMap(trace, oracle, 0.15, end_t=json.loads((out / "summary.json").read_text())["n_frames"])
    names = sorted({t["alias"] for ln in trace for t in ln["tracks"]} | {t["alias"] for t in json.loads((out / "summary.json").read_text())["tracks"]})
    arms = json.loads((out / "summary.json").read_text()).get("arms") or [None]
    direct = relation_accuracy(_jsonl(out / "rows_backward.jsonl"), oracle, tmap, names, arms, res["tolerance_frames"])
    assert json.dumps(res["forward+backward"]["relations"], sort_keys=True) == json.dumps(direct, sort_keys=True)


# --- P2: an R10 first answer that was not adopted is no B2 neighbour ----------------------------------------------------

def test_rejected_first_answer_is_not_a_b2_neighbour():
    k = ("in", "cream_cheese_A", "basket_B")

    def r(t, v, rejected=False, rechecked=None, p=0.9):
        row = Row(t, "keyframe", None, None, [(k, v)], {k: (p, "obs")}, [], "0/1", 0, 1, "?", [], {"main": None})
        if rejected:
            row.rejected = [k]
        if rechecked:
            row.rechecked = {k: rechecked}
        return row

    # t=228: the VDM said 0 against the geometry, the wrist camera could not tell (pending 0.5): not adopted
    rows = [r(228, "0", rejected=True, p=0.5), r(258, "1"), r(288, "0", p=0.1)]
    log: list = []
    bw.remove_noise(rows, log)
    assert log == [] and rows[1].observed == [(k, "1")]  # the 1 at t=258 has one real neighbour only: kept
    # without the flag the rejected 0 would have "corrected" it (the artificial fix the review found)
    rows2 = [r(228, "0", p=0.5), r(258, "1"), r(288, "0", p=0.1)]
    bw.remove_noise(rows2, log)
    assert [x["rule"] for x in log] == ["B2"]
    # an adopted look re-check counts as an answer at its row
    rows3 = [r(228, "0", rejected=True, rechecked="1"), r(258, "0", p=0.1), r(288, "1")]
    log3: list = []
    bw.remove_noise(rows3, log3)
    assert [(x["t"], x["was"], x["now"]) for x in log3] == [(258, "0", "1")]
    from capx.bggen.serialize import row_to_json

    j = row_to_json(rows3[0])
    assert j["rejected"] == ["(in cream_cheese_A basket_B)"] and j["rechecked"] == {"(in cream_cheese_A basket_B)": "1"}
    assert "rejected" not in row_to_json(rows3[1])


# --- C: lifted 4.3 cm and put back on the same plate (t02 demo_2) ---------------------------------------------------

def _put_back_world(lift: float) -> SwapWorld:
    """bowl_1 on plate_2: approach, close, lift ``lift`` m, come down 3 cm to the side and 2.5 cm higher than the close
    (the release EEF), the bowl back at its old place, open, go home."""
    w = SwapWorld()
    bowls = {"bowl_1": np.array([0.6, 0.0, BOWL_Z]), "bowl_2": np.array([0.6, -0.2, BOWL_Z])}
    st = {"eef": HOME.copy(), "width": OPEN_W, "held": None}

    def push(n=1):
        for _ in range(n):
            if st["held"] is not None:
                bowls[st["held"]] = st["eef"].copy()
            w.eef.append(st["eef"].copy())
            w.width.append(st["width"])
            w.bowls.append({k: v.copy() for k, v in bowls.items()})
            w.held.append(st["held"])

    def move(to, n):
        a = st["eef"].copy()
        for i in range(n):
            st["eef"] = a + (np.asarray(to, float) - a) * (i + 1) / n
            push()

    push(20)
    old = bowls["bowl_1"].copy()
    move([0.6, 0.0, 0.20], 40)
    move(old, 20)
    for v in np.linspace(OPEN_W, HELD_W, 10):
        st["width"] = float(v)
        push()
    st["held"] = "bowl_1"
    w.grasp_t.append(len(w.eef) - 1)
    push(6)
    move(old + [0, 0, lift], 20)
    push(10)
    move(old + [0.03, 0.0, 0.025], 20)
    st["held"] = None
    bowls["bowl_1"] = old.copy()  # set down where it was
    w.release_t.append(len(w.eef))
    for v in np.linspace(HELD_W, OPEN_W, 10):
        st["width"] = float(v)
        push()
    move(st["eef"] + [0, 0, 0.2], 20)
    move(HOME, 30)
    push(20)
    return w


def _run_put_back(tmp_path, lift):
    w = _put_back_world(lift)
    g, r = w.grasp_t[0], w.release_t[0]
    write_swap_episode(tmp_path, w)
    with FakeServer(TruthBrain(w)) as srv:  # the bowl is not detected while the hand is closed (no check before the release)
        s = run_episode(tmp_path, tmp_path / "out", lm_cfg("A"), Models.from_env(env=srv.env()),
                        FakeDetector(w, hide=lambda t, n: g - 8 <= t < r - 3 and n == "bowl_1"), "libero_mem")
    return s, json.loads((tmp_path / "out" / "backward_log.json").read_text())


def test_lifted_and_put_back_is_a_grasp_with_pick_and_place_labels(tmp_path):
    s, log = _run_put_back(tmp_path, 0.043)
    c = s["grasp_checks"][0]
    assert (c["result"], c["at"], c["evidence"]) == ("confirmed_put_back", "release", "lifted_and_put_back")
    assert c["at_old"] == 1 and c["lifted"] is False and c["max_lift"] >= 0.04  # what used to retract it
    seg = log["segments"][0]
    assert (seg["kind"], seg["retracted"], seg["label_pick"], seg["label_place"]) == \
        ("object", False, "pick(bowl_A, plate_B)", "place(bowl_A, plate_B)")
    assert seg["effect"] == []  # same plate: no state change, reported as such (B3)


def test_a_small_rise_is_still_retracted_but_keeps_pick_and_place(tmp_path):
    s, log = _run_put_back(tmp_path / "x", 0.02)  # 2 cm < carry_lift 3 cm: fingers closed on nothing
    assert s["grasp_checks"][0]["result"] == "retracted"
    seg = log["segments"][0]
    assert seg["kind"] == "object" and seg["retracted"] is True
    assert seg["label_pick"] != seg["label_place"] and not seg["label_pick"].startswith("manipulate")


# --- review 2: chained merges, shadowed chains, re-check 0 counted once -------------------------------------------------

def test_chained_merges_keep_every_alias_and_rename_shadowed_entries():
    p = "countertop_A"
    kw, kx, ky = ("on", "hotdog_W", p), ("on", "hotdog_X", p), ("on", "hotdog_Y", p)
    row = Row(10, "keyframe", None, None, [], {ky: (0.9, "mem"), kx: (0.8, "mem"), kw: (0.1, "obs")}, [], "0/1", 0, 1, "?",
              [], {"main": None})
    bw.unify_ids([row], [{"from": "hotdog_W", "into": "hotdog_X", "prefer_from": True},
                         {"from": "hotdog_X", "into": "hotdog_Y", "prefer_from": True}], [])
    assert row.belief == {ky: (0.1, "obs")}  # W's observed value survives both merges
    assert row.b1b_origin == {ky: ["hotdog_W", "hotdog_X"]}
    # X's own entry (displaced by W at the first merge) is renamed by the second one and keeps its path
    assert sorted(row.b1b_shadowed, key=str) == sorted([(ky, (0.8, "mem"), ["hotdog_X"]), (ky, (0.9, "mem"), [])], key=str)
    from capx.bggen.serialize import row_to_json

    j = row_to_json(row)
    assert j["b1b_origin"] == {"(on hotdog_Y countertop_A)": ["hotdog_W", "hotdog_X"]}


def _chain_run(tmp_path, mid_src):
    """Forward names: bowl_B, then (cut1, cut2] ``mid_src``'s track called bowl_W, after cut2 bowl_B called bowl_C.
    Backward merges C -> W -> B (a 2-step chain over 3 aliases); bowl_B's statements after cut2 passed through C and W."""
    w = write_swap_episode(tmp_path)
    out = tmp_path / "out"
    with FakeServer(TruthBrain(w)) as srv:
        run_episode(tmp_path, out, lm_cfg("A"), Models.from_env(env=srv.env()), FakeDetector(w), "libero_mem")
    cut1 = w.release_t[1] + 40
    cut2 = w.release_t[2] + 20
    trace = _jsonl(out / "trace.jsonl")
    for ln in trace:
        ren = {mid_src: "bowl_W"} if cut1 < ln["t"] <= cut2 else {"bowl_B": "bowl_C"} if ln["t"] > cut2 else {}
        ln["tracks"] = [{**t, "alias": ren.get(t["alias"], t["alias"])} for t in ln["tracks"]]
    (out / "trace.jsonl").write_text("".join(json.dumps(x) + "\n" for x in trace))
    rows = _jsonl(out / "rows_backward.jsonl")
    for r in rows:
        if r["t"] > cut2:
            r["b1b_origin"] = {s: ["bowl_C", "bowl_W"] for s in r["belief"] if "bowl_B" in s}
            r["b1b_shadowed"] = []
    (out / "rows_backward.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    log = json.loads((out / "backward_log.json").read_text())
    log["merges_backward"] = [{"t": cut2 + 1, "from": "bowl_C", "into": "bowl_W"}, {"t": cut1 + 1, "from": "bowl_W", "into": "bowl_B"}]
    (out / "backward_log.json").write_text(json.dumps(log))
    return out


def test_chain_of_right_merges_scores_right(tmp_path):
    from capx.bggen.evaluate import evaluate

    res = evaluate(_chain_run(tmp_path, "bowl_B"), tmp_path)
    assert [(c["from"], c["final"], c["identity_from"], c["identity_into"], c["right"]) for c in res["merges_backward"]] == \
        [("bowl_C", "bowl_B", "bowl_2", "bowl_2", True), ("bowl_W", "bowl_B", "bowl_2", "bowl_2", True)]
    rb = res["forward+backward"]["relations"]
    assert rb["merge_wrong"] == 0 and rb["accuracy"] >= 0.98


def test_chain_with_one_wrong_merge_makes_its_statements_wrong(tmp_path):
    from capx.bggen.evaluate import evaluate

    out = _chain_run(tmp_path, "bowl_A")  # bowl_W was really bowl_A's object (bowl_1): W -> B is wrong, C -> W -> B right for C
    res = evaluate(out, tmp_path)
    checks = {c["from"]: c for c in res["merges_backward"]}
    assert checks["bowl_C"]["right"] is True and checks["bowl_W"]["right"] is False
    assert checks["bowl_W"]["identity_from"] == "bowl_1" and checks["bowl_W"]["identity_into"] == "bowl_2"
    rb = res["forward+backward"]["relations"]
    # statements that passed through C and W: the earlier-right / later-wrong path is wrong (not scored as C's right merge)
    assert rb["merge_wrong"] > 0 and all("bowl_W" in str(e["truth"]) for e in rb["errors"] if "wrong merge" in str(e["truth"]))
    assert "bowl_W -> bowl_B: WRONG" in (out / "eval.md").read_text()


def test_adopted_recheck_zero_counts_once_and_an_isolated_zero_is_corrected():
    k = ("in", "cream_cheese_A", "basket_B")

    def r(t, v, p, **kw):
        row = Row(t, "keyframe", None, None, [(k, v)], {k: (p, "obs")}, [], "0/1", 0, 1, "?", [], {"main": None})
        for a, b in kw.items():
            setattr(row, a, b)
        return row

    # 1, then 0 confirmed by the wrist re-check (first 0 superseded), then 1: one 0 at t=258, isolated -> corrected
    rows = [r(228, "1", 0.9), r(258, "0", 0.1, rejected=[k], rechecked={k: "0"}), r(288, "1", 0.9)]
    log: list = []
    bw.remove_noise(rows, log)
    assert [(x["t"], x["was"], x["now"]) for x in log] == [(258, "0", "1")]
    assert rows[1].rechecked == {k: "1"} and rows[1].belief[k][0] == 0.9
