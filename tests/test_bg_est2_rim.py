"""bg-est2-rim: geometric veto of an Observe 1 for an object resting on a support's rim, marks only for
the tracks detected now (none on the arm), the "on means flat" Observe rule, the veto logs and the
shadow's veto accuracy. Numbers from the 20261101_bgest2_vocab_smoke task_06 diagnosis."""

from __future__ import annotations

import itertools
import json

import numpy as np
import pytest

from capx.belief.estimator2 import Est2Config, flat_veto
from capx.belief.prompts_v2 import ON_RULE, marks_now_text, observe_initial_prompt, observe_turn_prompt
from capx.belief.som import mark_radius
from tests.belief_fakes import SHORT_IDS, FakeWorld, swap_world
from tests.bg_est2_fakes import SWAP2_GOAL, AnyDetector, declared
from tests.test_bg_est2 import A, SWAP, make_mem, place_code, setup, signals, turn

CM = 0.01
CFG = Est2Config()
FLAT = CFG.flat_dz["bowl/plate"]


def veto(d_xy_cm: float, r_cm: float, dz_cm: float | None) -> list[str]:
    return flat_veto(d_xy_cm * CM, r_cm * CM, None if dz_cm is None else dz_cm * CM, FLAT,
                     CFG.veto_xy_margin, CFG.veto_dz_margin)


# --- the criterion on the trial data ----------------------------------------------------------------

def test_defaults_match_the_spec():
    assert CFG.veto and CFG.veto_xy_margin == pytest.approx(0.005) and CFG.veto_dz_margin == pytest.approx(0.010)
    assert FLAT == pytest.approx(0.018)


@pytest.mark.parametrize("d_xy,r,dz", list(itertools.product((6.4, 6.85, 7.3), (5.4, 5.85, 6.3), (3.0, 3.3, 3.6))))
def test_rim_cases_of_task06_are_vetoed(d_xy, r, dz):
    """vocab smoke t06: the 9 'VDM 1, truth 0' on-statements had d_xy 6.4-7.3 cm, r 5.4-6.3 cm, dz 3.0-3.6 cm
    (every combination of the ranges, so each of the 9 is covered)."""
    why = veto(d_xy, r, dz)
    assert "dz" in why  # dz alone separates them (> 1.8 + 1.0 cm)


def test_plate_prompt_run_rim_case_is_vetoed():
    """The plate-prompt run's one wrong 1: d_xy 7.2 > r 5.1 (+ 0.5), whatever dz was."""
    assert "d_xy" in veto(7.2, 5.1, None) and "d_xy" in veto(7.2, 5.1, 1.8)


@pytest.mark.parametrize("d_xy,r,dz", list(itertools.product((0.0, 1.5, 3.0, 4.5), (5.1, 5.4, 6.3), (1.2, 1.8, 2.4, 2.75))))
def test_flat_placements_are_not_vetoed(d_xy, r, dz):
    """Bowls resting flat (plate run, truth-1 23/23): centered within the plate, dz about 1.8 cm."""
    assert veto(d_xy, r, dz) == []


def test_boundaries():
    assert veto(5.9, 5.4, 1.8) == [] and veto(6.0, 5.4, 1.8) == ["d_xy"]  # r + 0.5 cm
    assert veto(0.0, 6.0, 2.7) == [] and veto(0.0, 6.0, 2.9) == ["dz"]  # 1.8 + 1.0 cm
    assert flat_veto(0.0, 0.06, 0.20, None) == []  # no flat dz known for the pair: d_xy only


# --- the veto in the estimator ----------------------------------------------------------------------

PLATE_R = 0.06  # the real plates' estimated radius (5.4-6.3 cm), not the fake vocabulary's 0.10


class RealSizeDetector(AnyDetector):
    def detect(self, obs, categories):
        out = super().detect(obs, categories)
        for d in out:
            if d.category == "plate":
                d.size = np.array([2 * PLATE_R, 2 * PLATE_R, 0.02])
        return out


def put(world, name, support, dx=0.0, dz=0.018):
    """Move ``name`` (sim name) to ``support`` + (dx, 0, dz) with the gripper raised and open."""
    p = world.objects[support]["pos"]
    world.objects[name]["pos"] = np.array([p[0] + dx, p[1], p[2] + dz])
    world.held, world.gripper = None, 1.0
    world.eef = np.array([0.3, -0.3, 0.4])


def test_rim_place_vetoed_uncertain_then_cleared():
    world = swap_world()
    est, rec, vdm, _, first = setup(world, SWAP2_GOAL, SWAP, detector=RealSizeDetector(world))
    assert first.vetoes == []  # bowls flat on their plates at turn 0
    world.pick_place(A, "plate_3")
    put(world, A, "plate_3", dx=0.068, dz=0.033)  # on the rim: d_xy 6.8 > 6.0 + 0.5, dz 3.3 > 1.8 + 1.0
    vdm.sticky = {"(on bowl_A plate_C)": "1"}
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    key = ("on", "bowl_A", "plate_C")
    assert [v["key"] for v in r.vetoes] == ["(on bowl_A plate_C)"]
    v = r.vetoes[0]
    assert v["observe"] == "1" and v["why"] == ["d_xy", "dz"] and v["kind"] == "rim"
    assert v["d_xy"] == pytest.approx(0.068, abs=1e-3) and v["r"] == pytest.approx(PLATE_R)
    assert v["dz"] == pytest.approx(0.033, abs=1e-3) and v["flat_dz"] == pytest.approx(0.018)
    assert v["xy_limit"] == pytest.approx(0.065) and v["dz_limit"] == pytest.approx(0.028)
    e = est.graph.edges[key]
    assert e.p == 0.5 and e.source == "conflict" and est.origin[key] == "conflict"
    # Not confirmed: the place stays pending, completion is suppressed, S4 says why.
    assert 0 not in est.confirmed_by and r.suppress_completion
    assert [s["status"] for s in r.judged][-1] == "pending"
    assert "bowl_A may be resting on the rim of plate_C (tilted), not on it" in " ".join(r.s4_items)
    fired = signals(est, r)
    assert "S4" in fired and "S5b" not in fired
    # Logged in the key record (belief_trace) and the turn log (memory_est).
    rec_k = est.key_records()["(on bowl_A plate_C)"]
    assert rec_k["source"] == "conflict" and rec_k["veto"]["why"] == ["d_xy", "dz"]
    assert est.trace()["vetoes"]["(on bowl_A plate_C)"]["kind"] == "rim"
    assert json.loads(json.dumps(r.to_log()))["vetoes"][0]["key"] == "(on bowl_A plate_C)"
    # ? next turn: still vetoed (uncertain), not back to an unqualified memory.
    vdm.sticky = {"(on bowl_A plate_C)": "?"}
    r = turn(est, world, rec, 2, declared("look(bowl_A)", "", "..."))
    assert key in est.veto and est.origin[key] == "conflict" and r.suppress_completion
    # The bowl settles flat: Observe 1 accepted, the veto is gone and the step is confirmed.
    put(world, A, "plate_3", dx=0.01, dz=0.018)
    vdm.sticky = {}
    r = turn(est, world, rec, 3, declared("look(bowl_A)", "", "..."))
    assert key not in est.veto and r.vetoes == [] and est.origin[key] == "vdm"
    assert est.graph.edges[key].p == pytest.approx(0.9) and est.confirmed_by.get(0) == "vdm"


def test_far_apart_keeps_the_wrong_mark_wording():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP, detector=RealSizeDetector(world))
    vdm.sticky = {"(on bowl_A plate_C)": "1"}  # bowl_A is still on plate_A, 0.4 m away
    r = turn(est, world, rec, 1, declared("place(bowl_A, plate_C)", "(on bowl_A plate_C) 0>1", "..."))
    assert r.vetoes and r.vetoes[0]["kind"] == "apart"
    assert any("a mark may be on the wrong object" in t for t in r.s4_items)


def test_observe_0_vs_geometry_on_unchanged():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP, detector=RealSizeDetector(world))
    world.pick_place(A, "plate_3")
    put(world, A, "plate_3", dx=0.0, dz=0.018)  # flat, centered: geometry says On
    vdm.sticky = {"(on bowl_A plate_C)": "0"}
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    e = est.graph.edges[("on", "bowl_A", "plate_C")]
    assert e.p == pytest.approx(0.1) and est.origin[e.key] == "vdm" and r.vetoes == []


def test_veto_off_and_unlisted_pair_measured():
    world = swap_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP, detector=RealSizeDetector(world), veto=False)
    world.pick_place(A, "plate_3")
    put(world, A, "plate_3", dx=0.068, dz=0.033)
    vdm.sticky = {"(on bowl_A plate_C)": "1"}
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    assert r.vetoes == [] and est.graph.edges[("on", "bowl_A", "plate_C")].p == pytest.approx(0.9)
    # A pair missing from flat_dz: its first accepted Observe 1 sets its flat dz (no dz check before).
    world = swap_world()
    est, rec, vdm, _, first = setup(world, SWAP2_GOAL, SWAP, detector=RealSizeDetector(world), flat_dz={})
    assert est.flat_measured["bowl/plate"] == pytest.approx(0.02) and first.vetoes == []
    world.pick_place(A, "plate_3")
    put(world, A, "plate_3", dx=0.0, dz=0.036)
    vdm.sticky = {"(on bowl_A plate_C)": "1"}
    r = turn(est, world, rec, 1, place_code("bowl_A", "plate_C"))
    assert r.vetoes and r.vetoes[0]["why"] == ["dz"] and r.vetoes[0]["flat_dz_source"] == "measured"


def test_shadow_veto_accuracy(tmp_path):
    mem, low, vdm = make_mem(tmp_path)
    world = low.world
    world.pick_place(SHORT_IDS["b1"], SHORT_IDS["p3"])
    put(world, SHORT_IDS["b1"], SHORT_IDS["p3"], dx=0.085, dz=0.033)  # off the plate for the truth (> 8 cm)
    low.history.append(low.supports())
    vdm.sticky = {"(on bowl_A plate_C)": "1"}
    mem.after_turn(1, "vdm text", executed=True, code=place_code("bowl_A", "plate_C"))
    log = mem.turn_log()
    assert log["est"]["vetoes"][0]["key"] == "(on bowl_A plate_C)"
    assert log["shadow"]["vetoes"] == [{"key": "(on bowl_A plate_C)", "why": ["dz"], "kind": "rim",
                                        "oracle": False, "right": True}]
    tot = mem.state()["shadow_totals"]
    assert tot["veto"] == [1, 1] and tot["veto_no_truth"] == 0
    s4 = [s for s in log["shadow"]["signals"] if s["id"] == "S4"]
    assert s4 and s4[0]["right"] is True  # the uncertainty was warranted
    line = json.loads((tmp_path / "belief_trace.jsonl").read_text().splitlines()[-1])
    k = line["keys"]["(on bowl_A plate_C)"]
    assert k["veto"]["dz"] == pytest.approx(0.033, abs=1e-3) and k["source"] == "conflict" and k["shadow"] is False
    assert line["est"]["vetoes"][0]["why"] == ["dz"]


# --- marks ------------------------------------------------------------------------------------------

W, H = 320, 256


def camera_obs(world):
    """The fake world's observation with a 320x256 front camera looking down from 1.5 m."""
    obs = FakeWorld.get_observation(world)
    obs["agentview"] = {
        "images": {"rgb": np.zeros((H, W, 3), np.uint8), "depth": np.ones((H, W), np.float32)},
        "intrinsics": np.array([[300.0, 0, W / 2], [0, 300.0, H / 2], [0, 0, 1]]),
        "pose_mat": np.array([[1.0, 0, 0, 0.5], [0, -1.0, 0, 0], [0, 0, -1.0, 1.5], [0, 0, 0, 1]]),
    }
    return obs


def m5_text(vdm):
    return "\n".join(c.get("text", "") for c in vdm.prompts[-1][1]["content"])


def turn_cam(est, world, rec, n, code=None):
    return est.end_turn(n, camera_obs(world), rec.drain(), code=code)


def cam_world():
    world = swap_world()
    world.get_observation = lambda: camera_obs(world)
    return world


def test_undetected_track_gets_no_mark_and_is_listed():
    world = cam_world()
    det = AnyDetector(world)
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP, detector=det)
    world.eef = np.array([0.3, -0.3, 0.4])  # out of the way
    det.miss = {"plate_3"}  # plate_C hidden this turn (by the arm, in the trial)
    r = turn_cam(est, world, rec, 1, declared("look(plate_C)", "", "..."))
    assert "plate_C" not in est.marked_now and r.marks_hidden == {"not_detected": ["plate_C"]}
    assert "Marks now: A = bowl_A (pixel" in m5_text(vdm) and "| 3: not detected" in m5_text(vdm)
    assert "3 = plate_C" not in m5_text(vdm)
    assert est.trace()["nodes"]["plate_C"]["marked_now"] is False
    # Statements about it stay: the VDM can answer ?.
    assert any("plate_C" in s for s in r.statements)


def test_no_mark_near_the_fingertip_except_the_held_object():
    world = cam_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    # The arm hovers low over the empty plate_C: its mark would be drawn on the gripper.
    plate_c = world.objects[SHORT_IDS["p3"]]["pos"]
    world.eef = plate_c + np.array([0.0, 0.0, 0.05])
    r = turn_cam(est, world, rec, 1, declared("look(plate_C)", "", "..."))
    assert r.marks_hidden.get("near_gripper") == ["plate_C"] and "plate_C" not in est.marked_now
    assert "3: under the gripper, not marked" in m5_text(vdm)
    # Holding bowl_A at the gripper: its own mark stays.
    world.pick_place(A, "plate_3")
    world.eef = world.objects[A]["pos"] + np.array([0.0, 0.0, 0.02])
    world.api["close_gripper"]()
    world.api["goto_pose"](world.eef + np.array([0.0, 0.0, 0.10]))
    r = turn_cam(est, world, rec, 2, declared("pick(bowl_A, plate_C)", "(inhand bowl_A) 0>1", "..."))
    assert est.robot["holding"] == "bowl_A" and "bowl_A" in est.marked_now


@pytest.mark.parametrize("seed", range(20))
def test_drawn_marks_never_near_the_gripper(seed):
    """Random fingertip positions over the table: no drawn mark (other than the held object's) lies within
    the guard radius of the fingertip's pixel."""
    rng = np.random.default_rng(seed)
    world = cam_world()
    est, rec, vdm, _, _ = setup(world, SWAP2_GOAL, SWAP)
    for n in range(1, 4):
        world.eef = np.array([rng.uniform(0.35, 0.65), rng.uniform(-0.3, 0.3), rng.uniform(0.02, 0.2)])
        turn_cam(est, world, rec, n, declared("look(bowl_A)", "", "..."))
        obs = camera_obs(world)
        gp = est._gripper_pixel(obs)
        hidden: dict = {}
        _, shown = est._marked(obs["agentview"]["images"]["rgb"], sorted(est.seen), gp, hidden)
        near = est.cfg.gripper_mark_radii * mark_radius((H, W, 3))
        for _, a, (u, v) in shown:
            if a != est._held_now():
                assert (u - gp[0]) ** 2 + (v - gp[1]) ** 2 >= near ** 2, (a, (u, v), gp)
        assert {a for _, a, _ in shown} | set(sum(hidden.values(), [])) == set(est.aliases)


def test_marks_now_text_near_gripper():
    assert marks_now_text([("A", "bowl_A", (1, 2))], ["B"], ["3"]) == \
        "A = bowl_A (pixel 1, 2) | B: not detected | 3: under the gripper, not marked"
    assert marks_now_text([("A", "bowl_A", (1, 2))], ["B"]) == "A = bowl_A (pixel 1, 2) | B: not detected"


# --- the "on means flat" rule -----------------------------------------------------------------------

def test_on_rule_in_m2_and_m5():
    assert ON_RULE == ('"on" means resting flat on top; an object leaning on the rim or tilted against it is 0, '
                       "and say so in Notes.")
    m2 = observe_initial_prompt("t", "data:x", [("A", "bowl_A", "black bowl", (1, 2))], ["(on bowl_A plate_A)"])
    m5 = observe_turn_prompt("t", None, "data:x", "A = bowl_A (pixel 1, 2)", None, ["(on bowl_A plate_A)"])
    for p in (m2, m5):
        text = "\n".join(c.get("text", "") for c in p[1]["content"])
        assert text.count(ON_RULE) == 1 and text.index(ON_RULE) < text.index("Statements:")


def test_other_modes_do_not_get_the_on_rule():
    import inspect

    from capx.belief import memory, verify

    for mod in (memory, verify):
        assert "resting flat on top" not in inspect.getsource(mod)
