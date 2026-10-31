"""bg_est2 SAM3 prompt candidates: order, re-detection triggers, names shown to models, v1 unchanged,
goal re-parse after an unknown-object rejection, logs (memory_est / belief_trace)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from capx.belief.estimator import BeliefEstimator
from capx.belief.estimator2 import BeliefEstimatorV2, Est2Config
from capx.belief.goal import GoalResult, unknown_objects
from capx.belief.hooks import ApiCallRecorder
from capx.belief.memory2 import BgEst2Memory
from capx.belief.observe import VdmObserver
from capx.belief.perception import VOCABULARY, VOCABULARY_V2, Sam3PerceptionSource
from tests.belief_fakes import ScriptedModel, WorldEnv, swap_world
from tests.bg_est2_fakes import BASKET_GOAL, SWAP2_GOAL, AnyDetector, FakeObserveVdm, basket_world
from tests.test_bg_est2 import BASKET, SWAP

V2 = {c.key: c for c in VOCABULARY_V2}
# SAM3-only strings: they describe the render, never what the object is; no model may see them.
SAM3_ONLY = ("small blue box", "blue box", "blue package", "saucer", "wicker basket")


class PromptDetector(AnyDetector):
    """AnyDetector that finds a category only under the SAM3 prompts in ``sees[key]`` (keys not in
    ``sees``: any prompt). Records every (category, prompt) run."""

    def __init__(self, world, sees: dict[str, set[str]], seed: int = 0) -> None:
        super().__init__(world, seed)
        self.sees = sees
        self.runs: list[tuple[str, str]] = []

    def detect(self, obs, categories):
        self.runs += [(c.key, c.sam3_text) for c in categories]
        ok = [c for c in categories if c.key not in self.sees or c.sam3_text in self.sees[c.key]]
        out = super().detect(obs, ok)
        self.calls += len(categories) - len(ok)
        return out


def build(world, detector, goal_model, instruction, **cfg):
    rec = ApiCallRecorder(world.proprio, lambda: world.steps)
    world.api = rec.wrap_functions(world.functions())
    holder = {}
    vdm = FakeObserveVdm(world, lambda: holder.get("est"))
    est = BeliefEstimatorV2(detector, Est2Config(**cfg), goal_query_fn=goal_model, observer=VdmObserver(vdm))
    holder["est"] = est
    first = est.start(world.get_observation(), instruction)
    return est, rec, vdm, first


def end(est, world, rec, n):
    return est.end_turn(n, world.get_observation(), rec.drain(), code=None)


# --- vocabulary --------------------------------------------------------------------------------------

def test_candidate_order_and_names():
    assert V2["cream_cheese"].candidates == ("small blue box", "blue box", "blue package")
    assert V2["plate"].candidates == ("saucer", "plate")
    assert V2["bowl"].candidates == ("bowl", "black bowl")
    assert V2["bottle"].candidates == ("bottle", "wine bottle")
    assert V2["basket"].candidates == ("basket", "wicker basket")
    for c in VOCABULARY_V2:
        assert c.sam3_text == c.candidates[0]
    # Shown names stay semantic.
    assert {k: V2[k].prompt for k in ("cream_cheese", "plate", "bowl", "bottle", "basket")} == {
        "cream_cheese": "cream cheese", "plate": "plate", "bowl": "black bowl", "bottle": "wine bottle",
        "basket": "basket"}
    # Other categories: their one v1 prompt.
    v1 = {c.key: c for c in VOCABULARY}
    for k in ("mug", "ketchup", "soup", "butter", "milk", "juice", "tray"):
        assert V2[k].candidates == (v1[k].prompt,) and V2[k].prompt == v1[k].prompt


def test_v1_vocabulary_unchanged():
    assert [(c.key, c.prompt) for c in VOCABULARY] == [
        ("cream_cheese", "cream cheese box"), ("bowl", "black bowl"), ("plate", "plate"), ("bottle", "wine bottle"),
        ("basket", "basket"), ("mug", "mug"), ("ketchup", "ketchup bottle"), ("soup", "alphabet soup can"),
        ("butter", "butter box"), ("milk", "milk carton"), ("juice", "orange juice carton"), ("tray", "tray")]
    for c in VOCABULARY:
        assert c.prompts == () and c.query == "" and c.candidates == (c.prompt,) and c.sam3_text == c.prompt
    assert BeliefEstimator.__init__.__defaults__[-2] is VOCABULARY  # v1 estimator default
    assert BeliefEstimatorV2.__init__.__defaults__[-2] is VOCABULARY_V2


def _cam_obs():
    return {"agentview": {"images": {"rgb": np.zeros((8, 8, 3), np.uint8), "depth": np.ones((8, 8), np.float32)},
                          "intrinsics": np.eye(3), "pose_mat": np.eye(4)}}


def test_sam3_source_sends_the_active_prompt():
    sent = []
    src = Sam3PerceptionSource(segment_fn=lambda rgb, text_prompt: sent.append(text_prompt) or [])
    v1 = {c.key: c for c in VOCABULARY}
    src.detect(_cam_obs(), [v1["cream_cheese"], v1["plate"], v1["bowl"]])
    assert sent == ["cream cheese box", "plate", "black bowl"]  # v1 exactly as before
    sent.clear()
    src.detect(_cam_obs(), [V2["cream_cheese"], V2["plate"], V2["cream_cheese"].with_query("blue box")])
    assert sent == ["small blue box", "saucer", "blue box"]


# --- turn 0 ------------------------------------------------------------------------------------------

def test_turn0_redetect_next_candidate():
    # task_08: the cheese is not found as "small blue box" but is as "blue box".
    world = basket_world()
    det = PromptDetector(world, {"cream_cheese": {"blue box"}})
    est, rec, vdm, first = build(world, det, ScriptedModel(BASKET_GOAL), BASKET)
    assert "cream_cheese_A" in est.tracks and est.goal.branches is not None
    assert est.prompts()["cream_cheese"] == "blue box" and est.prompts()["basket"] == "basket"
    assert est.tracks["cream_cheese_A"].category.sam3_text == "blue box"
    assert [(r["category"], r["prompt"], r["detections"], r["reason"]) for r in first.redetects] == [
        ("cream_cheese", "blue box", 1, "turn0_none")]
    assert ("cream_cheese", "blue package") not in det.runs  # stops at the first hit
    # Second turn-0 pass and later turns use the prompt that worked.
    assert det.runs.count(("cream_cheese", "blue box")) >= 2
    assert first.to_log()["prompts"]["cream_cheese"] == "blue box"


def test_turn0_no_redetect_when_found():
    world = basket_world()
    det = PromptDetector(world, {})
    est, _, _, first = build(world, det, ScriptedModel(BASKET_GOAL), BASKET)
    assert first.redetects == [] and est.redetect_log == []
    assert {k for k, _ in det.runs} == {"cream_cheese", "basket"}
    assert {p for _, p in det.runs} == {"small blue box", "basket"}


def test_turn0_all_candidates_then_keep_first():
    world = basket_world()
    det = PromptDetector(world, {"cream_cheese": set()})
    est, _, _, first = build(world, det, ScriptedModel(BASKET_GOAL), BASKET)
    assert [r["prompt"] for r in first.redetects] == ["blue box", "blue package"]
    assert est.prompts()["cream_cheese"] == "small blue box"  # nothing worked: the best one stays


def test_turn0_instruction_categories_first():
    world = basket_world()
    det = PromptDetector(world, {"cream_cheese": set(), "bowl": set()})
    est = BeliefEstimatorV2(det, Est2Config(), categories=[V2["bowl"], V2["cream_cheese"], V2["basket"]])
    first = est.start(world.get_observation(), BASKET)
    assert [r["category"] for r in first.redetects] == ["cream_cheese", "cream_cheese", "bowl"]


# --- later turns -------------------------------------------------------------------------------------

def test_later_turns_redetect_after_two_missed_turns_once_per_candidate():
    world = basket_world()
    det = PromptDetector(world, {})
    est, rec, _, _ = build(world, det, ScriptedModel(BASKET_GOAL), BASKET)
    det.sees["cream_cheese"] = set()  # lost from now on, under every prompt
    r1 = end(est, world, rec, 1)
    assert r1.redetects == []
    r2 = end(est, world, rec, 2)
    assert [(r["prompt"], r["detections"], r["reason"]) for r in r2.redetects] == [("blue box", 0, "none_2_turns")]
    assert r2.prompts["cream_cheese"] == "small blue box"
    assert end(est, world, rec, 3).redetects == []
    assert [r["prompt"] for r in end(est, world, rec, 4).redetects] == ["blue package"]
    for n in range(5, 9):  # every candidate tried once: no more SAM3 runs for it
        assert end(est, world, rec, n).redetects == []
    assert [r["prompt"] for r in est.redetect_log] == ["blue box", "blue package"]


def test_later_turn_redetect_hit_switches_prompt():
    world = basket_world()
    det = PromptDetector(world, {})
    est, rec, _, _ = build(world, det, ScriptedModel(BASKET_GOAL), BASKET)
    det.sees["basket"] = {"wicker basket"}
    end(est, world, rec, 1)
    r2 = end(est, world, rec, 2)
    assert [(r["prompt"], r["detections"]) for r in r2.redetects] == [("wicker basket", 2)]
    assert r2.prompts["basket"] == "wicker basket" and {"basket_A", "basket_B"} <= est.seen
    r3 = end(est, world, rec, 3)
    assert r3.redetects == [] and ("basket", "wicker basket") in det.runs[-2:]
    assert est.miss_streak["basket"] == 0  # a detection resets the streak


# --- goal re-parse -----------------------------------------------------------------------------------

def test_unknown_objects_from_goal_errors():
    g = GoalResult(None, "llm", errors=["unknown object 'cream_cheese_A' in ['in', 'cream_cheese_A', 'basket_A']",
                                        "unknown object 'cream cheese' in [...]", "query failed: x"])
    assert unknown_objects(g) == ["cream_cheese_A", "cream cheese"]
    assert unknown_objects(GoalResult([[]], "llm", errors=["unknown object 'x' in y"])) == []
    assert unknown_objects(GoalResult(None, "llm", errors=["no JSON object in the reply"])) == []


def test_goal_unknown_object_redetects_and_reparses_once():
    # Only the cheese is a category (max_categories=1): the goal names the baskets -> rejected; the
    # baskets are found from the instruction, the initial belief is rebuilt, the goal parsed again.
    world = basket_world()
    det = PromptDetector(world, {})
    model = ScriptedModel(BASKET_GOAL)
    est, rec, vdm, first = build(world, det, model, BASKET, max_categories=1)
    assert len(model.prompts) == 3  # 2 attempts (one retry with the error) + 1 after the re-detection
    assert first.goal_retry["unknown"] == ["basket_A"] and first.goal_retry["categories"] == ["basket"]
    assert first.goal_retry["retried"] and first.goal_retry["parsed"]
    assert first.goal_retry["new_tracks"] == ["basket_A", "basket_B"]
    assert est.goal.branches is not None and est.goal.attempts == 3 and len(est.goal.errors) == 2
    assert [r["reason"] for r in first.redetects] == ["goal_unknown_object"]
    assert "basket_A" in model.prompts[-1][1]["content"][0]["text"]
    assert len(vdm.prompts) == 2  # the initial Observe again, over the new objects
    assert any(k[1] == "basket_A" or k[2] == "basket_A" for k in est.graph.edges)
    assert first.progress.known and est.prompts() == {"cream_cheese": "small blue box", "basket": "basket"}
    # The rest of the trial works with the added category.
    r1 = end(est, world, rec, 1)
    assert r1.new_tracks == [] and {"basket_A", "basket_B", "cream_cheese_A"} <= est.seen


def test_goal_unknown_object_nothing_new_no_reparse():
    world = basket_world()
    det = PromptDetector(world, {"cream_cheese": set()})  # never visible
    model = ScriptedModel(BASKET_GOAL)
    est, _, vdm, first = build(world, det, model, BASKET)
    assert est.goal.branches is None and len(model.prompts) == 2  # no third goal call
    assert first.goal_retry == {"unknown": ["cream_cheese_A"], "categories": ["cream_cheese"], "new_tracks": [],
                                "retried": False}
    # Turn 0 already ran every cheese candidate on this image: nothing re-run for the goal.
    assert [r["reason"] for r in first.redetects] == ["turn0_none", "turn0_none"]
    assert len(vdm.prompts) == 1


# --- names shown to models ---------------------------------------------------------------------------

def _texts(prompts):
    out = []
    for p in prompts:
        for m in p:
            c = m["content"]
            out += [c] if isinstance(c, str) else [x.get("text", "") for x in c]
    return "\n".join(out)


def test_sam3_prompts_never_reach_model_prompts():
    world = basket_world()
    det = PromptDetector(world, {"cream_cheese": {"blue box"}})
    model = ScriptedModel(BASKET_GOAL)
    est, rec, vdm, _ = build(world, det, model, BASKET)
    end(est, world, rec, 1)
    det.sees["basket"] = {"wicker basket"}
    end(est, world, rec, 2)
    end(est, world, rec, 3)
    assert est.prompts() == {"cream_cheese": "blue box", "basket": "wicker basket"}
    goal_text, vdm_text, render = _texts(model.prompts), _texts(vdm.prompts), est.render()
    assert "cream_cheese_A (cream cheese)" in goal_text and "= cream_cheese_A (cream cheese)" in vdm_text
    for text in (goal_text, vdm_text, render, est.scene_lines() and "\n".join(est.scene_lines()),
                 "\n".join(est.objects_items()), est.objects_text()):
        for s in SAM3_ONLY:
            assert s not in text, s


def test_plate_saucer_not_shown():
    world = swap_world()
    det = PromptDetector(world, {})
    model = ScriptedModel(SWAP2_GOAL)
    est, rec, vdm, _ = build(world, det, model, SWAP)
    assert ("plate", "saucer") in det.runs and ("bowl", "bowl") in det.runs
    text = _texts(model.prompts) + _texts(vdm.prompts) + est.render()
    assert "saucer" not in text and "(plate)" in text and "(black bowl)" in text


# --- logs: memory_est and belief_trace --------------------------------------------------------------

def test_prompts_in_memory_log_and_trace(tmp_path):
    low = WorldEnv()
    low.reset()
    world = low.world
    det = PromptDetector(world, {"plate": {"plate"}})  # "saucer" misses -> turn-0 re-detection
    mem = BgEst2Memory(perception="sam3", shadow=False)
    vdm = FakeObserveVdm(world, lambda: mem.estimator)
    code_env = SimpleNamespace(api_call_wrapper=None)
    trace = tmp_path / "belief_trace.jsonl"
    mem.reset(low, code_env=code_env, instruction=SWAP, goal_query_fn=ScriptedModel(SWAP2_GOAL), vdm_query_fn=vdm,
              perception_source=det, trace_path=str(trace))
    log0 = mem.turn_log()["est"]
    assert log0["prompts"] == {"bowl": "bowl", "plate": "plate"}
    assert [(r["category"], r["prompt"], r["previous"]) for r in log0["redetects"]] == [("plate", "plate", "saucer")]
    mem.after_turn(1, None, executed=True, code=None)
    assert mem.turn_log()["est"]["prompts"] == {"bowl": "bowl", "plate": "plate"}
    lines = [json.loads(x) for x in trace.read_text().splitlines()]
    assert lines[0]["prompts"]["plate"] == "plate" and lines[0]["redetects"][0]["reason"] == "turn0_none"
    assert lines[-1]["prompts"] == {"bowl": "bowl", "plate": "plate"} and len(lines[-1]["redetects"]) == 1
