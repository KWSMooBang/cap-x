"""bggen-batch: episode list, prepare / run skipping and failure recording, aggregation, LIBERO-Mem task checks and
the at(center) region relation (task 8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from capx.belief.perception import V2_PROMPTS
from capx.bggen import aggregate as agg
from capx.bggen import batch as bt
from capx.bggen.adapters import libero_mem as lm
from capx.bggen.configs import b1k as b1k_cfg
from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.evaluate import _truth, oracle_regions, relation_accuracy
from capx.bggen.generator import Generator, PlanError
from capx.bggen.oracle import Oracle
from capx.bggen.pipeline import region_check
from capx.bggen.records import load_episode
from capx.bggen.vocab import categories_for, parse_statement, statement_key
from tests.bggen_fakes import FakeDetector, write_swap_episode

LIST = Path(__file__).resolve().parents[1] / "scripts" / "bggen" / "episodes.yaml"


# --- episode list --------------------------------------------------------------------------------------

def test_episode_list_parses():
    eps = bt.load_episodes(LIST)
    lib = [e for e in eps if e.benchmark == "libero_mem"]
    b1k = [e for e in eps if e.benchmark == "b1k"]
    assert len(lib) == 15 and len(b1k) == 5
    assert sorted({e.task for e in lib}) == [0, 2, 6, 7, 8]
    assert all(sorted(e.demo for e in lib if e.task == t) == ["demo_1", "demo_2", "demo_3"] for t in (0, 2, 6, 7, 8))
    # the single-episode runs' folders keep their names (already prepared there -> skipped)
    assert "libero_mem_t06_demo_1" in {e.id for e in lib} and "b1k_t45_e00450010" in {e.id for e in b1k}
    assert [e.episode for e in b1k] == [450010, 450020, 450030, 450040, 450050]
    for e in lib:  # file <-> task_id agrees with the LIBERO-Mem task map
        assert e.file == lm.task_file(e.task)
        assert e.file.removesuffix("_demo.hdf5") == lm.TASK_NAMES[e.task]
    assert lm.task_file(6) == lm.SWAP_FILE


def test_episode_list_categories_match_bg_est2():
    import yaml

    d = yaml.safe_load(LIST.read_text())
    for t in d["libero_mem"]["tasks"]:
        cats = categories_for(t["instruction"])
        assert {c.key for c in cats} == set(t["categories"])
        for c in cats:
            assert tuple(t["categories"][c.key]) == V2_PROMPTS[c.key] == c.candidates  # bg_est2's measured prompts
            assert c.sam3_text == V2_PROMPTS[c.key][0]


def test_episode_list_only_and_duplicates(tmp_path):
    assert {e.id for e in bt.load_episodes(LIST, ["t08"])} == {f"libero_mem_t08_demo_{i}" for i in (1, 2, 3)}
    assert all(e.benchmark == "b1k" for e in bt.load_episodes(LIST, ["b1k"]))
    assert len(bt.load_episodes(LIST, ["libero_mem"])) == 15
    p = tmp_path / "e.yaml"
    p.write_text("b1k: {task: 45, episodes: [450010, 450010]}\n")
    with pytest.raises(ValueError, match="duplicate"):
        bt.load_episodes(p)
    p.write_text("libero_mem: {tasks: [{task_id: 1, demos: [demo_1]}]}\n")
    with pytest.raises(ValueError, match="file"):
        bt.load_episodes(p)


# --- prepare / run with a fake command runner -----------------------------------------------------------

def _write_prepared(root: Path) -> None:
    for f in ("episode/meta.json", "episode/proprio.npz", "oracle/meta.json", "oracle/states.jsonl"):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text("{}")


class FakeRunner:
    """Plays the cli: prep writes the episode files and prints its JSON; scripted failures by (step, episode)."""

    def __init__(self, fail: set[tuple[str, str]] = frozenset(), b1k_checks: dict | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail
        self.b1k_checks = b1k_checks or {}

    def __call__(self, cmd, log, env=None):
        self.calls.append(cmd)
        sub = cmd[3]
        arg = dict(zip(cmd[4::2], cmd[5::2], strict=False))
        ep_id = Path(arg.get("--root", arg.get("--out", "x"))).name
        if (sub, ep_id) in self.fail or (sub, "*") in self.fail:
            return bt.Result(1, "Traceback ...\nRuntimeError: boom")
        if sub in ("prep-libero", "prep-b1k"):
            _write_prepared(Path(arg["--root"]))
            checks = self.b1k_checks.get(ep_id, {"head_height_1_2m": True}) if sub == "prep-b1k" else {}
            return bt.Result(0, "warning: noise\n" + json.dumps({"checks": checks, "task_info": {"layout_ok": True}}, indent=1)
                             + "\nexit noise")
        if sub == "check-episode":
            return bt.Result(0, json.dumps({"event_counts": {"close": 3}}, indent=1))
        if sub == "run":
            out = Path(arg["--out"])
            out.mkdir(parents=True, exist_ok=True)
            (out / "summary.json").write_text(json.dumps({"init_failed": None, "goal": {"region_missing": False}}))
            return bt.Result(0, "{}")
        return bt.Result(0, "")


def test_prep_skips_prepared_records_failures_and_downloads_once(tmp_path):
    eps = bt.load_episodes(LIST, ["t06", "t08", "b1k_t45_e0045001", "b1k_t45_e00450020"])
    runner = FakeRunner(fail={("prep-libero", "libero_mem_t08_demo_2")},
                        b1k_checks={"b1k_t45_e00450020": {"head_height_1_2m": False}})
    b = bt.Batch(tmp_path, py="PY", py_lm="PYLM", runner=runner)
    _write_prepared(b.root(eps[0]))  # libero_mem_t06_demo_1 prepared by the single-episode runbook
    st = b.prep(eps)
    assert st["libero_mem_t06_demo_1"]["status"] == "skipped"
    assert st["libero_mem_t06_demo_2"]["status"] == "ok"
    assert st["libero_mem_t08_demo_2"]["status"] == "failed" and st["libero_mem_t08_demo_2"]["step"] == "prep"
    assert "boom" in st["libero_mem_t08_demo_2"]["log"]
    assert st["libero_mem_t08_demo_3"]["status"] == "ok"  # the next episode still ran
    assert st["b1k_t45_e00450010"]["status"] == "ok" and st["b1k_t45_e00450010"]["checks"]["task_info_layout_ok"]
    assert st["b1k_t45_e00450020"]["status"] == "check_failed"
    assert st["b1k_t45_e00450020"]["failed_checks"] == ["head_height_1_2m"]
    dl = [c for c in runner.calls if c[3] == "download-libero"]
    assert [c[c.index("--task-id") + 1] for c in dl] == ["6", "8"]  # one download per task file
    assert all(c[0] == "PYLM" for c in runner.calls if c[3] == "prep-libero")  # LIBERO-Mem venv
    assert json.loads((tmp_path / "episodes/prep_status.json").read_text()) == st
    # second pass: everything prepared is skipped, the failed one is retried
    runner.calls.clear()
    runner.fail = set()
    st = b.prep(eps)
    assert st["libero_mem_t08_demo_2"]["status"] == "ok"
    assert st["b1k_t45_e00450020"]["status"] == "check_failed" and st["b1k_t45_e00450020"]["skipped_again"]
    assert {c[3] for c in runner.calls} <= {"download-libero", "dump-libero", "prep-libero", "check-episode"}
    assert sum(c[3] == "prep-libero" for c in runner.calls) == 1


def test_run_skips_done_continues_after_failure(tmp_path):
    eps = bt.load_episodes(LIST, ["t00", "b1k_t45_e00450020"])
    runner = FakeRunner(fail={("run", "libero_mem_t00_demo_2")})
    b = bt.Batch(tmp_path, py="PY", runner=runner)
    for e in eps[:3]:
        _write_prepared(b.root(e))
    _write_prepared(b.root(eps[3]))
    (tmp_path / "episodes").mkdir(exist_ok=True)
    bt.Status(tmp_path / "episodes/prep_status.json").set(eps[3].id, {"status": "check_failed", "failed_checks": ["x"]})
    done = b.run_dir("t1", eps[0])
    done.mkdir(parents=True)
    (done / "summary.json").write_text(json.dumps({"init_failed": "init failed: no goal", "goal": {}}))
    st = b.run("t1", eps, jobs=2)
    assert st["libero_mem_t00_demo_1"]["status"] == "skipped" and st["libero_mem_t00_demo_1"]["init_failed"]
    assert st["libero_mem_t00_demo_2"]["status"] == "failed" and st["libero_mem_t00_demo_2"]["step"] == "run"
    assert st["libero_mem_t00_demo_3"]["status"] == "ok"
    assert st["b1k_t45_e00450020"]["status"] == "not_prepared" and "checks failed" in st["b1k_t45_e00450020"]["reason"]
    runs = [c for c in runner.calls if c[3] == "run"]
    assert {Path(c[c.index("--out") + 1]).name for c in runs} == {"libero_mem_t00_demo_2", "libero_mem_t00_demo_3"}
    assert all(c[c.index("--benchmark") + 1] == "libero_mem" for c in runs)
    assert json.loads((b.run_dir("t1", eps[2]) / "batch_episode.json").read_text())["group"] == "libero_mem t00"
    # the skipped run is evaluated again; ALLOW_CHECK_FAIL runs the check-failed episode
    assert any(c[3] == "eval" and c[c.index("--out") + 1] == str(done) for c in runner.calls)
    b.allow_check_fail = True
    assert b.run_one("t1", eps[3])["status"] == "ok"


def test_json_out_ignores_log_noise():
    assert bt._json_out('x\n{\n "a": {\n  "b": 1\n }\n}\ntrailing') == {"a": {"b": 1}}
    assert bt._json_out("{}\nlog\n{\n \"z\": 2\n}") == {"z": 2}
    assert bt._json_out("no json") is None


# --- aggregation ------------------------------------------------------------------------------------------

def _fake_run(d: Path, acc: tuple[float, float], per: dict, calls: int, goal_ok: bool, init_failed=None,
              region_missing=False) -> None:
    d.mkdir(parents=True)
    ev = {"name_consistency": {"rate": 1.0 if goal_ok else 0.9, "map_changes": 0},
          "goal_consistency": {"final_graph_satisfies_goal": goal_ok, "oracle_success": True,
                               "region_goal_missing": region_missing}}
    for p, a in zip(("forward", "forward+backward"), acc, strict=True):
        ev[p] = {"relations": {"accuracy": a, "decided": 100, "undecided_rate": 0.1, "unmappable_wrong": 0,
                               "per_pred": {k: {"right": r, "wrong": w, "undecided": u,
                                                "accuracy": round(r / (r + w), 4)} for k, (r, w, u) in per.items()}},
                 "relations_remap": {"accuracy": a},
                 "subtask_boundary": {"intent": {"mean": 8.0}, "contact": {"mean": 4.0}},
                 "reobservation": {"rate": 1.0, "n": 2}}
    sm = {"init_failed": init_failed, "agent_fallbacks": 1, "n_frames": 400, "rows": 30,
          "init": {"attempts": [{}] if not init_failed else [{}, {}, {}]},
          "goal": {"branches": [[["(on bowl_A plate_B)"], ["(on bowl_B plate_A)"], ["(on bowl_A plate_C)"]]], "best": 0,
                   "done": [3 if goal_ok else 1], "region_missing": region_missing},
          "models": {"init": {"model": "Qwen/Qwen3.8-27B", "calls": 1, "prompt_tokens": 1_000_000, "completion_tokens": 0,
                              "cut": 0, "empty": 1, "requeries": 1, "incomplete": 0, "seconds": 3.0},
                     "vdm": {"model": "Qwen/Qwen3.8-27B", "calls": calls - 1, "prompt_tokens": 0,
                             "completion_tokens": 1_000_000, "cut": 1, "empty": 0, "requeries": 1, "incomplete": 0,
                             "seconds": 10.0}}}
    (d / "eval.json").write_text(json.dumps(ev))
    (d / "summary.json").write_text(json.dumps(sm))


def test_aggregate_tables(tmp_path):
    runs = tmp_path / "runs" / "t1"
    _fake_run(runs / "libero_mem_t06_demo_1", (0.9, 0.95), {"on": (9, 1, 1), "inhand": (5, 0, 0)}, 30, True)
    _fake_run(runs / "libero_mem_t06_demo_2", (0.7, 0.75), {"on": (7, 3, 0)}, 20, False)
    _fake_run(runs / "libero_mem_t08_demo_1", (0.8, 0.8), {"in": (4, 1, 0), "at": (1, 1, 2)}, 25, False,
              init_failed="init failed: no goal", region_missing=True)
    (runs / "libero_mem_t08_demo_2").mkdir()  # a run that crashed before writing anything: ignored as a folder
    (runs / "b1k_t45_e00450010").mkdir()
    (runs / "b1k_t45_e00450010" / "summary.json").write_text(json.dumps({"init_failed": None, "goal": {}, "models": {}}))
    (runs / "batch_status.json").write_text(json.dumps({"libero_mem_t08_demo_2": {"status": "failed", "step": "run"},
                                                        "libero_mem_t06_demo_1": {"status": "ok"}}))
    res = agg.aggregate(runs)
    g6, g8 = res["groups"]["libero_mem t06"], res["groups"]["libero_mem t08"]
    assert g6["episodes"] == 2 and g6["forward"]["acc"] == {"mean": 0.8, "min": 0.7, "max": 0.9, "n": 2}
    assert g6["forward+backward"]["acc"]["mean"] == 0.85
    assert g6["forward"]["per_pred_pooled"]["on"] == {"right": 16, "wrong": 4, "undecided": 1, "accuracy": 0.8}
    assert g6["forward"]["per_pred"]["inhand"]["n"] == 1
    assert g6["goal_satisfied"] == 1 and g6["steps_done"] == 4 and g6["steps_total"] == 6
    assert g6["usage"]["calls"] == 50 and g6["usage"]["cut"] == 2 and g6["usage"]["empty"] == 2
    assert g6["cost_usd"] == pytest.approx(2 * (0.45 + 3.0))
    assert g8["init_failed"] == ["libero_mem_t08_demo_1"] and g8["init_requeried"] == 2
    assert res["all"]["region_missing"] == ["libero_mem_t08_demo_1"]
    b = res["groups"]["b1k t45"]
    assert b["evaluated"] == 0 and b["forward"]["acc"] is None  # eval.json missing: not averaged
    assert res["benchmarks"]["libero_mem"]["episodes"] == 3 and res["all"]["episodes"] == 4
    assert set(res["failed"]) == {"libero_mem_t08_demo_2"}
    md = agg.to_markdown(res)
    assert "| libero_mem t06 | fwd | 2/2 | 0.800 [0.700, 0.900] |" in md
    assert "| libero_mem t06 | fwd+bwd | 2/2 | 0.850 [0.750, 0.950] |" in md
    assert "REGION MISSING" in md and "libero_mem_t08_demo_2 (failed: run)" in md
    out = tmp_path / "agg"
    assert agg.main([str(runs), "--out", str(out), "--price-in", "1", "--price-out", "1"]) == 0
    assert json.loads(out.with_suffix(".json").read_text())["all"]["cost_usd"] == pytest.approx(6.0)
    assert out.with_suffix(".md").exists()


def test_group_from_folder_name():
    assert agg.group_of("libero_mem_t06_demo_1") == "libero_mem t06"
    assert agg.group_of("b1k_t45_e00450010") == "b1k t45"
    assert agg.group_of("smoke") == "other"


# --- LIBERO-Mem task check -------------------------------------------------------------------------------

def test_check_task_uses_bddl_name_and_env_language():
    # task 2: the file's problem_info language lacks "3 times" (LIBERO-Mem-Raw, checked 2026-11-01)
    r = lm.check_task(2, "chiliocosm/bddl_files/libero_mem/KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times.bddl",
                      "pick up the bowl and place it back on the plate",
                      "lift the bowl and place it back on the plate 3 times")
    assert r["matched_by"] == "bddl_file_name" and not r["language_same"]
    assert r["instruction"] == "lift the bowl and place it back on the plate 3 times"
    with pytest.raises(RuntimeError, match="task_id 6"):
        lm.check_task(6, "x/KITCHEN_SCENE1_8_rotate_the_3_bowls_on_their_plates_from_left_to_right_using_the_empty_plate.bddl",
                      "rotate ...", "swap ...")
    # no bddl attr: the HDF5 name decides; neither: the languages must agree
    assert lm.check_task(8, None, None, "put", lm.task_file(8))["matched_by"] == "hdf5 name"
    with pytest.raises(RuntimeError):
        lm.check_task(8, None, "a", "b")
    assert lm.check_task(8, None, "same", "Same ")["matched_by"] == "language"


def test_regions_of_on_targets():
    assert lm.regions_of(["basket_1", "kitchen_table", "kitchen_table_the_center", "table"], ["basket_1"]) == \
        {"center": "kitchen_table_the_center"}
    assert lm.regions_of(["plate_1", "table"], []) == {}


# --- at(center) ------------------------------------------------------------------------------------------

def test_statement_key_accepts_region():
    names = ["basket_A", "cream_cheese_A"]
    assert parse_statement("(at basket_A center)", names) == ("at", "basket_A", "center")
    assert statement_key("on", ["basket_a", "center"], names) == ("at", "basket_A", "center")  # BDDL wording
    assert statement_key("at", ["basket_A", "cream_cheese_A"], names) is None  # at needs a region
    assert statement_key("on", ["center", "basket_A"], names) is None
    assert statement_key("inhand", ["center"], names) is None
    assert statement_key("in", ["cream_cheese_A", "basket_A"], names) == ("in", "cream_cheese_A", "basket_A")
    assert statement_key("at", ["basket_A", "corner"], names) is None  # unknown region


def test_init_goal_with_center(tmp_path):
    w = write_swap_episode(tmp_path)
    gen = Generator(load_episode(tmp_path), FakeDetector(w), None, lm_cfg("A"))
    reply = json.dumps({"remember": [], "observe": ["(at bowl_A center)"],
                        "goal_branches": [[[["on", "bowl_A", "plate_B"]], [["at", "bowl_A", "center"]]],
                                          [[["on", "bowl_A", "plate_B"]], [["on", "bowl_A", "center"]]]]})
    parsed = gen._parse_init(reply)
    assert parsed["branches"][0][1] == (("at", "bowl_A", "center"),)
    assert parsed["branches"][1][1] == (("at", "bowl_A", "center"),)
    assert parsed["observe"] == [("at", "bowl_A", "center")]
    gen.cfg = b1k_cfg("A")  # B1K has no region relation
    with pytest.raises(PlanError):
        gen._parse_init(reply)


def test_region_check_flags_missing_goal_step():
    instr = "put the cream cheese in the nearest basket and place that basket in the center"
    ok = region_check(instr, [[["(in cream_cheese_A basket_A)"], ["(at basket_A center)"]]], ("on", "in", "at"))
    assert ok["region_words"] == ["center"] and not ok["region_missing"]
    bad = region_check(instr, [[["(in cream_cheese_A basket_A)"]]], ("on", "in", "at"))
    assert bad["region_missing"]
    assert not region_check("swap the 2 bowls", [[["(on bowl_A plate_B)"]]], ("on",))["region_missing"]


def _oracle(regions: dict | None, true_at: bool) -> Oracle:
    meta = {"objects": {"basket_1": "basket", "cream_cheese_1": "cream_cheese"}, "relations": ["on", "in", "inhand", "handempty"]}
    if regions is not None:
        meta["regions"] = regions
    st = [{"t": 0, "pos": {}, "true": [["in", "cream_cheese_1", "basket_1_contain_region"]], "inhand": {"main": None}},
          {"t": 10, "pos": {}, "true": [["in", "cream_cheese_1", "basket_1_contain_region"]]
           + ([["on", "basket_1", "kitchen_table_the_center"]] if true_at else []), "inhand": {"main": None}}]
    return Oracle(meta, st)


def test_eval_scores_at_center_against_the_oracle_region():
    o = _oracle({"center": "kitchen_table_the_center"}, true_at=True)
    amap = {"basket_A": "basket_1", "cream_cheese_A": "cream_cheese_1"}
    key = ("at", "basket_A", "center")
    reg = oracle_regions(o)
    assert _truth(key, o.states[0], amap, reg) is False and _truth(key, o.states[1], amap, reg) is True
    assert _truth(key, o.states[1], amap, {}) is None  # no region map: not scored
    # region map recovered from the states when meta.json has none (episodes prepared before this change)
    assert oracle_regions(_oracle(None, true_at=True)) == {"center": "kitchen_table_the_center"}
    assert oracle_regions(_oracle(None, true_at=False)) == {}
    rows = [{"t": 0, "belief": {"(at basket_A center)": (0.1, "obs"), "(in cream_cheese_A basket_A)": (0.9, "obs")}},
            {"t": 10, "belief": {"(at basket_A center)": (0.9, "obs")}}]
    res = relation_accuracy(rows, o, amap, ["basket_A", "cream_cheese_A"], [None])
    assert res["per_pred"]["at"]["right"] == 2 and res["per_pred"]["in"]["right"] == 1 and res["accuracy"] == 1.0
    res = relation_accuracy(rows, _oracle(None, true_at=False), amap, ["basket_A", "cream_cheese_A"], [None])
    assert "at" not in res["per_pred"]  # nothing to score against
