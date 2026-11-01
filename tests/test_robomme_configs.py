"""RoboMME configs, pilot resume logic and the pilot aggregator."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from capx.envs.launch import LaunchArgs
from capx.utils.launch_utils import _load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import analyze_robomme_pilot as analyze  # noqa: E402
import run_robomme_pilot as pilot  # noqa: E402
from gen_robomme_configs import TASKS  # noqa: E402

CAP_AGENT0 = yaml.safe_load((REPO_ROOT / "env_configs" / "libero" / "franka_libero_cap_agent0.yaml").read_text())
MEM_BASELINE = REPO_ROOT / "env_configs" / "capbg_baseline" / "libero_mem" / "task_06_swap_bowls.yaml"
CONFIGS = sorted((REPO_ROOT / "env_configs" / "robomme").glob("*.yaml"))


def test_generated_configs_are_current():
    result = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "gen_robomme_configs.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_16_tasks():
    from capx.integrations.robomme import ROBOMME_SUITES, ROBOMME_TASKS

    assert [t for t, _ in TASKS] == ROBOMME_TASKS
    assert {p.stem for p in CONFIGS} == {f"robomme_{t.lower()}" for t in ROBOMME_TASKS}
    assert all(t in ROBOMME_SUITES[s] for t, s in TASKS)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_config_matches_libero_mem_baseline(path):
    args = LaunchArgs(config_path=str(path))
    env_factory, config, api_servers = _load_config(args)
    base_args = LaunchArgs(config_path=str(MEM_BASELINE))
    _, base, base_servers = _load_config(base_args)
    assert args.max_tokens == base_args.max_tokens == 32768
    for key in ("use_parallel_ensemble", "use_multimodel", "use_img_differencing", "use_visual_feedback",
                "use_video_differencing", "use_wrist_camera", "record_video", "save_multiturn_prompts", "memory_mode"):
        assert config[key] == base[key], key
    assert config["memory_mode"] == "none"
    assert config["trial_timeout_s"] == 3600 and config["max_trial_retries"] == 1
    assert api_servers == base_servers == CAP_AGENT0["api_servers"]
    cfg = env_factory["cfg"]
    assert cfg["prompt"] == CAP_AGENT0["env"]["cfg"]["prompt"]
    assert cfg["multi_turn_prompt"] == CAP_AGENT0["env"]["cfg"]["multi_turn_prompt"]
    assert cfg["apis"] == ["FrankaRoboMMEApiReducedSkillLibrary"] and cfg["privileged"] is False
    low = cfg["low_level"]
    assert low["_target_"] == "capx.envs.simulators.robomme.RoboMMEEnv"
    assert low["dataset"] == "test" and low["max_steps"] == 1300 and low["condition_frames"] == 32
    assert low["privileged"] is False


def test_resume_idx_cli_overrides_yaml():
    args = LaunchArgs(config_path=str(CONFIGS[0]), resume_idx=3)
    _, config, _ = _load_config(args)
    assert config["resume_idx"] == 3
    _, config, _ = _load_config(LaunchArgs(config_path=str(CONFIGS[0])))
    assert config["resume_idx"] is None


def _fake_trial(task_dir: Path, trial: int, end: dict | None, responses=None, name_suffix="sandboxrc_0") -> Path:
    d = task_dir / f"trial_{trial:02d}_{name_suffix}"
    d.mkdir(parents=True)
    (d / "all_responses.json").write_text(json.dumps(responses or []))
    if end is not None:
        (d / "trial_end.json").write_text(json.dumps({"trial": trial, **end}))
    return d


def test_pilot_resume_helpers(tmp_path):
    task_dir = tmp_path / "robomme_pickxtimes"
    _fake_trial(task_dir, 1, {"reason": "finish"})
    _fake_trial(task_dir, 2, {"reason": "success"})
    _fake_trial(task_dir, 3, None)  # killed mid-trial
    done = pilot.finished_trials(task_dir)
    assert done == {1, 2}
    assert pilot.first_unfinished(done, 5) == 3
    assert pilot.first_unfinished({1, 2, 3, 4, 5}, 5) is None
    moved = pilot.quarantine_incomplete(task_dir, done, 3)
    assert moved == ["trial_03_sandboxrc_0"]
    assert not list(task_dir.glob("trial_03_*")) and list((task_dir / "_incomplete").iterdir())
    cmd = pilot.task_command(pilot.Args(run_name="r"), "PickXtimes", 3, tmp_path)
    assert cmd[cmd.index("--resume-idx") + 1] == "3" and cmd[cmd.index("--total-trials") + 1] == "5"
    assert cmd[cmd.index("--memory-mode") + 1] == "none"
    assert "--api-key" not in cmd  # added only when running, never stored in the manifest


def test_analyzer_tsv_and_summary(tmp_path):
    run = tmp_path / "run"
    model_dir = run / "moonshotai_Kimi-K3"
    usage = {"prompt_tokens": 100, "completion_tokens": 10}
    responses = [
        {"decision": "initial", "usage": usage, "vdm_usage": [{"prompt_tokens": 50, "completion_tokens": 5}]},
        {"decision": "regenerate", "usage": usage, "truncated_first_usage": usage,
         "vdm_usage": [{"prompt_tokens": 50, "completion_tokens": 5}]},
        {"decision": "truncated", "reason": "success"},
    ]
    _fake_trial(model_dir / "robomme_pickxtimes", 1,
                {"reason": "success", "task_completed": True, "turns": 2, "sim_steps": 420, "time_s": 300.5}, responses)
    _fake_trial(model_dir / "robomme_pickxtimes", 2,
                {"reason": "finish", "task_completed": False, "turns": 1, "sim_steps": 100, "time_s": 60.0})
    _fake_trial(model_dir / "robomme_videounmaskswap", 1,
                {"reason": "wall_timeout", "task_completed": False, "turns": 3, "sim_steps": 900, "time_s": 3600})
    (run / "pilot_manifest.jsonl").write_text(json.dumps({"memory_mode": "none", "trials": 2}) + "\n")
    assert analyze.main.__module__  # import ok
    sys_argv = sys.argv
    try:
        sys.argv = ["analyze", str(run)]
        assert analyze.main() == 0
    finally:
        sys.argv = sys_argv
    lines = (run / "results.tsv").read_text().strip().split("\n")
    assert lines[0].split("\t") == analyze.COLUMNS
    rows = [dict(zip(analyze.COLUMNS, ln.split("\t"))) for ln in lines[1:]]
    first = rows[0]
    assert (first["cond"], first["task"], first["trial"], first["reason"], first["success"]) == \
        ("none", "PickXtimes", "1", "success", "1")
    assert first["code_prompt_tok"] == "300" and first["vdm_completion_tok"] == "10"
    missing = [r for r in rows if r["reason"] == "missing"]
    assert [(r["task"], r["trial"]) for r in missing] == [("VideoUnmaskSwap", "2")]
    summary = (run / "SUMMARY.md").read_text()
    assert "| Counting | 1/2 | 50% |" in summary and "| Permanence | 0/2 | 0% |" in summary
    assert "wall_timeout 1" in summary
