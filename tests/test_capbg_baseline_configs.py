"""cap-bg baseline configs: generated files are current and resolve to the intended settings."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from capx.envs.launch import LaunchArgs
from capx.utils.launch_utils import _load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = REPO_ROOT / "env_configs" / "capbg_baseline"
CAP_AGENT0 = yaml.safe_load(
    (REPO_ROOT / "env_configs" / "libero" / "franka_libero_cap_agent0.yaml").read_text()
)
BASELINE_CONFIGS = sorted(BASELINE_DIR.rglob("*.yaml"))


def test_generated_configs_are_current():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "gen_capbg_baseline_configs.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_all_benchmarks_covered():
    names = {p.relative_to(BASELINE_DIR).as_posix() for p in BASELINE_CONFIGS}
    assert "libero_pro/franka_libero_object_swap_0.yaml" in names
    assert len([n for n in names if n.startswith("libero_mem/task_")]) == 10


@pytest.mark.parametrize("path", BASELINE_CONFIGS, ids=lambda p: p.relative_to(BASELINE_DIR).as_posix())
def test_baseline_config_resolves(path):
    args = LaunchArgs(config_path=str(path))
    env_factory, config, api_servers = _load_config(args)
    assert args.max_tokens == 32768

    assert config["use_parallel_ensemble"] is False
    assert config["use_multimodel"] is False
    assert config["use_img_differencing"] is True
    assert config["use_visual_feedback"] is False
    assert config["use_video_differencing"] is False
    assert config["use_wrist_camera"] is False
    assert config["record_video"] is True
    assert config["save_multiturn_prompts"] is True

    cfg = env_factory["cfg"]
    assert cfg["multi_turn_prompt"] == CAP_AGENT0["env"]["cfg"]["multi_turn_prompt"]
    assert cfg["prompt"] == CAP_AGENT0["env"]["cfg"]["prompt"]
    assert cfg["apis"] == ["FrankaLiberoApiReducedSkillLibrary"]
    assert api_servers == CAP_AGENT0["api_servers"]


def test_ensemble_ablation_cli_override():
    path = BASELINE_DIR / "libero_mem" / "task_06_swap_bowls.yaml"
    args = LaunchArgs(config_path=str(path), use_parallel_ensemble=True, use_multimodel=True)
    _, config, _ = _load_config(args)
    assert config["use_parallel_ensemble"] is True
    assert config["use_multimodel"] is True


def test_max_tokens_cli_overrides_yaml():
    path = BASELINE_DIR / "libero_mem" / "task_06_swap_bowls.yaml"
    args = LaunchArgs(config_path=str(path), max_tokens=4096)
    _load_config(args)
    assert args.max_tokens == 4096


def test_max_tokens_default_without_yaml_key():
    path = REPO_ROOT / "env_configs" / "libero" / "franka_libero_cap_agent0.yaml"
    assert "max_tokens" not in yaml.safe_load(path.read_text())
    args = LaunchArgs(config_path=str(path))
    _load_config(args)
    assert args.max_tokens == 20480
