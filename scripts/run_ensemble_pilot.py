"""Run the Nebius ensemble pilot (env_configs/capbg_ensemble/) on LIBERO-Mem or RoboMME, resumable.

Same scheme as scripts/run_robomme_pilot.py (whose resume helpers it uses): each task is one
``capx/envs/launch.py`` process with one sequential worker, ``--workers`` tasks run at a time, a trial is done
when its directory has ``trial_end.json``, and a rerun of the same command resumes every task at its first
unfinished trial (half-written trial directories go to ``_incomplete/``).

Trials: LIBERO-Mem trial k resets with seed k, which picks its init state (same inits as the baseline pilots,
trials 1-5); RoboMME trial k plays test episode k-1 (episodes 0-4). Models come from the configs (ensemble
candidates / synthesis); ``--model`` only names the output directory (launch.py inserts it) and is the synthesis model by default.

    export NEBIUS_SERVER_URL=... NEBIUS_API_KEY=...
    python scripts/run_ensemble_pilot.py --bench libero_mem --run-name 20261102_ens_mem --workers 2
    python scripts/run_ensemble_pilot.py --bench robomme --run-name 20261102_ens_robomme --workers 2
    python scripts/analyze_libero_mem_pilot.py outputs/capbg_ensemble/20261102_ens_mem
    python scripts/analyze_robomme_pilot.py outputs/capbg_ensemble/20261102_ens_robomme

Outputs: outputs/capbg_ensemble/<run>/<model>/<config stem>/trial_XX_*/, logs in logs/capbg_ensemble/<run>/,
the exact commands and ensemble settings in <run>/pilot_manifest.jsonl.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import tyro
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from gen_capbg_ensemble_configs import LIBERO_MEM_TASKS, ROBOMME_TASKS, SYNTHESIS_MODEL, VDM_MODEL  # noqa: E402
from run_robomme_pilot import finished_trials, first_unfinished, quarantine_incomplete  # noqa: E402

BENCHES = ("libero_mem", "robomme")


@dataclass
class Args:
    bench: str
    """libero_mem or robomme."""
    run_name: str
    """Run directory name, e.g. 20261102_ens_mem."""
    tasks: list[str] = field(default_factory=list)
    """LIBERO-Mem task indices (0 4 6 7 8) or RoboMME task names; default: the pilot's tasks."""
    workers: int = 2
    """Task processes running at the same time."""
    trials: int = 5
    """Trials per task (1..trials)."""
    memory_mode: str = "none"
    model: str = SYNTHESIS_MODEL
    """Output directory name only (the configs pick the ensemble models)."""
    vdm_model: str = VDM_MODEL
    server_url: str = os.environ.get("NEBIUS_SERVER_URL", "")
    """Chat-completions URL for all models (default: $NEBIUS_SERVER_URL)."""
    api_key_env: str = "NEBIUS_API_KEY"
    """Environment variable holding the API key (never on a logged command line)."""
    trial_timeout_s: int = 3600
    """Wall-clock budget per trial (the baseline pilots ran with 3600 s and one attempt)."""
    max_trial_retries: int = 1
    out_root: str = "outputs/capbg_ensemble"
    log_root: str = "logs/capbg_ensemble"
    config_dir: str | None = None
    """Default: env_configs/capbg_ensemble/<bench>."""
    python: str = sys.executable
    start_servers: bool = True
    dry_run: bool = False
    extra: list[str] = field(default_factory=list)
    """Extra launch.py flags, e.g. --extra --condition-video-mode --extra images."""


def task_configs(args: Args) -> dict[str, Path]:
    """Task label -> config path, in pilot order."""
    cdir = REPO_ROOT / (args.config_dir or f"env_configs/capbg_ensemble/{args.bench}")
    if args.bench == "libero_mem":
        wanted = [int(t) for t in args.tasks] if args.tasks else LIBERO_MEM_TASKS
        out = {}
        for idx in wanted:
            found = sorted(cdir.glob(f"task_{idx:02d}_*.yaml"))
            if not found:
                raise SystemExit(f"no config for LIBERO-Mem task {idx} in {cdir}")
            out[found[0].stem] = found[0]
        return out
    wanted = args.tasks or ROBOMME_TASKS
    out = {}
    for task in wanted:
        path = cdir / f"robomme_{task.lower()}.yaml"
        if not path.exists():
            raise SystemExit(f"no config for RoboMME task {task} in {cdir}")
        out[path.stem] = path
    return out


def task_command(args: Args, config: Path, resume_idx: int, out_dir: Path) -> list[str]:
    cmd = [
        args.python, str(REPO_ROOT / "capx" / "envs" / "launch.py"),
        "--config-path", str(config),
        "--model", args.model,
        "--visual-differencing-model", args.vdm_model,
        "--total-trials", str(args.trials),
        "--resume-idx", str(resume_idx),
        "--num-workers", "1",
        "--output-dir", str(out_dir),
        "--memory-mode", args.memory_mode,
        "--record-video", "True",
        "--trial-timeout-s", str(args.trial_timeout_s),
        "--max-trial-retries", str(args.max_trial_retries),
    ]
    if args.server_url:
        cmd += ["--server-url", args.server_url, "--visual-differencing-model-server-url", args.server_url]
    return cmd + list(args.extra)


def ensemble_settings(config: Path) -> dict:
    cfg = yaml.safe_load(config.read_text())
    return {"ensemble": cfg.get("ensemble"), "use_parallel_ensemble": cfg.get("use_parallel_ensemble"),
            "use_multimodel": cfg.get("use_multimodel"), "condition_video_mode": cfg.get("condition_video_mode"),
            "condition_video_model": cfg.get("condition_video_model"),
            "condition_video_segment_frames": cfg.get("condition_video_segment_frames")}


def main(args: Args) -> int:
    if args.bench not in BENCHES:
        raise SystemExit(f"--bench must be one of {BENCHES}")
    configs = task_configs(args)
    run_dir = REPO_ROOT / args.out_root / args.run_name
    log_dir = REPO_ROOT / args.log_root / args.run_name
    model_dir = run_dir / args.model.replace("/", "_")  # launch.py inserts the model name before the last part
    env = dict(os.environ)
    api_key = env.get(args.api_key_env)

    plan = []
    for stem, config in configs.items():
        task_dir = model_dir / stem
        done = finished_trials(task_dir)
        resume = first_unfinished(done, args.trials)
        plan.append((stem, config, task_dir, done, resume))
        print(f"{stem:40s} done {sorted(done)} -> {'skip' if resume is None else f'resume at trial {resume}'}")

    todo = [p for p in plan if p[4] is not None]
    first_cfg = next(iter(configs.values()))
    manifest = {
        "run_name": args.run_name, "bench": args.bench, "memory_mode": args.memory_mode, "model": args.model,
        "vdm_model": args.vdm_model, "trials": args.trials, "workers": args.workers, "tasks": list(configs),
        **ensemble_settings(first_cfg), "trial_timeout_s": args.trial_timeout_s,
        "max_trial_retries": args.max_trial_retries, "extra": list(args.extra),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commands": {s: task_command(args, c, r, run_dir / s) for s, c, _, _, r in todo},
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if not todo:
        print("all trials finished")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "pilot_manifest.jsonl").open("a") as f:
        f.write(json.dumps(manifest) + "\n")

    procs = []
    if args.start_servers:
        from capx.envs.runner import _start_api_servers

        cfg = yaml.safe_load(todo[0][1].read_text())
        procs = _start_api_servers(cfg.get("api_servers"), wait_timeout=600.0)

    def run_task(item) -> tuple[str, int]:
        stem, config, task_dir, done, resume = item
        moved = quarantine_incomplete(task_dir, done, resume)
        if moved:
            print(f"[{stem}] moved unfinished trial dirs to _incomplete/: {moved}")
        cmd = task_command(args, config, resume, run_dir / stem)
        if api_key:
            cmd += ["--api-key", api_key, "--visual-differencing-model-api-key", api_key]
        print(f"[{stem}] start (trials {resume}..{args.trials})")
        with (log_dir / f"{stem}.log").open("a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} resume_idx={resume}\n")
            log.flush()
            rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"[{stem}] exit {rc}; {len(finished_trials(task_dir))}/{args.trials} trials finished")
        return stem, rc

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = list(pool.map(run_task, todo))
    finally:
        for p in procs:
            p.terminate()
    failed = [s for s, rc in results if rc != 0]
    remaining = [s for s, _, d, _, _ in plan if first_unfinished(finished_trials(d), args.trials) is not None]
    print(f"process failures: {failed or 'none'}; tasks with unfinished trials: {remaining or 'none'}")
    if remaining:
        print("rerun the same command to resume")
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
