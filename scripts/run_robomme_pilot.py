"""Run the RoboMME CaP-X baseline pilot: N tasks x T trials in one pod, resumable.

Trial k of every task plays test episode k-1 (trials 1-5 -> episodes 0-4), so conditions can be compared
seed by seed. Each task runs as its own ``capx/envs/launch.py`` process (one sequential worker, so trials
finish in order); ``--workers`` task processes run at a time. The perception / IK servers from the configs
(SAM3 :8114, Contact-GraspNet :8115, pyroki :8116) are started once here and shared; the Molmo2-8B server
(:8122) must be started separately from .venv-molmo (see docs/capbg-baseline.md).

Resume: rerun the same command. A trial counts as done when its directory has ``trial_end.json``; per task the
run restarts at the first unfinished trial (``--resume-idx``) after moving half-written trial directories to
``_incomplete/``. Finished tasks are skipped.

    export NEBIUS_SERVER_URL=... NEBIUS_API_KEY=...
    python scripts/run_robomme_pilot.py --run-name 20261101_robomme_pilot --workers 2
    python scripts/analyze_robomme_pilot.py outputs/robomme_pilot/20261101_robomme_pilot

Outputs: outputs/robomme_pilot/<run>/<model>/robomme_<task>/trial_XX_*/ (launch.py inserts the model name),
logs in logs/robomme_pilot/<run>/<task>.log, the exact commands in <run>/pilot_manifest.json.
"""

from __future__ import annotations

import json
import os
import shutil
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
from gen_robomme_configs import TASKS  # noqa: E402

ALL_TASKS = [t for t, _ in TASKS]


@dataclass
class Args:
    run_name: str
    """Run directory name, e.g. 20261101_robomme_pilot."""
    workers: int = 2
    """Task processes running at the same time (each builds its own SAPIEN scene on the GPU)."""
    trials: int = 5
    """Trials per task (trial k = test episode k-1)."""
    tasks: list[str] = field(default_factory=lambda: list(ALL_TASKS))
    memory_mode: str = "none"
    model: str = "moonshotai/Kimi-K3"
    vdm_model: str = "Qwen/Qwen3.8-27B"
    server_url: str = os.environ.get("NEBIUS_SERVER_URL", "")
    """Chat-completions URL for the coding model and the VDM (default: $NEBIUS_SERVER_URL)."""
    api_key_env: str = "NEBIUS_API_KEY"
    """Environment variable holding the API key (the key itself never goes on a command line or into logs)."""
    out_root: str = "outputs/robomme_pilot"
    log_root: str = "logs/robomme_pilot"
    config_dir: str = "env_configs/robomme"
    python: str = sys.executable
    start_servers: bool = True
    dry_run: bool = False
    """Print the plan and commands without running anything."""
    extra: list[str] = field(default_factory=list)
    """Extra launch.py flags, e.g. --extra --trial-timeout-s 1800."""


def finished_trials(task_dir: Path) -> set[int]:
    done = set()
    for end in task_dir.glob("trial_*/trial_end.json"):
        try:
            done.add(int(json.loads(end.read_text())["trial"]))
        except Exception:
            continue
    return done


def first_unfinished(done: set[int], trials: int) -> int | None:
    for k in range(1, trials + 1):
        if k not in done:
            return k
    return None


def quarantine_incomplete(task_dir: Path, done: set[int], from_trial: int) -> list[str]:
    """Move trial dirs of unfinished trials >= from_trial (no trial_end.json) out of the way."""
    moved = []
    for d in sorted(task_dir.glob("trial_*")):
        try:
            k = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if k >= from_trial and k not in done and not (d / "trial_end.json").exists():
            dest = task_dir / "_incomplete" / f"{d.name}.{int(time.time())}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(d), str(dest))
            moved.append(d.name)
    return moved


def task_command(args: Args, task: str, resume_idx: int, out_dir: Path) -> list[str]:
    cmd = [
        args.python, str(REPO_ROOT / "capx" / "envs" / "launch.py"),
        "--config-path", str(REPO_ROOT / args.config_dir / f"robomme_{task.lower()}.yaml"),
        "--model", args.model,
        "--visual-differencing-model", args.vdm_model,
        "--total-trials", str(args.trials),
        "--resume-idx", str(resume_idx),
        "--num-workers", "1",
        "--output-dir", str(out_dir),
        "--memory-mode", args.memory_mode,
        "--record-video", "True",
    ]
    if args.server_url:
        cmd += ["--server-url", args.server_url, "--visual-differencing-model-server-url", args.server_url]
    return cmd + list(args.extra)


def main(args: Args) -> int:
    unknown = [t for t in args.tasks if t not in ALL_TASKS]
    if unknown:
        raise SystemExit(f"unknown tasks: {unknown}; choose from {ALL_TASKS}")
    run_dir = REPO_ROOT / args.out_root / args.run_name
    log_dir = REPO_ROOT / args.log_root / args.run_name
    model_dir = run_dir / args.model.replace("/", "_")  # launch.py inserts the model name before the last part
    env = dict(os.environ)
    api_key = env.get(args.api_key_env)  # passed like the documented `--api-key "$NEBIUS_API_KEY"`, never logged

    plan = []
    for task in args.tasks:
        task_dir = model_dir / f"robomme_{task.lower()}"
        done = finished_trials(task_dir)
        resume = first_unfinished(done, args.trials)
        plan.append((task, task_dir, done, resume))
        print(f"{task:18s} done {sorted(done)} -> {'skip' if resume is None else f'resume at trial {resume}'}")

    todo = [p for p in plan if p[3] is not None]
    manifest = {
        "run_name": args.run_name, "memory_mode": args.memory_mode, "model": args.model, "vdm_model": args.vdm_model,
        "trials": args.trials, "workers": args.workers, "tasks": args.tasks, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "commands": {t: task_command(args, t, r, run_dir / f"robomme_{t.lower()}") for t, _, _, r in todo},
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if not todo:
        print("all trials finished")
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    manifests = run_dir / "pilot_manifest.jsonl"
    with manifests.open("a") as f:
        f.write(json.dumps(manifest) + "\n")

    procs = []
    if args.start_servers:
        from capx.envs.runner import _start_api_servers

        cfg = yaml.safe_load((REPO_ROOT / args.config_dir / f"robomme_{todo[0][0].lower()}.yaml").read_text())
        procs = _start_api_servers(cfg.get("api_servers"), wait_timeout=600.0)

    def run_task(item) -> tuple[str, int]:
        task, task_dir, done, resume = item
        moved = quarantine_incomplete(task_dir, done, resume)
        if moved:
            print(f"[{task}] moved unfinished trial dirs to _incomplete/: {moved}")
        cmd = task_command(args, task, resume, run_dir / f"robomme_{task.lower()}")
        if api_key:
            cmd += ["--api-key", api_key, "--visual-differencing-model-api-key", api_key]
        print(f"[{task}] start (trials {resume}..{args.trials})")
        with (log_dir / f"{task}.log").open("a") as log:
            log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} resume_idx={resume}\n")
            log.flush()
            rc = subprocess.call(cmd, cwd=str(REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
        n_done = len(finished_trials(task_dir))
        print(f"[{task}] exit {rc}; {n_done}/{args.trials} trials finished")
        return task, rc

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            results = list(pool.map(run_task, todo))
    finally:
        for p in procs:
            p.terminate()
    failed = [t for t, rc in results if rc != 0]
    remaining = [t for t, d, _, _ in plan if first_unfinished(finished_trials(d), args.trials) is not None]
    print(f"process failures: {failed or 'none'}; tasks with unfinished trials: {remaining or 'none'}")
    if remaining:
        print("rerun the same command to resume")
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
