"""Many-episode driver of the BG generator: prepare, run and evaluate every episode of ``scripts/bggen/episodes.yaml``.

    python -m capx.bggen.batch prep --out OUT [--only t08,b1k] [--force]
    python -m capx.bggen.batch run  --out OUT --tag RUN_TAG [--jobs 3] [--only ...] [--force]
    python -m capx.bggen.batch eval --out OUT --tag RUN_TAG [--only ...]
    python -m capx.bggen.batch list [--only ...]

Layout under OUT (the kubectl runbook's ``/workspace/cap-bg/outputs/agent-bg-gen``):

    data/libero_mem/<task file>.hdf5, data/b1k/...       downloads (only the listed files)
    episodes/<id>/{episode,oracle}/                       prepared episodes (skipped when already there)
    episodes/prep_status.json                             per episode: ok / skipped / failed / check_failed
    runs/<RUN_TAG>/<id>/                                  one generator run + eval per episode
    runs/<RUN_TAG>/batch_status.json                      per episode: ok / skipped / failed / not_prepared (+ init_failed)
    logs/batch/<step>_<id>.log                            every command's output

One failing episode never stops the batch: the failure (step, exit code, log tail) is recorded and the next episode
starts. Every step runs ``python -m capx.bggen.cli ...`` as a subprocess (LIBERO-Mem prepare with ``--py-lm``, the
LIBERO-Mem venv), so a crash inside one episode cannot take the driver down.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_LIST = Path(__file__).resolve().parents[2] / "scripts" / "bggen" / "episodes.yaml"


@dataclass(frozen=True)
class EpisodeSpec:
    id: str
    benchmark: str  # libero_mem | b1k
    task: int  # LIBERO-Mem cap-x task_id (0-based) / B1K task index
    demo: str | None = None  # LIBERO-Mem
    file: str | None = None  # LIBERO-Mem HDF5 file name
    episode: int | None = None  # B1K
    instruction: str | None = None

    @property
    def group(self) -> str:
        return f"{self.benchmark} t{self.task:02d}"


def libero_id(task: int, demo: str) -> str:
    return f"libero_mem_t{task:02d}_{demo}"


def b1k_id(task: int, episode: int) -> str:
    return f"b1k_t{task}_e{episode:08d}"


def load_episodes(path: str | Path = DEFAULT_LIST, only: Sequence[str] = ()) -> list[EpisodeSpec]:
    """The episode list, LIBERO-Mem first. ``only``: keep episodes whose id or group contains any of these
    (e.g. ``b1k``, ``t08``, ``libero_mem_t06_demo_2``)."""
    import yaml

    d = yaml.safe_load(Path(path).read_text()) or {}
    out: list[EpisodeSpec] = []
    for t in (d.get("libero_mem") or {}).get("tasks") or []:
        tid = int(t["task_id"])
        demos = t.get("demos") or []
        if not demos or not t.get("file"):
            raise ValueError(f"libero_mem task {tid}: needs 'file' and 'demos'")
        for demo in demos:
            out.append(EpisodeSpec(libero_id(tid, str(demo)), "libero_mem", tid, demo=str(demo), file=str(t["file"]),
                                   instruction=t.get("instruction")))
    b = d.get("b1k") or {}
    for e in b.get("episodes") or []:
        out.append(EpisodeSpec(b1k_id(int(b["task"]), int(e)), "b1k", int(b["task"]), episode=int(e),
                               instruction=b.get("instruction")))
    ids = [s.id for s in out]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"duplicate episodes in {path}: {dup}")
    if only:
        out = [s for s in out if any(o in s.id or o in s.group.replace(" ", "_") or o == s.benchmark for o in only)]
    return out


def is_prepared(root: str | Path) -> bool:
    """Episode + oracle written (the adapters write the oracle last)."""
    r = Path(root)
    return all((r / f).exists() for f in ("episode/meta.json", "episode/proprio.npz", "oracle/meta.json", "oracle/states.jsonl"))


def is_run(run_dir: str | Path) -> bool:
    return (Path(run_dir) / "summary.json").exists()


# --- commands -----------------------------------------------------------------------------------------

@dataclass
class Result:
    rc: int
    out: str


Runner = Callable[[list[str], Path, dict[str, str] | None], Result]


def subprocess_runner(cmd: list[str], log: Path, env: dict[str, str] | None = None) -> Result:
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       env={**os.environ, **(env or {})})
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n{p.stdout}\n[exit {p.returncode}, {time.time() - t0:.0f} s]\n")
    return Result(p.returncode, p.stdout)


def _tail(text: str, n: int = 15) -> str:
    return "\n".join((text or "").strip().splitlines()[-n:])


def _json_out(text: str) -> dict[str, Any] | None:
    """The last top-level JSON object a cli command printed (``_print``: indent 1, so a top-level object starts a
    line with "{"); log lines before or after it are ignored."""
    s = text or ""
    dec = json.JSONDecoder()
    found = None
    starts = ([0] if s.startswith("{") else []) + [i + 1 for i in range(len(s)) if s.startswith("\n{", i)]
    for i in starts:
        try:
            obj, _ = dec.raw_decode(s, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found = obj
    return found


class Status:
    """A JSON file of per-episode records, written after every change (thread-safe)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()
        self.data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}

    def set(self, ep_id: str, rec: dict[str, Any]) -> None:
        with self.lock:
            self.data[ep_id] = {**rec, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1, default=str))
            tmp.replace(self.path)


@dataclass
class Batch:
    out: Path
    py: str = sys.executable
    py_lm: str | None = None  # LIBERO-Mem venv python (prepare); default: ``py``
    runner: Runner = subprocess_runner
    wrist: bool = True
    libero_stride: int = 2
    b1k_stride: int = 6
    allow_check_fail: bool = False

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        self.logs = self.out / "logs" / "batch"
        self._dl_lock = threading.Lock()
        self._downloaded: set[str] = set()

    def root(self, s: EpisodeSpec) -> Path:
        return self.out / "episodes" / s.id

    def cli(self, *args: str, py: str | None = None) -> list[str]:
        return [py or self.py, "-m", "capx.bggen.cli", *args]

    def _cmd(self, step: str, s: EpisodeSpec, cmd: list[str], env: dict[str, str] | None = None) -> Result:
        return self.runner(cmd, self.logs / f"{step}_{s.id}.log", env)

    # --- prepare ---------------------------------------------------------------------------------------

    def _download(self, s: EpisodeSpec) -> Result:
        """Once per LIBERO-Mem task file / B1K episode in this process."""
        key = f"{s.benchmark}:{s.file if s.benchmark == 'libero_mem' else s.episode}"
        with self._dl_lock:
            if key in self._downloaded:
                return Result(0, "already downloaded")
            if s.benchmark == "libero_mem":
                r = self._cmd("download", s, self.cli("download-libero", "--dest", str(self.out / "data/libero_mem"),
                                                      "--task-id", str(s.task)))
            else:
                r = self._cmd("download", s, self.cli("download-b1k", "--dest", str(self.out / "data/b1k"),
                                                      "--episode", str(s.episode), *(["--wrist"] if self.wrist else [])))
            if r.rc == 0:
                self._downloaded.add(key)
            return r

    def prep_one(self, s: EpisodeSpec, force: bool = False) -> dict[str, Any]:
        root = self.root(s)
        if is_prepared(root) and not force:
            return {"status": "skipped", "reason": "already prepared", "root": str(root)}
        r = self._download(s)
        if r.rc != 0:
            return {"status": "failed", "step": "download", "rc": r.rc, "log": _tail(r.out)}
        if s.benchmark == "libero_mem":
            h5 = str(self.out / "data/libero_mem" / str(s.file))
            r = self._cmd("prep", s, self.cli("dump-libero", "--h5", h5, "--out", str(self.out / "data/libero_mem" /
                                                                                     f"structure_t{s.task:02d}.json")))
            if r.rc != 0:
                return {"status": "failed", "step": "dump", "rc": r.rc, "log": _tail(r.out)}
            env = {"MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"), "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM", "egl")}
            cmd = self.cli("prep-libero", "--h5", h5, "--demo", str(s.demo), "--root", str(root), "--task-id", str(s.task),
                           "--image-stride", str(self.libero_stride), *(["--wrist"] if self.wrist else []), py=self.py_lm or self.py)
            r = self._cmd("prep", s, cmd, env)
        else:
            cmd = self.cli("prep-b1k", "--src", str(self.out / "data/b1k"), "--root", str(root), "--episode", str(s.episode),
                           "--image-stride", str(self.b1k_stride), *(["--wrist"] if self.wrist else []))
            r = self._cmd("prep", s, cmd)
        if r.rc != 0:
            return {"status": "failed", "step": "prep", "rc": r.rc, "log": _tail(r.out)}
        prep = _json_out(r.out) or {}
        checks = dict(prep.get("checks") or {})
        if "task_info" in prep:
            checks["task_info_layout_ok"] = bool(prep["task_info"].get("layout_ok"))
        r2 = self._cmd("check", s, self.cli("check-episode", "--root", str(root)))
        chk = _json_out(r2.out) or {}
        rec = {"status": "ok", "root": str(root), "checks": checks, "prep": {k: prep.get(k) for k in (
            "demo", "frames", "replay", "success", "goal_final", "instruction_file", "instruction_env", "task_check",
            "eef_hdf5_vs_capx_mean_m", "gripper_width", "base_path_m") if k in prep},
               "event_counts": chk.get("event_counts"), "check_rc": r2.rc}
        bad = [k for k, v in checks.items() if v is False]
        if bad:
            rec.update(status="check_failed", failed_checks=bad)
        return rec

    def prep(self, specs: Sequence[EpisodeSpec], force: bool = False) -> dict[str, Any]:
        st = Status(self.out / "episodes" / "prep_status.json")
        for s in specs:
            t0 = time.time()
            try:
                rec = self.prep_one(s, force)
            except Exception as exc:  # the driver itself must go on
                rec = {"status": "failed", "step": "driver", "error": repr(exc)}
            prev = st.data.get(s.id)
            if rec["status"] == "skipped" and prev and prev.get("status") in ("ok", "check_failed"):
                rec = {**prev, "skipped_again": True}  # keep the verdict of the prepare that wrote it (check_failed stays)
            else:
                rec["seconds"] = round(time.time() - t0, 1)
            st.set(s.id, rec)
            print(f"[batch prep] {s.id}: {rec['status']}" + (f" ({rec.get('step')}, rc {rec.get('rc')})" if rec["status"] == "failed" else "")
                  + (f" failed checks {rec['failed_checks']}" if rec.get("failed_checks") else ""), flush=True)
        return st.data

    # --- run / eval --------------------------------------------------------------------------------------

    def run_dir(self, tag: str, s: EpisodeSpec) -> Path:
        return self.out / "runs" / tag / s.id

    def _prep_ok(self, s: EpisodeSpec) -> tuple[bool, str]:
        if not is_prepared(self.root(s)):
            return False, "not prepared"
        st = Status(self.out / "episodes" / "prep_status.json").data.get(s.id, {})
        if st.get("status") == "check_failed" and not self.allow_check_fail:
            return False, f"data checks failed {st.get('failed_checks')} (ALLOW_CHECK_FAIL=1 to run anyway)"
        return True, ""

    def run_one(self, tag: str, s: EpisodeSpec, force: bool = False, mode: str = "A",
                extra: Sequence[str] = ()) -> dict[str, Any]:
        ok, why = self._prep_ok(s)
        if not ok:
            return {"status": "not_prepared", "reason": why}
        d = self.run_dir(tag, s)
        if is_run(d) and not force:
            rec: dict[str, Any] = {"status": "skipped", "reason": "summary.json exists"}
        else:
            d.mkdir(parents=True, exist_ok=True)
            (d / "batch_episode.json").write_text(json.dumps(asdict(s) | {"group": s.group}, indent=1))
            r = self._cmd(f"run_{tag}", s, self.cli("run", "--root", str(self.root(s)), "--out", str(d), "--benchmark",
                                                    s.benchmark, "--mode", mode, *extra))
            if r.rc != 0 or not is_run(d):
                return {"status": "failed", "step": "run", "rc": r.rc, "log": _tail(r.out)}
            rec = {"status": "ok"}
        r = self._cmd(f"eval_{tag}", s, self.cli("eval", "--root", str(self.root(s)), "--out", str(d)))
        if r.rc != 0:
            rec.update(status="failed", step="eval", rc=r.rc, log=_tail(r.out))
        summ = json.loads((d / "summary.json").read_text())
        rec["init_failed"] = summ.get("init_failed")
        rec["goal_region_missing"] = (summ.get("goal") or {}).get("region_missing")
        return rec

    def run(self, tag: str, specs: Sequence[EpisodeSpec], force: bool = False, jobs: int = 1, mode: str = "A",
            extra: Sequence[str] = ()) -> dict[str, Any]:
        st = Status(self.out / "runs" / tag / "batch_status.json")

        def one(s: EpisodeSpec) -> None:
            t0 = time.time()
            try:
                rec = self.run_one(tag, s, force, mode, extra)
            except Exception as exc:
                rec = {"status": "failed", "step": "driver", "error": repr(exc)}
            rec["seconds"] = round(time.time() - t0, 1)
            st.set(s.id, rec)
            print(f"[batch run {tag}] {s.id}: {rec['status']}"
                  + (f" ({rec.get('step')}, rc {rec.get('rc')})" if rec["status"] == "failed" else "")
                  + (f" INIT FAILED: {rec['init_failed']}" if rec.get("init_failed") else "")
                  + (" REGION MISSING FROM GOAL" if rec.get("goal_region_missing") else ""), flush=True)

        if jobs <= 1:
            for s in specs:
                one(s)
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                list(ex.map(one, specs))
        return st.data

    def eval(self, tag: str, specs: Sequence[EpisodeSpec]) -> dict[str, Any]:
        """Re-evaluate every finished run (e.g. after an evaluate.py change); runs nothing new."""
        out = {}
        for s in specs:
            d = self.run_dir(tag, s)
            if not is_run(d):
                out[s.id] = "no run"
                continue
            r = self._cmd(f"eval_{tag}", s, self.cli("eval", "--root", str(self.root(s)), "--out", str(d)))
            out[s.id] = "ok" if r.rc == 0 else f"failed rc {r.rc}"
            print(f"[batch eval {tag}] {s.id}: {out[s.id]}", flush=True)
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="capx.bggen.batch")
    ap.add_argument("step", choices=("list", "prep", "run", "eval"))
    ap.add_argument("--episodes", default=str(DEFAULT_LIST))
    ap.add_argument("--out", default=os.environ.get("OUT", "/workspace/cap-bg/outputs/agent-bg-gen"))
    ap.add_argument("--tag", default=os.environ.get("RUN_TAG", "batch"))
    ap.add_argument("--only", default=os.environ.get("ONLY", ""), help="comma list: id / group substrings (b1k, t08, ...)")
    ap.add_argument("--force", action="store_true", help="redo prepared episodes / finished runs")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("JOBS", "1")), help="episodes run at once (run step)")
    ap.add_argument("--py", default=sys.executable)
    ap.add_argument("--py-lm", default=os.environ.get("PY_LM"))
    ap.add_argument("--no-wrist", action="store_true")
    ap.add_argument("--allow-check-fail", action="store_true", default=os.environ.get("ALLOW_CHECK_FAIL") == "1")
    ap.add_argument("--mode", default="A", choices=("A", "B"))
    ap.add_argument("--run-arg", action="append", default=[], help="extra 'cli run' argument (repeat), e.g. --run-arg=--fake-models")
    a = ap.parse_args(argv)
    specs = load_episodes(a.episodes, [x for x in a.only.split(",") if x])
    b = Batch(Path(a.out), py=a.py, py_lm=a.py_lm, wrist=not a.no_wrist, allow_check_fail=a.allow_check_fail)
    if a.step == "list":
        for s in specs:
            print(f"{s.id}\t{s.group}\tprepared={is_prepared(b.root(s))}\trun[{a.tag}]={is_run(b.run_dir(a.tag, s))}")
        return 0
    if a.step == "prep":
        st = b.prep(specs, a.force)
        bad = [k for k in (s.id for s in specs) if st.get(k, {}).get("status") in ("failed", "check_failed")]
    elif a.step == "run":
        st = b.run(a.tag, specs, a.force, a.jobs, a.mode, a.run_arg)
        bad = [k for k in (s.id for s in specs) if st.get(k, {}).get("status") in ("failed", "not_prepared")]
    else:
        st = b.eval(a.tag, specs)
        bad = [k for k, v in st.items() if v.startswith("failed")]
    print(f"[batch {a.step}] {len(specs)} episodes, {len(bad)} failed: {bad}", flush=True)
    return 0  # failures are recorded per episode; the runbook goes on to the next step


if __name__ == "__main__":
    sys.exit(main())
