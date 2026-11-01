"""LIBERO regression digest (bggen-fix5 D): the synthetic swap episode run through the LIBERO-Mem preset with the
scripted FAKE models; a digest of every output row file, the decision parts of the trace and every prompt sent to the
models. ``tests/fixtures/bggen_libero_golden.json`` holds the digests made at d401b65 (bggen-fix4, before fix5); a
change of the LIBERO path changes a digest. Regenerate only for an intended LIBERO change:
``PYTHONPATH=. python -m tests.bggen_golden``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from capx.bggen.configs import libero_mem as lm_cfg
from capx.bggen.fake_models import FakeServer
from capx.bggen.models import Models
from capx.bggen.pipeline import run_episode
from tests.bggen_fakes import FakeDetector, TruthBrain, write_swap_episode

GOLDEN = Path(__file__).parent / "fixtures" / "bggen_libero_golden.json"
FILES = ("rows_forward.jsonl", "rows_backward.jsonl", "cot_rows_forward.jsonl", "cot_rows.jsonl")
TRACE_KEYS = ("t", "event", "graph", "goal", "observe", "s4", "decisions", "tracks", "track", "grasp", "grasp_retry",
              "grasp_verify", "plan", "marks", "pending", "goal_confirmed", "tools")


def _h(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


VARIANTS: dict[str, dict[str, Any]] = {
    "swap_A": {"mode": "A", "hide": False},
    # bowl_2 unseen from t=20 until 30 frames after its grasp: that grasp has no target with a position (the fix5 B1K
    # unseen-grasp binding must stay off on LIBERO)
    "swap_B_hidden": {"mode": "B", "hide": True},
}


def digest(root: Path, name: str) -> dict[str, str]:
    v = VARIANTS[name]
    w = write_swap_episode(root)
    hide = None
    if v["hide"]:
        until = w.grasp_t[1] + 30
        hide = lambda t, s: s == "bowl_2" and 20 <= t <= until  # noqa: E731
    flip_from = w.grasp_t[2] + 5
    brain = TruthBrain(w, flip=[("(on bowl_B plate_B)", flip_from, flip_from + 35)])
    out = root / "out"
    with FakeServer(brain) as srv:
        run_episode(root, out, lm_cfg(v["mode"]), Models.from_env(env=srv.env()), FakeDetector(w, hide=hide), "libero_mem")
    d = {f: hashlib.sha256((out / f).read_bytes()).hexdigest()[:16] for f in FILES if (out / f).exists()}
    trace = [json.loads(x) for x in (out / "trace.jsonl").read_text().splitlines() if x.strip()]
    for line in trace:
        for r in (line.get("agent") or {}).get("replies", []):
            r.pop("usage", None)
    d["trace"] = _h([{k: line.get(k) for k in TRACE_KEYS} for line in trace])
    d["agent_replies"] = _h([[r.get("reply") for r in (line.get("agent") or {}).get("replies", [])] for line in trace])
    d["prompts"] = _h(brain.prompts)
    d["n_prompts"] = str(len(brain.prompts))
    return d


def main() -> None:
    import tempfile

    res = {}
    for name in VARIANTS:
        with tempfile.TemporaryDirectory() as tmp:
            res[name] = digest(Path(tmp), name)
    GOLDEN.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
