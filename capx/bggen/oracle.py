"""Ground truth of a prepared episode -- EVALUATION ONLY. The generator never imports this module
(tests check it) and never opens ``<root>/oracle/``.

Files written by the adapters:

    oracle/meta.json       {"source", "frame": "base" | "world", "objects": {sim_name: category key},
                            "relations": [predicates covered], "arms": [...], "notes": [...]}
    oracle/states.jsonl    per sampled frame: {"t", "pos": {sim_name: [x, y, z]},
                            "true": [[pred, sim_a, sim_b | null], ...] (covered relations that hold),
                            "inhand": {arm: sim_name | null}}
    oracle/segments.json   [{"start", "end", "label", "objects": [...]}] -- subtask boundaries
                            (LIBERO-Mem: grasp intervals from the simulator; B1K: skill_annotation)
    oracle/final.json      {"success": bool | null}
"""

from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Oracle:
    meta: dict[str, Any]
    states: list[dict[str, Any]]
    segments: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.states.sort(key=lambda s: s["t"])
        self._ts = [s["t"] for s in self.states]

    @property
    def objects(self) -> dict[str, str]:
        return dict(self.meta.get("objects", {}))

    @property
    def relations(self) -> list[str]:
        return list(self.meta.get("relations", []))

    def at(self, t: int) -> dict[str, Any] | None:
        i = bisect_right(self._ts, t) - 1
        return self.states[i] if i >= 0 else None

    def truth(self, t: int) -> set[tuple[str, str, str | None]]:
        s = self.at(t)
        return set() if s is None else {(p[0], p[1], p[2] if len(p) > 2 else None) for p in s.get("true", [])}


def write_oracle(root: str | Path, meta: dict[str, Any], states: list[dict[str, Any]],
                 segments: list[dict[str, Any]] | None = None, final: dict[str, Any] | None = None) -> Path:
    d = Path(root) / "oracle"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    with open(d / "states.jsonl", "w") as f:
        for s in states:
            f.write(json.dumps(s) + "\n")
    (d / "segments.json").write_text(json.dumps(segments or [], indent=1))
    (d / "final.json").write_text(json.dumps(final or {"success": None}))
    return d


def load_oracle(root: str | Path) -> Oracle:
    d = Path(root) / "oracle"
    meta = json.loads((d / "meta.json").read_text())
    states = [json.loads(line) for line in (d / "states.jsonl").read_text().splitlines() if line.strip()]
    seg = json.loads((d / "segments.json").read_text()) if (d / "segments.json").exists() else []
    fin = json.loads((d / "final.json").read_text()) if (d / "final.json").exists() else {}
    return Oracle(meta, states, seg, fin)
