"""Compare two output directories of the same trials file by file (used for trial-mem-leak: before / after).

Every file must exist in both and be byte-identical, except for what differs between any two runs of the
same code: JSON wall-clock fields (``*_s``, ``latency*``, ``time*``, ``wall*``) are dropped, and the parallel
ensemble's candidates (kept in thread completion order) are compared as a set, in JSON
(``ensemble_calls[].candidates``) and in ``ensemble_candidates*.txt``.

    python scripts/compare_trial_outputs.py OUT_A OUT_B
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

TIMING = re.compile(r"(_s$|^latency|^time|^wall|_time$|^t$)")


def strip(o):
    if isinstance(o, dict):
        out = {k: strip(v) for k, v in o.items() if not TIMING.search(k)}
        if isinstance(out.get("candidates"), list):
            out["candidates"] = sorted(out["candidates"], key=lambda c: json.dumps(c, sort_keys=True))
        return out
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def candidate_blocks(x: bytes) -> list[bytes]:
    """ensemble_candidates*.txt: one block per candidate, each starting with a line of '='."""
    return sorted(b.strip() for b in re.split(rb"(?m)^(?==+\nModel: )", x))


def main(a: str, b: str) -> int:
    ra, rb = Path(a), Path(b)
    fa = {p.relative_to(ra) for p in ra.rglob("*") if p.is_file()}
    fb = {p.relative_to(rb) for p in rb.rglob("*") if p.is_file()}
    bad = 0
    for rel in sorted(fa ^ fb):
        print(f"ONLY IN {'A' if rel in fa else 'B'}: {rel}")
        bad += 1
    same = 0
    for rel in sorted(fa & fb):
        xa, xb = (ra / rel).read_bytes(), (rb / rel).read_bytes()
        if rel.suffix in (".json", ".jsonl"):
            try:
                if rel.suffix == ".json":
                    ok = strip(json.loads(xa)) == strip(json.loads(xb))
                else:
                    ok = [strip(json.loads(x)) for x in xa.splitlines()] == [strip(json.loads(x)) for x in xb.splitlines()]
            except ValueError:
                ok = xa == xb
        elif rel.name.startswith("ensemble_candidates") and rel.suffix == ".txt":
            ok = candidate_blocks(xa) == candidate_blocks(xb)
        else:
            ok = xa == xb
        if ok:
            same += 1
        else:
            print(f"DIFFERS: {rel}")
            bad += 1
    print(f"{same} files identical, {bad} differ or missing (of {len(fa | fb)})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
