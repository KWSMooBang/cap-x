"""Re-apply the current multi-turn decision parser to saved replies and list decisions that change.

Usage: python scripts/rejudge_decisions.py RUN_DIR [RUN_DIR ...] [--json OUT]

Walks every ``all_responses.json`` under the given directories. Replies are only saved for finish
decisions (``reasoning`` holds the reply when the model returned no separate reasoning) and for the
first reply of an S5(a) rejection (``s5_first_reply.reply``); regenerate entries keep only the parsed
code, and the parser never turns code into finish, so they cannot change. For each saved reply:

- changed: the new parser decides differently (e.g. ``fallback`` -> ``raw_code``). Expected to be
  exactly the misparsed_finish trials with unfenced code.
- text_mismatch: same decision, but a different ``decision_source`` than recorded. The stored text is
  probably the model's reasoning rather than its reply, so that entry cannot be re-judged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capx.utils.launch_utils import parse_multi_turn_decision  # noqa: E402


def _replies(entries: list[dict]):
    for idx, entry in enumerate(entries):
        first = entry.get("s5_first_reply")
        if isinstance(first, dict) and first.get("decision_source"):
            yield idx, "s5_first_reply", first.get("reply"), first
        if entry.get("decision") == "finish" and entry.get("decision_source"):
            yield idx, "finish", entry.get("reasoning"), entry


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, help="write the per-reply rows here")
    args = ap.parse_args()

    rows, counts = [], {"files": 0, "replies": 0, "unchanged": 0, "changed": 0, "text_mismatch": 0}
    for root in args.dirs:
        for path in sorted(root.rglob("all_responses.json")):
            counts["files"] += 1
            try:
                entries = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                print(f"skip {path}: {exc}", file=sys.stderr)
                continue
            for idx, where, text, meta in _replies(entries):
                counts["replies"] += 1
                old_source = meta["decision_source"]
                old_decision = "finish" if where == "finish" else _decision_of(old_source)
                if old_source in ("empty_retry", "truncated_skip"):
                    counts["unchanged"] += 1  # decided by the retry logic, not by the text alone
                    continue
                new = parse_multi_turn_decision(text)
                if new.decision != old_decision:
                    status = "changed"
                elif new.source != old_source:
                    status = "text_mismatch"
                else:
                    status = "unchanged"
                counts[status] += 1
                if status != "unchanged":
                    first_line = (text or "").strip().splitlines()[:1]
                    row = {"file": str(path), "entry": idx, "where": where, "status": status,
                           "old": f"{old_decision}/{old_source}", "new": f"{new.decision}/{new.source}",
                           "flags": new.flags, "first_line": first_line[0][:100] if first_line else ""}
                    rows.append(row)
                    print(f"{status:14s} {row['old']:>24s} -> {row['new']:<22s} {path.parent.name} "
                          f"#{idx} {where}: {row['first_line']!r}")
    print(json.dumps(counts))
    if args.json:
        args.json.write_text(json.dumps({"counts": counts, "rows": rows}, indent=2))
    return 0


def _decision_of(source: str) -> str:
    return "regenerate" if source in ("keyword", "code_block", "tool_call", "raw_code") else "finish"


if __name__ == "__main__":
    sys.exit(main())
