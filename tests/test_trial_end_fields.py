"""trial_end.json task_completed: bool / np.bool_ / "True" / "False" / 0 / 1 read correctly; unknown values are
errors, never successes (the analyzers used bool(), so the string "False" counted as a success)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import analyze_libero_mem_pilot as analyze_mem  # noqa: E402
import analyze_libero_pro_pilot as analyze_pro  # noqa: E402
import analyze_robomme_pilot as analyze_rm  # noqa: E402
from trial_end_fields import parse_success  # noqa: E402


@pytest.mark.parametrize("value, expected", [
    (True, 1), (False, 0), (np.bool_(True), 1), (np.bool_(False), 0),
    ("True", 1), ("False", 0), ("true", 1), ("false", 0), ("1", 1), ("0", 0),
    (1, 1), (0, 0), (np.int64(1), 1), (np.int64(0), 0), (None, 0),
])
def test_parse_success(value, expected):
    assert parse_success(value) == (expected, None)


@pytest.mark.parametrize("value", ["yes", "", "nan", 2, -1, 0.5, [True], {"a": 1}])
def test_unknown_values_are_errors_not_successes(value):
    success, err = parse_success(value)
    assert success == 0 and err and "unparsed task_completed" in err


def write_trial(task_dir: Path, trial: int, task_completed) -> None:
    d = task_dir / f"trial_{trial:02d}_x"
    d.mkdir(parents=True)
    (d / "trial_end.json").write_text(json.dumps({"trial": trial, "reason": "step_cap",
                                                  "task_completed": task_completed}))


def test_libero_mem_and_pro_collect_string_false(tmp_path):
    task_dir = tmp_path / "task_00_demo"
    write_trial(task_dir, 1, "False")
    write_trial(task_dir, 2, "True")
    write_trial(task_dir, 3, "maybe")
    rows = analyze_mem.collect(tmp_path, "c", 3, "m", "v", {})
    assert [r["success"] for r in rows] == [0, 1, 0]
    assert rows[2]["success_error"] and "maybe" in rows[2]["success_error"]
    assert "unreadable in 1 trial" in analyze_mem.summarize(rows, {})
    assert "unreadable in 1 trial" in analyze_pro.summarize(rows, {})


def test_robomme_collect_string_false(tmp_path):
    task_dir = tmp_path / "robomme_pickxtimes"
    write_trial(task_dir, 1, "False")
    write_trial(task_dir, 2, True)
    rows = analyze_rm.collect(tmp_path, "c", 2)
    assert [r["success"] for r in rows] == [0, 1]
    assert all(r["success_error"] is None for r in rows)
