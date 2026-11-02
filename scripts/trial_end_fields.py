"""Read fields of trial_end.json written by any version of capx/envs/trial.py.

Before 1cf4416, numpy values were written with ``default=str``, so ``task_completed`` could be the string
"True" / "False" (``bool("False")`` is True: never use ``bool()`` on it).
"""

from __future__ import annotations

from typing import Any

_TRUE = {"true", "1"}
_FALSE = {"false", "0"}


def parse_success(value: Any) -> tuple[int, str | None]:
    """``task_completed`` -> (0 | 1, error). Unknown values give (0, error): never counted as success.

    Accepts bool / np.bool_, int 0 / 1 (and numpy ints), "True" / "False" / "true" / "false" / "0" / "1",
    and None / missing (0, not completed).
    """
    if value is None:
        return 0, None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):  # numpy scalar
        try:
            value = value.item()
        except (TypeError, ValueError):
            return 0, f"unparsed task_completed: {value!r}"
    if isinstance(value, bool):
        return int(value), None
    if isinstance(value, int) and value in (0, 1):
        return value, None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE:
            return 1, None
        if s in _FALSE:
            return 0, None
    return 0, f"unparsed task_completed: {value!r}"


def success_of(end: dict[str, Any], where: Any = "") -> tuple[int, str | None]:
    """parse_success(end["task_completed"]); the error names the file (``where``)."""
    success, err = parse_success(end.get("task_completed"))
    if err:
        err = f"{where}: {err}" if where else err
    return success, err


def success_error_lines(rows: list[dict[str, Any]]) -> list[str]:
    """SUMMARY.md lines listing trials whose task_completed could not be read (counted as failures)."""
    errs = [r["success_error"] for r in rows if r.get("success_error")]
    if not errs:
        return []
    return [f"**task_completed unreadable in {len(errs)} trial(s) (counted as failures):**", "",
            *(f"- {e}" for e in errs), ""]
