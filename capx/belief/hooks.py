"""API-call hooks for the estimated Belief Graph (``bg_est``).

``ApiCallRecorder.wrap(name, fn)`` returns a function that calls ``fn`` with the same arguments and
returns its result (or re-raises its exception) unchanged, and records the call: the sim step
counter before and after, and the robot's end-effector position and gripper opening (from the
robot's own observation) before and after. Only motion and gripper calls are hooked; everything
else is returned as is.

The estimator turns the records into expected effects (``estimator.py``): an object in hand after
a close_gripper that did not close fully, the object put down on the support under the gripper
after an open_gripper, an object lost when the gripper closes fully while holding it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

HOOKED = ("close_gripper", "open_gripper", "move_to_joints", "goto_pose", "goto_home_joint_position")
GRIPPER_CALLS = ("close_gripper", "open_gripper")

Proprio = tuple[np.ndarray, float]  # (EEF xyz in the robot base frame, gripper 0 closed .. 1 open)


@dataclass
class CallRecord:
    index: int  # order of the call within the trial
    fn: str
    args: str  # short repr of the arguments
    step_before: int | None
    step_after: int | None
    eef_before: np.ndarray | None
    eef_after: np.ndarray | None
    gripper_before: float | None
    gripper_after: float | None
    error: str | None = None  # exception type if the call raised

    @property
    def desc(self) -> str:
        step = f" (sim step {self.step_after})" if self.step_after is not None else ""
        return f"{self.fn}{step}"

    def to_dict(self) -> dict[str, Any]:
        def r(v):
            return None if v is None else [round(float(x), 4) for x in v]

        return {
            "index": self.index, "fn": self.fn, "args": self.args,
            "step_before": self.step_before, "step_after": self.step_after,
            "eef_before": r(self.eef_before), "eef_after": r(self.eef_after),
            "gripper_before": None if self.gripper_before is None else round(self.gripper_before, 4),
            "gripper_after": None if self.gripper_after is None else round(self.gripper_after, 4),
            "error": self.error,
        }


def _short_args(args: tuple, kwargs: dict, limit: int = 120) -> str:
    def fmt(v: Any) -> str:
        if isinstance(v, np.ndarray):
            v = np.round(v.astype(np.float64), 3).tolist()
        return repr(v)

    text = ", ".join([fmt(a) for a in args] + [f"{k}={fmt(v)}" for k, v in kwargs.items()])
    return text if len(text) <= limit else text[: limit - 3] + "..."


class ApiCallRecorder:
    """Records hooked API calls. ``proprio_fn() -> (eef_xyz, gripper)`` and ``clock_fn() -> sim step``
    read the robot's own state; failures there are recorded as None and never affect the call."""

    def __init__(
        self,
        proprio_fn: Callable[[], Proprio | None] | None = None,
        clock_fn: Callable[[], int | None] | None = None,
        hooked: tuple[str, ...] = HOOKED,
    ) -> None:
        self.proprio_fn = proprio_fn
        self.clock_fn = clock_fn
        self.hooked = hooked
        self.records: list[CallRecord] = []
        self._drained = 0
        self._depth = 0  # nested hooked calls (an API calling another API) are recorded once

    def _proprio(self) -> Proprio | None:
        try:
            p = self.proprio_fn() if self.proprio_fn is not None else None
            return None if p is None else (np.array(p[0], dtype=np.float64, copy=True), float(p[1]))
        except Exception:
            return None

    def _clock(self) -> int | None:
        try:
            return self.clock_fn() if self.clock_fn is not None else None
        except Exception:
            return None

    def wrap(self, name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
        if name not in self.hooked or getattr(fn, "__bg_hooked__", False):
            return fn

        @functools.wraps(fn)
        def hooked(*args: Any, **kwargs: Any) -> Any:
            if self._depth:
                return fn(*args, **kwargs)
            before, step_before = self._proprio(), self._clock()
            error = None
            self._depth += 1
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                error = type(exc).__name__
                raise
            finally:
                self._depth -= 1
                self._record(name, args, kwargs, before, step_before, error)

        hooked.__bg_hooked__ = True  # type: ignore[attr-defined]
        return hooked

    def _record(self, name: str, args: tuple, kwargs: dict, before: Proprio | None, step_before: int | None,
                error: str | None) -> None:
        """Append a record; never raises, so the wrapped call's result or exception is unchanged."""
        try:
            after = self._proprio()
            try:
                arg_text = _short_args(args, kwargs)
            except Exception:
                arg_text = "<unprintable>"

            def vec(p: Proprio | None) -> np.ndarray | None:
                return None if p is None else np.array(p[0], dtype=np.float64, copy=True).reshape(-1)[:3]

            def grip(p: Proprio | None) -> float | None:
                return None if p is None else float(p[1])

            self.records.append(CallRecord(
                index=len(self.records), fn=name, args=arg_text,
                step_before=step_before, step_after=self._clock(),
                eef_before=vec(before), eef_after=vec(after),
                gripper_before=grip(before), gripper_after=grip(after), error=error,
            ))
        except Exception:
            pass

    def wrap_functions(self, fns: dict[str, Callable[..., Any]]) -> dict[str, Callable[..., Any]]:
        return {name: self.wrap(name, fn) for name, fn in fns.items()}

    def drain(self) -> list[CallRecord]:
        """Records since the last drain (one turn's calls)."""
        out = self.records[self._drained:]
        self._drained = len(self.records)
        return out

    def reset(self) -> None:
        self.records = []
        self._drained = 0
        self._depth = 0
