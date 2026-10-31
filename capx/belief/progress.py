"""Task progress from a BDDL goal: Sequence / Or-of-Sequence goals unrolled into ordered steps.

A step is a conjunction of predicates (one ``(And ...)`` of a ``(Sequence ...)``). An ``(Or (Sequence
...) ...)`` goal has several branches; a branch stays possible while its prefix matches the completed
steps. Delta (remaining) is the fewest steps left over the possible branches. Repeat tasks (the
same step several times in a row, e.g. "place the bowl back 3 times") get a completion counter.

Steps are canonical (sorted, de-duplicated predicates), so a step matches regardless of the
predicate order inside ``(And ...)``. If the completed steps match no branch, progress is unknown:
``remaining`` is None and no completion is claimed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

Pred = tuple[str, ...]  # ("on", "akita_black_bowl_1", "plate_3")
Step = tuple[Pred, ...]


def _pred(p: Sequence[Any]) -> Pred:
    return tuple(str(x).lower() if i == 0 else str(x) for i, x in enumerate(p))


def canonical_step(preds: Sequence[Sequence[Any]]) -> Step:
    """Order-independent step: the same set of predicates gives the same tuple."""
    return tuple(sorted({_pred(p) for p in preds}))


def _step(node: Sequence[Any]) -> Step:
    """``['and', p1, p2]`` or a bare predicate list -> canonical step."""
    if node and isinstance(node[0], str) and node[0].lower() == "and":
        return canonical_step(node[1:])
    if node and isinstance(node[0], str):
        return canonical_step([node])
    return canonical_step(node)


def goal_branches(goal_state: Sequence[Any]) -> list[list[Step]]:
    """Unroll a parsed BDDL goal (``parsed_problem["goal_state"]``) into step lists.

    Mirrors LIBERO-Mem's ``get_all_goals``: ``[['or', ['sequence', ...], ...]]`` -> one list per
    branch, ``[['sequence', ...]]`` -> one list, anything else (a plain conjunction) -> one step.
    """
    head = goal_state[0] if goal_state else None
    op = head[0].lower() if head and isinstance(head[0], str) else None
    if op == "or":
        return [[_step(s) for s in seq[1:]] for seq in head[1:]]
    if op == "sequence":
        return [[_step(s) for s in head[1:]]]
    if op == "and":
        return [[_step(head)]]
    return [[canonical_step(goal_state)]]


def format_pred(p: Pred) -> str:
    if len(p) == 3:
        return f"{p[1]} {p[0]} {p[2]}"
    return " ".join(p[1:] + p[:1])


def format_step(step: Step) -> str:
    return " and ".join(format_pred(p) for p in step)


@dataclass
class Progress:
    """Snapshot of task progress after a turn."""

    branches: list[list[Step]]
    done: list[Step]  # completed steps, in order
    overshot: bool = False  # the finished goal was undone; LIBERO-Mem then never succeeds
    possible: list[int] = field(default_factory=list)  # branch indices consistent with ``done``

    @classmethod
    def from_completed(cls, branches: list[list[Step]], done: list[Step], overshot: bool = False) -> Progress:
        branches = [[canonical_step(s) for s in b] for b in branches]
        done = [canonical_step(s) for s in done]
        possible = [i for i, b in enumerate(branches) if b[: len(done)] == done]
        return cls(branches=branches, done=done, overshot=overshot, possible=possible)

    @property
    def known(self) -> bool:
        """False when the completed steps match no goal branch (tracker and goal disagree)."""
        return bool(self.possible)

    @property
    def total(self) -> int:
        """Steps needed along the shortest possible branch (all branches if none is possible)."""
        idx = self.possible or range(len(self.branches))
        return min(len(self.branches[i]) for i in idx)

    @property
    def remaining(self) -> int | None:
        """Delta: steps still to do (0 = all steps verified complete); None if progress is unknown."""
        if not self.possible:
            return None
        return min(len(self.branches[i]) - len(self.done) for i in self.possible)

    @property
    def complete(self) -> bool:
        return self.known and self.remaining == 0 and not self.overshot

    def next_steps(self) -> list[Step]:
        """Distinct next steps over the possible branches (several only for an undecided Or)."""
        out: list[Step] = []
        for i in self.possible:
            b = self.branches[i]
            if len(self.done) < len(b) and b[len(self.done)] not in out:
                out.append(b[len(self.done)])
        return out

    def remaining_steps(self) -> list[list[Step]]:
        """Steps left on each possible branch."""
        return [self.branches[i][len(self.done):] for i in self.possible]

    def repeat_count(self) -> tuple[int, int] | None:
        """(completed, required) when every step of the goal is the same step, else None."""
        if len(self.branches) != 1 or len(self.branches[0]) < 2:
            return None
        steps = self.branches[0]
        if any(s != steps[0] for s in steps):
            return None
        return len(self.done), len(steps)


def progress_line(
    progress: Progress,
    current_truths: dict[Pred, bool] | None = None,
    complete_text: str = "all steps verified complete",
) -> str:
    """One-line progress summary for the prompt (``complete_text`` is added when no step remains)."""
    if not progress.known:
        done_text = "; ".join(format_step(s) for s in progress.done)
        return (
            f"Progress: progress unknown (WARNING: the {len(progress.done)} completed step(s) [{done_text}] "
            "match no branch of the goal; remaining steps cannot be determined)"
        )
    done, total, rem = len(progress.done), progress.total, progress.remaining
    rep = progress.repeat_count()
    if rep is not None:
        head = f"Progress: {rep[0]}/{rep[1]} repetitions done ({format_step(progress.branches[0][0])})"
    elif done:
        head = f"Progress: step {done}/{total} done ({'; '.join(format_step(s) for s in progress.done)})"
    else:
        head = f"Progress: step 0/{total} done"
    parts = [head]
    if progress.overshot:
        parts.append("OVERSHOT: the completed goal state was undone, the task can no longer succeed")
    nxt = progress.next_steps()
    if nxt:
        texts = []
        for s in nxt:
            t = format_step(s)
            if current_truths is not None and all(current_truths.get(p, False) for p in s):
                t += " (holds now but not counted yet)"
            texts.append(t)
        parts.append("next: " + (texts[0] if len(texts) == 1 else "one of " + " OR ".join(texts)))
    parts.append(f"remaining: {rem}")
    if rem == 0 and not progress.overshot:
        parts.append(complete_text)
    return " | ".join(parts)
