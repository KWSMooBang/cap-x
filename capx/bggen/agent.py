"""Agent reply -> evidence plan, with a fixed fallback (design: 증거 계획).

A reply is used when it holds a JSON object with a ``calls`` list of known tools whose statements and
names parse. Anything else (no JSON, unknown tool, unknown object, every statement malformed, hindsight
in the online mode) is a format failure: the plan becomes the default one -- relations of the objects
around the event + the verifier's re-checks + the next goal relations -- and the reason is logged.
Single bad statements inside an otherwise valid reply are dropped and listed.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .vocab import ARMS_ONE, EdgeKey, parse_statement, statement_text

TOOLS = ("observe", "geometry", "detect", "look", "hindsight", "robot_rel")


@dataclass
class Plan:
    observe: list[EdgeKey] = field(default_factory=list)
    geometry: list[tuple[str, str]] = field(default_factory=list)
    detect: list[str] = field(default_factory=list)
    camera: str | None = None
    hindsight: list[EdgeKey] = field(default_factory=list)
    hindsight_after_s: float = 1.0
    expect: list[tuple[EdgeKey, bool]] = field(default_factory=list)
    why: str = ""
    fallback: bool = False
    fallback_reason: str | None = None
    dropped: list[str] = field(default_factory=list)
    forced: list[str] = field(default_factory=list)  # statements the verifier forced in (re-checks)

    def to_dict(self) -> dict[str, Any]:
        return {"observe": [statement_text(k) for k in self.observe], "geometry": [list(p) for p in self.geometry],
                "detect": self.detect, "camera": self.camera, "hindsight": [statement_text(k) for k in self.hindsight],
                "expect": [[statement_text(k), int(v)] for k, v in self.expect], "why": self.why,
                "fallback": self.fallback, "fallback_reason": self.fallback_reason, "dropped": self.dropped,
                "forced": self.forced}


class PlanError(ValueError):
    pass


def extract_json(text: str) -> Any:
    text = re.sub(r"```(?:json)?", "", text or "")
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise PlanError("no JSON object in the reply")


def json_error(text: str) -> str | None:
    """bggen-fix6 D: where the reply's JSON breaks, for a re-query (None if there is no ``{`` or it parses). The span
    from the first ``{`` to the last ``}`` is decoded; the error's message, line / column and the text around it."""
    text = re.sub(r"```(?:json)?", "", text or "")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    body = text[start:end + 1]
    try:
        json.loads(body)
    except json.JSONDecodeError as exc:
        lo, hi = max(exc.pos - 60, 0), min(exc.pos + 20, len(body))
        near = body[lo:exc.pos] + " <HERE> " + body[exc.pos:hi]
        return f"{exc.msg} at line {exc.lineno} column {exc.colno} (character {exc.pos}): ...{near}..."
    return None


def parse_plan(text: str, names: Sequence[str], arms: Sequence[str | None] = ARMS_ONE, cameras: Sequence[str] = (),
               categories: Sequence[str] = (), offline: bool = False, max_obs: int = 10) -> Plan:
    data = extract_json(text)
    if not isinstance(data, dict) or not isinstance(data.get("calls"), list):
        raise PlanError("missing 'calls' list")
    plan = Plan(why=str(data.get("why", ""))[:300])
    known = set(names) | {"table"}
    n_ok = 0
    for c in data["calls"]:
        if not isinstance(c, dict) or c.get("tool") not in TOOLS:
            raise PlanError(f"unknown tool call {c!r}")
        tool = c["tool"]
        if tool in ("observe", "hindsight"):
            if tool == "hindsight" and not offline:
                raise PlanError("hindsight is offline-only")
            stmts = c.get("statements")
            if not isinstance(stmts, list):
                raise PlanError(f"{tool}: 'statements' must be a list")
            for s in stmts:
                k = parse_statement(str(s), names, arms)
                if k is None:
                    plan.dropped.append(str(s))
                    continue
                dst = plan.observe if tool == "observe" else plan.hindsight
                if k not in dst:
                    dst.append(k)
                    n_ok += 1
            if tool == "hindsight":
                try:
                    plan.hindsight_after_s = float(c.get("after_s", 1.0))
                except (TypeError, ValueError):
                    plan.hindsight_after_s = 1.0
        elif tool == "geometry":
            for p in c.get("pairs") or []:
                if not (isinstance(p, (list, tuple)) and len(p) == 2 and all(str(x) in known for x in p)):
                    plan.dropped.append(f"geometry {p!r}")
                    continue
                plan.geometry.append((str(p[0]), str(p[1])))
                n_ok += 1
        elif tool == "detect":
            for cat in c.get("categories") or []:
                if str(cat) in categories:
                    plan.detect.append(str(cat))
                    n_ok += 1
                else:
                    plan.dropped.append(f"detect {cat!r}")
        elif tool == "look":
            cam = str(c.get("camera", ""))
            if cam in cameras:
                plan.camera = cam
                n_ok += 1
            else:
                plan.dropped.append(f"look {cam!r}")
        else:  # robot_rel: always computed
            n_ok += 1
    for item in data.get("expect") or []:
        if not isinstance(item, dict):
            continue
        k = parse_statement(str(item.get("statement", "")), names, arms)
        if k is None or item.get("value") not in (0, 1, "0", "1", True, False):
            plan.dropped.append(f"expect {item!r}")
            continue
        plan.expect.append((k, str(item["value"]) in ("1", "True")))
    if data["calls"] and n_ok == 0:
        raise PlanError("no usable call (every statement / name malformed)")
    plan.observe = plan.observe[:max_obs]
    return plan


def default_plan(suggestions: Sequence[EdgeKey], geometry: Sequence[tuple[str, str]], reason: str,
                 max_obs: int = 10) -> Plan:
    return Plan(observe=list(dict.fromkeys(suggestions))[:max_obs], geometry=list(geometry), fallback=True,
                fallback_reason=reason, why="default plan")
