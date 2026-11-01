"""Model calls of the BG generator: one OpenAI-compatible chat client, one config per role.

Roles: ``init`` (scene + goal initialization, once per episode), ``agent`` (evidence plan, every event),
``vdm`` (Observe, marked images -> 1 / 0 / ?). Defaults: init / agent ``moonshotai/Kimi-K3``, vdm
``Qwen/Qwen3.8-27B``.

URLs and keys come only from the environment:

* ``BGGEN_<ROLE>_URL`` (e.g. ``BGGEN_VDM_URL``), else ``NEBIUS_SERVER_URL``. A full chat-completions URL
  (``.../v1/chat/completions``) or a base URL ending in ``/v1`` (vLLM; ``/chat/completions`` is added).
* ``BGGEN_<ROLE>_API_KEY``, else ``NEBIUS_API_KEY`` (may be empty for a local vLLM).
* ``BGGEN_<ROLE>_MODEL`` overrides the model name.

* ``BGGEN_<ROLE>_MAX_TOKENS`` overrides the output limit (default 32768 per role, the harness baseline: Kimi-K3
  spent a 4096 budget on reasoning and returned an empty or cut ``content``).

Every call is counted per role (calls, prompt / completion / reasoning tokens from ``usage``, seconds,
failures, replies cut by the limit, empty replies, re-queries). A reply cut by the limit (``finish_reason``
``length``) or with an empty ``content`` is asked once more with ``retry_max_tokens``; if that also fails the
reply is returned with ``incomplete`` set (the caller logs it and falls back). Every attempt's
``finish_reason`` and reasoning size are returned in ``attempts`` (the generator writes them to the trace).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_MODELS = {"init": "moonshotai/Kimi-K3", "agent": "moonshotai/Kimi-K3", "vdm": "Qwen/Qwen3.8-27B"}
RETRY_STATUS = (404, 408, 429, 500, 502, 503, 504, 529)


@dataclass
class RoleConfig:
    role: str
    model: str
    url: str = ""
    api_key: str = ""
    max_tokens: int = 32768
    retry_max_tokens: int = 65536  # one re-query with this limit after a cut / empty reply (<= max_tokens: none)
    temperature: float = 0.0
    timeout_s: float = 300.0
    retries: int = 4

    @classmethod
    def from_env(cls, role: str, model: str | None = None, env: dict[str, str] | None = None, **kw: Any) -> RoleConfig:
        env = dict(os.environ if env is None else env)
        R = role.upper()
        url = env.get(f"BGGEN_{R}_URL") or env.get("NEBIUS_SERVER_URL", "")
        key = env.get(f"BGGEN_{R}_API_KEY") or env.get("NEBIUS_API_KEY", "")
        model = env.get(f"BGGEN_{R}_MODEL") or model or DEFAULT_MODELS.get(role, "")
        kw = dict(kw)
        if env.get(f"BGGEN_{R}_MAX_TOKENS"):
            kw["max_tokens"] = int(env[f"BGGEN_{R}_MAX_TOKENS"])
            kw["retry_max_tokens"] = max(2 * kw["max_tokens"], kw.get("retry_max_tokens", 0))
        return cls(role, model, chat_url(url), key, **kw)


def chat_url(url: str) -> str:
    url = (url or "").rstrip("/")
    if url and not url.endswith("/chat/completions"):
        url = url + "/chat/completions"
    return url


@dataclass
class Usage:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0  # as the server reports (real APIs include image tokens; the fake server text only)
    images: int = 0  # images sent (counted here, independent of the server)
    completion_tokens: int = 0
    reasoning_tokens: int = 0  # part of completion_tokens, when the server reports it
    seconds: float = 0.0
    cut: int = 0  # replies with finish_reason "length"
    empty: int = 0  # replies with an empty content
    requeries: int = 0  # re-queries with retry_max_tokens
    incomplete: int = 0  # still cut / empty after the re-query (the caller falls back)

    def add(self, usage: dict[str, Any] | None, seconds: float, ok: bool = True, images: int = 0) -> None:
        self.calls += 1
        self.images += images
        self.failures += 0 if ok else 1
        self.seconds += seconds
        if usage:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.reasoning_tokens += reasoning_tokens(usage) or 0

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "seconds": round(self.seconds, 1)}


def reasoning_tokens(usage: dict[str, Any] | None) -> int | None:
    """Reasoning tokens from ``usage`` (OpenAI ``completion_tokens_details.reasoning_tokens`` or a flat
    ``reasoning_tokens``); None when the server does not report them."""
    if not usage:
        return None
    det = usage.get("completion_tokens_details") or {}
    v = det.get("reasoning_tokens") if isinstance(det, dict) else None
    if v is None:
        v = usage.get("reasoning_tokens")
    return None if v is None else int(v)


def count_images(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if isinstance(m.get("content"), list)
               for p in m["content"] if isinstance(p, dict) and p.get("type") == "image_url")


class ChatClient:
    """``query(messages) -> {"content", "usage", "finish_reason"}``; raises after the retries."""

    def __init__(self, cfg: RoleConfig, post: Callable[..., Any] | None = None) -> None:
        self.cfg = cfg
        self.usage = Usage()
        self._post = post or requests.post

    def query(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """One reply. Cut (``finish_reason`` length) or empty -> one re-query with ``retry_max_tokens``; still
        cut / empty -> returned with ``incomplete`` = "length" / "empty". ``attempts``: per attempt max_tokens,
        finish_reason, content chars, reasoning tokens / chars."""
        if not self.cfg.url:
            raise RuntimeError(f"no URL for role {self.cfg.role}: set BGGEN_{self.cfg.role.upper()}_URL or NEBIUS_SERVER_URL")
        limits = [self.cfg.max_tokens]
        if self.cfg.retry_max_tokens > self.cfg.max_tokens:
            limits.append(self.cfg.retry_max_tokens)
        attempts: list[dict[str, Any]] = []
        out: dict[str, Any] = {}
        for n, limit in enumerate(limits):
            if n:
                self.usage.requeries += 1
            out = self._once(messages, limit)
            bad = "length" if out["finish_reason"] == "length" else "empty" if not out["content"].strip() else None
            self.usage.cut += bad == "length"
            self.usage.empty += bad == "empty"
            attempts.append({"max_tokens": limit, "finish_reason": out["finish_reason"], "content_chars": len(out["content"]),
                             "reasoning_tokens": reasoning_tokens(out["usage"]), "reasoning_chars": out.pop("reasoning_chars"),
                             "completion_tokens": (out["usage"] or {}).get("completion_tokens"), "incomplete": bad})
            out["incomplete"] = bad
            if bad is None:
                break
        self.usage.incomplete += out.get("incomplete") is not None
        out["attempts"] = attempts
        out["reasoning_tokens"] = attempts[-1]["reasoning_tokens"]
        return out

    def _once(self, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        payload = {"model": self.cfg.model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": self.cfg.temperature}
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        t0 = time.time()
        last: Exception | None = None
        for attempt in range(self.cfg.retries + 1):
            try:
                r = self._post(self.cfg.url, headers=headers, data=json.dumps(payload), timeout=self.cfg.timeout_s)
                if r.status_code in RETRY_STATUS and attempt < self.cfg.retries:
                    time.sleep(min(5.0 * 2 ** attempt, 120.0))
                    continue
                r.raise_for_status()
                body = r.json()
                msg = body["choices"][0]["message"]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                out = {"content": msg.get("content") or "", "usage": body.get("usage"),
                       "finish_reason": body["choices"][0].get("finish_reason"), "model": body.get("model", self.cfg.model),
                       "reasoning_chars": len(reasoning) if isinstance(reasoning, str) else None}
                self.usage.add(out["usage"], time.time() - t0, images=count_images(messages))
                return out
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last = exc
                if attempt < self.cfg.retries:
                    time.sleep(min(5.0 * 2 ** attempt, 120.0))
                    continue
            except Exception as exc:  # HTTP error or malformed body: no retry
                last = exc
                break
        self.usage.add(None, time.time() - t0, ok=False)
        raise RuntimeError(f"{self.cfg.role} query failed: {last}")


@dataclass
class Models:
    init: ChatClient
    agent: ChatClient
    vdm: ChatClient
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, models: dict[str, str] | None = None, env: dict[str, str] | None = None,
                 max_tokens: dict[str, int] | None = None, **kw: Any) -> Models:
        """``max_tokens``: per-role output limits (config, e.g. ``GenConfig.max_tokens``); the re-query limit is
        twice that (at least the default). ``BGGEN_<ROLE>_MAX_TOKENS`` wins over both."""
        models, max_tokens = models or {}, max_tokens or {}

        def cfg(r: str) -> RoleConfig:
            extra = dict(kw)
            if r in max_tokens:
                extra.update(max_tokens=max_tokens[r], retry_max_tokens=max(2 * max_tokens[r], RoleConfig.retry_max_tokens))
            return RoleConfig.from_env(r, models.get(r), env, **extra)

        return cls(*(ChatClient(cfg(r)) for r in ("init", "agent", "vdm")))

    def usage(self) -> dict[str, Any]:
        return {r: {"model": c.cfg.model, **c.usage.to_dict()} for r, c in
                (("init", self.init), ("agent", self.agent), ("vdm", self.vdm))}
