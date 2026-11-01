"""LLM client utilities for querying language models.

Extracted from capx/utils/launch_utils.py to separate LLM query logic
from launch/config utilities.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import json
import os
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from capx.envs.launch import LaunchArgs

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

GPT_MODELS = [
    "openai/gpt-5.4",
    "openai/o4-mini",
]
VLM_MODELS = [
    "google/gemini-3.1-pro-preview",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-opus-4-5",
    "anthropic/claude-haiku-4-5",
    "openai/gpt-5.4",
    "openai/o1",
    "openai/o4-mini",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "moonshotai/kimi-k2",
    "moonshotai/Kimi-K3",
    "Qwen/Qwen3.8-27B",
]
CLAUDE_MODELS = ["anthropic/claude-opus-4-5", "anthropic/claude-haiku-4-5"]
OSS_MODELS = [
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-r1-0528",
    "deepseek/deepseek-r1",
    "qwen/qwen3.5-122b-a10b",
    "moonshotai/kimi-k2",
    "moonshotai/Kimi-K3",
    "Qwen/Qwen3.8-27B",
]
OPENROUTER_MODELS = [
    "openrouter/google/gemini-2.5-pro-preview",
    "openrouter/google/gemini-2.5-flash-preview",
    "openrouter/anthropic/claude-sonnet-4",
    "openrouter/anthropic/claude-opus-4",
    "openrouter/deepseek/deepseek-r1",
    "openrouter/deepseek/deepseek-chat-v3-0324",
    "openrouter/openai/gpt-4.1",
    "openrouter/openai/o4-mini",
    "openrouter/meta-llama/llama-4-maverick",
    "openrouter/qwen/qwen3-235b-a22b",
]
OPENROUTER_SERVER_URL = "http://localhost:8110/chat/completions"

# HTTP timeout for one model request (seconds). Reasoning models can take minutes per response.
MODEL_REQUEST_TIMEOUT_S = 600
# Extra attempts when a model request times out or the connection drops.
MODEL_REQUEST_RETRIES = 2

# ---------------------------------------------------------------------------
# Ensemble configuration
# ---------------------------------------------------------------------------

ENSEMBLE_CONFIGS = [
    # Gemini-3-Pro only — best single model per CaP-Bench (Figure 1).
    # 3 temps for diversity; synthesis still uses Gemini-3-Pro.
    # ~45% faster than full multimodel (no Claude/GPT latency bottleneck).
    ("openai/gpt-5.4", [0.1, 0.5, 0.9]),
]

# Models that take text only (Nebius catalogue). query_model refuses a prompt with images for these instead
# of letting the server drop or reject them; RoboMME's conditioning video then goes in as VDM text
# (condition_video_mode vdm_text / auto, see capx/envs/trial.py).
TEXT_ONLY_MODELS = [
    "openai/gpt-oss-120b",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "moonshotai/Kimi-K2.7-Code",
    "deepseek-ai/DeepSeek-V4-Pro",
    # Candidate replacements (IDs as in env_configs/pricing/nebius.yaml; check them in the catalogue).
    # Qwen/Qwen3.5-397B-A17B is not listed: it takes images (add it here if the Nebius endpoint is text-only).
    "zai-org/GLM-5.3",
    "nvidia/Nemotron-3-Ultra-550b",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
]

# Upstream synthesis model of query_model_ensemble (used when no synthesis model is configured).
DEFAULT_SYNTHESIS_MODEL = "openai/gpt-5.4"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TextOnlyModelError(ValueError):
    """A prompt with images was about to go to a model that only takes text."""


def is_text_only_model(model: str) -> bool:
    return model in TEXT_ONLY_MODELS


def count_prompt_images(prompt: list[dict]) -> int:
    """Number of image (or video) items in a chat prompt."""
    n = 0
    for message in prompt:
        content = message.get("content")
        if isinstance(content, list):
            n += sum(1 for c in content if isinstance(c, dict) and c.get("type") in ("image_url", "input_image"))
    return n


def check_prompt_images(model: str, prompt: list[dict]) -> None:
    """Raise TextOnlyModelError if ``prompt`` has images and ``model`` is text-only (never drop them silently)."""
    if is_text_only_model(model):
        n = count_prompt_images(prompt)
        if n:
            raise TextOnlyModelError(
                f"{model} is text-only but the prompt has {n} image(s). Describe them in text first "
                "(RoboMME conditioning video: condition_video_mode vdm_text or auto; no use_visual_feedback)."
            )


def parse_ensemble_candidates(spec: Any) -> list[tuple[str, list[float]]] | None:
    """Ensemble candidates as ``[(model, [temperatures])]``.

    Accepts None, a list of ``{"model": ..., "temperatures": [...]}`` dicts or ``[model, [temps]]`` pairs
    (YAML ``ensemble.candidates``), or a CLI string ``"model@0.1,0.5,0.9;model2@0.5"``.
    """
    if spec is None:
        return None
    items: list[tuple[str, list[float]]] = []
    if isinstance(spec, str):
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            if "@" not in part:
                raise ValueError(f"ensemble candidate {part!r}: expected model@t1,t2,...")
            model, temps = part.rsplit("@", 1)
            items.append((model.strip(), [float(t) for t in temps.split(",") if t.strip()]))
    else:
        for entry in spec:
            if isinstance(entry, dict):
                model, temps = entry["model"], entry.get("temperatures", entry.get("temps"))
            else:
                model, temps = entry
            if isinstance(temps, (int, float)):
                temps = [temps]
            items.append((str(model), [float(t) for t in temps]))
    if not items or any(not temps for _, temps in items):
        raise ValueError(f"ensemble candidates need at least one model with at least one temperature: {spec!r}")
    return items


def usage_tokens(usage: Any) -> dict[str, int]:
    """Input / output / reasoning tokens of an OpenAI-style usage dict (zeros when missing).

    ``completion`` already includes ``reasoning`` (OpenAI and Nebius count reasoning as completion tokens).
    """
    if not isinstance(usage, dict):
        return {"prompt": 0, "completion": 0, "reasoning": 0}
    details = usage.get("completion_tokens_details") or {}
    reasoning = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if reasoning is None:
        reasoning = usage.get("reasoning_tokens")
    return {
        "prompt": int(usage.get("prompt_tokens") or 0),
        "completion": int(usage.get("completion_tokens") or 0),
        "reasoning": int(reasoning or 0),
    }


def synthesis_format_issue(content: Any, is_multiturn: bool) -> str | None:
    """Why a synthesis reply breaks the synthesis prompt's OUTPUT FORMAT, or None if it follows it.

    Initial code: exactly one closed fenced code block. Multi-turn: a REGENERATE line followed by exactly one
    closed fenced code block, or a FINISH line (not both).
    """
    if not isinstance(content, str) or not content.strip():
        return "empty"
    n_open = content.count("```")
    n_blocks = content.count("```python")
    lines = {ln.strip().strip("*#").strip() for ln in content.splitlines()}
    if is_multiturn:
        regen, finish = "REGENERATE" in lines, "FINISH" in lines
        if regen and finish:
            return "both_keywords"
        if finish:
            return None if n_open == 0 else "finish_with_code"
        if not regen:
            return "no_keyword"
    if n_blocks == 0:
        return "no_fence"
    if n_open % 2:
        return "unclosed_fence"
    if n_blocks > 1 or n_open > 2:
        return "multiple_fences"
    return None


def is_openrouter_model(model: str) -> bool:
    """Return True if the model should be routed through the OpenRouter proxy."""
    return model.startswith("openrouter/") or model in OPENROUTER_MODELS


@dataclass
class ModelQueryArgs:
    """Arguments for querying a model."""

    model: str
    server_url: str
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 4096
    reasoning_effort: str = "medium"
    debug: bool = False


def collapse_text_image_inputs(messages: list[dict]) -> list[dict]:
    """
    Collapse a list of messages with sequential text into a single text input, images are still in the same relative position
    """
    new_prompt = []
    current_text_input = ""
    for message in messages:
        if message["type"] == "text":
            current_text_input += message["text"] + "\n"
        else:
            if current_text_input != "":
                new_prompt.append({"type": "text", "text": current_text_input})
                current_text_input = ""
            new_prompt.append(message)
    if current_text_input != "":
        new_prompt.append({"type": "text", "text": current_text_input})
    return new_prompt


def _completions_to_responses_convert_prompt(prompt: list[dict]) -> list[dict]:
    """Convert completions api format to responses api format.

    Args:
        prompt: The prompt in completions api format

    Returns:
        The prompt in responses api format

    Switch prompt structure to api responses api format e.g.:
    From
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image in detail."},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    To
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Describe the image in detail."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}"
                }
            ]
        }
    ]
    """

    for message in prompt:
        for content in message["content"]:
            if type(content) == str:
                continue
            if content.get("type") == "text":
                content["type"] = "input_text"
                content["text"] = content.pop("text")

            elif content.get("type") == "image_url":
                content["type"] = "input_image"
                content["image_url"] = content["image_url"]["url"]
    return prompt


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------


def _post_chat(server_url: str, headers: dict[str, str], payload: dict) -> requests.Response:
    """POST a model request until it gets a usable response.

    Timeouts and dropped connections are retried MODEL_REQUEST_RETRIES times, then re-raised.
    404/5xx responses keep being retried with a long backoff.
    """
    data = json.dumps(payload)
    retry = 1
    failures = 0
    while True:
        try:
            response = requests.post(
                server_url, headers=headers, data=data, timeout=MODEL_REQUEST_TIMEOUT_S
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
            if failures >= MODEL_REQUEST_RETRIES:
                raise
            failures += 1
            sleep_time = 10 * failures
            print(
                f"Model request failed ({exc!r}); "
                f"retry {failures}/{MODEL_REQUEST_RETRIES} in {sleep_time} s"
            )
            time.sleep(sleep_time)
            continue
        if response.status_code not in [404, 500, 502, 503, 504]:
            return response
        sleep_time = 240 + random.uniform(-90, 90)
        print(f"Retry {retry}. Model query failed with status code {response.status_code}. Error: {response.text}. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
        retry += 1


def query_model(args: "LaunchArgs | ModelQueryArgs", prompt: list[dict]) -> str:
    """Query vLLM server for code generation.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
    Returns:
        Model response content
    """

    check_prompt_images(args.model, prompt)

    # Route OpenRouter models to the OpenRouter proxy server
    if is_openrouter_model(args.model):
        server_url = OPENROUTER_SERVER_URL
    else:
        server_url = args.server_url

    if args.model in GPT_MODELS:
        if "codex" in args.model:
            prompt = _completions_to_responses_convert_prompt(prompt)
            payload = {
                "model": args.model,
                "input": prompt,
            }
        else:
            payload = {
                "model": args.model,
                "reasoning_effort": args.reasoning_effort,
                "max_completion_tokens": args.max_tokens,  # Total completion tokens = reasoning + output tokens
                "messages": prompt,
            }
    elif is_openrouter_model(args.model):
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
        }
    elif args.model in OSS_MODELS:
        payload = {
            "model": args.model,
            "messages": prompt,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
        }
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"
    start_time = time.time()

    response = _post_chat(server_url, headers, payload)

    end_time = time.time()
    print(f"Time taken to query model: {end_time - start_time:.2f} seconds")
    response.raise_for_status()
    body = response.json()
    out = {}
    if args.debug:
        print(json.dumps(body, indent=2))
    try:
        if args.model in GPT_MODELS and "codex" in args.model:
            out["content"] = body["output_text"]
        else:
            out["content"] = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response format: {body}") from exc
    if body.get("choices") is not None:
        out["reasoning"] = body.get("choices")[0].get("message").get("reasoning", None)
        # "length" means the reply hit max_tokens (reasoning tokens count against it on some APIs).
        out["finish_reason"] = body.get("choices")[0].get("finish_reason")
    else:
        out["reasoning"] = None
        out["finish_reason"] = body.get("status")
    out["usage"] = body.get("usage")
    return out  # type: ignore[return-value]


def query_model_streaming(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
) -> Iterable[dict]:
    """Query model with streaming enabled, yielding partial responses.

    Yields dictionaries with:
      - {"type": "content_delta", "content": "partial text"}
      - {"type": "reasoning_delta", "content": "partial reasoning"} (if supported)
      - {"type": "done", "content": "full content", "reasoning": "full reasoning or None"}

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation

    Yields:
        Partial response chunks as they arrive
    """
    check_prompt_images(args.model, prompt)

    if args.model in GPT_MODELS:
        payload = {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
            "max_completion_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }
    elif args.model in CLAUDE_MODELS:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": prompt,
            "stream": True,
        }
    else:
        payload = {
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "messages": prompt,
            "stream": True,
        }

    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    elif os.getenv("OPENAI_API_KEY") is not None and args.model in GPT_MODELS:
        headers["Authorization"] = f"Bearer {os.getenv('OPENAI_API_KEY')}"

    full_content = ""
    full_reasoning = ""

    start_time = time.time()

    with requests.post(
        args.server_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=MODEL_REQUEST_TIMEOUT_S,
        stream=True,
    ) as response:
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        is_sse = "text/event-stream" in content_type
        is_json = "application/json" in content_type

        # If it's a regular JSON response (server doesn't support streaming),
        # fall back to non-streaming behavior
        if is_json and not is_sse:
            print("Warning: Server returned JSON instead of SSE stream, falling back to non-streaming")
            body = response.json()
            try:
                full_content = body["choices"][0]["message"]["content"]
                full_reasoning = body.get("choices", [{}])[0].get("message", {}).get("reasoning")
                if full_reasoning:
                    print(f"Reasoning extracted ({len(full_reasoning)} chars)")
                else:
                    print("No reasoning returned by model")
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Unexpected response format: {body}") from exc

            yield {"type": "content_delta", "content": full_content}
            yield {
                "type": "done",
                "content": full_content,
                "reasoning": full_reasoning if full_reasoning else None,
            }
            end_time = time.time()
            print(f"Time taken to query model (streaming fallback): {end_time - start_time:.2f} seconds")
            return

        for line in response.iter_lines():
            if not line:
                continue

            line_str = line.decode("utf-8")

            # SSE format: "data: {...}" or "data: [DONE]"
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # Remove "data: " prefix

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # Handle content delta
                    content_delta = delta.get("content", "")
                    if content_delta:
                        full_content += content_delta
                        yield {"type": "content_delta", "content": content_delta}

                    # Handle reasoning delta (some APIs support this)
                    reasoning_delta = delta.get("reasoning", "")
                    if reasoning_delta:
                        full_reasoning += reasoning_delta
                        yield {"type": "reasoning_delta", "content": reasoning_delta}

                except json.JSONDecodeError:
                    continue
            else:
                # Try parsing as raw JSON (non-SSE format)
                try:
                    data = json.loads(line_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content_delta = delta.get("content", "")
                        if content_delta:
                            full_content += content_delta
                            yield {"type": "content_delta", "content": content_delta}
                except json.JSONDecodeError:
                    continue

    end_time = time.time()
    print(f"Time taken to query model (streaming): {end_time - start_time:.2f} seconds")
    if full_reasoning:
        print(f"Reasoning extracted ({len(full_reasoning)} chars)")
    else:
        print("No reasoning returned by model")

    yield {
        "type": "done",
        "content": full_content,
        "reasoning": full_reasoning if full_reasoning else None,
    }


def _timed_query(query_args: ModelQueryArgs, prompt: list[dict]) -> tuple[dict[str, Any], float]:
    start = time.time()
    result = query_model(query_args, prompt)
    return result, time.time() - start


def _call_log(model: str, temp: float, ok: bool, latency_s: float, result: dict | None, error: str | None) -> dict:
    """One model call for the trial log (all_responses.json): no reply text, which is in the ensemble_*.txt files."""
    usage = result.get("usage") if result else None
    return {
        "model": model,
        "temp": temp,
        "ok": ok,
        "latency_s": round(latency_s, 2),
        "finish_reason": result.get("finish_reason") if result else None,
        "usage": usage,
        "tokens": usage_tokens(usage),
        **({"error": error[:500]} if error else {}),
    }


def query_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    synthesis_model: str | None = None,
    is_multiturn = False,
    candidates: list[tuple[str, list[float]]] | None = None,
) -> dict[str, Any]:
    """Query the candidate models (each at its temperatures) in parallel and synthesize one final output.

    Args:
        candidates: ``[(model, [temperatures])]``; None uses the upstream ENSEMBLE_CONFIGS.
        synthesis_model: None uses the upstream DEFAULT_SYNTHESIS_MODEL.

    With neither set, the requests are exactly the upstream ones. Failed candidates are left out of the
    synthesis (it fails only if all of them fail). When either is given, the result also carries the
    synthesis reply's ``finish_reason`` and ``usage`` (so the trial's cut-off handling applies to it);
    ``ensemble_log`` always holds per-call usage, latency and failures for the trial log.
    """
    configured = candidates is not None or synthesis_model is not None
    if synthesis_model is None:
        synthesis_model = DEFAULT_SYNTHESIS_MODEL
    if candidates is None:
        candidates = ENSEMBLE_CONFIGS
    # A text-only candidate would fail on an image prompt and drop out quietly: refuse up front instead.
    for model, _ in candidates:
        check_prompt_images(model, prompt)
    ensemble_start = time.time()

    def query_single(model: str, temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        start = time.time()
        try:
            result, latency = _timed_query(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True,
                    "log": _call_log(model, temp, True, latency, result, None)}
        except Exception as e:
            error_msg = str(e)
            print(f"[Multimodel Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False,
                    "log": _call_log(model, temp, False, time.time() - start, None, error_msg)}

    # Build all (model, temp) pairs and query in parallel
    tasks = [(m, t) for m, temps in candidates for t in temps]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(9, len(tasks))) as executor:
        futures = {executor.submit(query_single, m, t): (m, t) for m, t in tasks}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Multimodel Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")
    candidates_s = time.time() - ensemble_start

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        exc = RuntimeError("All ensemble queries failed")
        exc.ensemble_log = _ensemble_log(  # type: ignore[attr-defined]
            "multimodel", responses, synthesis_model, None, 0.0, is_multiturn,
            candidates_s=candidates_s, wall_s=time.time() - ensemble_start,
        )
        raise exc

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate ({r['model']}, temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    synth_args = ModelQueryArgs(
        model=synthesis_model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final, synthesis_s = _synthesize(
        synth_args, synthesis_prompt, "multimodel", responses,
        is_multiturn, candidates_s, ensemble_start,
    )

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {synthesis_model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    out = {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
        "ensemble_log": _ensemble_log(
            "multimodel", responses, synthesis_model, final, synthesis_s, is_multiturn,
            candidates_s=candidates_s, wall_s=time.time() - ensemble_start,
        ),
    }
    if configured:
        out["finish_reason"] = final.get("finish_reason")
        out["usage"] = final.get("usage")
    return out


def _synthesize(
    synth_args: ModelQueryArgs,
    synthesis_prompt: list[dict],
    kind: str,
    responses: list[dict],
    is_multiturn: bool,
    candidates_s: float,
    ensemble_start: float,
) -> tuple[dict[str, Any], float]:
    """Run the synthesis query. If it raises, the exception carries ``ensemble_log`` (the finished candidates'
    usage and the failed synthesis) so the caller can still log and cost them; the exception type is kept."""
    start = time.time()
    try:
        return _timed_query(synth_args, synthesis_prompt)
    except Exception as exc:
        log = _ensemble_log(
            kind, responses, synth_args.model, None, time.time() - start, is_multiturn,
            candidates_s=candidates_s, wall_s=time.time() - ensemble_start, synthesis_error=str(exc),
        )
        with contextlib.suppress(AttributeError):  # exceptions with __slots__
            exc.ensemble_log = log  # type: ignore[attr-defined]
        raise


def _ensemble_log(
    kind: str,
    responses: list[dict],
    synthesis_model: str,
    final: dict | None,
    synthesis_s: float,
    is_multiturn: bool,
    *,
    candidates_s: float,
    wall_s: float,
    synthesis_error: str | None = None,
) -> dict[str, Any]:
    """Per-call record of one ensemble query for all_responses.json (``ensemble_calls``).

    ``final`` None: the synthesis failed (``synthesis_error``) or was never sent (all candidates failed;
    ``synthesis`` is then None).
    """
    if final is not None:
        synth = _call_log(synthesis_model, 0.2, True, synthesis_s, final, None)
        synth["format_issue"] = synthesis_format_issue(final.get("content"), is_multiturn)
    elif synthesis_error is not None:
        synth = _call_log(synthesis_model, 0.2, False, synthesis_s, None, synthesis_error)
        synth["format_issue"] = "failed"
    else:
        synth = None
    return {
        "kind": kind,
        "is_multiturn": bool(is_multiturn),
        "n_candidates": len(responses),
        "n_failed": sum(1 for r in responses if not r["ok"]),
        "candidates": [r["log"] for r in responses],
        "synthesis": synth,
        "candidates_s": round(candidates_s, 2),
        "wall_s": round(wall_s, 2),
    }


def query_single_model_ensemble(
    args: "LaunchArgs | ModelQueryArgs",
    prompt: list[dict],
    model: str,
    is_multiturn = False,
) -> dict[str, Any]:
    """Query the same model 9 times (with temperatures 0.1 to 0.9) and synthesize final output.

    Args:
        args: Configuration with server URL and model settings
        prompt: Full prompt containing environment observation and possibly multi-turn decision prompt
        model: The model to use for both candidate generation and synthesis

    Returns:
        Dictionary containing synthesized content, reasoning, all responses, and text artifacts
    """

    def query_single(temp: float) -> dict:
        query_args = ModelQueryArgs(
            model=model,
            server_url=args.server_url,
            api_key=args.api_key,
            temperature=temp,
            max_tokens=args.max_tokens,
            reasoning_effort=getattr(args, "reasoning_effort", "medium"),
        )
        start = time.time()
        try:
            result, latency = _timed_query(query_args, copy.deepcopy(prompt))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True,
                    "log": _call_log(model, temp, True, latency, result, None)}
        except Exception as e:
            error_msg = str(e)
            print(f"[Single Model Ensemble] {model} temp={temp} FAILED: {error_msg}")
            return {"model": model, "temp": temp, "content": error_msg, "ok": False,
                    "log": _call_log(model, temp, False, time.time() - start, None, error_msg)}

    # Query same model with 9 different temperatures (0.1 to 0.9)
    temperatures = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    check_prompt_images(model, prompt)
    ensemble_start = time.time()
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=9) as executor:
        futures = {executor.submit(query_single, t): t for t in temperatures}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp['ok']:
                print(f"[Single Model Ensemble] {resp['model']} temp={resp['temp']} ok={resp['ok']}")
    candidates_s = time.time() - ensemble_start

    successful = [r for r in responses if r["ok"]]
    if not successful:
        # Print all errors for debugging
        print("\n=== All single model ensemble queries failed. Errors: ===")
        for r in responses:
            print(f"  {r['model']} temp={r['temp']}: {r['content']}")
        exc = RuntimeError("All single model ensemble queries failed")
        exc.ensemble_log = _ensemble_log(  # type: ignore[attr-defined]
            "single_model", responses, model, None, 0.0, is_multiturn,
            candidates_s=candidates_s, wall_s=time.time() - ensemble_start,
        )
        raise exc

    # Build synthesis prompt
    original_text = ""
    for msg in prompt:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c

    candidates = "\n\n".join(
        f"--- Candidate (temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )

    # Detect if this is a multiturn decision (candidates contain REGENERATE/FINISH)
    regenerate_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "REGENERATE" in r["content"])
    finish_count = sum(1 for r in successful if isinstance(r.get("content"), str) and "FINISH" in r["content"])

    if is_multiturn:
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate responses for a multi-turn robot control task.

    DECISION ANALYSIS:
    - {regenerate_count} candidates voted REGENERATE
    - {finish_count} candidates voted FINISH

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach
    5. Combine best code ideas from REGENERATE candidates

    OUTPUT FORMAT (strict):
    - You may include brief reasoning first
    - Then output "REGENERATE" on its own line followed by exactly ONE fenced code block, OR output "FINISH" on its own line
    """
    else: # first generation has no REGEN/FINISH candidates
        synthesis_system_prompt = f"""You are synthesizing {len(successful)} candidate Python solutions into one optimal program.

    SYNTHESIS RULES:
    1. Analyze critically and assume no candidate is fully correct
    2. Prefer explicit checks over assumptions
    3. Combine the best ideas from multiple candidates when appropriate
    4. If candidates disagree fundamentally, choose the more robust approach

    OUTPUT FORMAT (strict):
    You may include reasoning before the fenced code block.
    Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
    Do NOT include any other code blocks or code snippets outside this single block.
    """

    synthesis_user_prompt = f"""Synthesize the best solution.

    <original_task_description>
    {original_text}
    </original_task_description>

    <candidate_solutions>
    {candidates}
    </candidate_solutions>
    """

    synthesis_prompt = [
        {
            "role": "system",
            "content": synthesis_system_prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": synthesis_user_prompt,
                }
            ],
        },
    ]

    # Use the same model for synthesis
    synth_args = ModelQueryArgs(
        model=model,
        server_url=args.server_url,
        api_key=args.api_key,
        temperature=0.2,
        max_tokens=args.max_tokens,
    )
    final, synthesis_s = _synthesize(
        synth_args, synthesis_prompt, "single_model", responses,
        is_multiturn, candidates_s, ensemble_start,
    )

    # Build text content for saving
    candidates_txt = "\n\n".join(
        f"{'='*60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n{'='*60}\n{r['content']}"
        for r in responses
    )
    synthesis_txt = f"Model: {model}\n\n"
    synthesis_txt += f"{'='*60}\nREASONING\n{'='*60}\n{final.get('reasoning') or '(none)'}\n\n"
    synthesis_txt += f"{'='*60}\nOUTPUT\n{'='*60}\n{final['content']}"

    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": responses,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
        "ensemble_log": _ensemble_log(
            "single_model", responses, model, final, synthesis_s, is_multiturn,
            candidates_s=candidates_s, wall_s=time.time() - ensemble_start,
        ),
    }


# ---------------------------------------------------------------------------
# Backward-compatible aliases (underscore-prefixed names)
# ---------------------------------------------------------------------------

_query_model = query_model
_query_model_streaming = query_model_streaming
_query_model_ensemble = query_model_ensemble
_query_single_model_ensemble = query_single_model_ensemble
