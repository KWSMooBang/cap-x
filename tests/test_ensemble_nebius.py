"""Configurable multimodel ensemble, text-only models, conditioning video as VDM text, usage / cost logs.

Ensemble requests go to a fake chat-completions HTTP server (no API key or network needed).
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

import capx.envs.trial as trial_mod
import capx.llm.client as client
from capx.envs.launch import LaunchArgs
from capx.llm.client import (
    ModelQueryArgs,
    TextOnlyModelError,
    parse_ensemble_candidates,
    query_model,
    query_model_ensemble,
    synthesis_format_issue,
)
from capx.utils.launch_utils import _load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_libero_mem_pilot as analyze_mem  # noqa: E402
import analyze_robomme_pilot as analyze_rm  # noqa: E402
import model_costs  # noqa: E402
import run_ensemble_pilot as ens_pilot  # noqa: E402
from test_robomme_api import make_code_env, patched_models  # noqa: E402, F401

ENS_DIR = REPO_ROOT / "env_configs" / "capbg_ensemble"
NEBIUS = [("openai/gpt-oss-120b", [0.1, 0.5, 0.9]), ("Qwen/Qwen3-235B-A22B-Instruct-2507", [0.1, 0.5, 0.9]),
          ("deepseek-ai/DeepSeek-V4-Flash-0731", [0.1, 0.5, 0.9])]
SYNTH = "deepseek-ai/DeepSeek-V4-Pro"
# Conditioning-video describer of the video tasks (models.json condition_video; lead, 2026-11-01).
CV_MODEL = "moonshotai/Kimi-K3"
CV_TASKS = ["VideoUnmaskSwap", "VideoPlaceOrder", "MoveCube", "PatternLock"]
CODE = "```python\nopen_gripper()\n```"
IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


# --------------------------------------------------------------------------------------------------
# Fake chat-completions server
# --------------------------------------------------------------------------------------------------

class FakeServer:
    """Records every request body; replies by model. ``fail`` models get HTTP 400 (not retried)."""

    def __init__(self, fail: set[str] = frozenset(), reply=None) -> None:
        self.requests: list[dict[str, Any]] = []
        self.fail = set(fail)
        self.reply = reply or (lambda body: CODE)
        lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with lock:
                    outer.requests.append(body)
                # Candidates finish in temperature order, so the synthesis prompt's candidate order is fixed.
                time.sleep(0.02 * float(body.get("temperature", 0) or 0) * 10)
                if body["model"] in outer.fail:
                    data, code = b'{"error": "bad request"}', 400
                else:
                    msg = {"role": "assistant", "content": outer.reply(body), "reasoning": "thinking"}
                    data = json.dumps({
                        "choices": [{"message": msg, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1000, "completion_tokens": 200, "total_tokens": 1200,
                                  "completion_tokens_details": {"reasoning_tokens": 150}},
                    }).encode()
                    code = 200
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/chat/completions"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()

    def synthesis(self, model: str) -> dict[str, Any]:
        (req,) = [r for r in self.requests if r["model"] == model and r["messages"][0]["role"] == "system"]
        return req

    def candidates(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["messages"][0]["role"] != "system"]


@pytest.fixture
def server():
    s = FakeServer()
    yield s
    s.close()


def _args(url: str) -> ModelQueryArgs:
    return ModelQueryArgs(model="unused", server_url=url, api_key="k", max_tokens=777, reasoning_effort="low")


PROMPT = [{"role": "user", "content": [{"type": "text", "text": "Goal: pick the cube."}]}]


# --------------------------------------------------------------------------------------------------
# Ensemble calls
# --------------------------------------------------------------------------------------------------

def test_default_ensemble_requests_unchanged(server):
    """No candidates / synthesis given: upstream ENSEMBLE_CONFIGS (gpt-5.4 x 0.1/0.5/0.9) and gpt-5.4 synthesis."""
    out = query_model_ensemble(_args(server.url), PROMPT, is_multiturn=False)
    cands = server.candidates()
    assert len(cands) == 3
    for body in cands:  # GPT payload: no temperature, reasoning_effort + max_completion_tokens
        assert body == {"model": "openai/gpt-5.4", "reasoning_effort": "low", "max_completion_tokens": 777,
                        "messages": PROMPT}
    synth = server.synthesis("openai/gpt-5.4")
    assert set(synth) == {"model", "reasoning_effort", "max_completion_tokens", "messages"}
    assert synth["reasoning_effort"] == "medium"  # upstream: synthesis ModelQueryArgs default
    sys_prompt = synth["messages"][0]["content"]
    assert sys_prompt.startswith("You are synthesizing 3 candidate Python solutions into one optimal program.")
    user = synth["messages"][1]["content"][0]["text"]
    assert user.startswith("Synthesize the best solution.\n\n    <original_task_description>\n    Goal: pick the cube.")
    assert "--- Candidate (openai/gpt-5.4, temp=0.1) ---\n" + CODE in user
    # Upstream result keys are unchanged; no finish_reason / usage at top level (trial behaviour unchanged).
    assert {"content", "reasoning", "all_responses", "ensemble_candidates_txt", "ensemble_synthesis_txt"} <= set(out)
    assert "finish_reason" not in out and "usage" not in out
    assert out["ensemble_log"]["n_candidates"] == 3 and out["ensemble_log"]["synthesis"]["model"] == "openai/gpt-5.4"


def test_configured_candidates_and_synthesis(server):
    server.fail = {"deepseek-ai/DeepSeek-V4-Flash-0731"}
    out = query_model_ensemble(_args(server.url), PROMPT, synthesis_model=SYNTH, candidates=NEBIUS)
    cands = server.candidates()
    assert sorted((b["model"], b["temperature"]) for b in cands) == sorted((m, t) for m, ts in NEBIUS for t in ts)
    for b in cands:  # non-GPT payload
        assert set(b) == {"model", "temperature", "max_tokens", "messages"} and b["max_tokens"] == 777
    synth = server.synthesis(SYNTH)
    assert synth["temperature"] == 0.2
    assert synth["messages"][0]["content"].startswith("You are synthesizing 6 candidate Python solutions")
    text = synth["messages"][1]["content"][0]["text"]
    assert "DeepSeek-V4-Flash-0731" not in text and text.count("--- Candidate (") == 6  # failed ones left out
    log = out["ensemble_log"]
    assert (log["n_candidates"], log["n_failed"]) == (9, 3)
    ok = [c for c in log["candidates"] if c["ok"]]
    assert all(c["tokens"] == {"prompt": 1000, "completion": 200, "reasoning": 150} for c in ok)
    assert all(c["latency_s"] >= 0 for c in log["candidates"])
    assert all("error" in c and c["usage"] is None for c in log["candidates"] if not c["ok"])
    assert log["synthesis"]["model"] == SYNTH and log["synthesis"]["format_issue"] is None
    assert log["synthesis"]["tokens"]["reasoning"] == 150
    assert out["finish_reason"] == "stop" and out["usage"]["completion_tokens"] == 200
    assert "Success: False" in out["ensemble_candidates_txt"]


def test_all_candidates_fail_raises(server):
    server.fail = {m for m, _ in NEBIUS}
    with pytest.raises(RuntimeError, match="All ensemble queries failed"):
        query_model_ensemble(_args(server.url), PROMPT, synthesis_model=SYNTH, candidates=NEBIUS)
    assert not [r for r in server.requests if r["model"] == SYNTH]


def test_text_only_models_never_get_images(server):
    prompt = [{"role": "user", "content": [{"type": "text", "text": "Goal"}, IMAGE]}]
    with pytest.raises(TextOnlyModelError):
        query_model_ensemble(_args(server.url), prompt, synthesis_model=SYNTH, candidates=NEBIUS)
    with pytest.raises(TextOnlyModelError):
        query_model(ModelQueryArgs(model=SYNTH, server_url=server.url), prompt)
    assert server.requests == []  # refused before any request
    query_model(ModelQueryArgs(model="Qwen/Qwen3.8-27B", server_url=server.url), prompt)  # VLM: fine
    assert len(server.requests) == 1


@pytest.mark.parametrize("content, multiturn, issue", [
    (CODE, False, None),
    ("reasoning\n" + CODE, False, None),
    ("x = 1", False, "no_fence"),
    (CODE + "\n" + CODE, False, "multiple_fences"),
    ("```python\nx = 1\n", False, "unclosed_fence"),
    ("", False, "empty"),
    ("REGENERATE\n" + CODE, True, None),
    ("FINISH", True, None),
    ("**FINISH**", True, None),
    ("Looks done. FINISH", True, "no_keyword"),
    ("REGENERATE\nFINISH", True, "both_keywords"),
    ("REGENERATE\nno code", True, "no_fence"),
    ("FINISH\n" + CODE, True, "finish_with_code"),
    ("REGENERATE\n" + CODE + CODE, True, "multiple_fences"),
])
def test_synthesis_format_issue(content, multiturn, issue):
    assert synthesis_format_issue(content, multiturn) == issue


def test_parse_ensemble_candidates():
    assert parse_ensemble_candidates(None) is None
    assert parse_ensemble_candidates("a/b@0.1,0.5;c@0.9") == [("a/b", [0.1, 0.5]), ("c", [0.9])]
    assert parse_ensemble_candidates([{"model": "a", "temperatures": [0.1]}, ["b", 0.5]]) == [("a", [0.1]), ("b", [0.5])]
    with pytest.raises(ValueError):
        parse_ensemble_candidates("a")
    with pytest.raises(ValueError):
        parse_ensemble_candidates([{"model": "a", "temperatures": []}])


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------

ENS_CONFIGS = sorted(ENS_DIR.rglob("*.yaml"))


def test_generated_ensemble_configs_are_current():
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "gen_capbg_ensemble_configs.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    names = {p.relative_to(ENS_DIR).as_posix() for p in ENS_CONFIGS}
    assert len([n for n in names if n.startswith("libero_mem/")]) == 5
    assert len([n for n in names if n.startswith("robomme/")]) == 8
    assert {n.split("/")[1][:7] for n in names if n.startswith("libero_mem/")} == {
        "task_00", "task_04", "task_06", "task_07", "task_08"}


@pytest.mark.parametrize("path", ENS_CONFIGS, ids=lambda p: p.relative_to(ENS_DIR).as_posix())
def test_ensemble_config_matches_baseline_except_ensemble(path):
    if path.parent.name == "libero_mem":
        base = REPO_ROOT / "env_configs" / "capbg_baseline" / "libero_mem" / path.name
    else:
        base = REPO_ROOT / "env_configs" / "robomme" / path.name
    ens, ref = yaml.safe_load(path.read_text()), yaml.safe_load(base.read_text())
    changed = {"use_parallel_ensemble", "use_multimodel", "ensemble", "condition_video_mode", "output_dir",
               "condition_video_stride", "condition_video_segment_frames", "condition_video_overlap",
               "condition_video_max_segments", "condition_video_model"}
    assert {k: v for k, v in ens.items() if k not in changed} == {k: v for k, v in ref.items() if k not in changed}
    assert ens["output_dir"].startswith("./outputs/capbg_ensemble/")
    video_task = path.name in {f"robomme_{t.lower()}.yaml" for t in CV_TASKS}
    assert ens.get("condition_video_model") == (CV_MODEL if video_task else None)
    if path.parent.name == "robomme":
        assert ens["condition_video_segment_frames"] == (0 if video_task else 20)

    _, config, _ = _load_config(LaunchArgs(config_path=str(path)))
    assert config["use_parallel_ensemble"] is True and config["use_multimodel"] is True
    assert config["ensemble_candidates"] == [[m, t] for m, t in NEBIUS]
    assert config["ensemble_synthesis_model"] == SYNTH
    assert config["condition_video_mode"] == ("auto" if path.parent.name == "robomme" else "images")
    assert config["memory_mode"] == "none"
    assert config["condition_video_model"] == (CV_MODEL if video_task else None)
    if path.parent.name == "robomme":
        assert trial_mod._condition_video_mode(SimpleNamespace(model=SYNTH), config) == "vdm_text"


def test_baseline_config_has_no_ensemble_settings():
    path = REPO_ROOT / "env_configs" / "robomme" / "robomme_videounmaskswap.yaml"
    _, config, _ = _load_config(LaunchArgs(config_path=str(path)))
    assert (config["ensemble_candidates"], config["ensemble_synthesis_model"], config["condition_video_mode"]) == \
        (None, None, "images")
    assert trial_mod._condition_video_mode(SimpleNamespace(model="moonshotai/Kimi-K3"), config) == "images"


def test_cli_overrides_and_validation(tmp_path):
    path = ENS_DIR / "robomme" / "robomme_movecube.yaml"
    args = LaunchArgs(config_path=str(path), ensemble_candidates="x/y@0.3", ensemble_synthesis_model="z",
                      condition_video_mode="images")
    _, config, _ = _load_config(args)
    assert config["ensemble_candidates"] == [["x/y", [0.3]]] and config["ensemble_synthesis_model"] == "z"
    assert config["condition_video_mode"] == "images"
    with pytest.raises(ValueError, match="use_multimodel"):
        _load_config(LaunchArgs(config_path=str(path), use_multimodel=False))
    with pytest.raises(ValueError, match="condition_video_mode"):
        _load_config(LaunchArgs(config_path=str(path), condition_video_mode="video"))


# --------------------------------------------------------------------------------------------------
# Trial: conditioning video as VDM text, ensemble logs
# --------------------------------------------------------------------------------------------------

VDM_TEXT = "1. t = 0.20 s: the gripper lifts the red cup off the blue cube and puts it back."


def _trial_config(tmp_path: Path, **kw) -> dict[str, Any]:
    return {
        "output_dir": str(tmp_path), "use_parallel_ensemble": True, "use_multimodel": True,
        "use_visual_feedback": False, "use_img_differencing": True, "use_video_differencing": False,
        "use_wrist_camera": False, "record_video": True, "use_oracle_code": False, "save_multiturn_prompts": True,
        "ensemble_candidates": [[m, t] for m, t in NEBIUS], "ensemble_synthesis_model": SYNTH,
        "condition_video_mode": "auto", **kw,
    }


TRIAL_ARGS = SimpleNamespace(
    model=SYNTH, server_url="", api_key=None, max_tokens=100, temperature=1.0, reasoning_effort="low",
    debug=False, use_legacy_multi_turn_decision_prompt=False, visual_differencing_model="Qwen/Qwen3.8-27B",
    visual_differencing_model_server_url="", visual_differencing_model_api_key=None,
)


def _usage(p=100, c=20, r=5):
    return {"prompt_tokens": p, "completion_tokens": c, "completion_tokens_details": {"reasoning_tokens": r}}


def _run_trial(monkeypatch, tmp_path, config, n_demo=12, wrap=None):
    calls: list[tuple[str, Any]] = []

    def fake(args, prompt):
        client.check_prompt_images(args.model, prompt)  # the real guard (query_model is replaced here)
        calls.append((args.model, prompt))
        system = prompt[0]["content"] if prompt[0]["role"] == "system" else ""
        if "demonstration video" in system:
            text = VDM_TEXT
        elif args.model == "Qwen/Qwen3.8-27B":
            text = "A table with cups."
        elif "multi-turn" in system:
            text = "FINISH"
        elif args.model == SYNTH and system:
            text = CODE
        else:
            text = "FINISH" if "console stdout" in str(prompt).lower() or "Executed:" in str(prompt) else CODE
        return {"content": text, "reasoning": None, "finish_reason": "stop", "usage": _usage()}

    if wrap is not None:
        fake = wrap(fake)
    monkeypatch.setattr(trial_mod, "_query_model", fake)
    monkeypatch.setattr(client, "query_model", fake)
    monkeypatch.setattr(trial_mod, "_write_video", lambda frames, base_dir, suffix="": None)
    code_env = make_code_env(monkeypatch, task_id="VideoUnmaskSwap", n_demo=n_demo)
    trial_mod._run_single_trial(code_env, 1, TRIAL_ARGS, config,
                                "Executed:\n{executed_code}\n{console_stdout}\n{console_stderr}")
    trial_dir = next(tmp_path.glob("trial_01_*"))
    return calls, json.loads((trial_dir / "all_responses.json").read_text())


def _has_images(prompt) -> bool:
    return client.count_prompt_images(prompt) > 0


def test_trial_vdm_text_single_call(monkeypatch, patched_models, tmp_path):  # noqa: F811
    """condition_video_segment_frames 0: the first version, one VDM call with the sampled frames."""
    calls, responses = _run_trial(monkeypatch, tmp_path, _trial_config(tmp_path, condition_video_segment_frames=0))
    cv_calls = [p for m, p in calls if "demonstration video" in str(p[0]["content"])]
    assert len(cv_calls) == 1 and client.count_prompt_images(cv_calls[0]) == 12  # one VDM call with all frames
    coding = [(m, p) for m, p in calls if m != "Qwen/Qwen3.8-27B"]
    assert coding and not any(_has_images(p) for _, p in coding)
    first_cands = [p for m, p in coding if m == "openai/gpt-oss-120b"][:1]
    text = str(first_cands[0])
    assert "<conditioning_video_description>" in text and VDM_TEXT in text and "Video frame" not in text
    # The synthesis prompt carries the description too (upstream: text of the original prompt).
    synth = [p for m, p in coding if m == SYNTH][0]
    assert VDM_TEXT in synth[1]["content"][0]["text"]
    initial = responses[0]
    cv = initial["condition_video"]
    assert cv["mode"] == "vdm_text" and cv["config_mode"] == "auto" and cv["description"] == VDM_TEXT
    assert cv["n_frames"] == 12 and cv["vdm_calls"][0]["usage"]["prompt_tokens"] == 100
    assert not _has_images(initial["initial_prompt"])
    assert len(initial["vdm_usage"]) == 2  # conditioning video + initial scene description
    (ens,) = initial["ensemble_calls"]
    assert ens["n_candidates"] == 9 and ens["n_failed"] == 0 and ens["synthesis"]["model"] == SYNTH
    later = [r for r in responses[1:] if r.get("decision") in ("finish", "regenerate")]
    assert later and all(len(r["ensemble_calls"]) == 1 and r["ensemble_calls"][0]["is_multiturn"] for r in later)


def test_trial_images_mode_with_text_only_models_fails_loudly(monkeypatch, patched_models, tmp_path):  # noqa: F811
    with pytest.raises(TextOnlyModelError):
        _run_trial(monkeypatch, tmp_path, _trial_config(tmp_path, condition_video_mode="images"))


def test_trial_auto_keeps_images_for_vision_models(monkeypatch, patched_models, tmp_path):  # noqa: F811
    config = _trial_config(tmp_path, ensemble_candidates=[["Qwen/Qwen3.8-27B", [0.1]]],
                           ensemble_synthesis_model="moonshotai/Kimi-K3")
    calls, responses = _run_trial(monkeypatch, tmp_path, config)
    assert responses[0]["condition_video"]["mode"] == "images"
    assert "description" not in responses[0]["condition_video"]
    first_coding = [p for m, p in calls if m == "Qwen/Qwen3.8-27B" and "Goal" in str(p) and "Video frame" in str(p)]
    assert first_coding and client.count_prompt_images(first_coding[0]) == 12


def test_baseline_trial_logs_unchanged(monkeypatch, patched_models, tmp_path):  # noqa: F811
    """Ensemble off, no condition_video_mode key: no ensemble_calls key, images in the first prompt."""
    config = _trial_config(tmp_path, use_parallel_ensemble=False, use_multimodel=False)
    for k in ("ensemble_candidates", "ensemble_synthesis_model", "condition_video_mode"):
        config.pop(k)
    args = SimpleNamespace(**{**vars(TRIAL_ARGS), "model": "moonshotai/Kimi-K3"})
    monkeypatch.setattr(sys.modules[__name__], "TRIAL_ARGS", args)
    calls, responses = _run_trial(monkeypatch, tmp_path, config)
    assert "ensemble_calls" not in responses[0]
    assert responses[0]["condition_video"]["mode"] == "images"
    assert client.count_prompt_images(responses[0]["initial_prompt"]) == 12


# --------------------------------------------------------------------------------------------------
# Usage and cost
# --------------------------------------------------------------------------------------------------

def _ens_call(failed: int = 0, issue: str | None = None, multiturn: bool = False) -> dict[str, Any]:
    cands = []
    for i, (m, temps) in enumerate(NEBIUS):
        for t in temps:
            ok = not (i == 2 and failed and len([c for c in cands if not c["ok"]]) < failed)
            cands.append({"model": m, "temp": t, "ok": ok, "latency_s": 2.0,
                          "usage": _usage(1_000_000, 100_000, 50_000) if ok else None})
    return {"kind": "multimodel", "is_multiturn": multiturn, "n_candidates": 9, "n_failed": failed,
            "candidates": cands, "candidates_s": 3.0, "wall_s": 5.0,
            "synthesis": {"model": SYNTH, "ok": True, "latency_s": 2.0, "usage": _usage(2_000_000, 200_000, 0),
                          "format_issue": issue}}


def _ensemble_responses() -> list[dict[str, Any]]:
    return [
        {"decision": "initial", "usage": _usage(2_000_000, 200_000), "ensemble_calls": [_ens_call()],
         "vdm_usage": [_usage(1_000_000, 100_000)]},
        {"decision": "regenerate", "usage": _usage(2_000_000, 200_000),
         "ensemble_calls": [_ens_call(failed=3, issue="no_keyword", multiturn=True), _ens_call(multiturn=True)],
         "vdm_usage": [_usage(1_000_000, 100_000)]},
    ]


def test_trial_usage_and_cost():
    prices = model_costs.load_prices()
    assert prices["deepseek-ai/DeepSeek-V4-Pro"] == (1.75, 3.5) and prices["deepseek-ai/DeepSeek-V4-Flash-0731"] == (0.14, 0.28)
    u = model_costs.trial_usage(_ensemble_responses(), "unused", "Qwen/Qwen3.8-27B")
    bm = u["by_model"]
    assert bm["openai/gpt-oss-120b"]["calls"] == 9 and bm["openai/gpt-oss-120b"]["prompt"] == 9_000_000
    assert bm["deepseek-ai/DeepSeek-V4-Flash-0731"]["calls"] == 9 and bm["deepseek-ai/DeepSeek-V4-Flash-0731"]["failed"] == 3
    assert bm["deepseek-ai/DeepSeek-V4-Flash-0731"]["prompt"] == 6_000_000
    assert bm[SYNTH]["calls"] == 3 and bm[SYNTH]["prompt"] == 6_000_000  # entry "usage" not counted twice
    assert bm["Qwen/Qwen3.8-27B"]["calls"] == 2 and "unused" not in bm
    assert bm["openai/gpt-oss-120b"]["reasoning"] == 9 * 50_000
    ens = u["ensemble"]
    assert (ens["ensemble_queries"], ens["candidate_calls"], ens["failed_candidates"]) == (3, 27, 3)
    assert (ens["synthesis_calls"], ens["synthesis_format_errors"], ens["format_issues"]) == (3, 1, {"no_keyword": 1})
    cost = model_costs.cost_by_model(bm, prices)
    assert cost["openai/gpt-oss-120b"] == pytest.approx(9 * (0.15 + 0.1 * 0.6))
    assert cost["Qwen/Qwen3-235B-A22B-Instruct-2507"] == pytest.approx(9 * (0.2 + 0.1 * 0.6))
    assert cost["deepseek-ai/DeepSeek-V4-Flash-0731"] == pytest.approx(6 * (0.14 + 0.1 * 0.28))
    assert cost[SYNTH] == pytest.approx(3 * (2 * 1.75 + 0.2 * 3.5))
    assert cost["Qwen/Qwen3.8-27B"] == pytest.approx(2 * (0.45 + 0.1 * 3))
    assert cost["total"] == pytest.approx(sum(v for k, v in cost.items() if k not in ("total", "unpriced")))


def test_single_model_usage_and_unpriced():
    responses = [{"decision": "initial", "usage": _usage(1_000_000, 1_000_000), "vdm_usage": []},
                 {"decision": "finish", "usage": _usage(1_000_000, 0), "truncated_first_usage": _usage(1_000_000, 0),
                  "s5_first_reply": {"usage": _usage(1_000_000, 0)}}]
    u = model_costs.trial_usage(responses, "moonshotai/Kimi-K3", "Qwen/Qwen3.8-27B")
    assert u["by_model"]["moonshotai/Kimi-K3"]["prompt"] == 4_000_000
    cost = model_costs.cost_by_model(u["by_model"], model_costs.load_prices())
    assert cost["total"] == pytest.approx(4 * 3 + 15)
    cost = model_costs.cost_by_model({"x/unknown": {"prompt": 5, "completion": 1}}, {})
    assert cost["x/unknown"] is None and cost["unpriced"] == ["x/unknown"]


def _fake_trial(task_dir: Path, trial: int, end: dict, responses) -> None:
    d = task_dir / f"trial_{trial:02d}_sandboxrc_0"
    d.mkdir(parents=True)
    (d / "all_responses.json").write_text(json.dumps(responses))
    (d / "trial_end.json").write_text(json.dumps({"trial": trial, **end}))


def _run_main(module, argv):
    old = sys.argv
    try:
        sys.argv = argv
        return module.main()
    finally:
        sys.argv = old


def test_robomme_analyzer_cost_and_ensemble(tmp_path):
    run = tmp_path / "run"
    task = run / "deepseek-ai_DeepSeek-V4-Pro" / "robomme_videounmaskswap"
    _fake_trial(task, 1, {"reason": "success", "task_completed": True, "turns": 2, "sim_steps": 400, "time_s": 900},
                _ensemble_responses())
    (run / "pilot_manifest.jsonl").write_text(json.dumps({"memory_mode": "none", "trials": 1, "model": SYNTH,
                                                         "vdm_model": "Qwen/Qwen3.8-27B"}) + "\n")
    assert _run_main(analyze_rm, ["analyze", str(run)]) == 0
    lines = (run / "results.tsv").read_text().strip().split("\n")
    row = dict(zip(analyze_rm.COLUMNS, lines[1].split("\t")))
    expected = model_costs.cost_by_model(
        model_costs.trial_usage(_ensemble_responses(), SYNTH, "Qwen/Qwen3.8-27B")["by_model"],
        model_costs.load_prices())["total"]
    assert float(row["cost_usd"]) == pytest.approx(expected, abs=1e-3)
    assert (row["ens_queries"], row["ens_failed_candidates"], row["synth_format_errors"]) == ("3", "3", "1")
    assert int(row["code_prompt_tok"]) == 24_000_000 + 6_000_000  # 24 ok candidates + 3 synthesis calls
    summary = (run / "SUMMARY.md").read_text()
    assert "## Cost by model" in summary and "deepseek-ai/DeepSeek-V4-Flash-0731" in summary
    assert "failed candidates: 3 / 27" in summary and "synthesis format errors: 1 / 3" in summary


def test_libero_mem_analyzer(tmp_path):
    run = tmp_path / "run"
    task = run / "deepseek-ai_DeepSeek-V4-Pro" / "task_06_swap_bowls"
    responses = _ensemble_responses()
    responses[0]["init_source"] = "upstream"
    _fake_trial(task, 1, {"reason": "finish", "task_completed": True, "turns": 2, "sim_steps": 4000,
                          "time_s": 1200}, responses)
    (run / "pilot_manifest.jsonl").write_text(json.dumps({"bench": "libero_mem", "use_parallel_ensemble": True,
                                                         "trials": 2, "model": SYNTH}) + "\n")
    assert _run_main(analyze_mem, ["analyze", str(run)]) == 0
    rows = [dict(zip(analyze_mem.COLUMNS, ln.split("\t")))
            for ln in (run / "results.tsv").read_text().strip().split("\n")[1:]]
    assert [(r["task"], r["trial"], r["reason"]) for r in rows] == [
        ("task_06_swap_bowls", "1", "finish"), ("task_06_swap_bowls", "2", "missing")]
    assert rows[0]["cond"] == "ensemble" and rows[0]["init_source"] == "upstream" and float(rows[0]["cost_usd"]) > 0
    summary = (run / "SUMMARY.md").read_text()
    assert "| task_06_swap_bowls | 1/2 | 50% |" in summary and "## Ensemble" in summary


def test_ensemble_pilot_commands():
    args = ens_pilot.Args(bench="libero_mem", run_name="r")
    cfgs = ens_pilot.task_configs(args)
    assert [s[:7] for s in cfgs] == ["task_00", "task_04", "task_06", "task_07", "task_08"]
    cmd = ens_pilot.task_command(args, next(iter(cfgs.values())), 2, Path("/tmp/x"))
    assert cmd[cmd.index("--resume-idx") + 1] == "2" and cmd[cmd.index("--total-trials") + 1] == "5"
    assert cmd[cmd.index("--trial-timeout-s") + 1] == "3600" and cmd[cmd.index("--model") + 1] == SYNTH
    assert "capbg_ensemble/libero_mem" in cmd[cmd.index("--config-path") + 1]
    assert "--api-key" not in cmd
    rm = ens_pilot.task_configs(ens_pilot.Args(bench="robomme", run_name="r"))
    assert len(rm) == 8 and "robomme_videounmaskswap" in rm
    assert ens_pilot.ensemble_settings(rm["robomme_movecube"])["condition_video_mode"] == "auto"


# --------------------------------------------------------------------------------------------------
# Review fixes (7eef7ff)
# --------------------------------------------------------------------------------------------------

def test_streaming_refuses_images_for_text_only_models(server):
    prompt = [{"role": "user", "content": [{"type": "text", "text": "Goal"}, IMAGE]}]
    gen = client.query_model_streaming(ModelQueryArgs(model=SYNTH, server_url=server.url), prompt)
    with pytest.raises(TextOnlyModelError):
        next(gen)
    assert server.requests == []


def test_synthesis_failure_keeps_candidate_log(server):
    import requests

    server.fail = {SYNTH}
    with pytest.raises(requests.HTTPError) as info:  # exception type unchanged (timeouts stay timeouts)
        query_model_ensemble(_args(server.url), PROMPT, synthesis_model=SYNTH, candidates=NEBIUS)
    log = info.value.ensemble_log
    assert (log["n_candidates"], log["n_failed"]) == (9, 0)
    assert all(c["tokens"]["prompt"] == 1000 for c in log["candidates"])
    assert log["synthesis"]["ok"] is False and log["synthesis"]["format_issue"] == "failed"
    assert "error" in log["synthesis"]
    u = model_costs.trial_usage([{"ensemble_calls": [log]}], "x", "y")
    assert u["by_model"]["openai/gpt-oss-120b"]["prompt"] == 3000
    assert u["ensemble"]["synthesis_failed"] == 1 and u["ensemble"]["synthesis_format_errors"] == 0
    assert u["by_model"][SYNTH]["failed"] == 1


def test_all_candidates_failed_log(server):
    server.fail = {m for m, _ in NEBIUS}
    with pytest.raises(RuntimeError) as info:
        query_model_ensemble(_args(server.url), PROMPT, synthesis_model=SYNTH, candidates=NEBIUS)
    log = info.value.ensemble_log
    assert log["n_failed"] == 9 and log["synthesis"] is None


def test_failed_query_log_reaches_timeout_summary(monkeypatch, tmp_path):
    import capx.envs.runner as runner_mod
    from test_truncation import make_config

    err = RuntimeError("synthesis timed out")
    err.ensemble_log = {"kind": "multimodel", "n_candidates": 9, "n_failed": 0, "candidates": [], "synthesis": None}

    def boom(*a, **k):
        raise err

    monkeypatch.setattr(trial_mod, "_query_model_ensemble", boom)
    trial_mod._take_ensemble_calls()
    trial_mod._take_vdm_usage()
    with pytest.raises(RuntimeError):
        trial_mod._query_ensemble(SimpleNamespace(model="m"), {"use_multimodel": True,
                                  "ensemble_candidates": [["a", [0.1]]]}, PROMPT, False)
    trial_mod._VDM_USAGE.append({"prompt_tokens": 5})
    pa = {"all_responses": [{"decision": "initial"}], "trial_start_time": time.time()}
    runner_mod._build_timeout_summary(1, 10, pa, make_config(tmp_path), TimeoutError("t"))
    responses = json.loads(next(tmp_path.glob("trial_01_*/all_responses.json")).read_text())
    assert responses[-1]["decision"] == "aborted"
    assert responses[-1]["ensemble_calls"] == [err.ensemble_log]
    assert responses[-1]["vdm_usage"] == [{"prompt_tokens": 5}]
    assert trial_mod._take_ensemble_calls() == []


def test_ensemble_settings_need_parallel_ensemble():
    path = ENS_DIR / "robomme" / "robomme_movecube.yaml"
    with pytest.raises(ValueError, match="use_parallel_ensemble"):
        _load_config(LaunchArgs(config_path=str(path), use_parallel_ensemble=False))


def test_condition_video_latency_in_costs():
    responses = [{"decision": "initial", "vdm_usage": [_usage(10, 2), _usage(10, 2)],
                  "condition_video": {"mode": "vdm_text", "vdm_model": "Qwen/Qwen3.8-27B",
                                      "vdm_calls": [{"latency_s": 12.5, "usage": _usage(10, 2)}]}}]
    u = model_costs.trial_usage(responses, "x", "Qwen/Qwen3.8-27B")
    q = u["by_model"]["Qwen/Qwen3.8-27B"]
    assert (q["calls"], q["prompt"], q["latency_s"], q["timed_calls"]) == (2, 20, 12.5, 1)  # tokens not doubled
    table = "\n".join(model_costs.cost_table(u["by_model"], model_costs.load_prices(), 1))
    assert "| Qwen/Qwen3.8-27B | 2 | 0 | 20 | 4 | 10 | 12.5 |" in table


# --------------------------------------------------------------------------------------------------
# Segmented conditioning-video description
# --------------------------------------------------------------------------------------------------

from capx.envs.tasks.robomme import robomme_env as rme  # noqa: E402


def test_segment_planning():
    assert rme.strided_frame_indices(10, 3) == [0, 3, 6, 9]
    assert rme.strided_frame_indices(11, 3) == [0, 3, 6, 9, 10]  # last frame always included
    segs = rme.split_into_segments(list(range(50)), 20, 4)
    assert [(s[0], s[-1], len(s)) for s in segs] == [(0, 19, 20), (16, 35, 20), (32, 49, 18)]
    assert rme.split_into_segments(list(range(12)), 20, 4) == [list(range(12))]
    segs = rme.split_into_segments(list(range(102)), 20, 4)  # 2 frames left after part 6: merged into it
    assert [(s[0], s[-1], len(s)) for s in segs][-2:] == [(64, 83, 20), (80, 101, 22)] and len(segs) == 6
    stride, segs = rme.plan_condition_video_segments(400, 1, 20, 4, 12)
    assert (stride, len(segs)) == (2, 12)  # 400 frames: stride 1 -> 25 parts, stride 2 -> 12
    assert segs[-1][-1] == 399 and segs[0][0] == 0


@pytest.mark.parametrize("reply, events, state", [
    ("EVENTS:\n1. a\nSTATE:\n- cube under cup A", "1. a", "- cube under cup A"),
    ("**EVENTS:**\n1. a\n\n**STATE:**\nx", "1. a", "x"),
    ("## EVENTS\nnone\n## STATE\ny", "none", "y"),
    ("just text", "just text", "just text"),
])
def test_split_segment_reply(reply, events, state):
    assert rme.split_segment_reply(reply) == (events, state)


def test_describe_segmented_passes_state_between_parts():
    import numpy as np

    frames = [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(50)]
    prompts = []

    def query(args, prompt):
        prompts.append(prompt)
        k = len(prompts)
        return {"content": f"EVENTS:\n{k}. event of part {k}\nSTATE:\nstate after part {k}",
                "finish_reason": "stop", "usage": _usage()}

    desc, log = rme.describe_condition_video_segmented(
        query, SimpleNamespace(model="v"), "find the cube", frames, stride=1, segment_frames=20, overlap=4,
        max_segments=12)
    assert len(prompts) == 3 and [c["n_frames"] for c in log["vdm_calls"]] == [20, 20, 18]
    assert [client.count_prompt_images(p) for p in prompts] == [20, 20, 18]
    text = [str(p) for p in prompts]
    assert "state_so_far" not in text[0]
    assert "<state_so_far>\\nstate after part 1\\n</state_so_far>" in text[1]
    assert "state after part 2" in text[2] and "Its first 4 frames are the last frames of part 2" in text[2]
    assert "part 1 of 3" in text[0] and "frames 17-36" in text[1]
    assert "find the cube" in text[0] and "EVENTS:" in text[0] and "STATE:" in text[0]
    assert desc.index("1. event of part 1") < desc.index("2. event of part 2") < desc.index("3. event of part 3")
    assert desc.endswith("Final state (end of the video):\nstate after part 3")
    assert "state after part 1" not in desc  # only the last state goes into the prompt
    assert log["segmentation"]["segments"] == [[0, 19, 20], [16, 35, 20], [32, 49, 18]]
    assert log["n_frames"] == 50 and [p["reply"] for p in log["parts"]][0].startswith("EVENTS:")
    assert "50 of its frames (every frame, in 3 overlapping parts" in log["how"]


def test_describe_segmented_retries_empty_part():
    import numpy as np

    replies = iter(["", "EVENTS:\nnone\nSTATE:\ns"])
    desc, log = rme.describe_condition_video_segmented(
        lambda a, p: {"content": next(replies), "finish_reason": "stop", "usage": None},
        SimpleNamespace(model="v"), None, [np.zeros((4, 4, 3), np.uint8)] * 5,
        stride=1, segment_frames=20, overlap=4, max_segments=12)
    assert [c["attempt"] for c in log["vdm_calls"]] == [1, 2] and desc.endswith("s")
    with pytest.raises(RuntimeError):
        rme.describe_condition_video_segmented(
            lambda a, p: {"content": "", "finish_reason": "stop"}, SimpleNamespace(model="v"), None,
            [np.zeros((4, 4, 3), np.uint8)] * 5, stride=1, segment_frames=20, overlap=4, max_segments=12)


def test_trial_vdm_text_segmented(monkeypatch, patched_models, tmp_path):  # noqa: F811
    config = _trial_config(tmp_path, condition_video_segment_frames=20, condition_video_overlap=4)
    calls, responses = _run_trial(monkeypatch, tmp_path, config, n_demo=50)
    cv_calls = [p for m, p in calls if "demonstration video" in str(p[0]["content"])]
    assert [client.count_prompt_images(p) for p in cv_calls] == [20, 20, 18]
    assert "<state_so_far>" in str(cv_calls[1])
    cv = responses[0]["condition_video"]
    assert cv["mode"] == "vdm_text" and cv["segmentation"]["n_segments"] == 3 and len(cv["vdm_calls"]) == 3
    assert cv["n_frames"] == 50 and len(cv["parts"]) == 3 and all(p["reply"] == VDM_TEXT for p in cv["parts"])
    assert cv["description"].startswith("Part 1/3") and VDM_TEXT in cv["description"]
    assert len(responses[0]["vdm_usage"]) == 4  # 3 parts + initial scene description
    first = [p for m, p in calls if m == "openai/gpt-oss-120b"][0]
    text = str(first)
    assert "50 of its frames (every frame, in 3 overlapping parts" in text and not _has_images(first)
    assert "<conditioning_video_description>\\nPart 1/3" in text


def test_segment_settings_config():
    path = ENS_DIR / "robomme" / "robomme_pickxtimes.yaml"  # the video tasks use segment_frames 0 (Kimi-K3)
    _, config, _ = _load_config(LaunchArgs(config_path=str(path)))
    assert trial_mod._condition_video_segment_settings(config) == {
        "stride": 1, "segment_frames": 20, "overlap": 4, "max_segments": 12}
    _, config, _ = _load_config(LaunchArgs(config_path=str(path), condition_video_stride=3,
                                           condition_video_segment_frames=16))
    assert trial_mod._condition_video_segment_settings(config)["stride"] == 3
    with pytest.raises(ValueError, match="overlap"):
        _load_config(LaunchArgs(config_path=str(path), condition_video_overlap=20))
    base = REPO_ROOT / "env_configs" / "robomme" / "robomme_movecube.yaml"
    _, config, _ = _load_config(LaunchArgs(config_path=str(base)))
    assert trial_mod._condition_video_segment_settings(config) == rme.CONDITION_VIDEO_SEGMENT_DEFAULTS


# --------------------------------------------------------------------------------------------------
# Candidate replacement in the config generator
# --------------------------------------------------------------------------------------------------

import gen_capbg_ensemble_configs as gen_ens  # noqa: E402


def test_generator_candidates_argument():
    swapped = [(m if m != "deepseek-ai/DeepSeek-V4-Flash-0731" else "zai-org/GLM-5.3", t) for m, t in NEBIUS]
    files = gen_ens.build(swapped, SYNTH)
    text = files[Path("robomme") / "robomme_videounmaskswap.yaml"]
    cfg = yaml.safe_load(text)
    assert [c["model"] for c in cfg["ensemble"]["candidates"]] == [m for m, _ in swapped]
    assert "GLM-5.3" in text.split("\n")[3] and "All coding-side models are text-only" in text
    mixed = gen_ens.build([("Qwen/Qwen3.8-27B", [0.5])], SYNTH)[Path("libero_mem") / "task_06_swap_bowls.yaml"]
    assert "Text-only: DeepSeek-V4-Pro." in mixed
    assert gen_ens.load_models() == ([(m, t) for m, t in NEBIUS], SYNTH)
    # --check with a different choice reports stale files and writes nothing.
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "gen_capbg_ensemble_configs.py"), "--check",
                        "--replace", "deepseek-ai/DeepSeek-V4-Flash-0731=zai-org/GLM-5.3"], capture_output=True, text=True)
    assert r.returncode == 1 and "models.json" in r.stdout
    r = subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "gen_capbg_ensemble_configs.py"), "--check",
                        "--replace", "nope=x"], capture_output=True, text=True)
    assert r.returncode == 2
    assert gen_ens.load_models() == ([(m, t) for m, t in NEBIUS], SYNTH)


def test_new_prices():
    prices = model_costs.load_prices()
    assert prices["zai-org/GLM-5.3"] == (1.4, 4.4) and prices["nvidia/Nemotron-3-Ultra-550b-a55b"] == (1.0, 3.0)
    assert prices["Qwen/Qwen3.5-397B-A17B"] == (0.6, 3.6)
    assert prices["deepseek-ai/DeepSeek-V4-Flash-0731"] == (0.14, 0.28)


# --------------------------------------------------------------------------------------------------
# Review fixes (65cdcc4): cut-off parts, describe script methods, model IDs
# --------------------------------------------------------------------------------------------------

def _seg_query(cut_parts: set[int]):
    import numpy as np  # noqa: F401

    prompts: list = []

    def query(args, prompt):
        prompts.append(prompt)
        text = str(prompt)
        part = int(text.split("This is part ")[1].split(" of")[0])
        if part in cut_parts:
            return {"content": f"EVENTS:\n{part}. partial event\nSTATE:\n- half a sta", "finish_reason": "length",
                    "usage": _usage()}
        return {"content": f"EVENTS:\n{part}. event\nSTATE:\nstate after part {part}", "finish_reason": "stop",
                "usage": _usage()}

    return query, prompts


def _frames(n):
    import numpy as np

    return [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)]


def test_cut_off_part_is_marked_and_its_state_not_passed_on():
    query, prompts = _seg_query({2})
    desc, log = rme.describe_condition_video_segmented(
        query, SimpleNamespace(model="v"), None, _frames(50), stride=1, segment_frames=20, overlap=4, max_segments=12)
    assert len(prompts) == 4  # part 2 asked twice
    assert log["truncated_parts"] == [2] and [p["truncated"] for p in log["parts"]] == [False, True, False]
    assert "half a sta" not in str(prompts[3]) and "state after part 1" in str(prompts[3])  # part 3 gets part 1's
    assert log["parts"][2]["state_in"] == "state after part 1"
    part2 = desc.split("Part 2/3")[1].split("Part 3/3")[0]
    assert "2. partial event" in part2 and rme.TRUNCATED_MARK in part2
    assert desc.endswith("Final state (end of the video):\nstate after part 3")


def test_cut_off_last_part_falls_back_to_last_complete_state():
    query, _ = _seg_query({3})
    desc, log = rme.describe_condition_video_segmented(
        query, SimpleNamespace(model="v"), None, _frames(50), stride=1, segment_frames=20, overlap=4, max_segments=12)
    assert log["truncated_parts"] == [3] and "half a sta" not in desc.split("Final state")[1]
    assert "Final state: unknown (the description of the last part(s) was cut off). State at the end of part 2" in desc
    assert desc.endswith("state after part 2")


def test_single_call_cut_off_marked():
    calls = []

    def query(args, prompt):
        calls.append(prompt)
        return {"content": "1. t = 0.1 s: the cup", "finish_reason": "length", "usage": _usage()}

    desc, log = rme.describe_condition_video_single(query, SimpleNamespace(model="v"), None,
                                                    list(enumerate(_frames(3))), 3)
    assert len(calls) == 2 and log["truncated"] and desc.endswith(rme.TRUNCATED_MARK)


def test_trial_logs_truncated_parts(monkeypatch, patched_models, tmp_path):  # noqa: F811
    def cut_part_2(model_fn):
        def f(args, prompt):
            out = model_fn(args, prompt)
            if "demonstration video" in str(prompt[0]["content"]) and "part 2 of" in str(prompt):
                out = {**out, "finish_reason": "length"}
            return out
        return f

    config = _trial_config(tmp_path, condition_video_segment_frames=20, condition_video_overlap=4)
    _, responses = _run_trial(monkeypatch, tmp_path, config, n_demo=50, wrap=cut_part_2)
    cv = responses[0]["condition_video"]
    assert cv["truncated_parts"] == [2] and rme.TRUNCATED_MARK in cv["description"]
    assert [c["part"] for c in cv["vdm_calls"]] == [1, 2, 2, 3]


import describe_condition_video as describe_cv  # noqa: E402


def test_describe_script_segment_frames_zero_is_single_call():
    calls = []

    def query(args, prompt):
        calls.append(prompt)
        return {"content": "1. t = 0.0 s: cup", "finish_reason": "stop", "usage": _usage(100, 10)}

    frames = _frames(60)
    sampled = list(enumerate(frames))[::2]
    seg = {"stride": 1, "segment_frames": 0, "overlap": 4, "max_segments": 12}
    rec = describe_cv.describe_task(query, SimpleNamespace(model="v"), "g", frames, sampled, seg, also_single=False)
    assert len(calls) == 1 and client.count_prompt_images(calls[0]) == 30  # not split frame by frame
    assert rec["method"] == "single" and rec["description"] == "1. t = 0.0 s: cup"
    assert rec["methods"] == {"single": {"calls": 1, "wall_s": rec["methods"]["single"]["wall_s"],
                                         "prompt_tokens": 100, "completion_tokens": 10}}
    assert rec["frame_indices"] == [i for i, _ in sampled]


def test_describe_script_comparison_counted_separately(tmp_path):
    calls = []

    def query(args, prompt):
        calls.append(prompt)
        seg = "This is part" in str(prompt)
        return {"content": "EVENTS:\n1. e\nSTATE:\ns" if seg else "single text", "finish_reason": "stop",
                "usage": _usage(100, 10)}

    frames = _frames(50)
    seg = {"stride": 1, "segment_frames": 20, "overlap": 4, "max_segments": 12}
    rec = describe_cv.describe_task(query, SimpleNamespace(model="v"), "g", frames, list(enumerate(frames))[:32],
                                    seg, also_single=True)
    assert len(calls) == 4
    assert rec["method"] == "segmented" and rec["methods"]["segmented"]["calls"] == 3
    assert rec["methods"]["single (comparison)"]["calls"] == 1
    assert rec["methods"]["single (comparison)"]["prompt_tokens"] == 100
    assert rec["single"]["description"] == "single text"
    describe_cv.write_markdown(tmp_path / "x.md", {**rec, "task": "T", "episode": 0})
    md = (tmp_path / "x.md").read_text()
    assert "- segmented: 3 calls" in md and "- single (comparison): 1 calls" in md and "single text" in md


def test_text_only_ids():
    assert client.is_text_only_model("nvidia/Nemotron-3-Ultra-550b-a55b")
    assert client.is_text_only_model("Qwen/Qwen3.5-397B-A17B")
    assert not client.is_text_only_model("nvidia/Nemotron-3-Ultra-550b")


# --------------------------------------------------------------------------------------------------
# condition_video_model: another model describes the conditioning video, turn VDM unchanged
# --------------------------------------------------------------------------------------------------

def _recording(sink: list):
    """Wrap the fake model: record every request (all ModelQueryArgs fields + prompt) as JSON text."""
    import dataclasses

    def wrap(model_fn):
        def f(args, prompt):
            sink.append(json.dumps({"args": dataclasses.asdict(args), "prompt": prompt}, sort_keys=True))
            return model_fn(args, prompt)
        return f
    return wrap


def _canonical(reqs: list[str]) -> list[str]:
    """Requests as a sorted multiset; in synthesis prompts the lines are sorted too. Candidates run in parallel, so
    their request order and their order in the synthesis prompt (completion order, as upstream) vary run to run."""
    out = []
    for r in reqs:
        j = json.loads(r)
        if "synthesizing" in str(j["prompt"][0]["content"]):
            j["prompt"] = sorted(line.strip() for line in json.dumps(j["prompt"]).split("\\n"))
        out.append(json.dumps(j, sort_keys=True))
    return sorted(out)


def _is_cv_request(req: str) -> bool:
    return "demonstration video" in str(json.loads(req)["prompt"][0]["content"])


@pytest.mark.parametrize("segment_frames, n_demo", [(0, 12), (20, 50)])
def test_condition_video_model_only_moves_condition_video_calls(monkeypatch, patched_models, tmp_path,  # noqa: F811
                                                                segment_frames, n_demo):
    """Unset (or = the VDM): requests byte for byte as before. Set: only the conditioning-video requests change,
    and only in ``model``; every turn VDM / coding request is the same."""
    runs = {}
    for name, extra in [("unset", {}), ("same", {"condition_video_model": "Qwen/Qwen3.8-27B"}),
                        ("kimi", {"condition_video_model": CV_MODEL})]:
        d = tmp_path / name
        d.mkdir()
        reqs: list[str] = []
        config = _trial_config(d, condition_video_segment_frames=segment_frames, **extra)
        _, responses = _run_trial(monkeypatch, d, config, n_demo=n_demo, wrap=_recording(reqs))
        runs[name] = (reqs, responses)
    unset, same, kimi = (runs[k][0] for k in ("unset", "same", "kimi"))
    assert _canonical(same) == _canonical(unset)
    # Conditioning-video requests are sequential and come first: compare them in order.
    cv_unset, cv_kimi = [r for r in unset if _is_cv_request(r)], [r for r in kimi if _is_cv_request(r)]
    n_cv = len(cv_unset)
    assert n_cv == (1 if segment_frames == 0 else 3) and len(cv_kimi) == n_cv
    assert unset[:n_cv] == cv_unset and kimi[:n_cv] == cv_kimi
    for a, b in zip(cv_unset, cv_kimi):
        ja, jb = json.loads(a), json.loads(b)
        assert ja["args"]["model"] == "Qwen/Qwen3.8-27B" and jb["args"]["model"] == CV_MODEL
        jb["args"]["model"] = ja["args"]["model"]
        assert jb == ja  # same prompt, server, max_tokens, temperature, ...
    # Everything else (turn VDM, candidates, synthesis) byte for byte the same.
    assert _canonical(kimi[n_cv:]) == _canonical(unset[n_cv:])
    vdm_models = [json.loads(r)["args"]["model"] for r in kimi if not _is_cv_request(r)]
    assert "Qwen/Qwen3.8-27B" in vdm_models and CV_MODEL not in vdm_models  # turn VDM stays Qwen

    # Logs: unset keeps the old log (conditioning-video usage in vdm_usage); set keeps it out of vdm_usage.
    first = {k: runs[k][1][0] for k in runs}
    cv_unset, cv_kimi = first["unset"]["condition_video"], first["kimi"]["condition_video"]
    assert "usage_in_vdm_usage" not in cv_unset and cv_unset["vdm_model"] == "Qwen/Qwen3.8-27B"
    assert cv_kimi["usage_in_vdm_usage"] is False and cv_kimi["vdm_model"] == CV_MODEL
    assert len(first["unset"]["vdm_usage"]) == n_cv + 1 and len(first["kimi"]["vdm_usage"]) == 1
    assert first["same"]["vdm_usage"] == first["unset"]["vdm_usage"]

    # Costs: the conditioning-video calls at Kimi-K3 prices, the turn VDM at Qwen's; VDM-role totals unchanged.
    prices = model_costs.load_prices()
    u_unset = model_costs.trial_usage(runs["unset"][1], SYNTH, "Qwen/Qwen3.8-27B")
    u_kimi = model_costs.trial_usage(runs["kimi"][1], SYNTH, "Qwen/Qwen3.8-27B")
    k, q = u_kimi["by_model"][CV_MODEL], u_kimi["by_model"]["Qwen/Qwen3.8-27B"]
    q0 = u_unset["by_model"]["Qwen/Qwen3.8-27B"]
    assert (k["calls"], k["prompt"], k["completion"], k["timed_calls"]) == (n_cv, 100 * n_cv, 20 * n_cv, n_cv)
    assert q["calls"] == q0["calls"] - n_cv and q["prompt"] == q0["prompt"] - 100 * n_cv
    assert q0["timed_calls"] == n_cv and q["timed_calls"] == 0
    for key in ("calls", "prompt", "completion"):
        assert u_kimi["by_role"]["vdm"][key] == u_unset["by_role"]["vdm"][key]
    cost = model_costs.cost_by_model(u_kimi["by_model"], prices)
    assert prices[CV_MODEL] == (3.0, 15.0)
    assert cost[CV_MODEL] == pytest.approx(n_cv * (100 * 3.0 + 20 * 15.0) / 1e6)
    assert cost["Qwen/Qwen3.8-27B"] == pytest.approx(q["calls"] * (100 * 0.45 + 20 * 3.0) / 1e6)


def test_condition_video_model_costs_from_log():
    """A logged trial with condition_video_model: calls, tokens and latency under that model, none twice."""
    responses = [{"decision": "initial", "vdm_usage": [_usage(10, 2)],
                  "condition_video": {"mode": "vdm_text", "vdm_model": CV_MODEL, "usage_in_vdm_usage": False,
                                      "vdm_calls": [{"latency_s": 30.0, "usage": _usage(1_000_000, 100_000)}]}}]
    u = model_costs.trial_usage(responses, "x", "Qwen/Qwen3.8-27B")
    k, q = u["by_model"][CV_MODEL], u["by_model"]["Qwen/Qwen3.8-27B"]
    assert (k["calls"], k["prompt"], k["completion"], k["latency_s"]) == (1, 1_000_000, 100_000, 30.0)
    assert (q["calls"], q["prompt"], q["timed_calls"]) == (1, 10, 0)
    assert u["by_role"]["vdm"]["calls"] == 2
    cost = model_costs.cost_by_model(u["by_model"], model_costs.load_prices())
    assert cost[CV_MODEL] == pytest.approx(3.0 + 0.1 * 15.0)


def test_condition_video_model_config_and_validation():
    video = ENS_DIR / "robomme" / "robomme_videoplaceorder.yaml"
    other = ENS_DIR / "robomme" / "robomme_pickxtimes.yaml"
    _, config, _ = _load_config(LaunchArgs(config_path=str(video)))
    assert config["condition_video_model"] == CV_MODEL
    assert trial_mod._condition_video_segment_settings(config)["segment_frames"] == 0
    _, config, _ = _load_config(LaunchArgs(config_path=str(other)))
    assert config["condition_video_model"] is None
    base = REPO_ROOT / "env_configs" / "robomme" / "robomme_videoplaceorder.yaml"
    _, config, _ = _load_config(LaunchArgs(config_path=str(base)))
    assert config["condition_video_model"] is None
    _, config, _ = _load_config(LaunchArgs(config_path=str(video), condition_video_model="Qwen/Qwen3.8-27B"))
    assert config["condition_video_model"] == "Qwen/Qwen3.8-27B"  # CLI > YAML
    with pytest.raises(ValueError, match="must take images"):
        _load_config(LaunchArgs(config_path=str(video), condition_video_model="openai/gpt-oss-120b"))
    with pytest.raises(ValueError, match="must take images"):
        _load_config(LaunchArgs(config_path=str(other), condition_video_model="x/not-a-vlm"))


def test_trial_refuses_text_only_condition_video_model(monkeypatch, patched_models, tmp_path):  # noqa: F811
    with pytest.raises(ValueError, match="must take images"):
        _run_trial(monkeypatch, tmp_path, _trial_config(tmp_path, condition_video_model="openai/gpt-oss-120b"))


def test_generator_condition_video_override():
    files_none = gen_ens.build(condition_video=None)
    files = gen_ens.build()
    assert gen_ens.load_condition_video() == {"model": CV_MODEL, "tasks": CV_TASKS, "segment_frames": 0,
                                              "condition_frames": 32}
    video = {Path("robomme") / f"robomme_{t.lower()}.yaml" for t in CV_TASKS}
    assert len(files) == 13
    for rel, text in files.items():
        if rel not in video:
            assert text == files_none[rel]  # the other 9 configs: no change
            continue
        a, b = yaml.safe_load(text), yaml.safe_load(files_none[rel])
        assert a["condition_video_model"] == CV_MODEL and a["condition_video_segment_frames"] == 0
        assert a["env"]["cfg"]["low_level"]["condition_frames"] == 32
        for key in ("condition_video_model", "condition_video_segment_frames"):
            a.pop(key, None)
            b.pop(key, None)
        assert a == b
    # condition_frames other than 32 goes into the env block; segment_frames unset keeps the default.
    alt = gen_ens.build(condition_video={"model": CV_MODEL, "tasks": ["MoveCube"], "segment_frames": None,
                                         "condition_frames": 16})[Path("robomme") / "robomme_movecube.yaml"]
    assert "      condition_frames: 16\n" in alt and "condition_video_segment_frames: 20\n" in alt
    gen = [sys.executable, str(REPO_ROOT / "scripts" / "gen_capbg_ensemble_configs.py"), "--check"]
    r = subprocess.run(gen + ["--condition-video-model", "openai/gpt-oss-120b"], capture_output=True, text=True)
    assert r.returncode == 2 and "must take images" in r.stderr
    r = subprocess.run(gen + ["--condition-frames", "16"], capture_output=True, text=True)
    assert r.returncode == 2
    r = subprocess.run(gen + ["--condition-video-model", "none"], capture_output=True, text=True)
    assert r.returncode == 1 and "models.json" in r.stdout  # would change; --check writes nothing
    assert gen_ens.load_condition_video()["model"] == CV_MODEL
