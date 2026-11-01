"""Describe RoboMME conditioning videos with the VDM only (no simulation steps, no coding model).

Builds each task's RoboMME env (GPU / Vulkan host, .venv-robomme), resets it to the chosen episode, and runs exactly
the vdm_text description the trial uses (capx.envs.tasks.robomme.robomme_env.describe_condition_video_segmented,
same prompts and segmentation keys as the configs). Use it to check and tune the description before a pilot.

    export NEBIUS_SERVER_URL=... NEBIUS_API_KEY=...
    python scripts/describe_condition_video.py --run-name 20261102_cvdesc \\
        --tasks VideoUnmaskSwap VideoPlaceOrder MoveCube PatternLock --episode 0 [--also-single]

Writes outputs/condition_video_desc/<run>/<task>_ep<k>.{json,md}: the description that would go into the first
prompt, every part's raw reply, per-call latency / tokens, and (default) the frames shown to the VDM as PNGs in
<task>_ep<k>_frames/. ``--also-single`` also runs the old one-call description of the 32 sampled frames.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import tyro
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

VIDEO_TASKS = ["VideoUnmaskSwap", "VideoPlaceOrder", "MoveCube", "PatternLock"]


@dataclass
class Args:
    run_name: str
    tasks: list[str] = field(default_factory=lambda: list(VIDEO_TASKS))
    episode: int = 0
    """Test episode index (trial k of a pilot plays episode k-1)."""
    config_dir: str = "env_configs/capbg_ensemble/robomme"
    vdm_model: str = "Qwen/Qwen3.8-27B"
    server_url: str = os.environ.get("NEBIUS_SERVER_URL", "")
    api_key_env: str = "NEBIUS_API_KEY"
    stride: int | None = None
    """Default: the config's condition_video_stride."""
    segment_frames: int | None = None
    overlap: int | None = None
    max_segments: int | None = None
    max_tokens: int | None = None
    """Default: the config's max_tokens."""
    temperature: float = 1.0
    """As launch.py's default (the trial passes --temperature to the VDM)."""
    also_single: bool = False
    """Also run the single-call description of the condition_frames sampled frames (the first version) for
    comparison; reported as its own method. With --segment-frames 0 the single call is the description itself."""
    save_frames: bool = True
    out_root: str = "outputs/condition_video_desc"


def _tokens(usage) -> tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def _method_stats(calls: list[dict], wall_s: float) -> dict:
    p_tok = sum(_tokens(c.get("usage"))[0] for c in calls)
    c_tok = sum(_tokens(c.get("usage"))[1] for c in calls)
    return {"calls": len(calls), "wall_s": round(wall_s, 2), "prompt_tokens": p_tok, "completion_tokens": c_tok}


def describe_task(query, vdm_args, goal: str | None, frames: list, sampled: list, seg: dict[str, int],
                  also_single: bool) -> dict:
    """Description(s) of one conditioning video, as the trial would make them.

    ``seg["segment_frames"] > 0``: the part-by-part description (plus, with ``also_single``, the single-call one
    for comparison). ``segment_frames`` 0: the single-call description of ``sampled`` (what the trial then uses).
    ``methods`` holds calls / wall time / tokens per method, so a comparison call is never folded into another.
    """
    from capx.envs.tasks.robomme.robomme_env import (
        describe_condition_video_segmented,
        describe_condition_video_single,
    )

    record: dict = {"goal": goal, "total_frames": len(frames), "settings": seg, "methods": {}}
    if seg["segment_frames"] > 0:
        start = time.time()
        description, log = describe_condition_video_segmented(query, vdm_args, goal, frames, **seg)
        record.update(method="segmented", description=description, **log)
        record["methods"]["segmented"] = _method_stats(log["vdm_calls"], time.time() - start)
    if seg["segment_frames"] <= 0 or also_single:
        start = time.time()
        single, slog = describe_condition_video_single(query, vdm_args, goal, sampled, len(frames))
        stats = _method_stats(slog["vdm_calls"], time.time() - start)
        if seg["segment_frames"] <= 0:
            record.update(method="single", description=single, vdm_calls=slog["vdm_calls"],
                          frame_indices=slog["frame_indices"], n_frames=slog["n_frames"],
                          truncated_parts=[1] if slog["truncated"] else [], parts=[])
            record["methods"]["single"] = stats
        else:
            record["single"] = {"description": single, **slog}
            record["methods"]["single (comparison)"] = stats
    return record


def main(a: Args) -> int:
    from PIL import Image

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.tasks.robomme.robomme_env import CONDITION_VIDEO_SEGMENT_DEFAULTS
    from capx.llm.client import ModelQueryArgs, query_model
    from model_costs import cost_by_model, load_prices

    out_dir = REPO_ROOT / a.out_root / a.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = load_prices()
    summary = []
    for task in a.tasks:
        cfg_path = REPO_ROOT / a.config_dir / f"robomme_{task.lower()}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        seg = {k: int(a.__dict__[k] if a.__dict__[k] is not None else cfg.get(f"condition_video_{k}", v))
               for k, v in CONDITION_VIDEO_SEGMENT_DEFAULTS.items()}
        vdm_args = ModelQueryArgs(
            model=a.vdm_model, server_url=a.server_url, api_key=os.environ.get(a.api_key_env),
            max_tokens=int(a.max_tokens or cfg.get("max_tokens", 32768)), temperature=a.temperature,
        )
        low_cfg = dict(cfg["env"]["cfg"]["low_level"])
        low_cfg["episode_idx"] = a.episode
        low = instantiate(low_cfg)  # RoboMMEEnv: loads the episode and resets (no control steps)
        try:
            frames = list(low.handle.conditioning_frames("front"))
            if not frames:
                print(f"{task}: no conditioning video")
                continue
            record = describe_task(query_model, vdm_args, low.handle.task_language, frames,
                                   low.conditioning_frames(), seg, a.also_single)
            record.update(task=task, episode=a.episode, vdm_model=a.vdm_model, max_tokens=vdm_args.max_tokens)
            stem = f"{task}_ep{a.episode}"
            (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
            write_markdown(out_dir / f"{stem}.md", record)
            if a.save_frames:
                fdir = out_dir / f"{stem}_frames"
                fdir.mkdir(exist_ok=True)
                for i in record["frame_indices"]:
                    Image.fromarray(frames[i]).save(fdir / f"frame_{i + 1:04d}.png")
            parts = []
            for name, m in record["methods"].items():
                cost = cost_by_model({a.vdm_model: {"prompt": m["prompt_tokens"],
                                                    "completion": m["completion_tokens"]}}, prices)["total"]
                parts.append(f"{name}: {m['calls']} calls, {m['wall_s']} s, tokens {m['prompt_tokens']:,} in / "
                             f"{m['completion_tokens']:,} out, ${cost:.3f}")
            cut = record.get("truncated_parts") or []
            summary.append(f"{task}: " + "; ".join(parts) + (f"; TRUNCATED parts {cut}" if cut else "")
                           + f" -> {out_dir / (stem + '.md')}")
            print(summary[-1])
        finally:
            try:
                low.handle.env.close()
            except Exception:
                pass
    (out_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n")
    return 0


def write_markdown(path: Path, record: dict) -> None:
    seg = record["settings"]
    md = [f"# {record.get('task', '?')} episode {record.get('episode', '?')}", "", f"Goal: {record['goal']}", "",
          f"{record['total_frames']} frames; method {record['method']}; settings {seg}; VDM max_tokens "
          f"{record.get('max_tokens')}"]
    for name, m in record["methods"].items():
        md.append(f"- {name}: {m['calls']} calls, {m['wall_s']} s, {m['prompt_tokens']:,} / "
                  f"{m['completion_tokens']:,} tokens")
    if record.get("truncated_parts"):
        md.append(f"- cut off at max_tokens (both attempts): parts {record['truncated_parts']}")
    md += ["", "## Description (as inserted into the first prompt)", "", "```text", record["description"], "```", ""]
    for p in record.get("parts") or []:
        flag = " (cut off)" if p.get("truncated") else ""
        md += [f"## Part {p['part']} raw reply (frames {p['first'] + 1}-{p['last'] + 1}){flag}", "",
               "```text", p["reply"], "```", ""]
    if "single" in record:
        md += ["## Single-call description (comparison; sampled frames)", "", "```text",
               str(record["single"]["description"]), "```", ""]
    path.write_text("\n".join(md))


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
