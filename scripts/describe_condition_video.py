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
    """Also run the single-call description of the condition_frames sampled frames (the first version)."""
    save_frames: bool = True
    out_root: str = "outputs/condition_video_desc"


def _tokens(usage) -> tuple[int, int]:
    if not isinstance(usage, dict):
        return 0, 0
    return int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)


def main(a: Args) -> int:
    from PIL import Image

    from capx.envs.configs.instantiate import instantiate
    from capx.envs.tasks.robomme.robomme_env import (
        CONDITION_VIDEO_SEGMENT_DEFAULTS,
        condition_video_vdm_prompt,
        describe_condition_video_segmented,
    )
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
            goal = low.handle.task_language
            if not frames:
                print(f"{task}: no conditioning video")
                continue
            start = time.time()
            description, log = describe_condition_video_segmented(query_model, vdm_args, goal, frames, **seg)
            record = {"task": task, "episode": a.episode, "goal": goal, "total_frames": len(frames),
                      "vdm_model": a.vdm_model, "settings": seg, "wall_s": round(time.time() - start, 2),
                      "description": description, **log}
            if a.also_single:
                sampled = low.conditioning_frames()
                t0 = time.time()
                out = query_model(vdm_args, condition_video_vdm_prompt(goal, sampled, len(frames)))
                record["single"] = {"n_frames": len(sampled), "latency_s": round(time.time() - t0, 2),
                                    "usage": out.get("usage"), "finish_reason": out.get("finish_reason"),
                                    "description": out.get("content")}
            stem = f"{task}_ep{a.episode}"
            (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=2, default=str))
            md = [f"# {task} episode {a.episode}", "", f"Goal: {goal}", "",
                  f"{len(frames)} frames; settings {seg} -> stride {log['segmentation']['stride']}, "
                  f"{log['segmentation']['n_segments']} parts, {len(log['vdm_calls'])} VDM calls, "
                  f"{record['wall_s']} s", "", "## Description (as inserted into the first prompt)", "",
                  "```text", description, "```", ""]
            for p in log["parts"]:
                md += [f"## Part {p['part']} raw reply (frames {p['first'] + 1}-{p['last'] + 1})", "",
                       "```text", p["reply"], "```", ""]
            if a.also_single:
                md += ["## Single-call description (32 sampled frames)", "", "```text",
                       str(record["single"]["description"]), "```", ""]
            (out_dir / f"{stem}.md").write_text("\n".join(md))
            if a.save_frames:
                fdir = out_dir / f"{stem}_frames"
                fdir.mkdir(exist_ok=True)
                for i in log["frame_indices"]:
                    Image.fromarray(frames[i]).save(fdir / f"frame_{i + 1:04d}.png")
            usages = [c["usage"] for c in log["vdm_calls"]] + (
                [record["single"]["usage"]] if a.also_single else [])
            p_tok = sum(_tokens(u)[0] for u in usages)
            c_tok = sum(_tokens(u)[1] for u in usages)
            cost = cost_by_model({a.vdm_model: {"prompt": p_tok, "completion": c_tok}}, prices)["total"]
            summary.append(f"{task}: {len(log['vdm_calls'])} calls, {record['wall_s']} s, "
                           f"tokens {p_tok:,} in / {c_tok:,} out, ${cost:.3f} -> {out_dir / (stem + '.md')}")
            print(summary[-1])
        finally:
            try:
                low.handle.env.close()
            except Exception:
                pass
    (out_dir / "SUMMARY.txt").write_text("\n".join(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
