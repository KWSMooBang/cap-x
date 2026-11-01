"""Command line of the BG generator (the kubectl runbook ``scripts/bggen/kubectl_runbook.sh`` calls these).

    python -m capx.bggen.cli download-libero --dest D
    python -m capx.bggen.cli dump-libero --h5 FILE --out structure.json
    python -m capx.bggen.cli prep-libero --h5 FILE --demo demo_0 --root EP          (LIBERO-Mem venv, renderer)
    python -m capx.bggen.cli download-b1k --dest D [--wrist]
    python -m capx.bggen.cli dump-b1k --src D --out structure.json
    python -m capx.bggen.cli prep-b1k --src D --root EP [--wrist] [--image-stride 6]
    python -m capx.bggen.cli check-episode --root EP                                 (episode files + a marked frame)
    python -m capx.bggen.cli run --root EP --out OUT --benchmark libero_mem|b1k --mode A|B \\
        [--fake-models] [--detector sam3|format_fake]
    python -m capx.bggen.cli eval --root EP --out OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _print(d: object) -> None:
    print(json.dumps(d, indent=1, default=str))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="capx.bggen.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    req, flag, opt = {"required": True}, {"action": "store_true"}, {}
    ep = ("--episode", {"type": int, "default": 450010})
    specs: dict[str, list[tuple[str, dict]]] = {
        "download-libero": [("--dest", req)],
        "dump-libero": [("--h5", req), ("--out", opt)],
        "prep-libero": [("--h5", req), ("--demo", {"default": "demo_0"}), ("--root", req), ("--task-id", {"type": int, "default": 6}),
                        ("--image-stride", {"type": int, "default": 2}), ("--wrist", flag), ("--max-frames", {"type": int})],
        "download-b1k": [("--dest", req), ("--wrist", flag), ep],
        "dump-b1k": [("--src", req), ("--out", opt), ep],
        "prep-b1k": [("--src", req), ("--root", req), ("--wrist", flag), ("--image-stride", {"type": int, "default": 6}), ep,
                     ("--max-frames", {"type": int})],
        "check-episode": [("--root", req)],
        "run": [("--root", req), ("--out", req), ("--benchmark", {"choices": ("libero_mem", "b1k"), "required": True}),
                ("--mode", {"default": "A", "choices": ("A", "B")}),
                ("--fake-models", {"action": "store_true", "help": "scripted local server instead of models (format smoke)"}),
                ("--detector", {"default": "sam3", "choices": ("sam3", "format_fake")}), ("--init-model", opt),
                ("--agent-model", opt), ("--vdm-model", opt), ("--no-touch-rule", flag), ("--keyframe-s", {"type": float})],
        "eval": [("--root", req), ("--out", req)],
    }
    for name, args in specs.items():
        s = sub.add_parser(name)
        for flag_name, kw in args:
            s.add_argument(flag_name, **kw)
    a = ap.parse_args(argv)

    if a.cmd == "download-libero":
        from .adapters import libero_mem
        print(libero_mem.download(a.dest))
    elif a.cmd == "dump-libero":
        from .adapters import libero_mem
        d = libero_mem.dump_structure(a.h5)
        if a.out:
            Path(a.out).write_text(json.dumps(d, indent=1))
        _print(d)
    elif a.cmd == "prep-libero":
        from .adapters import libero_mem
        _print(libero_mem.prepare(a.h5, a.root, a.demo, a.task_id, a.image_stride, wrist=a.wrist, max_frames=a.max_frames))
    elif a.cmd == "download-b1k":
        from .adapters import b1k
        cams = ("head", "left_wrist", "right_wrist") if a.wrist else ("head",)
        print(b1k.download(a.dest, episode=a.episode, cams=cams))
    elif a.cmd == "dump-b1k":
        from .adapters import b1k
        d = b1k.dump_structure(a.src, episode=a.episode)
        if a.out:
            Path(a.out).write_text(json.dumps(d, indent=1, default=str))
        _print(d)
    elif a.cmd == "prep-b1k":
        from .adapters import b1k
        cams = ("head", "left_wrist", "right_wrist") if a.wrist else ("head",)
        d = b1k.prepare(a.src, a.root, episode=a.episode, cams=cams, image_stride=a.image_stride, max_frames=a.max_frames)
        _print({"checks": d["checks"], "task_info": {k: d["task_info"][k] for k in ("assumed_len", "actual_len", "layout_ok")},
                "gripper_width": d["gripper_width"], "base_path_m": d["base_path_m"], "cameras": d["cameras"]})
    elif a.cmd == "check-episode":
        _print(check_episode(a.root))
    elif a.cmd == "run":
        return run(a)
    elif a.cmd == "eval":
        from .evaluate import evaluate
        res = evaluate(a.out, a.root)
        print((Path(a.out) / "eval.md").read_text())
        return 0 if res else 1
    return 0


def check_episode(root: str) -> dict:
    """Episode sanity: counts, gripper / base ranges, event candidates, depth range, camera height."""
    import numpy as np

    from .configs import preset
    from .events import detect_events
    from .records import load_episode

    ep = load_episode(root)
    out: dict = {"instruction": ep.instruction, "frames": ep.n_frames, "fps": ep.fps, "arms": ep.arms, "cameras": {}}
    for arm in ep.arms:
        g = ep.gripper(arm)
        out[f"gripper_{arm}"] = {"min": round(float(g.min()), 4), "max": round(float(g.max()), 4)}
    bench = ep.meta.get("benchmark", "libero_mem")
    cfg = preset(bench if bench in ("libero_mem", "b1k") else "libero_mem")
    ev = detect_events(ep, cfg.events)
    out["events"] = [e.label() for e in ev if e.kind != "keyframe"]
    for c in ep.cameras:
        cam = ep.camera(c, 0)
        info = {"images": int(len(ep.camera_frames(c))), "shape": list(cam.rgb.shape) if cam else None}
        if cam is not None and cam.depth is not None:
            d = cam.depth[np.isfinite(cam.depth) & (cam.depth > 0)]
            info["depth_m"] = {"p5": round(float(np.percentile(d, 5)), 3), "p50": round(float(np.median(d)), 3),
                               "p95": round(float(np.percentile(d, 95)), 3)}
        if cam is not None and cam.T_base_cam is not None:
            info["cam_pos_base"] = cam.T_base_cam[:3, 3].round(3).tolist()
            info["optical_axis_base"] = cam.T_base_cam[:3, 2].round(3).tolist()
        out["cameras"][c] = info
    return out


def run(a: argparse.Namespace) -> int:
    from dataclasses import replace

    from .configs import preset
    from .models import Models
    from .perception import FormatFakeDetector, Sam3Detector
    from .pipeline import run_episode

    cfg = preset(a.benchmark, a.mode)
    if a.no_touch_rule:
        cfg = replace(cfg, verifier=replace(cfg.verifier, touch_rule=False))
    if a.keyframe_s is not None:
        cfg = replace(cfg, events=replace(cfg.events, keyframe_s=a.keyframe_s))
    det = FormatFakeDetector() if a.detector == "format_fake" else Sam3Detector()
    names = {k: v for k, v in (("init", a.init_model), ("agent", a.agent_model), ("vdm", a.vdm_model)) if v}
    if a.fake_models:
        from .fake_models import FakeBrain, FakeServer

        with FakeServer(FakeBrain(bad_every=5)) as srv:
            models = Models.from_env(env=srv.env())
            s = run_episode(a.root, a.out, cfg, models, det, a.benchmark)
    else:
        models = Models.from_env(names)
        s = run_episode(a.root, a.out, cfg, models, det, a.benchmark)
    _print({k: s.get(k) for k in ("events", "rows", "agent_fallbacks", "models", "backward")} | {"goal": s["goal"]["done"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
