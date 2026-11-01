"""BEHAVIOR-1K 2025 challenge demos (HF ``behavior-1k/2025-challenge-demos``, LeRobot v2.1, 30 fps, R1Pro)
-> the generator's episode directory + an oracle directory.

Sources of the layout (checked 2026-11-01; the 2025 demos were recorded with OmniGibson 3.7.0-alpha,
see ``meta/episodes/.../episode_*.json`` ``scene_file.versions``):

* ``observation.state`` (256 floats, names null in info.json): ``PROPRIOCEPTION_INDICES["R1Pro"]`` of
  BEHAVIOR-1K v3.7.2 ``OmniGibson/omnigibson/learning/utils/eval_utils.py`` (``STATE_SLICES`` below).
  eef_*_pos / eef_*_quat are in the robot base frame (quat xyzw), gripper_*_qpos two fingers 0..0.05 m.
  robot_pos / robot_2d_ori are GLOBAL (not allowed in the standard challenge track; here they are the
  robot's own odometry: the world <- base pose of a mobile robot).
* Cameras: head 720x720, wrists 480x480, intrinsics ``CAMERA_INTRINSICS["R1Pro"]`` (head fx = fy = 306,
  c = 360; wrist fx = fy = 388.6639, c = 240), constant over the episode.
* Depth videos: 14-bit log-quantized in the luma plane (yuv420p16le, read as gray16le):
  ``d = exp(q / 16383 * (log(10 + 3.5) - log(0 + 3.5)) + log(3.5)) - 3.5`` m (``dequantize_depth``,
  MIN_DEPTH 0, MAX_DEPTH 10, SHIFT 3.5 in v3.7.2; later versions use MIN_DEPTH 0.01).
* ``observation.cam_rel_poses`` (21) = 3 x [pos(3), quat xyzw(4)], camera -> robot base, in the order
  of ``ROBOT_CAMERA_NAMES["R1Pro"]`` = left_wrist, right_wrist, head; the camera frame is OpenGL-like and
  ``depth_to_pcd`` rotates it by 180 deg about x to OpenCV axes (done here).
  ASSUMPTION (order and quaternion convention): ``prepare`` checks that the head camera sits 1.0-2.0 m
  above the base and looks forward-down, and records the check in ``meta.json``.
* ``observation.task_info`` (oracle only): per task object, in ``meta/episodes`` ``task_obs_keys`` order,
  ``*_real`` (1), ``*_pos`` (3, world), ``*_ori_cos`` (3), ``*_ori_sin`` (3), ``*_in_gripper_left`` (1),
  ``*_in_gripper_right`` (1); the agent has no in_gripper fields. ASSUMPTION: sizes 1/3/3/3/1/1 -- checked
  against the vector length, the oracle positions are switched off on a mismatch.
* ``annotations/.../episode_*.json`` (oracle only): ``skill_annotation`` [{skill_description, object_id,
  frame_duration, ...}] -> subtask boundaries and an annotation-derived relation timeline (transitions at
  segment ends; initial state from the first "pick up from" source; doors closed and switches off at t=0).
* Not used by the generator: task_info, annotations, seg_instance_id videos (oracle only / ignored).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = "behavior-1k/2025-challenge-demos"
FPS = 30.0
TASK_INDEX = 45
EPISODE = 450010
CAMERAS = ("left_wrist", "right_wrist", "head")  # cam_rel_poses order (ASSUMPTION, checked)
RESOLUTION = {"head": 720, "left_wrist": 480, "right_wrist": 480}
INTRINSICS = {"head": (306.0, 360.0), "left_wrist": (388.6639, 240.0), "right_wrist": (388.6639, 240.0)}
STATE_SLICES = {
    "robot_pos": (140, 143), "robot_2d_ori": (149, 150),
    "eef_left_pos": (186, 189), "eef_left_quat": (189, 193), "gripper_left_qpos": (193, 195),
    "eef_right_pos": (225, 228), "eef_right_quat": (228, 232), "gripper_right_qpos": (232, 234),
    "base_qvel": (253, 256),
}
DEPTH_MIN, DEPTH_MAX, DEPTH_SHIFT, QMAX = 0.0, 10.0, 3.5, (1 << 14) - 1
# BDDL / scene name prefix -> generator category key
CATEGORY_OF = {"hotdog": "hotdog", "electric_refrigerator": "fridge", "fridge": "fridge", "microwave": "microwave",
               "countertop": "countertop"}


def files_for(task: int = TASK_INDEX, episode: int = EPISODE, cams: Sequence[str] = ("head",)) -> list[str]:
    ep = f"episode_{episode:08d}"
    out = [f"data/task-{task:04d}/{ep}.parquet", "meta/info.json", "meta/tasks.jsonl",
           f"meta/episodes/task-{task:04d}/{ep}.json", f"annotations/task-{task:04d}/{ep}.json"]
    for c in cams:
        out += [f"videos/task-{task:04d}/observation.images.rgb.{c}/{ep}.mp4",
                f"videos/task-{task:04d}/observation.images.depth.{c}/{ep}.mp4"]
    return out


def download(dest: str | Path, task: int = TASK_INDEX, episode: int = EPISODE, cams: Sequence[str] = ("head",)) -> Path:
    """Only this episode's files (parquet, videos of ``cams``, meta, annotation)."""
    from huggingface_hub import hf_hub_download

    dest = Path(dest)
    for f in files_for(task, episode, cams):
        hf_hub_download(REPO, f, repo_type="dataset", local_dir=dest)
    return dest


# --- pure conversions (unit-tested) -----------------------------------------------------------------

def dequantize_depth(q: np.ndarray) -> np.ndarray:
    lo, hi = np.log(DEPTH_MIN + DEPTH_SHIFT), np.log(DEPTH_MAX + DEPTH_SHIFT)
    d = np.exp(np.asarray(q, np.float64) / QMAX * (hi - lo) + lo) - DEPTH_SHIFT
    return np.clip(d, DEPTH_MIN, DEPTH_MAX).astype(np.float32)


def quantize_depth(d: np.ndarray) -> np.ndarray:
    lo, hi = np.log(DEPTH_MIN + DEPTH_SHIFT), np.log(DEPTH_MAX + DEPTH_SHIFT)
    n = (np.log(np.asarray(d, np.float64) + DEPTH_SHIFT) - lo) / (hi - lo)
    return np.clip(np.round(n * QMAX), 0, QMAX).astype(np.uint16)


def quat_xyzw_to_mat(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64) / (np.linalg.norm(q) or 1.0)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


RX_PI = np.diag([1.0, -1.0, -1.0])  # 180 deg about x: OpenGL camera axes -> OpenCV


def cam_pose(rel: Sequence[float]) -> np.ndarray:
    """[pos(3), quat xyzw(4)] camera -> base  ->  4x4 base <- camera with OpenCV axes."""
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_mat(rel[3:7]) @ RX_PI
    T[:3, 3] = rel[:3]
    return T


def intrinsics(cam: str) -> np.ndarray:
    f, c = INTRINSICS[cam]
    return np.array([[f, 0.0, c], [0.0, f, c], [0.0, 0.0, 1.0]])


def parse_state(state: np.ndarray) -> dict[str, np.ndarray]:
    """(N, 256) -> proprio arrays of the episode format (arms ``left`` / ``right``, base_pose)."""
    s = np.asarray(state, np.float64)
    g = {k: s[:, a:b] for k, (a, b) in STATE_SLICES.items()}
    out: dict[str, np.ndarray] = {}
    for arm in ("left", "right"):
        out[f"eef_pos_{arm}"] = g[f"eef_{arm}_pos"]
        q = g[f"eef_{arm}_quat"]
        out[f"eef_quat_{arm}"] = np.concatenate([q[:, 3:4], q[:, :3]], axis=1)  # xyzw -> wxyz
        out[f"gripper_{arm}"] = g[f"gripper_{arm}_qpos"].sum(axis=1)
    yaw = g["robot_2d_ori"][:, 0]
    T = np.tile(np.eye(4), (len(s), 1, 1))
    T[:, 0, 0], T[:, 0, 1], T[:, 1, 0], T[:, 1, 1] = np.cos(yaw), -np.sin(yaw), np.sin(yaw), np.cos(yaw)
    T[:, :3, 3] = g["robot_pos"]
    out["base_pose"] = T
    return out


TASK_INFO_SIZES = (("real", 1), ("pos", 3), ("ori_cos", 3), ("ori_sin", 3), ("in_gripper_left", 1), ("in_gripper_right", 1))


def task_info_layout(task_obs_keys: Sequence[str]) -> dict[str, tuple[int, int]]:
    """``task_obs_keys`` -> {key: (start, end)} with the ASSUMED field sizes."""
    sizes = dict(TASK_INFO_SIZES)
    out, i = {}, 0
    for k in task_obs_keys:
        field_ = next((f for f in sizes if k.endswith("_" + f)), None)
        n = sizes.get(field_, 1)
        out[k] = (i, i + n)
        i += n
    return out


def category_of(name: str) -> str | None:
    base = re.sub(r"(\.n\.\d+_\d+|_\w{6}_\d+|_\d+)$", "", name)
    return CATEGORY_OF.get(base)


def annotation_timeline(skills: Sequence[dict[str, Any]], n_frames: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """skill_annotation -> (segments, relation events). Relation events: [(t, (pred, a, b|None), bool)],
    approximate (a change takes effect at the end of its skill segment)."""
    segs, events = [], []
    first_src: dict[str, tuple[str, str]] = {}
    for s in skills:
        desc = (s.get("skill_description") or [""])[0]
        objs = (s.get("object_id") or [[]])[0]
        start, end = (int(x) for x in s.get("frame_duration", [0, 0]))
        segs.append({"start": start, "end": end, "label": desc, "objects": objs})
        if desc == "pick up from" and len(objs) == 2:
            first_src.setdefault(objs[0], (objs[0], objs[1]))
            events.append((end, ("in", objs[0], objs[1]), False))
            events.append((end, ("on", objs[0], objs[1]), False))
        elif desc in ("place in", "place on") and len(objs) == 2:
            events.append((end, ("in" if desc == "place in" else "on", objs[0], objs[1]), True))
        elif desc == "open door" and objs:
            events.append((end, ("open", objs[0], None), True))
        elif desc == "close door" and objs:
            events.append((end, ("open", objs[0], None), False))
        elif desc == "turn on switch" and objs:
            events.append((end, ("toggled_on", objs[0], None), True))
    init = []
    for a, b in first_src.values():
        cat = category_of(b)
        init.append((0, ("in" if cat in ("fridge", "microwave") else "on", a, b), True))
    for s in segs:
        for o in s["objects"]:
            if category_of(o) in ("fridge", "microwave"):
                init.append((0, ("open", o, None), False))
                if category_of(o) == "microwave":
                    init.append((0, ("toggled_on", o, None), False))
    return segs, {"events": sorted(init + events, key=lambda x: x[0]), "n_frames": n_frames}


def relations_at(timeline: dict[str, Any], t: int) -> set[tuple]:
    state: dict[tuple, bool] = {}
    for te, key, val in timeline["events"]:
        if te <= t:
            state[key] = val
    return {k for k, v in state.items() if v}


# --- video ---------------------------------------------------------------------------------------------

def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode_frames(path: str | Path, size: int, keep: set[int], depth: bool = False):
    """Yield (frame index, array) for frames in ``keep``: RGB uint8 (H, W, 3) or depth float32 m.
    PyAV if installed (as BEHAVIOR's loader), else an ffmpeg pipe (rgb24 / gray16le = the luma plane)."""
    try:
        import av  # type: ignore

        with av.open(str(path)) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i in keep:
                    yield i, (dequantize_depth(fr.reformat(format="gray16le").to_ndarray()) if depth
                              else fr.to_ndarray(format="rgb24"))
        return
    except ImportError:
        pass
    fmt, bpp = ("gray16le", 2) if depth else ("rgb24", 3)
    cmd = [_ffmpeg(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", fmt, "-"]
    n = size * size * bpp
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p:
        i = 0
        assert p.stdout is not None
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            if i in keep:
                if depth:
                    yield i, dequantize_depth(np.frombuffer(buf, "<u2").reshape(size, size))
                else:
                    yield i, np.frombuffer(buf, np.uint8).reshape(size, size, 3)
            i += 1


# --- prepare ---------------------------------------------------------------------------------------------

def dump_structure(src: str | Path, task: int = TASK_INDEX, episode: int = EPISODE) -> dict[str, Any]:
    import pandas as pd

    src = Path(src)
    ep = f"episode_{episode:08d}"
    df = pd.read_parquet(src / f"data/task-{task:04d}/{ep}.parquet")
    out: dict[str, Any] = {"rows": len(df), "columns": {}}
    for c in df.columns:
        v = df[c].iloc[0]
        arr = np.asarray(v)
        out["columns"][c] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "first": np.asarray(v).ravel()[:8].tolist()}
    st = np.stack(df["observation.state"].to_numpy())
    p = parse_state(st)
    out["gripper_width"] = {a: {"min": float(p[f"gripper_{a}"].min()), "max": float(p[f"gripper_{a}"].max()),
                                "p10": float(np.percentile(p[f"gripper_{a}"], 10)),
                                "p90": float(np.percentile(p[f"gripper_{a}"], 90))} for a in ("left", "right")}
    xy = p["base_pose"][:, :2, 3]
    out["base_path_m"] = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    rel = np.stack(df["observation.cam_rel_poses"].to_numpy())
    out["cameras"] = {}
    for i, c in enumerate(CAMERAS):
        T = cam_pose(rel[0, 7 * i: 7 * i + 7])
        out["cameras"][c] = {"pos_base_t0": T[:3, 3].round(3).tolist(), "optical_axis_base_t0": T[:3, 2].round(3).tolist()}
    head = out["cameras"]["head"]
    out["checks"] = {"head_height_1_2m": 1.0 <= head["pos_base_t0"][2] <= 2.0,
                     "head_looks_forward_down": head["optical_axis_base_t0"][0] > 0.2 and head["optical_axis_base_t0"][2] < 0.2}
    meta = json.loads((src / f"meta/episodes/task-{task:04d}/{ep}.json").read_text())
    keys = meta.get("task_obs_keys", [])
    lay = task_info_layout(keys)
    ti_len = len(np.asarray(df["observation.task_info"].iloc[0]))
    out["task_info"] = {"keys": keys, "assumed_len": max(e for _, e in lay.values()) if lay else 0, "actual_len": ti_len}
    out["task_info"]["layout_ok"] = out["task_info"]["assumed_len"] == ti_len
    out["episode_meta"] = {k: meta[k] for k in ("n_steps", "robot_type") if k in meta}
    return out


def prepare(src: str | Path, root: str | Path, task: int = TASK_INDEX, episode: int = EPISODE,
            cams: Sequence[str] = ("head",), image_stride: int = 6, max_frames: int | None = None) -> dict[str, Any]:
    import pandas as pd

    from ..records import write_episode, write_image

    src, root = Path(src), Path(root)
    ep = f"episode_{episode:08d}"
    df = pd.read_parquet(src / f"data/task-{task:04d}/{ep}.parquet")
    n = len(df) if max_frames is None else min(len(df), max_frames)
    st = np.stack(df["observation.state"].to_numpy()[:n])
    rel = np.stack(df["observation.cam_rel_poses"].to_numpy()[:n])
    act = np.stack(df["action"].to_numpy()[:n])
    proprio = parse_state(st)
    proprio.update(t=np.arange(n), action=act)
    tasks = [json.loads(x) for x in (src / "meta/tasks.jsonl").read_text().splitlines() if x.strip()]
    instruction = next(t["task"] for t in tasks if t["task_index"] == task)
    dump = dump_structure(src, task, episode)
    keep = sorted(set(range(0, n, image_stride)) | {n - 1})
    cam_index, cam_meta = {}, {}
    for c in cams:
        idx = CAMERAS.index(c)
        cam_index[c] = {"t": keep, "K": intrinsics(c), "T": np.stack([cam_pose(rel[t, 7 * idx: 7 * idx + 7]) for t in keep])}
        cam_meta[c] = {"depth": True, "width": RESOLUTION[c], "height": RESOLUTION[c]}
    meta = {"benchmark": "b1k", "episode_id": ep, "task_index": task, "instruction": instruction, "fps": FPS, "n_frames": n,
            "arms": ["left", "right"], "cameras": cam_meta, "image_stride": image_stride,
            "notes": ["eef / camera poses in the robot base frame; base_pose = world <- base from robot_pos + robot_2d_ori",
                      "depth: dequantized log-14-bit (v3.7.2 constants), float32 m",
                      f"cam_rel_poses order assumed {CAMERAS}; checks: {dump['checks']}"]}
    write_episode(root, meta, proprio, cam_index)
    keepset = set(keep)
    for c in cams:
        vd = src / f"videos/task-{task:04d}"
        depth = dict(decode_frames(vd / f"observation.images.depth.{c}/{ep}.mp4", RESOLUTION[c], keepset, depth=True))
        for t, rgb in decode_frames(vd / f"observation.images.rgb.{c}/{ep}.mp4", RESOLUTION[c], keepset):
            write_image(root, c, t, rgb, depth.get(t))
    _write_b1k_oracle(src, root, df, n, task, ep, dump)
    (root / "structure.json").write_text(json.dumps(dump, indent=1, default=str))
    return dump


def _write_b1k_oracle(src: Path, root: Path, df: Any, n: int, task: int, ep: str, dump: dict[str, Any]) -> None:
    from ..oracle import write_oracle

    meta = json.loads((src / f"meta/episodes/task-{task:04d}/{ep}.json").read_text())
    ann = json.loads((src / f"annotations/task-{task:04d}/{ep}.json").read_text())
    segs, timeline = annotation_timeline(ann.get("skill_annotation", []), n)
    keys = meta.get("task_obs_keys", [])
    lay = task_info_layout(keys)
    ok = dump["task_info"]["layout_ok"]
    ti = np.stack(df["observation.task_info"].to_numpy()[:n]) if ok else None
    bddl = sorted({re.sub(r"_(real|pos|ori_cos|ori_sin|in_gripper_left|in_gripper_right)$", "", k) for k in keys})
    # annotation names (hotdog_207) <-> task_info names (hotdog.n.02_1): the one whose in_gripper flag
    # turns on during the annotation's "pick up from" segment
    ann_to_bddl: dict[str, str] = {}
    if ti is not None:
        for s in segs:
            if s["label"] != "pick up from" or not s["objects"]:
                continue
            a = s["objects"][0]
            for b in bddl:
                rng = [lay.get(f"{b}_in_gripper_left"), lay.get(f"{b}_in_gripper_right")]
                if (any(r is not None and ti[s["start"]:s["end"] + 1, r[0]].max() > 0.5 for r in rng)
                        and category_of(b) == category_of(a) and b not in ann_to_bddl.values()):
                    ann_to_bddl[a] = b
                    break
        for b in bddl:  # single-instance categories (fridge, microwave)
            cat = category_of(b)
            names = {o for s in segs for o in s["objects"] if category_of(o) == cat}
            if cat and len(names) == 1 and len([x for x in bddl if category_of(x) == cat]) == 1:
                ann_to_bddl.setdefault(names.pop(), b)
    objects = {o: category_of(o) for s in segs for o in s["objects"] if category_of(o)}
    states = []
    for t in range(0, n, 3):
        pos, inhand = {}, {"left": None, "right": None}
        if ti is not None:
            for a, b in ann_to_bddl.items():
                r = lay.get(f"{b}_pos")
                if r is not None:
                    pos[a] = ti[t, r[0]:r[1]].tolist()
                for arm in ("left", "right"):
                    g = lay.get(f"{b}_in_gripper_{arm}")
                    if g is not None and ti[t, g[0]] > 0.5:
                        inhand[arm] = a
        true = [list(k) for k in relations_at(timeline, t)]
        states.append({"t": t, "pos": pos, "true": true, "inhand": inhand if ti is not None else {}})
    rels = ["in", "on", "open", "toggled_on"] + (["inhand", "handempty"] if ti is not None else [])
    write_oracle(root, {"source": "b1k annotations + task_info", "frame": "world", "objects": objects, "relations": rels,
                        "arms": ["left", "right"], "annotation_to_task_info": ann_to_bddl,
                        "notes": ["on/in/open/toggled_on from skill_annotation, changing at segment ends (approximate)",
                                  "inhand from task_info in_gripper flags; positions world frame from task_info"
                                  + ("" if ti is not None else " -- DISABLED: task_info layout mismatch")]},
                 states, segs, {"success": None})
