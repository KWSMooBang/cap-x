"""LIBERO-Mem demonstrations (HF ``libero-mem/LIBERO-Mem-Raw``, one HDF5 per task) -> the generator's
episode directory (agentview RGB-D + K + base <- camera, proprio) + an oracle directory (simulator).

The HF README says only "simulator initial states, proprioception, and control actions"; the HDF5 layout
is assumed to be LIBERO's (``data/demo_<i>/{actions, states, obs/{ee_pos, gripper_states, ...}}``, data attrs
``problem_info`` / ``env_args``) and ``dump_structure`` writes what is really there. Replay:

* ``states`` present (N, D): per step ``OffScreenRenderEnv.set_init_state(states[t])`` (exact; LIBERO's
  own observation refresh), then cap-x's ``get_observation()`` for agentview RGB, metric depth, K and the
  camera pose in the robot base frame (the frame bg_est2 uses).
* no ``states``: ``init_state`` (demo attr or ``states[0]``) then the recorded ``actions`` stepped through
  the OSC_POSE controller (LIBERO's ``create_dataset`` replay; may drift -- the dump compares the replayed
  EEF with the recorded one).

Proprio comes from the HDF5: ``obs/ee_pos`` (world, robosuite grip site) moved into the robot base frame
with the simulator's base pose, ``obs/gripper_states`` (two finger joints) -> width q0 - q1. If missing,
cap-x's ``robot_cartesian_pos`` (+ bg_est2's TCP offset) is used and the meta notes say so.

Oracle per sampled frame: ``OracleEstimator._candidates`` keys (on over objects / fixtures / table, in over
contain regions, inhand) evaluated by ``LiberoMemSource`` (BDDL predicates + contacts), object positions
(base frame), the LIBERO-Mem subgoal tracker, final ``task_success``. Grasp intervals (simulator grasp
check) -> subtask boundary segments.

Needs the LIBERO-Mem venv and a renderer (CPU: MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa NUMBA_DISABLE_JIT=1).
UNVERIFIED on the development server (no simulator there): run ``scripts/bggen/prep_libero_mem.py`` first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPO = "libero-mem/LIBERO-Mem-Raw"
# The swap task: file index 7 (1-based, KITCHEN_SCENE1_<k>) = cap-x LIBERO-Mem task_id 6 (0-based;
# env_configs/libero_mem/franka_libero_mem_6_swap_bowls.yaml). ``prepare`` checks the instruction.
SWAP_FILE = "KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate_demo.hdf5"
SWAP_TASK_ID = 6
SWAP_INSTRUCTION = "swap the 2 bowls on their plates using the empty plate"


def download(dest: str | Path, filename: str = SWAP_FILE) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, filename, repo_type="dataset", local_dir=str(dest)))


def dump_structure(h5path: str | Path, max_attr: int = 400) -> dict[str, Any]:
    import h5py

    out: dict[str, Any] = {"file": str(h5path), "datasets": {}, "attrs": {}}

    def visit(name: str, obj: Any) -> None:
        if isinstance(obj, h5py.Dataset) and name.count("/") <= 3 and (
                name.startswith("data/demo_0/") or not name.startswith("data/demo_")):
            out["datasets"][name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
        for k, v in obj.attrs.items():
            if name.startswith("data/demo_") and not name.startswith("data/demo_0"):
                continue
            s = v.decode() if isinstance(v, bytes) else str(v)
            out["attrs"][f"{name}@{k}"] = s[:max_attr]

    with h5py.File(h5path, "r") as f:
        f.visititems(visit)
        for k, v in f.attrs.items():
            out["attrs"][f"/@{k}"] = str(v)[:max_attr]
        demos = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[-1])) if "data" in f else []
        out["demos"] = len(demos)
        out["lengths"] = {d: int(f[f"data/{d}/actions"].shape[0]) for d in demos[:200] if f"data/{d}/actions" in f}
        if demos and f"data/{demos[0]}/obs/gripper_states" in f:
            g = np.asarray(f[f"data/{demos[0]}/obs/gripper_states"])
            w = g[:, 0] - g[:, 1] if g.shape[1] >= 2 else g[:, 0]
            out["gripper_width_demo0"] = {"min": float(w.min()), "max": float(w.max()),
                                          "p10": float(np.percentile(w, 10)), "p50": float(np.median(w))}
    return out


def instruction_of(h5path: str | Path) -> str | None:
    import h5py

    with h5py.File(h5path, "r") as f:
        pi = f["data"].attrs.get("problem_info") if "data" in f else None
        if pi is None:
            return None
        try:
            return json.loads(pi).get("language_instruction")
        except Exception:
            return None


def gripper_width(gs: np.ndarray) -> np.ndarray:
    gs = np.asarray(gs, np.float64)
    return gs[:, 0] - gs[:, 1] if gs.ndim == 2 and gs.shape[1] >= 2 else gs.reshape(-1)


def world_to_base(p_world: np.ndarray, base_pos: np.ndarray, base_rot: np.ndarray) -> np.ndarray:
    return (np.asarray(p_world) - base_pos) @ base_rot  # rot.T @ (p - base) for row vectors


def prepare(h5path: str | Path, root: str | Path, demo: str = "demo_0", task_id: int = SWAP_TASK_ID,
            image_stride: int = 2, oracle_stride: int = 2, wrist: bool = False, max_frames: int | None = None) -> dict[str, Any]:
    import h5py

    from capx.belief.oracle import LiberoMemSource, OracleEstimator, _quat_wxyz_to_mat
    from capx.belief.perception import proprio_from_obs
    from capx.envs.simulators.libero_mem import FrankaLiberoMemEnv

    from ..oracle import write_oracle
    from ..records import write_episode, write_image

    root = Path(root)
    with h5py.File(h5path, "r") as f:
        g = f[f"data/{demo}"]
        actions = np.asarray(g["actions"])
        states = np.asarray(g["states"]) if "states" in g else None
        obs = g.get("obs", {})
        ee_pos = np.asarray(obs["ee_pos"]) if "ee_pos" in obs else None
        grip = np.asarray(obs["gripper_states"]) if "gripper_states" in obs else None
        init_state = np.asarray(g.attrs["init_state"]) if "init_state" in g.attrs else (states[0] if states is not None else None)
    instr_file = instruction_of(h5path)
    n = len(actions) if max_frames is None else min(len(actions), max_frames)

    low = FrankaLiberoMemEnv(task_id=task_id, privileged=False, max_steps=10 ** 8, control_freq=20)
    low.reset(seed=0)
    instr_env = getattr(low.handle, "task_language", "")
    if instr_file and instr_env and instr_file.strip().lower() != instr_env.strip().lower():
        raise RuntimeError(f"task_id {task_id} is {instr_env!r} but the file is {instr_file!r}: pass the matching --task-id")
    henv = low.handle.env  # OffScreenRenderEnv
    sim = henv.sim
    src = LiberoMemSource(low)
    orc = OracleEstimator(src)
    base_id = sim.model.body_name2id("robot0_base")

    def refresh(o: dict[str, Any]) -> dict[str, Any]:
        low._current_obs = o
        low.gripper_link_wxyz_xyz = np.concatenate([sim.data.xquat[low.gripper_link_idx], sim.data.xpos[low.gripper_link_idx]])
        return low.get_observation()

    mode = "states" if states is not None else "actions"
    if mode == "actions":
        if init_state is None:
            raise RuntimeError("no states and no init_state in the demo: cannot replay")
        o = henv.set_init_state(init_state)
    cams = ["agentview"] + (["robot0_eye_in_hand"] if wrist else [])
    keep = set(range(0, n, image_stride)) | {n - 1}
    eef_b, width, capx_eef = [], [], []
    cam_t: dict[str, list[int]] = {c: [] for c in cams}
    cam_T: dict[str, list[np.ndarray]] = {c: [] for c in cams}
    cam_K: dict[str, np.ndarray] = {}
    ostates, grasped_prev, segs, seg_start, last_end = [], None, [], None, 0
    for t in range(n):
        if mode == "states":
            o = henv.set_init_state(states[t])
        elif t > 0:
            o, _, _, _ = low.handle.step(actions[t - 1].tolist())
        ob = refresh(o)
        bpos = np.asarray(sim.data.xpos[base_id], np.float64)
        brot = _quat_wxyz_to_mat(np.asarray(sim.data.xquat[base_id], np.float64))
        pr = proprio_from_obs(ob)
        capx_eef.append(pr[0] if pr else np.zeros(3))
        eef_b.append(world_to_base(ee_pos[t], bpos, brot) if ee_pos is not None else capx_eef[-1])
        width.append(float(gripper_width(grip[t:t + 1])[0]) if grip is not None
                     else float(ob["robot_cartesian_pos"][-1]) * float(getattr(low, "gripper_metric_length", 0.08)))
        if t in keep:
            for c in cams:
                im = ob[c]["images"]
                depth = im.get("depth")
                write_image(root, c, t, im["rgb"], None if depth is None else np.asarray(depth)[..., 0] if np.ndim(depth) == 3 else depth)
                cam_t[c].append(t)
                cam_T[c].append(np.asarray(ob[c]["pose_mat"]))
                cam_K[c] = np.asarray(ob[c]["intrinsics"])
        if t % oracle_stride == 0 or t == n - 1:
            truths = {k: orc._eval(k) for k in orc.candidates}
            inh = next((k[1] for k, v in truths.items() if k[0] == "inhand" and v), None)
            ostates.append({"t": t, "pos": {k: list(v) for k, v in src.positions().items()},
                            "true": [[k[0], k[1], k[2]] for k, v in truths.items() if v and k[0] != "inhand"],
                            "inhand": {"main": inh}, "subgoals": low.memory_state()})
            if inh != grasped_prev:
                if inh is not None:
                    segs.append({"start": last_end, "end": t, "label": "pick", "objects": [inh]})
                    seg_start = t
                elif seg_start is not None:
                    segs.append({"start": seg_start, "end": t, "label": "place", "objects": [grasped_prev]})
                    last_end, seg_start = t, None
                grasped_prev = inh
    success = bool(low.handle.task_success())
    proprio = {"t": np.arange(n), "eef_pos_main": np.stack(eef_b), "gripper_main": np.asarray(width),
               "action": actions[:n]}
    meta = {"benchmark": "libero_mem", "episode_id": f"{Path(h5path).stem}/{demo}", "task_id": task_id,
            "instruction": instr_file or instr_env, "fps": 20.0, "n_frames": n, "arms": ["main"],
            "cameras": {c: {"depth": True} for c in cams}, "image_stride": image_stride, "replay": mode,
            "notes": ["eef: HDF5 obs/ee_pos (grip site) in the robot base frame" if ee_pos is not None else
                      "eef: cap-x robot_cartesian_pos + TCP offset (no obs/ee_pos in the file)",
                      "gripper: HDF5 obs/gripper_states q0 - q1 (m)" if grip is not None else "gripper: cap-x fraction x metric length",
                      "camera pose / K / depth: cap-x get_observation() after the replayed state"]}
    write_episode(root, meta, proprio, {c: {"t": cam_t[c], "K": cam_K.get(c), "T": np.stack(cam_T[c])} for c in cams})
    objects = {}
    for name in src.movable_objects():
        for key in ("bowl", "plate", "bottle", "basket", "cream_cheese", "mug", "tray"):
            if key in name:
                objects[name] = key
                break
    write_oracle(root, {"source": "libero_mem simulator", "frame": "base", "objects": objects,
                        "relations": ["on", "in", "inhand", "handempty"], "arms": ["main"], "replay": mode},
                 ostates, segs, {"success": success})
    drift = float(np.linalg.norm(np.stack(eef_b) - np.stack(capx_eef), axis=1).mean())
    return {"frames": n, "replay": mode, "images": len(keep), "success": success, "instruction_file": instr_file,
            "instruction_env": instr_env, "eef_hdf5_vs_capx_mean_m": round(drift, 4),
            "oracle_segments": len(segs)}
