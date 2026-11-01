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
with the simulator's base pose, ``obs/gripper_states`` (two finger joints) -> width |q0| + |q1|. If missing,
cap-x's ``robot_cartesian_pos`` (+ bg_est2's TCP offset) is used and the meta notes say so.

Oracle per sampled frame: ``OracleEstimator._candidates`` keys (on over objects / fixtures / table, in over
contain regions, inhand) evaluated by ``LiberoMemSource`` (BDDL predicates + contacts), object positions
(base frame), the LIBERO-Mem subgoal tracker; success = the BDDL goal's final relations evaluated on the
last state (``goal_success``; the tracker does not advance under state replay and is kept as ``tracker_success``). Grasp intervals (simulator grasp
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
FINGER_QPOS = slice(7, 9)  # Panda finger joints in sim.data.qpos (states[:, 8:10]); width = q0 - q1 = |q0| + |q1|
SWAP_INSTRUCTION = "swap the 2 bowls on their plates using the empty plate"
# cap-x LIBERO-Mem task_id (0-based) -> LIBERO-Mem task name = BDDL file stem = HF file "<name>_demo.hdf5"
# (LIBERO-Mem libero/benchmark/libero_suite_task_map.py "libero_mem"; HF file list checked 2026-11-01).
TASK_NAMES: tuple[str, ...] = (
    "KITCHEN_SCENE1_1_pick_up_the_bowl_and_place_it_back_on_the_plate",
    "KITCHEN_SCENE1_2_lift_the_bottle_and_put_it_down_on_the_plate",
    "KITCHEN_SCENE1_3_lift_the_bowl_and_place_it_back_on_the_plate_3_times",
    "KITCHEN_SCENE1_4_pick_up_the_bottle_and_put_it_down_the_plate_3_times",
    "KITCHEN_SCENE1_5_lift_the_bowl_and_place_it_back_on_the_plate_5_times",
    "KITCHEN_SCENE1_6_pick_up_the_bowl_and_place_it_on_the_plate_7_times",
    "KITCHEN_SCENE1_7_swap_the_2_bowls_on_their_plates_using_the_empty_plate",
    "KITCHEN_SCENE1_8_rotate_the_3_bowls_on_their_plates_from_left_to_right_using_the_empty_plate",
    "KITCHEN_SCENE1_9_put_the_cream_cheese_in_the_nearest_basket_and_place_that_basket_in_the_center",
    "KITCHEN_SCENE1_10_put_the_cream_cheese_in_the_nearest_basket_and_place_the_empty_basket_in_the_center",
)


def task_file(task_id: int) -> str:
    return f"{TASK_NAMES[task_id]}_demo.hdf5"


def bddl_stem(path: str | None) -> str | None:
    return Path(str(path)).stem if path else None


def check_task(task_id: int, file_bddl: str | None, file_instruction: str | None, env_instruction: str | None,
               h5_name: str | None = None) -> dict[str, Any]:
    """Is the HDF5 a demo of cap-x task ``task_id``? The file's ``data@bddl_file_name`` stem (else the HDF5 name)
    must be the task's name. The languages are compared too but may differ: LIBERO-Mem-Raw task 2
    (KITCHEN_SCENE1_3 ... 3_times) stores "pick up the bowl and place it back on the plate" in problem_info while
    its BDDL says "... 3 times" (checked 2026-11-01). The episode instruction is the env's (BDDL) language, the one
    cap-x evaluates with; the file's is kept in the meta. Raises on a task mismatch."""
    name = TASK_NAMES[task_id] if 0 <= task_id < len(TASK_NAMES) else None
    h5 = Path(h5_name).name if h5_name else ""
    stem = bddl_stem(file_bddl) or (h5.removesuffix("_demo.hdf5") if h5.startswith("KITCHEN_") and h5.endswith("_demo.hdf5") else None)

    def norm(x: str | None) -> str:
        return (x or "").strip().lower()

    lang_same = bool(file_instruction and env_instruction) and norm(file_instruction) == norm(env_instruction)
    if stem is not None and name is not None:
        if stem != name:
            raise RuntimeError(f"task_id {task_id} is {name} but the file is {stem}: pass the matching --task-id")
        how = "bddl_file_name" if file_bddl else "hdf5 name"
    elif file_instruction and env_instruction and not lang_same:
        raise RuntimeError(f"task_id {task_id} is {env_instruction!r} but the file is {file_instruction!r}: "
                           "pass the matching --task-id")
    else:
        how = "language"
    return {"task_name": name, "matched_by": how, "language_same": lang_same, "file_instruction": file_instruction,
            "env_instruction": env_instruction, "instruction": env_instruction or file_instruction}


def regions_of(on_targets: list[str], objects: list[str]) -> dict[str, str]:
    """Goal regions among the oracle's On targets: ``kitchen_table_the_center`` -> {"center": ...}."""
    import re

    from ..vocab import REGIONS

    out = {}
    for n in on_targets:
        m = re.search(r"_the_([a-z]+)$", n)
        if m and m.group(1) in REGIONS and n not in objects:
            out[m.group(1)] = n
    return out


def download(dest: str | Path, filename: str = SWAP_FILE) -> Path:
    """One task's HDF5 (``task_file(task_id)``); hf_hub_download skips a complete local copy."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(REPO, filename, repo_type="dataset", local_dir=str(dest)))


def dump_structure(h5path: str | Path, max_attr: int = 400) -> dict[str, Any]:
    import h5py

    out: dict[str, Any] = {"file": str(h5path), "datasets": {}, "attrs": {}}
    with h5py.File(h5path, "r") as f:
        demos = list_demos(f)
        first = demos[0] if demos else None
        out["demos"], out["first_demo"] = len(demos), first

        def visit(name: str, obj: Any) -> None:
            # every dataset outside data/, and those of the first demo (all demos share the layout)
            other_demo = name.startswith("data/demo_") and (first is None or not (name == f"data/{first}"
                                                                                  or name.startswith(f"data/{first}/")))
            if other_demo:
                return
            if isinstance(obj, h5py.Dataset):
                out["datasets"][name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
            for k, v in obj.attrs.items():
                s = v.decode() if isinstance(v, bytes) else str(v)
                out["attrs"][f"{name}@{k}"] = s[:max_attr]

        f.visititems(visit)
        for k, v in f.attrs.items():
            out["attrs"][f"/@{k}"] = str(v)[:max_attr]
        out["lengths"] = {d: int(f[f"data/{d}/actions"].shape[0]) for d in demos[:200] if f"data/{d}/actions" in f}
        g = None
        if first and f"data/{first}/obs/gripper_states" in f:
            g, out["gripper_source"] = np.asarray(f[f"data/{first}/obs/gripper_states"], np.float64), "obs/gripper_states"
        elif first and f"data/{first}/states" in f:  # robosuite flattened state [time, qpos...]: fingers qpos[7:9]
            g, out["gripper_source"] = np.asarray(f[f"data/{first}/states"], np.float64)[:, 8:10], "states[:, 8:10]"
        if g is not None:
            w = gripper_width(g)
            out["gripper_first_demo"] = {
                "fingers": [{"min": float(g[:, i].min()), "max": float(g[:, i].max())} for i in range(g.shape[1])],
                "width_abs_sum": {"min": float(w.min()), "max": float(w.max()), "p10": float(np.percentile(w, 10)),
                                  "p50": float(np.median(w))}}
    return out


def list_demos(f: Any) -> list[str]:
    """Demo groups in numeric order (LIBERO-Mem-Raw starts at demo_1)."""
    if "data" not in f:
        return []
    return sorted((k for k in f["data"] if k.startswith("demo_")), key=lambda s: int(s.split("_")[-1]))


def first_demo(h5path: str | Path) -> str:
    import h5py

    with h5py.File(h5path, "r") as f:
        demos = list_demos(f)
    if not demos:
        raise RuntimeError(f"no demo_* group in {h5path}")
    return demos[0]


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


def bddl_of(h5path: str | Path) -> str | None:
    import h5py

    with h5py.File(h5path, "r") as f:
        v = f["data"].attrs.get("bddl_file_name") if "data" in f else None
    return v.decode() if isinstance(v, bytes) else (str(v) if v is not None else None)


def gripper_width(gs: np.ndarray) -> np.ndarray:
    """Finger opening = |q0| + |q1|: right for both sign conventions of the second finger joint
    (robosuite Panda: q1 negative when open; some exports store both positive)."""
    gs = np.asarray(gs, np.float64)
    return np.abs(gs[:, 0]) + np.abs(gs[:, 1]) if gs.ndim == 2 and gs.shape[1] >= 2 else np.abs(gs.reshape(-1))


def final_goal_preds(branches: Any) -> list[list[tuple]]:
    """Per goal branch, the relations that must hold at the end: steps applied in order, a later on / in of
    the same object replaces the earlier one (swap: A->plate_A, B->plate_B, A->plate_C => A on plate_C, B on
    plate_B). Repeats ("3 times") collapse to the final arrangement (the count is not checked)."""
    out = []
    for b in branches:
        loc: dict[str, tuple] = {}
        other: list[tuple] = []
        for step in b:
            for pr in step:
                if pr[0] in ("on", "in") and len(pr) == 3:
                    loc[pr[1]] = tuple(pr)
                elif tuple(pr) not in other:
                    other.append(tuple(pr))
        out.append(list(loc.values()) + other)
    return out


def goal_success(src: Any) -> dict[str, Any]:
    """Success from the final simulator state: some branch's final relations all hold (``eval_predicate``)."""
    from capx.belief.progress import goal_branches

    res = []
    for preds in final_goal_preds(goal_branches(src.goal_state())):
        vals = {" ".join(p): bool(src.eval_predicate(p)) for p in preds}
        res.append(vals)
    return {"success": any(all(v.values()) for v in res if v), "branches": res}


def world_to_base(p_world: np.ndarray, base_pos: np.ndarray, base_rot: np.ndarray) -> np.ndarray:
    return (np.asarray(p_world) - base_pos) @ base_rot  # rot.T @ (p - base) for row vectors


def prepare(h5path: str | Path, root: str | Path, demo: str | None = None, task_id: int = SWAP_TASK_ID,
            image_stride: int = 2, oracle_stride: int = 2, wrist: bool = False, max_frames: int | None = None) -> dict[str, Any]:
    import h5py

    from capx.belief.oracle import LiberoMemSource, OracleEstimator, _quat_wxyz_to_mat
    from capx.belief.perception import proprio_from_obs
    from capx.belief.progress import goal_branches
    from capx.envs.simulators.libero_mem import FrankaLiberoMemEnv

    from ..oracle import write_oracle
    from ..records import write_episode, write_image

    root = Path(root)
    demo = demo or first_demo(h5path)
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
    task = check_task(task_id, bddl_of(h5path), instr_file, instr_env, Path(h5path).name)
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
        # finger joints: HDF5 obs/gripper_states if present, else the replayed simulator qpos[7:9] (= states[:, 8:10],
        # robosuite flattened state [time, qpos...]; kubectl 1a: q0 0.0021..0.040, q1 -0.040..-0.0021)
        fingers = grip[t:t + 1] if grip is not None else np.asarray(sim.data.qpos[FINGER_QPOS], np.float64)[None]
        width.append(float(gripper_width(fingers)[0]))
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
    tracker_success = bool(low.handle.task_success())  # the subgoal tracker does not advance under state replay
    goal = goal_success(src)
    success = goal["success"]
    proprio = {"t": np.arange(n), "eef_pos_main": np.stack(eef_b), "gripper_main": np.asarray(width),
               "action": actions[:n]}
    meta = {"benchmark": "libero_mem", "episode_id": f"{Path(h5path).stem}/{demo}", "task_id": task_id,
            "instruction": task["instruction"], "instruction_file": instr_file, "task_check": task, "fps": 20.0, "n_frames": n, "arms": ["main"],
            "cameras": {c: {"depth": True} for c in cams}, "image_stride": image_stride, "replay": mode,
            "notes": ["eef: HDF5 obs/ee_pos (grip site) in the robot base frame" if ee_pos is not None else
                      "eef: cap-x robot_cartesian_pos + TCP offset (no obs/ee_pos in the file)",
                      "gripper: HDF5 obs/gripper_states |q0| + |q1| (m)" if grip is not None else "gripper: replayed sim qpos[7:9] (= states[:, 8:10]) |q0| + |q1| (m)",
                      "camera pose / K / depth: cap-x get_observation() after the replayed state"]}
    write_episode(root, meta, proprio, {c: {"t": cam_t[c], "K": cam_K.get(c), "T": np.stack(cam_T[c])} for c in cams})
    objects = {}
    for name in src.movable_objects():
        for key in ("bowl", "plate", "bottle", "basket", "cream_cheese", "mug", "tray"):
            if key in name:
                objects[name] = key
                break
    write_oracle(root, {"source": "libero_mem simulator", "frame": "base", "objects": objects,
                        "relations": ["on", "in", "inhand", "handempty"], "arms": ["main"], "replay": mode,
                        "regions": regions_of(list(src.on_targets()) + [p[2] for b in final_goal_preds(goal_branches(src.goal_state()))
                                                                          for p in b if p[0] == "on" and len(p) == 3],
                                              src.movable_objects())},
                 ostates, segs, {"success": success, "source": "final goal predicates (BDDL) on the last state",
                                 "branches": goal["branches"], "tracker_success": tracker_success})
    drift = float(np.linalg.norm(np.stack(eef_b) - np.stack(capx_eef), axis=1).mean())
    return {"demo": demo, "frames": n, "replay": mode, "images": len(keep), "success": success,
            "tracker_success": tracker_success, "goal_final": goal["branches"], "instruction_file": instr_file,
            "instruction_env": instr_env, "task_check": task, "eef_hdf5_vs_capx_mean_m": round(drift, 4),
            "oracle_segments": len(segs)}
