"""capx.bggen adapters' prepare() on mocked sources: a tiny LeRobot-v2.1 directory with stubbed video
decoding (B1K) and a stubbed LIBERO-Mem simulator. Checks the episode / oracle files and the generator's
loader on them; the real data paths run on kubectl (scripts/bggen/kubectl_runbook.sh)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from capx.bggen.adapters import b1k, libero_mem
from capx.bggen.events import detect_events
from capx.bggen.configs import b1k as b1k_cfg
from capx.bggen.oracle import load_oracle
from capx.bggen.records import load_episode
from tests.test_bggen import B1K_SKILLS, B1K_TASK_OBS_KEYS

N = 120


def _fake_b1k_src(src: Path, flags: bool = True) -> None:
    """``flags=False``: in_gripper flags all -1 as in the real 2025 demos; hotdog_1 moves with the left EEF 45-85."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    ep = "episode_00450010"
    (src / "data/task-0045").mkdir(parents=True)
    (src / "meta/episodes/task-0045").mkdir(parents=True)
    (src / "annotations/task-0045").mkdir(parents=True)
    state = np.zeros((N, 256), np.float32)
    w = np.r_[np.full(30, 0.04), np.linspace(0.04, 0.012, 10), np.full(40, 0.012), np.linspace(0.012, 0.04, 10), np.full(30, 0.04)]
    state[:, 193], state[:, 194] = w / 2, w / 2  # left gripper closes on something and opens
    state[:, 232:234] = 0.025
    state[:, 186:189] = [0.5, 0.2, 0.9]
    state[:, 189:193] = [0, 0, 0, 1]
    state[:, 140] = np.linspace(6.0, 6.5, N)  # the base drives 0.5 m in x
    rel = np.zeros((N, 21), np.float32)
    for i, (x, z) in enumerate(((0.4, 0.9), (0.4, 0.9), (0.1, 1.5))):
        rel[:, 7 * i: 7 * i + 3] = [x, 0.0, z]
        # camera looking forward-down: OpenGL camera with -z along base +x, y up -> quat of that rotation (xyzw)
        rel[:, 7 * i + 3: 7 * i + 7] = [0.5, -0.5, -0.5, 0.5]
    ti = np.zeros((N, 70), np.float32)
    lay = b1k.task_info_layout(B1K_TASK_OBS_KEYS)
    a, _ = lay["hotdog.n.02_1_in_gripper_left"]
    p0, p1 = lay["hotdog.n.02_1_pos"]
    q0, q1 = lay["hotdog.n.02_2_pos"]
    ti[:, q0:q1] = [7.5, -0.5, 0.9]  # hotdog_2 never moves
    if flags:
        ti[45:85, a] = 1.0
        ti[:, p0:p1] = [6.8, 0.2, 0.9]
    else:
        for k, (s0, _) in lay.items():
            if "_in_gripper_" in k:
                ti[:, s0] = -1.0
        eef_w = np.c_[np.linspace(6.0, 6.5, N) + 0.5, np.full(N, 0.2), np.full(N, 0.9)]  # base x + eef x (no yaw)
        ti[:, p0:p1] = eef_w[45]
        ti[45:85, p0:p1] = eef_w[45:85]  # carried by the left hand
        ti[85:, p0:p1] = eef_w[84]
    df = pd.DataFrame({"observation.state": list(state), "observation.cam_rel_poses": list(rel), "observation.task_info": list(ti),
                       "action": list(np.zeros((N, 23), np.float32)), "timestamp": np.arange(N) / 30.0, "episode_index": 450010,
                       "index": np.arange(N)})
    df.to_parquet(src / f"data/task-0045/{ep}.parquet")
    (src / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 45, "task_name": "cook_hot_dogs", "task":
                                                      "Take the two hot dogs out of the refrigerator in the kitchen and cook them in the microwave until both are cooked."}) + "\n")
    (src / f"meta/episodes/task-0045/{ep}.json").write_text(json.dumps({"task_obs_keys": B1K_TASK_OBS_KEYS, "n_steps": N,
                                                                         "robot_type": "R1Pro"}))
    skills = [{"skill_description": [d], "object_id": [o], "frame_duration": [min(f[0] // 60, N - 1), min(f[1] // 60, N - 1)]}
              for d, o, f in B1K_SKILLS]
    skills[2]["frame_duration"] = [40, 90]  # hotdog_207 picked while in_gripper flags are on
    (src / f"annotations/task-0045/{ep}.json").write_text(json.dumps({"skill_annotation": skills}))


def test_b1k_prepare_mocked(tmp_path, monkeypatch):
    src, root = tmp_path / "src", tmp_path / "ep"
    _fake_b1k_src(src)

    def fake_decode(path, size, keep, depth=False):
        for i in sorted(keep):
            yield i, (np.full((size, size), 1.5, np.float32) if depth else np.full((size, size, 3), 100, np.uint8))

    monkeypatch.setattr(b1k, "decode_frames", fake_decode)
    dump = b1k.prepare(src, root, image_stride=10)
    assert dump["task_info"]["layout_ok"] and dump["checks"] == {"head_height_1_2m": True, "head_looks_forward_down": True}
    assert dump["cameras"]["head"]["optical_axis_base_t0"] == [1.0, 0.0, 0.0]
    ep = load_episode(root)
    assert ep.arms == ["left", "right"] and ep.mobile and ep.n_frames == N and ep.instruction.startswith("Take the two hot dogs")
    cam = ep.camera("head", 55)
    assert cam.t == 50 and cam.has_geometry and cam.depth.shape == (720, 720) and np.allclose(cam.K[0, 0], 306.0)
    assert np.isclose(ep.gripper("left")[0], 0.04)
    kinds = [e.kind for e in detect_events(ep, b1k_cfg().events) if e.kind not in ("keyframe", "start", "end")]
    assert kinds.count("grasp") == 1 and kinds.count("release") == 1 and "base_start" in kinds
    orc = load_oracle(root)
    assert orc.meta["annotation_to_task_info"].get("hotdog_207") == "hotdog.n.02_1"
    assert orc.at(60)["inhand"]["left"] == "hotdog_207" and orc.at(10)["inhand"]["left"] is None
    assert orc.meta["frame"] == "world" and "hotdog_207" in orc.at(0)["pos"]
    assert not (root / "episode" / "oracle").exists() and (root / "structure.json").exists()


def test_libero_prepare_mocked(tmp_path, monkeypatch):
    import h5py

    n = 12
    h5 = tmp_path / "demo.hdf5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("data")
        g.attrs["problem_info"] = json.dumps({"language_instruction": libero_mem.SWAP_INSTRUCTION})
        # as LIBERO-Mem-Raw (kubectl 1a): demos start at demo_1; only actions, init_state, states (no obs/);
        # states = robosuite [time, qpos ...] with the Panda fingers at states[:, 8:10]
        d = g.create_group("demo_1")
        d.create_dataset("actions", data=np.zeros((n, 7)))
        states = np.zeros((n, 12))
        states[:, 0] = np.arange(n)
        states[:, 8], states[:, 9] = 0.04, -0.04
        states[6:, 8], states[6:, 9] = 0.0021, -0.0021  # closed on the bowl rim: 4.2 mm
        d.create_dataset("states", data=states)
        d.create_dataset("init_state", data=states[0])

    class Sim:
        def __init__(self):
            self.model = types.SimpleNamespace(body_name2id=lambda name: 0)
            self.data = types.SimpleNamespace(xpos=np.array([[1.0, 0.0, 0.9]]), xquat=np.array([[1.0, 0, 0, 0]]),
                                              qpos=np.zeros(11))

    class HEnv:
        def __init__(self):
            self.sim, self.state = Sim(), None

        def set_init_state(self, s):
            self.state = s
            self.sim.data.qpos = np.asarray(s[1:])
            return {"t": int(s[0])}

    class Env:
        def __init__(self, **kw):
            # the subgoal tracker does not advance under state replay: task_success() stays False
            self.handle = types.SimpleNamespace(env=HEnv(), task_language=libero_mem.SWAP_INSTRUCTION, task_success=lambda: False,
                                                step=None)
            self.gripper_link_idx = 0

        def reset(self, seed=0):
            pass

        def get_observation(self):
            t = self._current_obs["t"]
            cam = {"images": {"rgb": np.full((8, 10, 3), t, np.uint8), "depth": np.full((8, 10, 1), 0.8)},
                   "pose_mat": np.eye(4), "intrinsics": np.eye(3)}
            return {"agentview": cam, "robot_cartesian_pos": np.r_[0.6, 0.0, 0.1, 1, 0, 0, 0, 1.0]}

        def memory_state(self):
            return {"num_satisfied_subgoals": 0}

    class Src:
        def __init__(self, low):
            self.low = low

        def positions(self):
            return {"akita_black_bowl_1": (0.6, 0.0, 0.04), "plate_1": (0.6, 0.0, 0.01)}

        def movable_objects(self):
            return ["akita_black_bowl_1", "plate_1"]

        def on_targets(self):
            return ["akita_black_bowl_1", "plate_1", "kitchen_table_the_center", "table"]

        def goal_state(self):
            return [["sequence", ["on", "akita_black_bowl_1", "plate_2"], ["on", "akita_black_bowl_1", "plate_1"]]]

        def eval_predicate(self, p):  # the last state: the bowl is on plate_1 (the final relation of the sequence)
            return tuple(p) == ("on", "akita_black_bowl_1", "plate_1")

    class Orc:
        def __init__(self, src):
            self.src = src
            self.candidates = [("on", "akita_black_bowl_1", "plate_1"), ("inhand", "akita_black_bowl_1", None)]

        def _eval(self, key):
            t = self.src.low.handle.env.state[0]
            return key[0] == "on" if t < 5 else key[0] == "inhand"

    monkeypatch.setitem(sys.modules, "capx.envs.simulators.libero_mem", types.SimpleNamespace(FrankaLiberoMemEnv=Env))
    import capx.belief.oracle as bo

    monkeypatch.setattr(bo, "LiberoMemSource", Src)
    monkeypatch.setattr(bo, "OracleEstimator", Orc)
    info = libero_mem.prepare(h5, tmp_path / "ep", image_stride=4)
    assert info["demo"] == "demo_1" and info["replay"] == "states" and info["frames"] == n
    assert info["success"] and not info["tracker_success"]  # (a) success from the final goal relations
    ep = load_episode(tmp_path / "ep")
    # no obs/ee_pos: cap-x robot_cartesian_pos + bg_est2's TCP offset (0.108 along the approach axis);
    # fingers from the replayed sim qpos[7:9]: width q0 - q1 (not q0)
    assert np.allclose(ep.eef("main")[0], [0.6, 0.0, 0.1 + 0.108])
    assert np.isclose(ep.gripper("main")[0], 0.08) and np.isclose(ep.gripper("main")[-1], 0.0042)
    assert list(ep.camera_frames("agentview")) == [0, 4, 8, 11] and ep.camera("agentview", 9).rgb[0, 0, 0] == 8
    orc = load_oracle(tmp_path / "ep")
    assert orc.objects == {"akita_black_bowl_1": "bowl", "plate_1": "plate"}
    assert orc.meta["regions"] == {"center": "kitchen_table_the_center"}  # at(X, center) is scored against it
    assert orc.at(0)["true"] == [["on", "akita_black_bowl_1", "plate_1"]] and orc.at(6)["inhand"]["main"] == "akita_black_bowl_1"
    assert [s["label"] for s in orc.segments] == ["pick"]
    assert orc.final["success"] is True and orc.final["tracker_success"] is False
    assert orc.final["branches"] == [{"on akita_black_bowl_1 plate_1": True}]


def test_b1k_oracle_inhand_by_comotion_when_flags_unfilled(tmp_path, monkeypatch):
    """kubectl 1b: the 2025 demos leave every *_in_gripper_* at -1 -> inhand and the annotation <-> task_info
    names come from motion (the object moving with the EEF / its movement onset during the pick)."""
    src, root = tmp_path / "src", tmp_path / "ep"
    _fake_b1k_src(src, flags=False)

    def fake_decode(path, size, keep, depth=False):
        for i in sorted(keep):
            yield i, (np.full((size, size), 1.5, np.float32) if depth else np.full((size, size, 3), 100, np.uint8))

    monkeypatch.setattr(b1k, "decode_frames", fake_decode)
    b1k.prepare(src, root, image_stride=30)
    orc = load_oracle(root)
    assert orc.meta["annotation_to_task_info"]["hotdog_207"] == "hotdog.n.02_1"
    assert orc.meta["inhand_source"].startswith("co-motion")
    assert orc.at(66)["inhand"]["left"] == "hotdog_207" and orc.at(66)["inhand"]["right"] is None
    assert orc.at(30)["inhand"]["left"] is None and orc.at(114)["inhand"]["left"] is None


def test_comotion_inhand_needs_moving_together():
    n = 60
    eef = np.c_[np.linspace(0, 0.6, n), np.zeros(n), np.full(n, 0.9)]
    carried = eef.copy()
    carried[:20] = eef[20]
    carried[40:] = eef[39]
    near_still = np.tile(eef[30] + [0.0, 0.05, 0.0], (n, 1))  # the hand passes by, the object stays
    out = b1k.comotion_inhand({"a": carried, "b": near_still}, {"left": eef})
    # carried from frame 20: seen as held once a full window of joint motion exists (frame 35)
    assert out["left"][36] == "a" and out["left"][10] is None
    assert out["left"][39] == "a" and out["left"][59] is None  # dropped at 40, the hand moves on
    assert "b" not in out["left"]
