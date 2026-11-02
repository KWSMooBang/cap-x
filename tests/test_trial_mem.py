"""Per-trial memory release (trial-mem-leak): video frames handed over without a second copy, and the
runner trimming the heap after every trial attempt. The RSS measurement on a real LIBERO-Mem task is
scripts/trial_mem_repro.py (needs the simulator); these checks run anywhere."""

from __future__ import annotations

import numpy as np
import pytest

import capx.envs.runner as runner
from capx.utils import mem_utils


def test_release_memory_runs_and_rss_is_read():
    assert isinstance(mem_utils.release_memory(), bool)
    rss = mem_utils.rss_mb()
    assert rss is None or rss > 0


def test_release_memory_without_glibc(monkeypatch):
    monkeypatch.setattr(mem_utils, "_LIBC_LOADED", True)
    monkeypatch.setattr(mem_utils, "_LIBC", None)
    assert mem_utils.release_memory() is False


def _summary(trial):
    return runner.TrialSummary(trial=trial, success=True, reward=1.0, terminated=True, truncated=False,
                               sandbox_rc=0, log="")


def test_runner_releases_memory_after_every_attempt(monkeypatch):
    calls: list[tuple[int, int]] = []
    attempts = {"n": 0}

    def fake_trial(**kw):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TimeoutError("slow")
        return _summary(kw["trial"])

    monkeypatch.setattr(runner, "_run_single_trial_with_timeout", fake_trial)
    monkeypatch.setattr(runner, "_release_trial_memory", lambda trial, attempt: calls.append((trial, attempt)))
    out = runner._run_trial_with_retries(object(), 7, None, {"max_trial_retries": 3}, None)
    assert out.trial == 7
    assert calls == [(7, 0), (7, 1), (7, 2)]


def test_runner_releases_memory_when_the_trial_raises(monkeypatch):
    calls: list[tuple[int, int]] = []

    def boom(**kw):
        raise RuntimeError("env broke")

    monkeypatch.setattr(runner, "_run_single_trial_with_timeout", boom)
    monkeypatch.setattr(runner, "_release_trial_memory", lambda trial, attempt: calls.append((trial, attempt)))
    with pytest.raises(RuntimeError):
        runner._run_trial_with_retries(object(), 2, None, {"max_trial_retries": 3}, None)
    assert calls == [(2, 0)]


def test_release_trial_memory_logs_rss(capsys):
    runner._release_trial_memory(3, 0)
    out = capsys.readouterr().out
    if mem_utils.rss_mb() is not None:
        assert "[mem] trial 3 attempt 1: rss" in out


def _frame_env(cls):
    env = object.__new__(cls)
    env._frame_buffer = [np.full((4, 5, 3), i, np.uint8) for i in range(3)]
    env._wrist_frame_buffer = [np.full((4, 5, 3), 10 + i, np.uint8) for i in range(2)]
    return env


def _check_handover(env):
    expected = [f.copy() for f in env._frame_buffer]
    kept = env.get_video_frames()  # no clear: copies, buffer unchanged
    assert len(env._frame_buffer) == 3 and all(a is not b for a, b in zip(kept, env._frame_buffer))
    buffer = env._frame_buffer
    frames = env.get_video_frames(clear=True)
    assert frames is buffer  # handed over, not copied
    assert env._frame_buffer == [] and env.get_video_frame_count() == 0
    assert [f.tobytes() for f in frames] == [f.tobytes() for f in expected]
    env._frame_buffer.append(np.zeros((4, 5, 3), np.uint8))  # recording into the new buffer
    assert len(frames) == 3
    wrist = env.get_wrist_video_frames(clear=True)
    assert [int(f[0, 0, 0]) for f in wrist] == [10, 11] and env._wrist_frame_buffer == []


def test_libero_frames_handed_over():
    libero = pytest.importorskip("capx.envs.simulators.libero", exc_type=ImportError)
    _check_handover(_frame_env(libero.FrankaLiberoEnv))


def test_robomme_frames_handed_over():
    robomme = pytest.importorskip("capx.envs.simulators.robomme", exc_type=ImportError)
    _check_handover(_frame_env(robomme.RoboMMEEnv))
