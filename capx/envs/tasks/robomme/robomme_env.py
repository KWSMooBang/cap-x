from capx.envs.tasks.base import CodeExecutionEnvBase


class RoboMMECodeEnv(CodeExecutionEnvBase):
    """Generic high-level code environment for RoboMME tasks.

    Prompt and APIs come from the YAML config; the task is selected by ``task_id`` /
    ``episode_idx`` on the low-level env (``RoboMMEEnv``).
    """


__all__ = ["RoboMMECodeEnv"]
