from capx.envs.tasks.base import CodeExecutionEnvBase


class FrankaLiberoMemCodeEnv(CodeExecutionEnvBase):
    """Generic high-level code environment for LIBERO-Mem tasks.

    Like ``FrankaLiberoCodeEnv``, prompt and task come from the YAML config; the task is
    selected by ``task_id`` on the low-level env (``FrankaLiberoMemEnv``).
    """


__all__ = ["FrankaLiberoMemCodeEnv"]
