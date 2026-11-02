"""JSON helpers for trial artifacts: numpy values (np.bool_, np.int64, ndarray, ...) become Python values."""

from __future__ import annotations

from typing import Any


def json_default(o: Any) -> Any:
    """``default=`` hook for json.dump(s): numpy scalar -> .item(), array -> .tolist(), anything else -> str.

    Only called for objects json cannot encode itself, so output for plain Python values is unchanged.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if np is not None:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    if isinstance(o, (set, frozenset)):
        return list(o)
    return str(o)
