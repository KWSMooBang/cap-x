"""Give memory a finished trial freed back to the OS.

A trial records up to ~2000 video frames (1.2 MB each at 800x512) and every sim step renders two
cameras with depth (~5.6 MB of short-lived arrays). After the trial these are freed, but glibc keeps
the freed heap pages (main and per-thread arenas) in the process, so a task process that runs its
trials one after another grew by about the size of one trial's frames per trial. ``malloc_trim(0)``
returns the free pages; it changes no Python object.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import gc

_LIBC = None
_LIBC_LOADED = False


def _libc():
    global _LIBC, _LIBC_LOADED
    if not _LIBC_LOADED:
        _LIBC_LOADED = True
        try:
            name = ctypes.util.find_library("c") or "libc.so.6"
            lib = ctypes.CDLL(name)
            lib.malloc_trim.argtypes = [ctypes.c_size_t]
            lib.malloc_trim.restype = ctypes.c_int
            _LIBC = lib
        except (OSError, AttributeError):  # not glibc (e.g. musl, macOS): nothing to trim
            _LIBC = None
    return _LIBC


def rss_mb() -> float | None:
    """Resident set size of this process in MB (Linux; None elsewhere)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def release_memory() -> bool:
    """gc.collect() and, on glibc, malloc_trim(0). True if memory was trimmed."""
    gc.collect()
    lib = _libc()
    if lib is None:
        return False
    return bool(lib.malloc_trim(0))
