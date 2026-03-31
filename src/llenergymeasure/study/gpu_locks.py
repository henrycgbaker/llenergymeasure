"""Per-GPU advisory locks for study concurrency safety.

Uses filelock.FileLock (kernel-backed via fcntl.flock on Linux) which
auto-releases on process death including SIGKILL. Lock files live at
~/.cache/llem/gpu-{N}.lock.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from filelock import FileLock, Timeout

from llenergymeasure.utils.exceptions import StudyError

__all__ = ["acquire_gpu_locks", "release_gpu_locks"]


def acquire_gpu_locks(
    gpu_indices: list[int],
    lock_dir: Path | None = None,
) -> list[FileLock]:
    """Acquire advisory file locks for the given GPU indices, in sorted order.

    Sorted acquisition (Dijkstra's resource ordering) prevents deadlocks when
    multiple studies attempt to acquire overlapping GPU sets concurrently.

    The locks are non-blocking (timeout=0). If any GPU is already locked by
    another process, all previously acquired locks are released (atomic all-or-none
    rollback) and a StudyError is raised.

    Args:
        gpu_indices: GPU device indices to lock (e.g. [0, 1]).
        lock_dir: Directory for lock files. Defaults to ~/.cache/llem.

    Returns:
        List of acquired FileLock objects in sorted index order.

    Raises:
        StudyError: If any GPU is locked by another process.
    """
    if lock_dir is None:
        lock_dir = Path.home() / ".cache" / "llem"

    lock_dir.mkdir(parents=True, exist_ok=True)

    # Sort to prevent deadlocks (Dijkstra's resource ordering)
    sorted_indices = sorted(gpu_indices)

    acquired: list[FileLock] = []
    failed_indices: list[int] = []

    for idx in sorted_indices:
        lock_path = lock_dir / f"gpu-{idx}.lock"
        lock = FileLock(str(lock_path), timeout=0)
        try:
            lock.acquire()
            acquired.append(lock)
        except Timeout:
            failed_indices.append(idx)
            # Atomic rollback: release all already-acquired locks
            for held_lock in acquired:
                with contextlib.suppress(Exception):
                    held_lock.release()
            raise StudyError(
                f"GPU(s) {failed_indices} locked by another process. Use --no-lock to override."
            ) from None

    return acquired


def release_gpu_locks(locks: list[FileLock]) -> None:
    """Release all acquired GPU locks, suppressing any errors.

    Args:
        locks: List of FileLock objects previously returned by acquire_gpu_locks.
    """
    for lock in locks:
        with contextlib.suppress(Exception):
            lock.release()
