#!/usr/bin/env python3
"""PoC-RT-2 — subprocess-crash JSONL state.

Adversarial findings B2/B3: when a subprocess SIGKILLs mid-write to a JSONL
file, what's on disk? Decides Path P (per-experiment JSONL) vs Path Q (shared
cache) for M3.

B3 argued per-experiment JSONL (P) may be overcorrection from multi-process
concerns that don't apply — O_APPEND is POSIX-atomic for writes ≤ PIPE_BUF
(typically 4096 bytes on Linux). This probe tests empirically.

Test matrix:
  Record sizes: 200B (typical warning), 1KB (medium log), 5KB (big traceback),
                10KB (very big traceback). 5KB and 10KB cross the PIPE_BUF
                boundary.
  Writers:
    (W1) buffered + flush + fsync (preview pattern)
    (W2) buffered only (no flush)
    (W3) os.write(fd, ...) with O_APPEND — single-syscall per record
  Scenarios:
    (S1) Single-writer SIGKILL partway through
    (S2) Multi-writer concurrent append (no kills, but stress-test interleaving)
    (S3) Mid-write SIGKILL — worker writes record in two halves, SIGKILL
         between halves, checks for partial-record corruption

Reference: .claude/plans/m3-design-discussion-2026-04-24.md

Written by autonomous overnight PoC run, 2026-04-24.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

PIPE_BUF = os.pathconf("/", "PC_PIPE_BUF")


# -----------------------------------------------------------------------------
# Workers — module-level so multiprocessing spawn can pickle them
# -----------------------------------------------------------------------------


def _record_for(worker_id: int, i: int, size: int) -> str:
    """Build a JSON record of approximately `size` bytes (+/- small overhead)."""
    meta = {"worker": worker_id, "i": i}
    overhead = len(json.dumps({**meta, "d": ""})) + 2  # closing + newline
    payload = "x" * max(1, size - overhead)
    record = {**meta, "d": payload}
    return json.dumps(record) + "\n"


def worker_w1_buffered_fsync(
    path_str: str, worker_id: int, size: int, total: int, delay_ms: float
) -> None:
    """(W1) open('a') + write + flush + fsync (preview pattern)."""
    with open(path_str, "a", encoding="utf-8") as fh:
        for i in range(total):
            line = _record_for(worker_id, i, size)
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)


def worker_w2_buffered_only(
    path_str: str, worker_id: int, size: int, total: int, delay_ms: float
) -> None:
    """(W2) open('a') + write only — relies on Python buffer."""
    fh = open(path_str, "a", encoding="utf-8")
    for i in range(total):
        line = _record_for(worker_id, i, size)
        fh.write(line)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
    # Don't close — we want to see what's in Python buffer at crash time


def worker_w3_oswrite(
    path_str: str, worker_id: int, size: int, total: int, delay_ms: float
) -> None:
    """(W3) os.write(fd, ...) with O_APPEND — single syscall per record, no Python buffering."""
    fd = os.open(path_str, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    for i in range(total):
        line = _record_for(worker_id, i, size).encode("utf-8")
        os.write(fd, line)
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)


WORKERS: dict[str, Callable] = {
    "W1_buffered_fsync": worker_w1_buffered_fsync,
    "W2_buffered_only": worker_w2_buffered_only,
    "W3_oswrite": worker_w3_oswrite,
}


def worker_midwrite_split(
    path_str: str, worker_id: int, size: int, total: int, delay_ms: float
) -> None:
    """Writes each record in two halves with a pause between (mid-write SIGKILL target)."""
    fd = os.open(path_str, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    for i in range(total):
        line = _record_for(worker_id, i, size).encode("utf-8")
        half = len(line) // 2
        os.write(fd, line[:half])
        time.sleep(delay_ms / 1000.0)  # SIGKILL can land here
        os.write(fd, line[half:])


# -----------------------------------------------------------------------------
# File inspection
# -----------------------------------------------------------------------------


@dataclass
class FileStats:
    total_bytes: int
    complete: int = 0  # lines that parse as JSON
    torn: int = 0  # non-empty trailing segment that doesn't parse
    torn_midfile: int = 0  # non-final lines that don't parse
    interleaved: int = 0  # lines where JSON parses but values look garbled
    trailing_no_newline: bool = False
    per_worker: dict[int, int] = field(default_factory=dict)


def inspect(path: Path) -> FileStats:
    raw = path.read_bytes()
    stats = FileStats(total_bytes=len(raw))
    if not raw:
        return stats
    has_trailing_newline = raw.endswith(b"\n")
    stats.trailing_no_newline = not has_trailing_newline

    text = raw.decode("utf-8", errors="replace")
    segments = text.split("\n")
    # If has_trailing_newline, final element is "" — discard
    # If not, final element is the torn tail
    for idx, seg in enumerate(segments):
        is_last = idx == len(segments) - 1
        if seg == "":
            continue
        try:
            obj = json.loads(seg)
            stats.complete += 1
            if isinstance(obj, dict) and "worker" in obj:
                wid = obj["worker"]
                stats.per_worker[wid] = stats.per_worker.get(wid, 0) + 1
        except json.JSONDecodeError:
            if is_last and not has_trailing_newline:
                stats.torn += 1
            else:
                stats.torn_midfile += 1
    return stats


# -----------------------------------------------------------------------------
# Scenarios
# -----------------------------------------------------------------------------


def scenario_s1_single_sigkill(
    worker_fn: Callable,
    size: int,
    total: int,
    kill_after_ms: int,
    delay_ms: float,
) -> FileStats:
    """Spawn single worker; SIGKILL after kill_after_ms."""
    ctx = multiprocessing.get_context("spawn")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        path = Path(tf.name)
    path.write_bytes(b"")

    p = ctx.Process(target=worker_fn, args=(str(path), 1, size, total, delay_ms))
    p.start()
    time.sleep(kill_after_ms / 1000.0)
    if p.is_alive():
        os.kill(p.pid, signal.SIGKILL)
    p.join(timeout=5)

    stats = inspect(path)
    path.unlink()
    return stats


def scenario_s2_multi_concurrent(
    worker_fn: Callable,
    size: int,
    total_per_worker: int,
    n_writers: int,
    delay_ms: float,
) -> FileStats:
    """Spawn n_writers concurrent workers; no kills; measure interleaving."""
    ctx = multiprocessing.get_context("spawn")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        path = Path(tf.name)
    path.write_bytes(b"")

    procs = []
    for wid in range(n_writers):
        p = ctx.Process(target=worker_fn, args=(str(path), wid, size, total_per_worker, delay_ms))
        p.start()
        procs.append(p)

    for p in procs:
        p.join(timeout=30)

    stats = inspect(path)
    path.unlink()
    return stats


def scenario_s3_midwrite_sigkill(size: int, total: int, kill_after_ms: int) -> FileStats:
    """Mid-write SIGKILL — worker writes each record in two halves with pause between."""
    ctx = multiprocessing.get_context("spawn")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        path = Path(tf.name)
    path.write_bytes(b"")

    # Delay between halves large enough that kill can land mid-record
    mid_delay_ms = 5.0
    p = ctx.Process(target=worker_midwrite_split, args=(str(path), 1, size, total, mid_delay_ms))
    p.start()
    time.sleep(kill_after_ms / 1000.0)
    if p.is_alive():
        os.kill(p.pid, signal.SIGKILL)
    p.join(timeout=5)

    stats = inspect(path)
    path.unlink()
    return stats


# -----------------------------------------------------------------------------
# Parent-side sentinel feasibility (adversarial B2)
# -----------------------------------------------------------------------------


def scenario_parent_sentinel(
    worker_fn: Callable, size: int, total: int, kill_after_ms: int
) -> dict:
    """Simulate: child crashes, parent writes sentinel. Confirm feasibility."""
    ctx = multiprocessing.get_context("spawn")
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tf:
        path = Path(tf.name)
    path.write_bytes(b"")

    p = ctx.Process(target=worker_fn, args=(str(path), 1, size, total, 1.0))
    p.start()
    time.sleep(kill_after_ms / 1000.0)
    if p.is_alive():
        os.kill(p.pid, signal.SIGKILL)
    p.join(timeout=5)
    exit_code = p.exitcode

    # Parent now writes the sentinel
    fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    sentinel = {
        "outcome": "subprocess_died",
        "exit_code": exit_code,
        "reason": "SIGKILL simulated",
        "worker": 1,
    }
    os.write(fd, (json.dumps(sentinel) + "\n").encode("utf-8"))
    os.close(fd)

    stats = inspect(path)
    # Check the last line is the sentinel
    raw = path.read_bytes().decode("utf-8", errors="replace")
    last_line = raw.rstrip("\n").split("\n")[-1]
    sentinel_ok = False
    try:
        parsed = json.loads(last_line)
        if parsed.get("outcome") == "subprocess_died":
            sentinel_ok = True
    except json.JSONDecodeError:
        sentinel_ok = False

    path.unlink()
    return {"exit_code": exit_code, "sentinel_ok": sentinel_ok, "stats": stats}


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------


SIZES = [200, 1024, 5120, 10240]
REPEATS_S1 = 20
REPEATS_S2 = 15
REPEATS_S3 = 30


def run_s1(worker_name: str, worker_fn: Callable, size: int, repeats: int) -> dict:
    """Aggregate S1 over `repeats` runs.

    Tuned for spawn-context startup overhead (~300-500ms on this host).
    kill_after_ms=1500 + delay_ms=2 + total=500 means:
      - spawn + import takes ~300ms
      - first write at ~300ms
      - kill at 1500ms = ~600 records written by then
    Sufficient to exercise the writer and catch any SIGKILL-mid-write cases.
    """
    torn_total = 0
    torn_midfile_total = 0
    complete_runs = []
    no_writes_runs = 0
    for _ in range(repeats):
        stats = scenario_s1_single_sigkill(
            worker_fn, size=size, total=3000, kill_after_ms=1500, delay_ms=5.0
        )
        torn_total += stats.torn
        torn_midfile_total += stats.torn_midfile
        complete_runs.append(stats.complete)
        if stats.complete == 0 and stats.torn == 0:
            no_writes_runs += 1
    return {
        "worker": worker_name,
        "size": size,
        "repeats": repeats,
        "torn_total": torn_total,
        "torn_midfile_total": torn_midfile_total,
        "avg_complete": sum(complete_runs) / len(complete_runs),
        "min_complete": min(complete_runs),
        "max_complete": max(complete_runs),
        "no_writes_runs": no_writes_runs,
    }


def run_s2(worker_name: str, worker_fn: Callable, size: int, repeats: int) -> dict:
    torn_total = 0
    torn_midfile_total = 0
    complete_runs = []
    per_worker_counts = []
    for _ in range(repeats):
        stats = scenario_s2_multi_concurrent(
            worker_fn, size=size, total_per_worker=50, n_writers=4, delay_ms=0.0
        )
        torn_total += stats.torn
        torn_midfile_total += stats.torn_midfile
        complete_runs.append(stats.complete)
        per_worker_counts.append(stats.per_worker)
    return {
        "worker": worker_name,
        "size": size,
        "repeats": repeats,
        "torn_total": torn_total,
        "torn_midfile_total": torn_midfile_total,
        "avg_complete": sum(complete_runs) / len(complete_runs),
        "expected_per_run": 4 * 50,
    }


def run_s3(size: int, repeats: int) -> dict:
    torn_total = 0
    torn_midfile_total = 0
    complete_runs = []
    for _ in range(repeats):
        stats = scenario_s3_midwrite_sigkill(size=size, total=100, kill_after_ms=50)
        torn_total += stats.torn
        torn_midfile_total += stats.torn_midfile
        complete_runs.append(stats.complete)
    return {
        "size": size,
        "repeats": repeats,
        "torn_total": torn_total,
        "torn_midfile_total": torn_midfile_total,
        "avg_complete": sum(complete_runs) / len(complete_runs),
    }


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"PIPE_BUF: {PIPE_BUF} bytes")
    print("PoC-RT-2 — subprocess-crash JSONL state\n")

    # Scenario S1: single-writer SIGKILL
    print("=" * 80)
    print(f"Scenario S1: single-writer SIGKILL mid-run ({REPEATS_S1} repeats per cell)")
    print("=" * 80)
    print(
        f"{'Writer':22s} {'Size':>16s} {'TearTail':>9s} {'TearMid':>8s} {'AvgDone':>8s} {'NoWrite':>8s}"
    )
    s1_results = []
    for worker_name, worker_fn in WORKERS.items():
        for size in SIZES:
            r = run_s1(worker_name, worker_fn, size, REPEATS_S1)
            s1_results.append(r)
            xpipe = " (>PIPE_BUF)" if size > PIPE_BUF else ""
            size_str = f"{size}B{xpipe}"
            print(
                f"{worker_name:22s} {size_str:>16s} {r['torn_total']:>9d} {r['torn_midfile_total']:>8d} {r['avg_complete']:>8.1f} {r['no_writes_runs']:>8d}"
            )

    # Scenario S2: multi-writer concurrent
    print()
    print("=" * 72)
    print("Scenario S2: multi-writer concurrent (4 writers × 50 records, 15 repeats)")
    print("=" * 72)
    print(
        f"{'Writer':22s} {'Size':>6s} {'Torn(tail)':>11s} {'Torn(mid)':>10s} {'Avg complete':>13s} {'Expected':>9s}"
    )
    s2_results = []
    for worker_name, worker_fn in WORKERS.items():
        for size in SIZES:
            r = run_s2(worker_name, worker_fn, size, REPEATS_S2)
            s2_results.append(r)
            xpipe = " (>PIPE_BUF)" if size > PIPE_BUF else ""
            print(
                f"{worker_name:22s} {size:>5d}B{xpipe:<12s} {r['torn_total']:>11d} {r['torn_midfile_total']:>10d} {r['avg_complete']:>13.1f} {r['expected_per_run']:>9d}"
            )

    # Scenario S3: mid-write SIGKILL
    print()
    print("=" * 72)
    print("Scenario S3: mid-write SIGKILL (os.write in 2 halves, 30 repeats)")
    print("=" * 72)
    print(f"{'Size':>10s} {'Torn(tail)':>11s} {'Torn(mid)':>10s} {'Avg complete':>13s}")
    s3_results = []
    for size in SIZES:
        r = run_s3(size, REPEATS_S3)
        s3_results.append(r)
        xpipe = " (>PIPE_BUF)" if size > PIPE_BUF else ""
        print(
            f"{size:>5d}B{xpipe:<5s} {r['torn_total']:>11d} {r['torn_midfile_total']:>10d} {r['avg_complete']:>13.1f}"
        )

    # Parent-sentinel feasibility
    print()
    print("=" * 72)
    print("Parent-side sentinel feasibility (adversarial B2)")
    print("=" * 72)
    r_sentinel = scenario_parent_sentinel(worker_w3_oswrite, size=200, total=50, kill_after_ms=20)
    print(f"Child exit_code: {r_sentinel['exit_code']}")
    print(f"Parent successfully appended sentinel: {r_sentinel['sentinel_ok']}")
    print(f"Final file state: {r_sentinel['stats']}")

    # Summary
    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    any_torn = any(
        r["torn_total"] + r["torn_midfile_total"] > 0 for r in s1_results + s2_results + s3_results
    )
    if any_torn:
        print("! Torn records detected in at least one cell — inspect raw table above.")
    else:
        print("✓ No torn records in any cell.")
    for r in s1_results:
        if r["torn_total"] + r["torn_midfile_total"] > 0:
            print(
                f"  S1 {r['worker']} {r['size']}B: {r['torn_total']} tail-torn, {r['torn_midfile_total']} mid-file torn"
            )
    for r in s2_results:
        if r["torn_total"] + r["torn_midfile_total"] > 0:
            print(
                f"  S2 {r['worker']} {r['size']}B: {r['torn_total']} tail-torn, {r['torn_midfile_total']} mid-file torn"
            )
    for r in s3_results:
        if r["torn_total"] + r["torn_midfile_total"] > 0:
            print(
                f"  S3 {r['size']}B: {r['torn_total']} tail-torn, {r['torn_midfile_total']} mid-file torn"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
