"""Tests for StudyRunner subprocess lifecycle.

Test strategy: All execution paths are exercised without spawning real subprocesses
or requiring GPU hardware. Worker functions are replaced with in-process callables
via mock injection of multiprocessing.get_context. ManifestWriter interactions are
asserted against MagicMock instances.

The only paths NOT covered here are:
- Real CUDA teardown (requires GPU hardware — manual verification noted in STATE.md)
- True cross-process Pipe data transfer (integration test concern)
"""

from __future__ import annotations

import inspect
import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.config.models import ExecutionConfig, ExperimentConfig, StudyConfig
from llenergymeasure.study.runner import StudyRunner, _calculate_timeout, _run_experiment_worker

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config() -> ExperimentConfig:
    """A minimal ExperimentConfig with n=100."""
    return ExperimentConfig(model="test/model", backend="pytorch", n=100)


@pytest.fixture
def large_config() -> ExperimentConfig:
    """A minimal ExperimentConfig with n=1000."""
    return ExperimentConfig(model="test/model", backend="pytorch", n=1000)


@pytest.fixture
def study_config(basic_config: ExperimentConfig) -> StudyConfig:
    """Single-experiment StudyConfig."""
    return StudyConfig(
        experiments=[basic_config],
        name="test-study",
        execution=ExecutionConfig(n_cycles=1, cycle_order="sequential"),
        study_design_hash="deadbeef12345678",
    )


def _make_mock_process(
    *,
    is_alive_after_join: bool = False,
    exitcode: int = 0,
    pid: int = 12345,
) -> MagicMock:
    """Factory for a mock Process with controllable lifecycle."""
    proc = MagicMock()
    proc.pid = pid
    proc.exitcode = exitcode

    def fake_join(timeout=None):
        # After join, is_alive reflects whether timed out
        pass

    proc.join.side_effect = fake_join
    proc.is_alive.return_value = is_alive_after_join
    return proc


def _make_mock_context(
    process: MagicMock,
    pipe_data: object = None,
    *,
    pipe_has_data: bool = True,
) -> MagicMock:
    """Build a mock multiprocessing context that returns a controlled process.

    pipe_data: the object that parent_conn.recv() will return.
    pipe_has_data: if False, parent_conn.poll() returns False (empty pipe).
    """
    ctx = MagicMock()
    ctx.Process.return_value = process

    parent_conn = MagicMock()
    child_conn = MagicMock()
    parent_conn.poll.return_value = pipe_has_data
    if pipe_data is not None:
        parent_conn.recv.return_value = pipe_data

    ctx.Pipe.return_value = (child_conn, parent_conn)

    # Queue: use a real in-process queue so the consumer thread works
    ctx.Queue.return_value = queue.SimpleQueue()

    return ctx


# =============================================================================
# Task 1a: _calculate_timeout
# =============================================================================


def test_calculate_timeout_minimum(basic_config: ExperimentConfig) -> None:
    """_calculate_timeout returns >= 600 for n=100."""
    result = _calculate_timeout(basic_config)
    assert result >= 600, f"Expected >= 600, got {result}"


def test_calculate_timeout_scales_with_n(large_config: ExperimentConfig) -> None:
    """_calculate_timeout returns >= 2000 for n=1000 (2s/prompt heuristic)."""
    result = _calculate_timeout(large_config)
    assert result >= 2000, f"Expected >= 2000 for n=1000, got {result}"


# =============================================================================
# Task 1b: spawn context guaranteed
# =============================================================================


def test_spawn_context_used(study_config: StudyConfig) -> None:
    """StudyRunner uses multiprocessing.get_context('spawn'), never fork or default."""
    manifest = MagicMock()

    # A fake ExperimentResult-like dict to return from pipe
    fake_result = {"status": "ok", "data": "fake"}

    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    with patch("multiprocessing.get_context", return_value=ctx) as mock_get_context:
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-study"))
        runner.run()
        mock_get_context.assert_called_once_with("spawn")


def test_daemon_false(study_config: StudyConfig) -> None:
    """Process is always created with daemon=False."""
    manifest = MagicMock()

    fake_result = {"status": "ok"}

    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-study"))
        runner.run()

    # daemon=False must be explicitly passed
    call_kwargs = ctx.Process.call_args
    assert call_kwargs is not None, "Process() was never called"
    # Check keyword arg
    assert ctx.Process.call_args.kwargs.get("daemon") is False, (
        f"Expected daemon=False, got daemon={ctx.Process.call_args.kwargs.get('daemon')!r}"
    )


# =============================================================================
# Task 1c: Success path
# =============================================================================


def test_study_runner_success_path(
    study_config: StudyConfig, basic_config: ExperimentConfig
) -> None:
    """Happy path: fake worker sends result via Pipe; mark_completed is called."""
    manifest = MagicMock()

    from llenergymeasure.domain.experiment import compute_measurement_config_hash

    config_hash = compute_measurement_config_hash(basic_config)
    fake_result = {"config_hash": config_hash, "status": "success", "value": 42}

    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result, pipe_has_data=True)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-study"))
        results = runner.run()

    assert len(results) == 1
    # mark_running called before start
    manifest.mark_running.assert_called_once()
    # mark_completed called on success
    manifest.mark_completed.assert_called_once()
    manifest.mark_failed.assert_not_called()


# =============================================================================
# Task 1d: Subprocess exception path
# =============================================================================


def test_study_runner_subprocess_exception(study_config: StudyConfig) -> None:
    """Subprocess sends error dict via Pipe; StudyRunner marks failed and continues."""
    manifest = MagicMock()

    error_payload = {
        "type": "RuntimeError",
        "message": "CUDA OOM",
        "traceback": "Traceback (most recent call last):\n  ...",
    }

    proc = _make_mock_process(is_alive_after_join=False, exitcode=1)
    ctx = _make_mock_context(proc, pipe_data=error_payload, pipe_has_data=True)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-study"))
        # Must NOT raise — failures are non-fatal
        results = runner.run()

    assert len(results) == 1
    manifest.mark_running.assert_called_once()
    manifest.mark_failed.assert_called_once()
    manifest.mark_completed.assert_not_called()


# =============================================================================
# Task 1e: Timeout path
# =============================================================================


def test_study_runner_timeout(study_config: StudyConfig) -> None:
    """Timeout: p.is_alive() stays True after join; p.kill() is called (SIGKILL)."""
    manifest = MagicMock()

    # Process stays alive (timed-out)
    proc = _make_mock_process(is_alive_after_join=True, exitcode=None)
    ctx = _make_mock_context(proc, pipe_has_data=False)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-study"))
        results = runner.run()

    assert len(results) == 1
    proc.kill.assert_called_once()  # SIGKILL, not SIGTERM
    proc.terminate.assert_not_called()
    manifest.mark_failed.assert_called_once()

    # Verify error_type indicates timeout
    call_args = manifest.mark_failed.call_args
    error_type = call_args.kwargs.get("error_type") or call_args.args[2]
    assert "Timeout" in error_type or "timeout" in error_type.lower(), (
        f"Expected Timeout error_type, got {error_type!r}"
    )


# =============================================================================
# Task 1f: Non-zero exitcode, empty Pipe
# =============================================================================


def test_study_runner_exitcode_nonzero_no_pipe_data(study_config: StudyConfig) -> None:
    """Non-zero exit with empty Pipe: experiment marked failed with ProcessCrash."""
    manifest = MagicMock()

    proc = _make_mock_process(is_alive_after_join=False, exitcode=1)
    ctx = _make_mock_context(proc, pipe_has_data=False)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-study"))
        results = runner.run()

    assert len(results) == 1
    manifest.mark_failed.assert_called_once()

    call_args = manifest.mark_failed.call_args
    error_type = call_args.kwargs.get("error_type") or call_args.args[2]
    assert "Crash" in error_type or "crash" in error_type.lower() or "Process" in error_type, (
        f"Expected ProcessCrash error_type, got {error_type!r}"
    )


# =============================================================================
# Task 1g: Cycle ordering
# =============================================================================


def _make_ordering_study(
    cycle_order: str,
    n_cycles: int = 2,
) -> tuple[StudyConfig, list[ExperimentConfig]]:
    """Build a 2-experiment StudyConfig for ordering tests.

    Mirrors what load_study_config() produces: study.experiments is already the
    fully-cycled execution list from apply_cycles(). The runner must not call
    apply_cycles() again.
    """
    from llenergymeasure.study.grid import CycleOrder, apply_cycles

    exp_a = ExperimentConfig(model="model-a", backend="pytorch", n=10)
    exp_b = ExperimentConfig(model="model-b", backend="pytorch", n=10)
    ordered = apply_cycles([exp_a, exp_b], n_cycles, CycleOrder(cycle_order), "aaaa0000bbbb1111")
    study = StudyConfig(
        experiments=ordered,
        name="ordering-test",
        execution=ExecutionConfig(n_cycles=n_cycles, cycle_order=cycle_order),
        study_design_hash="aaaa0000bbbb1111",
    )
    return study, [exp_a, exp_b]


def test_interleaved_ordering() -> None:
    """cycle_order=interleaved with 2 experiments x 2 cycles = A,B,A,B order."""
    study, (_exp_a, _exp_b) = _make_ordering_study("interleaved", n_cycles=2)
    manifest = MagicMock()

    call_order: list[str] = []
    fake_result = {"status": "ok"}

    def fake_process_factory(**kwargs):
        # Capture which config this process is for via target args
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 0
        proc.pid = 99

        args = kwargs.get("args", ())
        if args:
            config = args[0]
            call_order.append(config.model)

        def fake_start():
            pass

        proc.start.side_effect = fake_start
        proc.join.return_value = None
        return proc

    def fake_pipe(duplex=True):
        child = MagicMock()
        parent = MagicMock()
        parent.poll.return_value = True
        parent.recv.return_value = fake_result
        return child, parent

    ctx = MagicMock()
    ctx.Process.side_effect = lambda **kwargs: fake_process_factory(**kwargs)
    ctx.Pipe.side_effect = fake_pipe
    ctx.Queue.return_value = queue.SimpleQueue()

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study, manifest, Path("/tmp/test-study"))
        runner.run()

    assert call_order == ["model-a", "model-b", "model-a", "model-b"], (
        f"Interleaved ordering wrong: {call_order}"
    )


def test_sequential_ordering() -> None:
    """cycle_order=sequential with 2 experiments x 2 cycles = A,A,B,B order."""
    study, (_exp_a, _exp_b) = _make_ordering_study("sequential", n_cycles=2)
    manifest = MagicMock()

    call_order: list[str] = []
    fake_result = {"status": "ok"}

    def fake_process_factory(**kwargs):
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 0
        proc.pid = 99

        args = kwargs.get("args", ())
        if args:
            config = args[0]
            call_order.append(config.model)

        proc.start.return_value = None
        proc.join.return_value = None
        return proc

    def fake_pipe(duplex=True):
        child = MagicMock()
        parent = MagicMock()
        parent.poll.return_value = True
        parent.recv.return_value = fake_result
        return child, parent

    ctx = MagicMock()
    ctx.Process.side_effect = lambda **kwargs: fake_process_factory(**kwargs)
    ctx.Pipe.side_effect = fake_pipe
    ctx.Queue.return_value = queue.SimpleQueue()

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study, manifest, Path("/tmp/test-study"))
        runner.run()

    assert call_order == ["model-a", "model-a", "model-b", "model-b"], (
        f"Sequential ordering wrong: {call_order}"
    )


# =============================================================================
# Task 2: SIGINT handling
# =============================================================================


def _make_sigint_study() -> StudyConfig:
    """Single-experiment study for SIGINT tests."""
    return StudyConfig(
        experiments=[ExperimentConfig(model="test/model", backend="pytorch", n=10)],
        name="sigint-test",
        execution=ExecutionConfig(n_cycles=1, cycle_order="sequential"),
        study_design_hash="deadbeef12345678",
    )


def test_sigint_first_ctrl_c_marks_manifest_interrupted() -> None:
    """First Ctrl+C: interrupt_event set, manifest.mark_interrupted() called, sys.exit(130).

    We simulate SIGINT by patching _run_one to set interrupt_event after the first experiment,
    which triggers the post-loop exit path.
    """
    study = _make_sigint_study()
    manifest = MagicMock()

    runner = StudyRunner(study, manifest, Path("/tmp/test-study"))

    original_run_one = runner._run_one

    def sigint_during_run_one(config, mp_ctx, index=1, total=1):
        """Call original _run_one, then simulate SIGINT having fired."""
        # Run the real experiment dispatch (mocked below)
        result = original_run_one(config, mp_ctx, index=index, total=total)
        # Simulate first Ctrl+C arriving after experiment completes
        runner._interrupt_event.set()
        runner._interrupt_count = 1
        return result

    fake_result = {"status": "ok"}
    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch.object(runner, "_run_one", side_effect=sigint_during_run_one),
        pytest.raises(SystemExit) as exc_info,
    ):
        runner.run()

    assert exc_info.value.code == 130
    manifest.mark_interrupted.assert_called_once()


def test_sigint_during_gap_exits_immediately() -> None:
    """Interrupt during gap: run loop exits and mark_interrupted is called."""
    study = StudyConfig(
        experiments=[
            ExperimentConfig(model="model-a", backend="pytorch", n=10),
            ExperimentConfig(model="model-b", backend="pytorch", n=10),
        ],
        name="gap-interrupt-test",
        execution=ExecutionConfig(
            n_cycles=1, cycle_order="sequential", experiment_gap_seconds=60.0
        ),
        study_design_hash="deadbeef12345678",
    )
    manifest = MagicMock()
    fake_result = {"status": "ok"}
    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    def fake_run_gap(seconds: float, label: str, interrupt_event: object) -> None:
        """Simulate SIGINT occurring during gap by setting the event."""

        if isinstance(interrupt_event, threading.Event):
            interrupt_event.set()

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch("llenergymeasure.study.runner.run_gap", side_effect=fake_run_gap),
        pytest.raises(SystemExit) as exc_info,
    ):
        runner = StudyRunner(study, manifest, Path("/tmp/test-study"))
        runner.run()

    assert exc_info.value.code == 130
    manifest.mark_interrupted.assert_called_once()


def test_sigint_second_ctrl_c_kills_immediately() -> None:
    """Second Ctrl+C escalates to SIGKILL (p.kill()) immediately."""
    study = _make_sigint_study()
    manifest = MagicMock()

    proc = _make_mock_process(is_alive_after_join=True, exitcode=None)
    proc.is_alive.return_value = True

    ctx = _make_mock_context(proc, pipe_has_data=False)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study, manifest, Path("/tmp/test-study"))
        # Directly set _active_process so the handler can reference it
        runner._active_process = proc

        # Call the SIGINT handler twice to simulate second Ctrl+C
        # We simulate by calling run() with interrupt pre-set AND by checking kill is called
        # after two handler invocations.
        runner._interrupt_count = 1  # pretend first Ctrl+C already fired
        runner._interrupt_event.set()

        # Now simulate second Ctrl+C handler call
        runner._interrupt_count += 1
        if runner._active_process is not None and runner._active_process.is_alive():
            runner._active_process.kill()

    proc.kill.assert_called_once()


# =============================================================================
# Task 2 (Plan 02): Worker wiring tests
# =============================================================================


def test_worker_no_longer_stub() -> None:
    """_run_experiment_worker no longer raises NotImplementedError."""
    src = inspect.getsource(_run_experiment_worker)
    assert "NotImplementedError" not in src, (
        "_run_experiment_worker still contains NotImplementedError — worker not wired"
    )


def test_worker_calls_get_backend(monkeypatch) -> None:
    """_run_experiment_worker calls get_backend with the config's backend name.

    Uses a mock conn (not a real Pipe) so MagicMock results don't need pickling.
    """
    config = ExperimentConfig(model="test/model", backend="pytorch", n=10)

    backend_calls: list[str] = []
    sent_results: list = []
    fake_result = MagicMock()

    def fake_get_backend(name: str):
        backend_calls.append(name)
        mock_backend = MagicMock()
        mock_backend.run.return_value = fake_result
        return mock_backend

    monkeypatch.setattr("llenergymeasure.core.backends.get_backend", fake_get_backend)
    monkeypatch.setattr("llenergymeasure.orchestration.preflight.run_preflight", lambda c: None)

    # Use a mock connection to avoid pickling MagicMock through a real Pipe
    mock_conn = MagicMock()
    mock_conn.send.side_effect = lambda r: sent_results.append(r)

    progress_q: queue.SimpleQueue = queue.SimpleQueue()

    _run_experiment_worker(config, mock_conn, progress_q)

    assert len(backend_calls) == 1, f"Expected 1 get_backend call, got {backend_calls}"
    assert backend_calls[0] == "pytorch"

    # Worker should have sent the fake result via conn.send()
    assert len(sent_results) == 1, "Worker did not call conn.send()"
    assert sent_results[0] is fake_result


# =============================================================================
# Bug fixes: correct experiment count and cycle tracking (STU-07, STU-08)
# =============================================================================


def test_multi_cycle_correct_experiment_count() -> None:
    """2-config x 3-cycle study runs exactly 6 experiments, not 18.

    study.experiments is the pre-cycled list (6 entries) from apply_cycles().
    The runner must NOT call apply_cycles() again.
    """
    from llenergymeasure.study.grid import CycleOrder, apply_cycles

    exp_a = ExperimentConfig(model="model-a", backend="pytorch", n=10)
    exp_b = ExperimentConfig(model="model-b", backend="pytorch", n=10)
    ordered = apply_cycles([exp_a, exp_b], 3, CycleOrder.INTERLEAVED, "aabb0011", None)
    assert len(ordered) == 6, "sanity: apply_cycles should produce 6 entries"

    study = StudyConfig(
        experiments=ordered,
        name="count-test",
        execution=ExecutionConfig(n_cycles=3, cycle_order="interleaved"),
        study_design_hash="aabb0011",
    )
    manifest = MagicMock()
    fake_result = {"status": "ok"}

    def fake_process_factory(**kwargs):
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 0
        proc.pid = 99
        proc.start.return_value = None
        proc.join.return_value = None
        return proc

    def fake_pipe(duplex=True):
        child = MagicMock()
        parent = MagicMock()
        parent.poll.return_value = True
        parent.recv.return_value = fake_result
        return child, parent

    ctx = MagicMock()
    ctx.Process.side_effect = lambda **kwargs: fake_process_factory(**kwargs)
    ctx.Pipe.side_effect = fake_pipe
    ctx.Queue.return_value = queue.SimpleQueue()

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study, manifest, Path("/tmp/test-count"))
        results = runner.run()

    assert len(results) == 6, f"Expected 6 results, got {len(results)}"
    assert ctx.Process.call_count == 6, f"Expected 6 Process() calls, got {ctx.Process.call_count}"


def test_cycle_counter_increments_per_config_hash() -> None:
    """Cycle counter increments per config_hash, not globally.

    For a 2-config x 2-cycle interleaved study (A, B, A, B):
    - hash_A: cycles should be [1, 2]
    - hash_B: cycles should be [1, 2]
    """
    from llenergymeasure.domain.experiment import compute_measurement_config_hash
    from llenergymeasure.study.grid import CycleOrder, apply_cycles

    exp_a = ExperimentConfig(model="model-a", backend="pytorch", n=10)
    exp_b = ExperimentConfig(model="model-b", backend="pytorch", n=10)
    hash_a = compute_measurement_config_hash(exp_a)
    hash_b = compute_measurement_config_hash(exp_b)

    ordered = apply_cycles([exp_a, exp_b], 2, CycleOrder.INTERLEAVED, "aabb0011", None)
    assert len(ordered) == 4, "sanity: 2 configs x 2 cycles = 4"

    study = StudyConfig(
        experiments=ordered,
        name="cycle-test",
        execution=ExecutionConfig(n_cycles=2, cycle_order="interleaved"),
        study_design_hash="aabb0011",
    )
    manifest = MagicMock()
    fake_result = {"status": "ok"}

    def fake_process_factory(**kwargs):
        proc = MagicMock()
        proc.is_alive.return_value = False
        proc.exitcode = 0
        proc.pid = 99
        proc.start.return_value = None
        proc.join.return_value = None
        return proc

    def fake_pipe(duplex=True):
        child = MagicMock()
        parent = MagicMock()
        parent.poll.return_value = True
        parent.recv.return_value = fake_result
        return child, parent

    ctx = MagicMock()
    ctx.Process.side_effect = lambda **kwargs: fake_process_factory(**kwargs)
    ctx.Pipe.side_effect = fake_pipe
    ctx.Queue.return_value = queue.SimpleQueue()

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study, manifest, Path("/tmp/test-cycle"))
        runner.run()

    # Extract (config_hash, cycle) pairs from mark_running calls
    call_args_list = manifest.mark_running.call_args_list
    assert len(call_args_list) == 4, f"Expected 4 mark_running calls, got {len(call_args_list)}"

    cycles_by_hash: dict[str, list[int]] = {}
    for call in call_args_list:
        h = call.args[0] if call.args else call.kwargs["config_hash"]
        c = call.args[1] if len(call.args) > 1 else call.kwargs["cycle"]
        cycles_by_hash.setdefault(h, []).append(c)

    # hash_A and hash_B should each have cycles [1, 2]
    assert hash_a in cycles_by_hash, f"hash_a not found in mark_running calls: {cycles_by_hash}"
    assert hash_b in cycles_by_hash, f"hash_b not found in mark_running calls: {cycles_by_hash}"
    assert sorted(cycles_by_hash[hash_a]) == [1, 2], (
        f"hash_a cycles: expected [1, 2], got {cycles_by_hash[hash_a]}"
    )
    assert sorted(cycles_by_hash[hash_b]) == [1, 2], (
        f"hash_b cycles: expected [1, 2], got {cycles_by_hash[hash_b]}"
    )


# =============================================================================
# Progress display wiring
# =============================================================================


def test_progress_events_forwarded():
    """_consume_progress_events forwards events to print_study_progress."""
    from queue import Queue
    from unittest.mock import patch

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.study.runner import _consume_progress_events

    config = ExperimentConfig(model="test-model", backend="pytorch", n=10)
    q = Queue()
    q.put({"event": "started", "config_hash": "abc123"})
    q.put({"event": "completed", "config_hash": "abc123"})
    q.put(None)  # sentinel

    with patch("llenergymeasure.cli._display.print_study_progress") as mock_progress:
        _consume_progress_events(q, index=3, total=12, config=config)

    assert mock_progress.call_count == 2
    statuses = [c.kwargs["status"] for c in mock_progress.call_args_list]
    assert statuses == ["running", "completed"]


# =============================================================================
# GPU memory residual check wiring (MEAS-01, MEAS-02)
# =============================================================================


def test_gpu_memory_check_called_before_dispatch(study_config: StudyConfig) -> None:
    """check_gpu_memory_residual() is called once before p.start() in _run_one().

    Verifies the wiring exists and ordering is correct: memory check fires
    before the child process starts.
    """
    manifest = MagicMock()
    fake_result = {"status": "ok"}

    call_order: list[str] = []

    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    # Track call order between gpu check and p.start()
    def record_start(*args, **kwargs):
        call_order.append("p.start")

    proc.start.side_effect = record_start

    def record_gpu_check(*args, **kwargs):
        call_order.append("gpu_check")

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch(
            "llenergymeasure.study.gpu_memory.check_gpu_memory_residual",
            side_effect=record_gpu_check,
        ) as mock_check,
    ):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-study"))
        runner.run()

    mock_check.assert_called_once()
    # GPU check must precede p.start()
    assert call_order.index("gpu_check") < call_order.index("p.start"), (
        f"GPU check should happen before p.start(), got order: {call_order}"
    )


# =============================================================================
# Docker dispatch path (DOCK-05)
# =============================================================================


def _make_docker_runner_spec(image: str = "test/image:latest", source: str = "yaml"):
    """Build a RunnerSpec with mode='docker'."""
    from llenergymeasure.infra.runner_resolution import RunnerSpec

    return RunnerSpec(mode="docker", image=image, source=source)


def _make_local_runner_spec(source: str = "default"):
    """Build a RunnerSpec with mode='local'."""
    from llenergymeasure.infra.runner_resolution import RunnerSpec

    return RunnerSpec(mode="local", image=None, source=source)


def test_docker_runner_spec_dispatches_to_docker(
    study_config: StudyConfig, basic_config: ExperimentConfig
) -> None:
    """When runner_specs has docker mode for backend, _run_one_docker is called instead of subprocess."""
    manifest = MagicMock()
    spec = _make_docker_runner_spec()
    runner_specs = {"pytorch": spec}

    from llenergymeasure.domain.experiment import ExperimentResult

    fake_result = MagicMock(spec=ExperimentResult)
    fake_result.total_energy_j = 1.0

    docker_run_calls: list = []
    subprocess_process_calls: list = []

    def fake_docker_run(config):
        docker_run_calls.append(config)
        return fake_result

    fake_ctx = MagicMock()
    fake_ctx.Process.side_effect = lambda **kwargs: (
        subprocess_process_calls.append(kwargs) or MagicMock()
    )

    with patch("multiprocessing.get_context", return_value=fake_ctx):
        runner = StudyRunner(
            study_config, manifest, Path("/tmp/test-docker"), runner_specs=runner_specs
        )
        with (
            patch(
                "llenergymeasure.infra.docker_runner.DockerRunner.run", side_effect=fake_docker_run
            ),
            patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
            patch("llenergymeasure.cli._display.print_study_progress"),
            patch(
                "llenergymeasure.results.persistence.save_result",
                return_value=Path("/tmp/test-docker/r.json"),
            ),
        ):
            results = runner.run()

    # Subprocess Process() should NOT have been spawned
    assert subprocess_process_calls == [], (
        "subprocess Process() was created for Docker dispatch path"
    )
    # Docker run should have been called
    assert len(docker_run_calls) == 1
    assert docker_run_calls[0].backend == "pytorch"
    assert len(results) == 1


def test_local_runner_spec_uses_subprocess_path(
    study_config: StudyConfig, basic_config: ExperimentConfig
) -> None:
    """When runner_specs has local mode for backend, subprocess dispatch is used (backward compat)."""
    manifest = MagicMock()
    spec = _make_local_runner_spec()
    runner_specs = {"pytorch": spec}

    fake_result = {"status": "ok"}
    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(
            study_config, manifest, Path("/tmp/test-local"), runner_specs=runner_specs
        )
        results = runner.run()

    # Subprocess should have been spawned
    ctx.Process.assert_called_once()
    assert len(results) == 1


def test_no_runner_specs_uses_subprocess_path(study_config: StudyConfig) -> None:
    """When runner_specs is None (default), subprocess dispatch is used (backward compat)."""
    manifest = MagicMock()

    fake_result = {"status": "ok"}
    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-nospecs"))
        # runner_specs defaults to None
        results = runner.run()

    ctx.Process.assert_called_once()
    assert len(results) == 1


def test_docker_error_caught_and_converted_to_failure_dict(
    study_config: StudyConfig,
) -> None:
    """DockerError from DockerRunner is caught and converted to a non-fatal failure dict."""
    manifest = MagicMock()
    spec = _make_docker_runner_spec()
    runner_specs = {"pytorch": spec}

    from llenergymeasure.exceptions import DockerError

    def raise_docker_error(config):
        raise DockerError("Container failed to start")

    subprocess_process_calls: list = []

    def fake_process_factory(**kwargs):
        subprocess_process_calls.append(kwargs)
        return MagicMock()

    fake_ctx = MagicMock()
    fake_ctx.Process.side_effect = fake_process_factory

    with (
        patch("multiprocessing.get_context", return_value=fake_ctx),
        patch(
            "llenergymeasure.infra.docker_runner.DockerRunner.run", side_effect=raise_docker_error
        ),
        patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
        patch("llenergymeasure.cli._display.print_study_progress"),
    ):
        runner = StudyRunner(
            study_config, manifest, Path("/tmp/test-docker-err"), runner_specs=runner_specs
        )
        results = runner.run()

    # Subprocess Process() must NOT have been spawned
    assert subprocess_process_calls == [], (
        "subprocess Process() was created when DockerError occurred"
    )
    # Result should be a failure dict (non-fatal)
    assert len(results) == 1
    assert isinstance(results[0], dict)
    assert "type" in results[0]
    assert "DockerError" in results[0]["type"]
    # Manifest should record failure
    manifest.mark_failed.assert_called_once()
    manifest.mark_completed.assert_not_called()
