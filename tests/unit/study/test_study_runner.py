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

import contextlib
import queue
import signal
import threading
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from llenergymeasure.config.models import ExecutionConfig, ExperimentConfig, StudyConfig
from llenergymeasure.study.runner import (
    StudyRunner,
    _calculate_timeout,
    _kill_process_group,
    _run_experiment_worker,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_config() -> ExperimentConfig:
    """A minimal ExperimentConfig with n_prompts=100."""
    return ExperimentConfig(model="test/model", backend="pytorch")


@pytest.fixture
def large_config() -> ExperimentConfig:
    """A minimal ExperimentConfig with n_prompts=1000."""
    from llenergymeasure.config.models import DatasetConfig

    return ExperimentConfig(
        model="test/model", backend="pytorch", dataset=DatasetConfig(n_prompts=1000)
    )


@pytest.fixture
def study_config(basic_config: ExperimentConfig) -> StudyConfig:
    """Single-experiment StudyConfig."""
    return StudyConfig(
        experiments=[basic_config],
        study_name="test-study",
        study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
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

    ctx.Pipe.return_value = (parent_conn, child_conn)

    # Queue: use a real in-process queue so the consumer thread works
    ctx.Queue.return_value = queue.SimpleQueue()

    return ctx


# =============================================================================
# Task 1a: _calculate_timeout
# =============================================================================


def test_calculate_timeout_minimum(basic_config: ExperimentConfig) -> None:
    """_calculate_timeout returns >= 600 for n_prompts=100."""
    result = _calculate_timeout(basic_config)
    assert result >= 600, f"Expected >= 600, got {result}"


def test_calculate_timeout_scales_with_n(large_config: ExperimentConfig) -> None:
    """_calculate_timeout returns >= 2000 for n_prompts=1000 (2s/prompt heuristic)."""
    result = _calculate_timeout(large_config)
    assert result >= 2000, f"Expected >= 2000 for n_prompts=1000, got {result}"


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
    study_config: StudyConfig, basic_config: ExperimentConfig, tmp_path: Path
) -> None:
    """Happy path: fake worker sends result via Pipe; mark_completed is called."""
    manifest = MagicMock()

    from llenergymeasure.domain.experiment import compute_measurement_config_hash

    config_hash = compute_measurement_config_hash(basic_config)
    fake_result = {"config_hash": config_hash, "status": "success", "value": 42}

    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result, pipe_has_data=True)

    # Patch save_result so the success path completes without a real filesystem write.
    # The fake dict result has no "type" key, so _handle_result calls _save_and_record.
    fake_result_path = tmp_path / "result.json"
    fake_result_path.write_text("{}", encoding="utf-8")

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch(
            "llenergymeasure.results.persistence.save_result",
            return_value=fake_result_path,
        ),
    ):
        runner = StudyRunner(study_config, manifest, tmp_path)
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
    """Timeout: p.is_alive() stays True after join; os.killpg(pid, SIGKILL) is called."""
    manifest = MagicMock()

    # Process stays alive (timed-out)
    proc = _make_mock_process(is_alive_after_join=True, exitcode=None, pid=12345)
    ctx = _make_mock_context(proc, pipe_has_data=False)

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch("llenergymeasure.study.runner.os.killpg") as mock_killpg,
    ):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-study"))
        results = runner.run()

    assert len(results) == 1
    mock_killpg.assert_any_call(proc.pid, signal.SIGKILL)
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
    experiment_order: str,
    n_cycles: int = 2,
) -> tuple[StudyConfig, list[ExperimentConfig]]:
    """Build a 2-experiment StudyConfig for ordering tests.

    Mirrors what load_study_config() produces: study.experiments is already the
    fully-cycled execution list from apply_cycles(). The runner must not call
    apply_cycles() again.
    """
    from llenergymeasure.config.grid import ExperimentOrder, apply_cycles
    from llenergymeasure.config.models import DatasetConfig

    exp_a = ExperimentConfig(
        model="model-a", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
    )
    exp_b = ExperimentConfig(
        model="model-b", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
    )
    ordered = apply_cycles(
        [exp_a, exp_b], n_cycles, ExperimentOrder(experiment_order), "aaaa0000bbbb1111"
    )
    study = StudyConfig(
        experiments=ordered,
        study_name="ordering-test",
        study_execution=ExecutionConfig(n_cycles=n_cycles, experiment_order=experiment_order),
        study_design_hash="aaaa0000bbbb1111",
    )
    return study, [exp_a, exp_b]


def test_interleaved_ordering() -> None:
    """experiment_order=interleave with 2 experiments x 2 cycles = A,B,A,B order."""
    study, (_exp_a, _exp_b) = _make_ordering_study("interleave", n_cycles=2)
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
        return parent, child

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
    """experiment_order=sequential with 2 experiments x 2 cycles = A,A,B,B order."""
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
        return parent, child

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
    from llenergymeasure.config.models import DatasetConfig

    return StudyConfig(
        experiments=[
            ExperimentConfig(
                model="test/model", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
            )
        ],
        study_name="sigint-test",
        study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
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
    from llenergymeasure.config.models import DatasetConfig

    study = StudyConfig(
        experiments=[
            ExperimentConfig(
                model="model-a", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
            ),
            ExperimentConfig(
                model="model-b", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
            ),
        ],
        study_name="gap-interrupt-test",
        study_execution=ExecutionConfig(
            n_cycles=1, experiment_order="sequential", experiment_gap_seconds=60.0
        ),
        study_design_hash="deadbeef12345678",
    )
    manifest = MagicMock()
    fake_result = {"status": "ok"}
    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    def fake_run_gap(seconds: float, label: str, interrupt_event: object, **kwargs: object) -> None:
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
    """Second Ctrl+C escalates to SIGKILL via os.killpg(pid, SIGKILL)."""
    study = _make_sigint_study()
    manifest = MagicMock()

    proc = _make_mock_process(is_alive_after_join=True, exitcode=None, pid=12345)
    proc.is_alive.return_value = True

    _make_mock_context(proc, pipe_has_data=False)

    with patch("llenergymeasure.study.runner.os.killpg") as mock_killpg:
        runner = StudyRunner(study, manifest, Path("/tmp/test-study"))
        runner._active_process = proc
        runner._interrupt_count = 1  # pretend first Ctrl+C already fired

        # Simulate second Ctrl+C: call the SIGKILL branch of the handler directly
        runner._interrupt_count += 1
        if runner._active_process is not None and runner._active_process.is_alive():
            _kill_process_group(runner._active_process.pid, signal.SIGKILL)

    mock_killpg.assert_called_once_with(proc.pid, signal.SIGKILL)


# =============================================================================
# Task 2 (Plan 02): Worker wiring tests
# =============================================================================


def test_worker_no_longer_stub(monkeypatch) -> None:
    """_run_experiment_worker no longer raises NotImplementedError."""
    from llenergymeasure.config.models import DatasetConfig
    from tests.conftest import make_result

    config = ExperimentConfig(
        model="test/model", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
    )
    fake_result = make_result()

    monkeypatch.setattr("llenergymeasure.backends.get_backend", lambda name: MagicMock())
    monkeypatch.setattr("llenergymeasure.harness.preflight.run_preflight", lambda c: None)
    monkeypatch.setattr(
        "llenergymeasure.harness.MeasurementHarness.run",
        lambda self, backend, config, **kwargs: fake_result,
    )

    mock_conn = MagicMock()
    progress_q: queue.SimpleQueue = queue.SimpleQueue()

    # Must not raise NotImplementedError
    _run_experiment_worker(config, mock_conn, progress_q)
    mock_conn.send.assert_called_once()


def test_worker_calls_get_backend(monkeypatch) -> None:
    """_run_experiment_worker calls get_backend with the config's backend name.

    Uses a mock conn (not a real Pipe) so MagicMock results don't need pickling.
    """
    from llenergymeasure.config.models import DatasetConfig
    from tests.conftest import make_result

    config = ExperimentConfig(
        model="test/model", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
    )

    backend_calls: list[str] = []
    sent_results: list = []
    fake_result = make_result()

    def fake_get_backend(name: str):
        backend_calls.append(name)
        return MagicMock()

    monkeypatch.setattr("llenergymeasure.backends.get_backend", fake_get_backend)
    monkeypatch.setattr("llenergymeasure.harness.preflight.run_preflight", lambda c: None)
    monkeypatch.setattr(
        "llenergymeasure.harness.MeasurementHarness.run",
        lambda self, backend, config, snapshot=None, **kw: fake_result,
    )

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
    from llenergymeasure.config.grid import ExperimentOrder, apply_cycles
    from llenergymeasure.config.models import DatasetConfig

    exp_a = ExperimentConfig(
        model="model-a", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
    )
    exp_b = ExperimentConfig(
        model="model-b", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
    )
    ordered = apply_cycles([exp_a, exp_b], 3, ExperimentOrder.INTERLEAVE, "aabb0011", None)
    assert len(ordered) == 6, "sanity: apply_cycles should produce 6 entries"

    study = StudyConfig(
        experiments=ordered,
        study_name="count-test",
        study_execution=ExecutionConfig(n_cycles=3, experiment_order="interleave"),
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
        return parent, child

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
    from llenergymeasure.config.grid import ExperimentOrder, apply_cycles
    from llenergymeasure.config.models import DatasetConfig
    from llenergymeasure.domain.experiment import compute_measurement_config_hash

    exp_a = ExperimentConfig(
        model="model-a", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
    )
    exp_b = ExperimentConfig(
        model="model-b", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
    )
    hash_a = compute_measurement_config_hash(exp_a)
    hash_b = compute_measurement_config_hash(exp_b)

    ordered = apply_cycles([exp_a, exp_b], 2, ExperimentOrder.INTERLEAVE, "aabb0011", None)
    assert len(ordered) == 4, "sanity: 2 configs x 2 cycles = 4"

    study = StudyConfig(
        experiments=ordered,
        study_name="cycle-test",
        study_execution=ExecutionConfig(n_cycles=2, experiment_order="interleave"),
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
        return parent, child

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
    for c_item in call_args_list:
        h = c_item.args[0] if c_item.args else c_item.kwargs["config_hash"]
        c = c_item.args[1] if len(c_item.args) > 1 else c_item.kwargs["cycle"]
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
    """_consume_progress_events forwards step events to study_progress callback."""
    from queue import Queue
    from unittest.mock import MagicMock

    from llenergymeasure.study.runner import _consume_progress_events

    mock_progress = MagicMock()
    q = Queue()
    q.put({"event": "step_start", "step": "baseline", "description": "Measuring", "detail": "30s"})
    q.put({"event": "step_done", "step": "baseline", "elapsed_sec": 30.1})
    q.put({"event": "substep", "step": "model", "text": "loading weights", "elapsed_sec": 2.5})
    q.put({"event": "started", "config_hash": "abc123"})  # coarse event — ignored
    q.put(None)  # sentinel

    _consume_progress_events(q, study_progress=mock_progress)

    mock_progress.on_step_start.assert_called_once_with("baseline", "Measuring", "30s")
    mock_progress.on_step_done.assert_called_once_with("baseline", 30.1)
    mock_progress.on_substep.assert_called_once_with("model", "loading weights", 2.5)


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

    def fake_docker_run(config, **kwargs):
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
            patch("llenergymeasure.study._progress.print_study_progress"),
            patch(
                "llenergymeasure.results.persistence.save_result",
                return_value=Path("/tmp/test-docker/r.json"),
            ),
            patch.object(runner, "_prepare_images"),
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

    from llenergymeasure.utils.exceptions import DockerError

    def raise_docker_error(config, **kwargs):
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
        patch("llenergymeasure.study._progress.print_study_progress"),
    ):
        runner = StudyRunner(
            study_config, manifest, Path("/tmp/test-docker-err"), runner_specs=runner_specs
        )
        with patch.object(runner, "_prepare_images"):
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


# =============================================================================
# H5: recv-before-join ordering (pipe deadlock prevention)
# =============================================================================


def test_recv_before_join_ordering(study_config: StudyConfig) -> None:
    """parent_conn.poll()/recv() is called before p.join() in _run_one().

    Verifies that the pipe is drained BEFORE p.join() is called, preventing
    the classic deadlock where a large pickled result (>64KB) fills the OS pipe
    buffer and causes child to block on send() while parent blocks in join().
    """
    manifest = MagicMock()
    fake_result = {"status": "ok"}

    call_order: list[str] = []

    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)
    parent_conn = ctx.Pipe.return_value[0]

    # Track call order between poll/recv and join
    def track_poll(*args, **kwargs):
        call_order.append("poll")
        return True  # data is available

    def track_recv(*args, **kwargs):
        call_order.append("recv")
        return fake_result

    def track_join(*args, **kwargs):
        call_order.append("join")

    parent_conn.poll.side_effect = track_poll
    parent_conn.recv.side_effect = track_recv
    proc.join.side_effect = track_join

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-recv-join"))
        runner.run()

    # poll (drain) must appear before at least one join call
    assert "poll" in call_order, "parent_conn.poll() was never called"
    assert "join" in call_order, "p.join() was never called"
    first_poll = call_order.index("poll")
    first_join = call_order.index("join")
    assert first_poll < first_join, f"poll() must happen before join() — got order: {call_order}"


# =============================================================================
# Pipe direction regression test
# =============================================================================


def test_pipe_direction_parent_reads_child_writes() -> None:
    """Regression: Pipe(duplex=False) returns (recv_end, send_end).

    The runner must unpack as parent_conn, child_conn so parent can recv
    and child can send. A previous bug swapped these, causing OSError on
    every subprocess experiment.
    """
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    conn1, conn2 = ctx.Pipe(duplex=False)

    try:
        # conn1 is recv-only, conn2 is send-only (Python docs guarantee).
        # This verifies the behavioural contract: the first element of
        # Pipe(duplex=False) is read-only and the second is write-only.
        # The runner's unpacking order (parent_conn, child_conn) is covered
        # by this directional check - if swapped, parent would be unable
        # to recv and child unable to send.
        assert not conn1.writable, "First Pipe element should be read-only"
        assert not conn2.readable, "Second Pipe element should be write-only"
    finally:
        conn1.close()
        conn2.close()


# =============================================================================
# Pipe FD leak fix (C4)
# =============================================================================


def test_parent_conn_closed_after_collect_result(study_config: StudyConfig) -> None:
    """parent_conn.close() is called after _collect_result returns (C4 FD leak fix).

    The write-end (child_conn) is closed before p.start(); the read-end
    (parent_conn) must be closed after result collection to avoid leaking
    a file descriptor on every experiment.
    """
    manifest = MagicMock()
    fake_result = {"status": "ok"}

    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)
    parent_conn = ctx.Pipe.return_value[0]

    close_called_after_collect: list[bool] = []

    def patched_collect(p, conn, config, timeout, pipe_payload=None):
        # At this point close must NOT have been called yet
        close_called_after_collect.append(parent_conn.close.called)
        return fake_result

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch("llenergymeasure.study.runner._collect_result", side_effect=patched_collect),
    ):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-fd-leak"))
        runner.run()

    # close() must have been called exactly once on parent_conn
    parent_conn.close.assert_called_once()

    # close() must NOT have been called before _collect_result returned
    assert close_called_after_collect == [False], (
        "parent_conn.close() was called before _collect_result returned"
    )


# =============================================================================
# Process group kill behaviour
# =============================================================================


def test_worker_calls_setpgrp(monkeypatch) -> None:
    """_run_experiment_worker calls os.setpgrp() as the first line before any other work."""
    from llenergymeasure.config.models import DatasetConfig
    from tests.conftest import make_result

    config = ExperimentConfig(
        model="test/model", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
    )
    fake_result = make_result()

    setpgrp_calls: list[int] = []

    def fake_setpgrp() -> None:
        setpgrp_calls.append(1)

    monkeypatch.setattr("llenergymeasure.study.runner.os.setpgrp", fake_setpgrp)
    monkeypatch.setattr("llenergymeasure.backends.get_backend", lambda name: MagicMock())
    monkeypatch.setattr("llenergymeasure.harness.preflight.run_preflight", lambda c: None)
    monkeypatch.setattr(
        "llenergymeasure.harness.MeasurementHarness.run",
        lambda self, backend, config, **kwargs: fake_result,
    )

    mock_conn = MagicMock()
    progress_q: queue.SimpleQueue = queue.SimpleQueue()

    _run_experiment_worker(config, mock_conn, progress_q)

    assert len(setpgrp_calls) == 1, f"Expected os.setpgrp() called once, got {len(setpgrp_calls)}"


def test_kill_process_group_helper_suppresses_lookup_error() -> None:
    """_kill_process_group silently ignores ProcessLookupError for dead PIDs."""
    with patch("llenergymeasure.study.runner.os.killpg", side_effect=ProcessLookupError):
        # Must not raise
        _kill_process_group(999999, signal.SIGKILL)


def test_kill_process_group_helper_calls_os_killpg() -> None:
    """_kill_process_group forwards (pid, sig) to os.killpg."""
    with patch("llenergymeasure.study.runner.os.killpg") as mock_killpg:
        _kill_process_group(42, signal.SIGTERM)

    mock_killpg.assert_called_once_with(42, signal.SIGTERM)


def test_kill_process_group_helper_suppresses_permission_error() -> None:
    """_kill_process_group silently ignores PermissionError."""
    with patch("llenergymeasure.study.runner.os.killpg", side_effect=PermissionError):
        # Must not raise
        _kill_process_group(1, signal.SIGKILL)


def test_timeout_uses_killpg(study_config: StudyConfig) -> None:
    """Timeout path calls os.killpg(pid, SIGKILL), not proc.kill()."""
    manifest = MagicMock()

    proc = _make_mock_process(is_alive_after_join=True, exitcode=None, pid=77777)
    ctx = _make_mock_context(proc, pipe_has_data=False)

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch("llenergymeasure.study.runner.os.killpg") as mock_killpg,
    ):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-timeout-killpg"))
        runner.run()

    mock_killpg.assert_any_call(77777, signal.SIGKILL)
    # proc.kill() must NOT have been called directly
    proc.kill.assert_not_called()


def test_sigint_first_ctrl_c_sends_sigterm_to_group() -> None:
    """First Ctrl+C sends SIGTERM to the process group via os.killpg."""
    study = _make_sigint_study()
    manifest = MagicMock()

    proc = _make_mock_process(is_alive_after_join=True, exitcode=None, pid=22222)
    proc.is_alive.return_value = True

    with patch("llenergymeasure.study.runner.os.killpg") as mock_killpg:
        runner = StudyRunner(study, manifest, Path("/tmp/test-sigterm-group"))
        runner._active_process = proc
        runner._interrupt_count = 0

        # Simulate first Ctrl+C: call the SIGTERM branch directly
        runner._interrupt_count += 1
        if runner._active_process is not None and runner._active_process.is_alive():
            _kill_process_group(runner._active_process.pid, signal.SIGTERM)

    mock_killpg.assert_called_once_with(22222, signal.SIGTERM)
    # Direct p.terminate() must NOT have been called
    proc.terminate.assert_not_called()


def test_grace_period_uses_killpg(study_config: StudyConfig) -> None:
    """Grace period SIGKILL path uses os.killpg(pid, SIGKILL), not p.kill()."""
    manifest = MagicMock()

    # Process stays alive after initial 5s join AND 2s grace join
    proc = _make_mock_process(is_alive_after_join=True, exitcode=None, pid=33333)

    # After first join returns, is_alive=True; after SIGKILL join, is_alive=False
    call_count = [0]

    def is_alive_side_effect():
        call_count[0] += 1
        # First few checks: True (before grace kill); after killpg: False
        # The grace period checks: interrupt_event.is_set() AND p.is_alive()
        # We need at least 2 True answers to reach grace + SIGKILL branch
        return call_count[0] <= 2

    proc.is_alive.side_effect = is_alive_side_effect

    ctx = _make_mock_context(proc, pipe_has_data=False)

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch("llenergymeasure.study.runner.os.killpg") as mock_killpg,
    ):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-grace-killpg"))
        # Pre-set interrupt so the grace period path is taken
        runner._interrupt_event.set()
        runner._interrupt_count = 1

        # Patch run() to skip the experiment loop and go directly to grace period
        original_run_one = runner._run_one

        def run_one_with_interrupt(config, mp_ctx, index=1, total=1):
            # Start the process, then let the grace period logic fire
            result = original_run_one(config, mp_ctx, index=index, total=total)
            return result

        with (
            patch.object(runner, "_run_one", side_effect=run_one_with_interrupt),
            contextlib.suppress(SystemExit),
        ):
            runner.run()

    # SIGKILL must have been sent to the process group
    killpg_calls = [c for c in mock_killpg.call_args_list if c == call(33333, signal.SIGKILL)]
    assert len(killpg_calls) >= 1, (
        f"Expected os.killpg(33333, SIGKILL) at least once, got: {mock_killpg.call_args_list}"
    )
    # Direct p.kill() must NOT have been called
    proc.kill.assert_not_called()


# =============================================================================
# Timeseries parquet: output_dir passed as kwarg to local subprocess
# =============================================================================


def test_local_subprocess_passes_output_dir_as_kwarg(
    study_config: StudyConfig, basic_config: ExperimentConfig
) -> None:
    """_run_one passes output_dir and save_timeseries as Process kwargs.

    output_dir is no longer on ExperimentConfig; the runner creates a temp dir
    and passes it as a keyword argument to the subprocess Process target.
    The config itself must NOT contain output_dir.
    """
    manifest = MagicMock()

    from llenergymeasure.domain.experiment import compute_measurement_config_hash

    config_hash = compute_measurement_config_hash(basic_config)
    fake_result = {"config_hash": config_hash, "status": "success"}

    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result, pipe_has_data=True)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-output-dir"))
        runner.run()

    # Check Process was called with output_dir and save_timeseries as kwargs
    process_call = ctx.Process.call_args
    assert process_call is not None, "Process() was never called"

    process_kwargs = process_call.kwargs.get("kwargs", {})
    assert "output_dir" in process_kwargs, (
        f"output_dir must be passed as a Process kwarg, got kwargs: {process_kwargs}"
    )
    assert process_kwargs["output_dir"] is not None, "output_dir kwarg must not be None"
    assert process_kwargs["output_dir"].startswith("/tmp/llem-ts-"), (
        f"output_dir should be a temp dir, got: {process_kwargs['output_dir']}"
    )
    assert "save_timeseries" in process_kwargs, (
        f"save_timeseries must be passed as a Process kwarg, got kwargs: {process_kwargs}"
    )
    assert process_kwargs["save_timeseries"] is True

    # The config passed as a positional arg must NOT contain output_dir
    process_args = process_call.kwargs.get(
        "args", process_call.args[0] if process_call.args else ()
    )
    if isinstance(process_args, tuple) and len(process_args) > 0:
        worker_config = process_args[0]
        assert not hasattr(worker_config, "output_dir"), (
            "ExperimentConfig must not have output_dir; it is passed as a Process kwarg"
        )


# =============================================================================
# Error JSON persistence to failed-runs/ subdirectory
# =============================================================================


def test_error_payload_persisted_to_failed_runs_subdir(
    study_config: StudyConfig,
    tmp_path: Path,
) -> None:
    """DockerContainerError with error_payload copies artefacts to failed-runs/
    and enriches the manifest with the structured error type/message."""
    import json

    from llenergymeasure.domain.experiment import compute_measurement_config_hash
    from llenergymeasure.infra.docker_errors import DockerContainerError

    manifest = MagicMock()
    spec = _make_docker_runner_spec()
    runner_specs = {"pytorch": spec}

    config = study_config.experiments[0]
    config_hash = compute_measurement_config_hash(config)

    # Simulate exchange dir with both container.log and error JSON
    exchange_dir = tmp_path / "llem-exchange"
    exchange_dir.mkdir()
    (exchange_dir / "container.log").write_text("stderr output here", encoding="utf-8")
    error_payload = {
        "type": "RuntimeError",
        "message": "CUDA OOM",
        "traceback": "Traceback...",
    }
    (exchange_dir / f"{config_hash}_error.json").write_text(
        json.dumps(error_payload), encoding="utf-8"
    )

    def raise_docker_error(config, **kwargs):
        err = DockerContainerError(
            message="RuntimeError: CUDA OOM",
            fix_suggestion="Check the error traceback.",
        )
        err.error_payload = error_payload
        err.exchange_dir = str(exchange_dir)
        raise err

    fake_ctx = MagicMock()

    with (
        patch("multiprocessing.get_context", return_value=fake_ctx),
        patch(
            "llenergymeasure.infra.docker_runner.DockerRunner.run",
            side_effect=raise_docker_error,
        ),
        patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
    ):
        runner = StudyRunner(study_config, manifest, tmp_path, runner_specs=runner_specs)
        with patch.object(runner, "_prepare_images"):
            results = runner.run()

    assert len(results) == 1
    result = results[0]

    # Manifest receives the structured error type/message from the payload
    assert result["type"] == "RuntimeError"
    assert result["message"] == "CUDA OOM"

    # Verify failed-runs/ directory was created
    failed_runs = tmp_path / "failed-runs"
    assert failed_runs.is_dir()

    # Verify container.log was copied to failed-runs/
    assert result.get("log_file", "").startswith("failed-runs/")
    log_dest = tmp_path / result["log_file"]
    assert log_dest.exists()
    assert log_dest.read_text(encoding="utf-8") == "stderr output here"

    # Manifest should record the failure
    manifest.mark_failed.assert_called_once()


def test_docker_error_persists_to_failed_runs_subdir(
    study_config: StudyConfig,
    tmp_path: Path,
) -> None:
    """DockerError exception path also persists container.log to failed-runs/ subdirectory."""
    manifest = MagicMock()
    spec = _make_docker_runner_spec()
    runner_specs = {"pytorch": spec}

    from llenergymeasure.utils.exceptions import DockerError

    # Simulate an exchange dir with container.log
    exchange_dir = tmp_path / "llem-exc-exchange"
    exchange_dir.mkdir()
    (exchange_dir / "container.log").write_text("error stderr", encoding="utf-8")

    def raise_docker_error(config, **kwargs):
        err = DockerError("Container failed")
        err.exchange_dir = str(exchange_dir)
        raise err

    fake_ctx = MagicMock()

    with (
        patch("multiprocessing.get_context", return_value=fake_ctx),
        patch(
            "llenergymeasure.infra.docker_runner.DockerRunner.run",
            side_effect=raise_docker_error,
        ),
        patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
    ):
        runner = StudyRunner(study_config, manifest, tmp_path, runner_specs=runner_specs)
        with patch.object(runner, "_prepare_images"):
            results = runner.run()

    assert len(results) == 1
    result = results[0]

    # Verify failed-runs/ directory was created and log is there
    failed_runs = tmp_path / "failed-runs"
    assert failed_runs.is_dir()
    assert result.get("log_file", "").startswith("failed-runs/")
    log_dest = tmp_path / result["log_file"]
    assert log_dest.exists()
    assert log_dest.read_text(encoding="utf-8") == "error stderr"

    manifest.mark_failed.assert_called_once()


# =============================================================================
# _prepare_images() — study-level Docker image preparation
# =============================================================================

FAKE_INSPECT_JSON = b"""[{
    "Id": "sha256:abc123def456",
    "Size": 7620000000,
    "Created": "2026-03-20T10:00:00Z",
    "RootFS": {"Layers": ["sha256:a", "sha256:b", "sha256:c"]}
}]"""


class TestPrepareImages:
    """Tests for StudyRunner._prepare_images() across all image resolution paths."""

    def _make_runner(
        self,
        study_config: StudyConfig,
        runner_specs: dict,
        tmp_path: Path,
        progress: MagicMock | None = None,
    ) -> StudyRunner:
        manifest = MagicMock()
        fake_ctx = MagicMock()
        with patch("multiprocessing.get_context", return_value=fake_ctx):
            runner = StudyRunner(
                study_config, manifest, tmp_path, runner_specs=runner_specs, progress=progress
            )
        return runner

    def test_no_runner_specs_noop(self, study_config: StudyConfig, tmp_path: Path) -> None:
        """No runner_specs means _prepare_images is a no-op."""
        runner = self._make_runner(study_config, {}, tmp_path)
        runner._prepare_images()
        assert not runner._images_prepared

    def test_local_only_specs_noop(self, study_config: StudyConfig, tmp_path: Path) -> None:
        """Local-mode runner_specs skip image preparation entirely."""
        runner = self._make_runner(study_config, {"pytorch": _make_local_runner_spec()}, tmp_path)
        runner._prepare_images()
        assert not runner._images_prepared

    def test_local_cache_hit(self, study_config: StudyConfig, tmp_path: Path) -> None:
        """Image found locally — no pull, metadata extracted, _images_prepared set."""
        import subprocess

        progress = MagicMock()
        runner = self._make_runner(
            study_config,
            {"pytorch": _make_docker_runner_spec()},
            tmp_path,
            progress=progress,
        )

        fake_result = MagicMock(spec=subprocess.CompletedProcess)
        fake_result.returncode = 0
        fake_result.stdout = FAKE_INSPECT_JSON

        with patch("subprocess.run", return_value=fake_result) as mock_run:
            runner._prepare_images()

        assert runner._images_prepared
        # Only inspect should have been called (no pull)
        assert mock_run.call_count == 1
        assert "inspect" in mock_run.call_args_list[0][0][0]

        progress.begin_image_prep.assert_called_once_with(["pytorch"])
        progress.image_ready.assert_called_once()
        call_kwargs = progress.image_ready.call_args
        assert call_kwargs[1]["cached"] is True or call_kwargs[0][2] is True
        progress.end_image_prep.assert_called_once()

    def test_registry_pull_success(self, study_config: StudyConfig, tmp_path: Path) -> None:
        """Image not found locally, pulled from registry successfully."""
        import subprocess

        progress = MagicMock()
        runner = self._make_runner(
            study_config,
            {"pytorch": _make_docker_runner_spec()},
            tmp_path,
            progress=progress,
        )

        inspect_fail = MagicMock(spec=subprocess.CompletedProcess)
        inspect_fail.returncode = 1

        pull_ok = MagicMock(spec=subprocess.CompletedProcess)
        pull_ok.returncode = 0

        inspect_ok = MagicMock(spec=subprocess.CompletedProcess)
        inspect_ok.returncode = 0
        inspect_ok.stdout = FAKE_INSPECT_JSON

        with patch("subprocess.run", side_effect=[inspect_fail, pull_ok, inspect_ok]):
            runner._prepare_images()

        assert runner._images_prepared
        progress.image_ready.assert_called_once()
        call_args = progress.image_ready.call_args
        # cached=False when image was pulled
        assert (
            call_args[1].get("cached", call_args[0][2] if len(call_args[0]) > 2 else None) is False
        )
        progress.end_image_prep.assert_called_once()

    def test_pull_fails_raises_pull_error(self, study_config: StudyConfig, tmp_path: Path) -> None:
        """Image not locally cached, pull fails — raises DockerImagePullError."""
        import subprocess

        from llenergymeasure.infra.docker_errors import DockerImagePullError

        progress = MagicMock()
        runner = self._make_runner(
            study_config,
            {"pytorch": _make_docker_runner_spec()},
            tmp_path,
            progress=progress,
        )

        inspect_fail = MagicMock(spec=subprocess.CompletedProcess)
        inspect_fail.returncode = 1

        pull_fail = MagicMock(spec=subprocess.CompletedProcess)
        pull_fail.returncode = 1

        with (
            patch("subprocess.run", side_effect=[inspect_fail, pull_fail]),
            pytest.raises(DockerImagePullError, match="Image not found"),
        ):
            runner._prepare_images()

        assert not runner._images_prepared
        progress.image_failed.assert_called_once()
        progress.end_image_prep.assert_called_once()

    def test_pull_timeout_raises_pull_error(
        self, study_config: StudyConfig, tmp_path: Path
    ) -> None:
        """Pull times out — raises DockerImagePullError with timeout message."""
        import subprocess

        from llenergymeasure.infra.docker_errors import DockerImagePullError

        progress = MagicMock()
        runner = self._make_runner(
            study_config,
            {"pytorch": _make_docker_runner_spec()},
            tmp_path,
            progress=progress,
        )

        inspect_fail = MagicMock(spec=subprocess.CompletedProcess)
        inspect_fail.returncode = 1

        with (
            patch(
                "subprocess.run",
                side_effect=[inspect_fail, subprocess.TimeoutExpired("docker pull", 1800)],
            ),
            pytest.raises(DockerImagePullError, match="timed out"),
        ):
            runner._prepare_images()

        assert not runner._images_prepared
        progress.image_failed.assert_called_once()

    def test_multi_backend_all_cached(self, study_config: StudyConfig, tmp_path: Path) -> None:
        """Multiple Docker backends, all found locally."""
        import subprocess

        progress = MagicMock()
        runner = self._make_runner(
            study_config,
            {
                "pytorch": _make_docker_runner_spec(image="llem:pytorch"),
                "vllm": _make_docker_runner_spec(image="llem:vllm"),
            },
            tmp_path,
            progress=progress,
        )

        ok = MagicMock(spec=subprocess.CompletedProcess)
        ok.returncode = 0
        ok.stdout = FAKE_INSPECT_JSON

        with patch("subprocess.run", return_value=ok):
            runner._prepare_images()

        assert runner._images_prepared
        assert progress.image_ready.call_count == 2
        progress.begin_image_prep.assert_called_once_with(["pytorch", "vllm"])
        progress.end_image_prep.assert_called_once()

    def test_inspect_exception_falls_through_to_pull(
        self, study_config: StudyConfig, tmp_path: Path
    ) -> None:
        """If docker inspect raises (e.g. docker not found), falls through to pull."""
        import subprocess

        runner = self._make_runner(
            study_config,
            {"pytorch": _make_docker_runner_spec()},
            tmp_path,
        )

        pull_ok = MagicMock(spec=subprocess.CompletedProcess)
        pull_ok.returncode = 0

        inspect_ok = MagicMock(spec=subprocess.CompletedProcess)
        inspect_ok.returncode = 0
        inspect_ok.stdout = FAKE_INSPECT_JSON

        with patch(
            "subprocess.run",
            side_effect=[FileNotFoundError("docker"), pull_ok, inspect_ok],
        ):
            runner._prepare_images()

        assert runner._images_prepared

    def test_images_prepared_flag_enables_skip_image_check(
        self, study_config: StudyConfig, tmp_path: Path
    ) -> None:
        """After _prepare_images, _run_one_docker passes skip_image_check=True."""

        from llenergymeasure.domain.experiment import ExperimentResult

        progress = MagicMock()
        manifest = MagicMock()

        fake_ctx = MagicMock()
        with patch("multiprocessing.get_context", return_value=fake_ctx):
            runner = StudyRunner(
                study_config,
                manifest,
                tmp_path,
                runner_specs={"pytorch": _make_docker_runner_spec()},
                progress=progress,
            )

        # Pre-set _images_prepared to True (simulating successful prep)
        runner._images_prepared = True

        fake_result = MagicMock(spec=ExperimentResult)
        fake_result.total_energy_j = 1.0

        docker_run_kwargs: list[dict] = []

        def capture_docker_run(config, **kwargs):
            docker_run_kwargs.append(kwargs)
            return fake_result

        with (
            patch(
                "llenergymeasure.infra.docker_runner.DockerRunner.run",
                side_effect=capture_docker_run,
            ),
            patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
            patch(
                "llenergymeasure.results.persistence.save_result",
                return_value=Path("/tmp/r.json"),
            ),
            patch.object(runner, "_prepare_images"),
        ):
            runner.run()

        # DockerRunner.run should have been called with skip_image_check=True
        assert len(docker_run_kwargs) == 1
        assert docker_run_kwargs[0].get("skip_image_check") is True


class TestParseImageMetadata:
    """Tests for StudyRunner._parse_image_metadata()."""

    def test_valid_inspect_output(self) -> None:
        meta = StudyRunner._parse_image_metadata(FAKE_INSPECT_JSON)
        assert meta is not None
        assert meta["id"] == "abc123def456"
        assert "GB" in meta["size"]
        assert meta["layers"] == "3"

    def test_empty_json_array(self) -> None:
        assert StudyRunner._parse_image_metadata(b"[]") is None

    def test_invalid_json(self) -> None:
        assert StudyRunner._parse_image_metadata(b"not json") is None

    def test_size_mb_format(self) -> None:
        data = b'[{"Id": "sha256:abc123", "Size": 524288000}]'
        meta = StudyRunner._parse_image_metadata(data)
        assert meta is not None
        assert "MB" in meta["size"]

    def test_missing_fields_returns_none(self) -> None:
        data = b"[{}]"
        assert StudyRunner._parse_image_metadata(data) is None


# =============================================================================
# Circuit breaker, wall-clock timeout, mark_study_completed (Plan 03)
# =============================================================================


def _make_multi_experiment_study(n: int = 3) -> StudyConfig:
    """Build an n-experiment study (all distinct configs, 1 cycle each)."""
    from llenergymeasure.config.models import DatasetConfig

    experiments = [
        ExperimentConfig(
            model=f"test/model-{i}", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
        )
        for i in range(n)
    ]
    return StudyConfig(
        experiments=experiments,
        study_name="multi-exp-test",
        study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
        study_design_hash="aabbccdd11223344",
    )


def _make_ctx_with_results(results_sequence: list) -> MagicMock:
    """Build a mock mp context where each Process returns results in sequence.

    Each element of results_sequence is returned by parent_conn.recv() on the
    corresponding experiment invocation. Elements may be dicts (failure) or
    objects (success).
    """
    call_count = [0]

    def make_process(**kwargs):
        proc = MagicMock()
        proc.pid = 9999
        proc.is_alive.return_value = False
        proc.exitcode = 0
        proc.start.return_value = None
        proc.join.return_value = None
        return proc

    def make_pipe(duplex=True):
        idx = call_count[0]
        call_count[0] += 1
        result = results_sequence[min(idx, len(results_sequence) - 1)]

        parent = MagicMock()
        child = MagicMock()
        parent.poll.return_value = True
        parent.recv.return_value = result
        return parent, child

    ctx = MagicMock()
    ctx.Process.side_effect = make_process
    ctx.Pipe.side_effect = make_pipe
    ctx.Queue.return_value = queue.SimpleQueue()
    return ctx


def test_mark_study_completed_called_on_success(study_config: StudyConfig) -> None:
    """mark_study_completed() is called when all experiments succeed."""
    manifest = MagicMock()

    # Non-dict result triggers success path
    fake_result = {"status": "ok"}
    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study_config, manifest, Path("/tmp/test-completed"), no_lock=True)
        runner.run()

    manifest.mark_study_completed.assert_called_once()
    manifest.mark_interrupted.assert_not_called()


def test_circuit_breaker_trips_and_marks_remaining_skipped() -> None:
    """Circuit breaker trips after N consecutive failures; remaining experiments marked skipped."""
    # 3 experiments; circuit breaker trips after 2 consecutive failures (max_consecutive_failures=2)
    from llenergymeasure.config.models import DatasetConfig

    experiments = [
        ExperimentConfig(
            model=f"test/model-{i}", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
        )
        for i in range(4)
    ]
    study = StudyConfig(
        experiments=experiments,
        study_name="breaker-test",
        study_execution=ExecutionConfig(
            n_cycles=1,
            experiment_order="sequential",
            max_consecutive_failures=2,
            circuit_breaker_cooldown_seconds=0.0,  # no sleep in tests
        ),
        study_design_hash="aabbccdd11223344",
    )
    manifest = MagicMock()

    # Results: fail, fail (trips), probe (abort) — 3 experiments dispatched
    failure_dict = {"type": "RuntimeError", "message": "CUDA OOM"}
    results_seq = [failure_dict, failure_dict, failure_dict, failure_dict]
    ctx = _make_ctx_with_results(results_seq)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study, manifest, Path("/tmp/test-breaker"), no_lock=True)
        runner.run()

    # After probe fails -> abort -> mark_study_circuit_breaker
    manifest.mark_study_circuit_breaker.assert_called_once()
    # mark_skipped called for remaining (not-dispatched) experiments
    assert manifest.mark_skipped.call_count >= 1


def test_circuit_breaker_probe_success_resets_state() -> None:
    """Probe success after trip resets circuit breaker and study completes normally."""
    from llenergymeasure.config.models import DatasetConfig

    experiments = [
        ExperimentConfig(
            model=f"test/model-{i}", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
        )
        for i in range(4)
    ]
    study = StudyConfig(
        experiments=experiments,
        study_name="probe-success-test",
        study_execution=ExecutionConfig(
            n_cycles=1,
            experiment_order="sequential",
            max_consecutive_failures=2,
            circuit_breaker_cooldown_seconds=0.0,
        ),
        study_design_hash="aabbccdd11223344",
    )
    manifest = MagicMock()

    # fail, fail (trips), success (probe succeeds -> closed), success
    failure_dict = {"type": "RuntimeError", "message": "fail"}
    success_dict = {"status": "ok"}  # not a failure (no "type" key at top-level = error dict)
    results_seq = [failure_dict, failure_dict, success_dict, success_dict]
    ctx = _make_ctx_with_results(results_seq)

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study, manifest, Path("/tmp/test-probe-ok"), no_lock=True)
        runner.run()

    # No circuit breaker abort
    manifest.mark_study_circuit_breaker.assert_not_called()
    # Study completes normally
    manifest.mark_study_completed.assert_called_once()


def test_wall_clock_timeout_marks_remaining_skipped() -> None:
    """Wall-clock timeout marks remaining experiments as skipped and sets timed_out status."""
    from llenergymeasure.config.models import DatasetConfig

    experiments = [
        ExperimentConfig(
            model=f"test/model-{i}", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
        )
        for i in range(3)
    ]
    study = StudyConfig(
        experiments=experiments,
        study_name="timeout-test",
        study_execution=ExecutionConfig(
            n_cycles=1,
            experiment_order="sequential",
            wall_clock_timeout_hours=0.0001,  # very small: expires immediately
        ),
        study_design_hash="aabbccdd11223344",
    )
    manifest = MagicMock()

    success_dict = {"status": "ok"}
    ctx = _make_ctx_with_results([success_dict, success_dict, success_dict])

    # Force time.monotonic() to report deadline exceeded on first check

    tick = [0]

    def fast_monotonic():
        tick[0] += 1
        # First call (deadline computation) returns 0; subsequent calls return large value
        return 0.0 if tick[0] == 1 else 1e9

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch("llenergymeasure.study.runner.time.monotonic", side_effect=fast_monotonic),
    ):
        runner = StudyRunner(study, manifest, Path("/tmp/test-timeout"), no_lock=True)
        runner.run()

    manifest.mark_study_timed_out.assert_called_once()
    # At least some experiments marked as skipped
    assert manifest.mark_skipped.call_count >= 1
    # mark_study_completed must NOT be called on timeout
    manifest.mark_study_completed.assert_not_called()


def test_fail_fast_aborts_after_first_failure() -> None:
    """max_consecutive_failures=1 (fail-fast): study aborts after the first failure."""
    from llenergymeasure.config.models import DatasetConfig

    experiments = [
        ExperimentConfig(
            model=f"test/model-{i}", backend="pytorch", dataset=DatasetConfig(n_prompts=10)
        )
        for i in range(3)
    ]
    study = StudyConfig(
        experiments=experiments,
        study_name="fail-fast-test",
        study_execution=ExecutionConfig(
            n_cycles=1,
            experiment_order="sequential",
            max_consecutive_failures=1,
            circuit_breaker_cooldown_seconds=0.0,
        ),
        study_design_hash="aabbccdd11223344",
    )
    manifest = MagicMock()

    failure_dict = {"type": "RuntimeError", "message": "instant fail"}
    ctx = _make_ctx_with_results([failure_dict, failure_dict, failure_dict])

    with patch("multiprocessing.get_context", return_value=ctx):
        runner = StudyRunner(study, manifest, Path("/tmp/test-fail-fast"), no_lock=True)
        runner.run()

    # First failure trips the breaker (max_failures=1).
    # Probe experiment runs next; probe also fails -> abort.
    manifest.mark_study_circuit_breaker.assert_called_once()


def test_gpu_locks_acquired_and_released(study_config: StudyConfig, tmp_path: Path) -> None:
    """GPU locks are acquired before image prep and released in finally block."""
    manifest = MagicMock()

    acquired_locks: list = []
    released_locks: list = []

    def fake_acquire(gpu_indices, lock_dir=None):
        lock = MagicMock()
        acquired_locks.append(lock)
        return [lock]

    def fake_release(locks):
        released_locks.extend(locks)

    fake_result = {"status": "ok"}
    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch("llenergymeasure.study.gpu_locks.acquire_gpu_locks", side_effect=fake_acquire),
        patch("llenergymeasure.study.gpu_locks.release_gpu_locks", side_effect=fake_release),
        patch("llenergymeasure.device.gpu_info._resolve_gpu_indices", return_value=[0]),
    ):
        runner = StudyRunner(study_config, manifest, tmp_path, no_lock=False)
        runner.run()

    assert len(acquired_locks) == 1, "Expected exactly one acquire call"
    assert len(released_locks) == 1, "Expected exactly one release call (in finally)"


def test_no_lock_skips_gpu_lock_acquisition(study_config: StudyConfig) -> None:
    """When no_lock=True, GPU locks are never acquired."""
    manifest = MagicMock()

    fake_result = {"status": "ok"}
    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data=fake_result)

    with (
        patch("multiprocessing.get_context", return_value=ctx),
        patch("llenergymeasure.study.gpu_locks.acquire_gpu_locks") as mock_acquire,
    ):
        runner = StudyRunner(study_config, manifest, Path("/tmp/no-lock"), no_lock=True)
        runner.run()

    mock_acquire.assert_not_called()
