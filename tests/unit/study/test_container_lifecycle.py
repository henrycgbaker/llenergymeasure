"""Tests for container_lifecycle module."""

from __future__ import annotations

import os
import signal
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.study.container_lifecycle import (
    cleanup_study_containers,
    generate_container_labels,
    generate_container_name,
    install_sigterm_bridge,
    reap_orphaned_containers,
    register_container_cleanup,
)

# ---------------------------------------------------------------------------
# generate_container_name
# ---------------------------------------------------------------------------


class TestGenerateContainerName:
    def test_standard_study_id(self) -> None:
        name = generate_container_name("abcdef1234567890", 1)
        assert name == "llem-abcdef12-0001"

    def test_zero_padded_index(self) -> None:
        name = generate_container_name("abcdef1234567890", 42)
        assert name == "llem-abcdef12-0042"

    def test_empty_study_id_falls_back_to_unknown(self) -> None:
        name = generate_container_name("", 42)
        assert name == "llem-unknown-0042"

    def test_short_study_id_used_as_is(self) -> None:
        name = generate_container_name("abc", 1)
        assert name == "llem-abc-0001"

    def test_large_index_zero_padded(self) -> None:
        name = generate_container_name("deadbeef12345678", 9999)
        assert name == "llem-deadbeef-9999"

    def test_index_zero(self) -> None:
        name = generate_container_name("abcdef1234567890", 0)
        assert name == "llem-abcdef12-0000"


# ---------------------------------------------------------------------------
# generate_container_labels
# ---------------------------------------------------------------------------


class TestGenerateContainerLabels:
    def test_returns_required_keys(self) -> None:
        labels = generate_container_labels("my-study-id")
        assert "llem.study_id" in labels
        assert "llem.parent_pid" in labels
        assert "llem.started_at" in labels

    def test_study_id_matches(self) -> None:
        labels = generate_container_labels("my-study-id")
        assert labels["llem.study_id"] == "my-study-id"

    def test_parent_pid_is_string_of_current_pid(self) -> None:
        labels = generate_container_labels("my-study-id")
        assert labels["llem.parent_pid"] == str(os.getpid())

    def test_started_at_is_iso8601(self) -> None:
        from datetime import datetime

        labels = generate_container_labels("my-study-id")
        # Should be parseable as a datetime with timezone
        dt = datetime.fromisoformat(labels["llem.started_at"])
        assert dt.tzinfo is not None


# ---------------------------------------------------------------------------
# cleanup_study_containers
# ---------------------------------------------------------------------------


class TestCleanupStudyContainers:
    def test_calls_docker_ps_with_label_filter(self) -> None:
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=0)
            cleanup_study_containers("test-study-id")

        first_call = mock_run.call_args_list[0]
        cmd = first_call[0][0]
        assert "docker" in cmd
        assert "ps" in cmd
        assert "label=llem.study_id=test-study-id" in " ".join(cmd)

    def test_stops_running_containers(self) -> None:
        ps_result = MagicMock(stdout="abc123\ndef456\n", returncode=0)
        stop_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[ps_result, stop_result, stop_result]) as mock_run:
            cleanup_study_containers("test-study-id")

        # Should have called docker stop for each container ID
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 2
        stop_cmds = [" ".join(c[0][0]) for c in stop_calls]
        assert any("abc123" in cmd for cmd in stop_cmds)
        assert any("def456" in cmd for cmd in stop_cmds)

    def test_suppresses_all_exceptions(self) -> None:
        with patch("subprocess.run", side_effect=RuntimeError("Docker unavailable")):
            # Should not raise
            cleanup_study_containers("test-study-id")

    def test_skips_empty_container_ids(self) -> None:
        # docker ps output with blank lines
        ps_result = MagicMock(stdout="abc123\n\n  \n", returncode=0)
        stop_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[ps_result, stop_result]) as mock_run:
            cleanup_study_containers("test-study-id")

        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 1

    def test_no_containers_running(self) -> None:
        ps_result = MagicMock(stdout="", returncode=0)

        with patch("subprocess.run", return_value=ps_result) as mock_run:
            cleanup_study_containers("test-study-id")

        # Only the docker ps call, no docker stop calls
        assert mock_run.call_count == 1


# ---------------------------------------------------------------------------
# register_container_cleanup
# ---------------------------------------------------------------------------


class TestRegisterContainerCleanup:
    def test_registers_atexit_handler(self) -> None:
        with patch("atexit.register") as mock_register:
            register_container_cleanup("my-study")

        mock_register.assert_called_once_with(cleanup_study_containers, "my-study")


# ---------------------------------------------------------------------------
# install_sigterm_bridge
# ---------------------------------------------------------------------------


class TestInstallSigtermBridge:
    def test_installs_handler_and_returns_original(self) -> None:
        original_handler = signal.getsignal(signal.SIGTERM)
        try:
            returned = install_sigterm_bridge()
            new_handler = signal.getsignal(signal.SIGTERM)

            assert returned is original_handler
            assert new_handler is not original_handler
            assert callable(new_handler)
        finally:
            # Restore to avoid affecting other tests
            signal.signal(signal.SIGTERM, original_handler)

    def test_installed_handler_calls_sys_exit(self) -> None:
        original_handler = signal.getsignal(signal.SIGTERM)
        try:
            install_sigterm_bridge()
            handler = signal.getsignal(signal.SIGTERM)

            with pytest.raises(SystemExit) as exc_info:
                handler(signal.SIGTERM, None)  # type: ignore[call-arg]

            assert exc_info.value.code == 0
        finally:
            signal.signal(signal.SIGTERM, original_handler)

    def test_returns_none_on_value_error(self) -> None:
        with patch("signal.getsignal", side_effect=ValueError("not main thread")):
            result = install_sigterm_bridge()

        assert result is None


# ---------------------------------------------------------------------------
# reap_orphaned_containers
# ---------------------------------------------------------------------------


class TestReapOrphanedContainers:
    def _make_ps_result(self, lines: str) -> MagicMock:
        return MagicMock(stdout=lines, returncode=0)

    def test_stops_container_with_dead_parent_pid(self) -> None:
        ps_result = self._make_ps_result("abc123 99999999\n")
        stop_result = MagicMock(returncode=0)

        with (
            patch("subprocess.run", side_effect=[ps_result, stop_result]) as mock_run,
            patch("os.kill", side_effect=ProcessLookupError),
        ):
            count = reap_orphaned_containers()

        assert count == 1
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 1
        assert "abc123" in " ".join(stop_calls[0][0][0])

    def test_skips_container_with_alive_parent_pid(self) -> None:
        ps_result = self._make_ps_result(f"abc123 {os.getpid()}\n")

        with (
            patch("subprocess.run", return_value=ps_result) as mock_run,
            patch("os.kill", return_value=None),
        ):
            count = reap_orphaned_containers()

        assert count == 0
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 0

    def test_skips_container_with_permission_error(self) -> None:
        ps_result = self._make_ps_result("abc123 1\n")

        with (
            patch("subprocess.run", return_value=ps_result) as mock_run,
            patch("os.kill", side_effect=PermissionError),
        ):
            count = reap_orphaned_containers()

        assert count == 0
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 0

    def test_skips_malformed_lines(self) -> None:
        # Lines missing pid field
        ps_result = self._make_ps_result("abc123\nxyz\n")

        with patch("subprocess.run", return_value=ps_result):
            count = reap_orphaned_containers()

        assert count == 0

    def test_handles_multiple_containers_mixed(self) -> None:
        """One orphan + one alive container."""
        lines = "dead111 11111111\nalive22 22222222\n"
        ps_result = self._make_ps_result(lines)
        stop_result = MagicMock(returncode=0)

        def kill_side_effect(pid: int, sig: int) -> None:
            if pid == 11111111:
                raise ProcessLookupError
            # 22222222 is "alive"

        with (
            patch("subprocess.run", side_effect=[ps_result, stop_result]) as mock_run,
            patch("os.kill", side_effect=kill_side_effect),
        ):
            count = reap_orphaned_containers()

        assert count == 1
        stop_calls = [c for c in mock_run.call_args_list if "stop" in c[0][0]]
        assert len(stop_calls) == 1
        assert "dead111" in " ".join(stop_calls[0][0][0])

    def test_suppresses_all_exceptions(self) -> None:
        with patch("subprocess.run", side_effect=RuntimeError("Docker down")):
            # Should not raise
            count = reap_orphaned_containers()

        assert count == 0

    def test_invalid_pid_string_treated_as_dead(self) -> None:
        ps_result = self._make_ps_result("abc123 not-a-pid\n")
        stop_result = MagicMock(returncode=0)

        with patch("subprocess.run", side_effect=[ps_result, stop_result]):
            # ValueError from int("not-a-pid") should be treated as orphan
            count = reap_orphaned_containers()

        assert count == 1

    def test_empty_output_returns_zero(self) -> None:
        ps_result = self._make_ps_result("")

        with patch("subprocess.run", return_value=ps_result):
            count = reap_orphaned_containers()

        assert count == 0
