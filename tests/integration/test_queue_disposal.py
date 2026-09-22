"""Abandoned queue disposal must not wait for an absent consumer."""

import subprocess
import sys

from flowlens.integration.worker_runtime import MultiprocessingWorkerRuntime
from tests.integration.test_worker_runtime import FakeQueue


def test_abandoned_real_queue_cleanup_returns_before_deadline() -> None:
    program = """
import multiprocessing
from flowlens.integration.worker_runtime import MultiprocessingWorkerRuntime

q = multiprocessing.get_context("spawn").Queue()
q.put(b"x" * 1_000_000)
assert MultiprocessingWorkerRuntime._close_queues((q, q, None)) == ()
print("disposed", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "disposed" in result.stdout


def test_cancel_failure_still_closes_remaining_queues() -> None:
    failing = FakeQueue(0, [], fail_cancel=True)
    normal = FakeQueue(0, [])
    errors = MultiprocessingWorkerRuntime._close_queues((failing, normal))
    assert errors == ("queue cancel_join_thread: OSError",)
    assert failing.closed
    assert normal.closed and normal.join_cancelled


def test_close_failure_does_not_prevent_next_queue_disposal() -> None:
    failing = FakeQueue(0, [], fail_close=True)
    normal = FakeQueue(0, [])
    errors = MultiprocessingWorkerRuntime._close_queues((failing, normal))
    assert errors == ("queue close: OSError",)
    assert failing.join_cancelled
    assert normal.closed and normal.join_cancelled
