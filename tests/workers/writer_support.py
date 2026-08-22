"""Shared helpers for Writer Worker process and in-process tests."""

from collections import deque
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from pathlib import Path
from queue import Empty
from threading import Event as ThreadEvent
from typing import cast

import pytest

from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import AudioSource, MessageType, ProcessSource
from flowlens.domain.messages import (
    AudioDrainFence,
    AudioWriteCommand,
    EventRecord,
    MessageEnvelope,
    TranscriptCommitted,
    TranscriptRecord,
    WriterAck,
    WriterFinalize,
    WriterOpenSession,
    WriterShutdown,
)
from flowlens.domain.session import SessionManifest
from flowlens.persistence.session_writer import SessionWriter
from flowlens.workers import writer as writer_module
from tests.factories import (
    make_discussion_state,
    make_finalize_command,
    make_manifest,
    make_transcript_record,
)

SESSION_ID = "01J00000000000000000000000"


class _RecordingResponseQueue:
    """In-process response queue that records fatal flush operations."""

    def __init__(self) -> None:
        self.items: list[object] = []
        self.close_calls = 0
        self.join_thread_calls = 0

    def put(self, item: object) -> None:
        self.items.append(item)

    def close(self) -> None:
        self.close_calls += 1

    def join_thread(self) -> None:
        self.join_thread_calls += 1


class _FakeSessionWriter:
    """Small persistence port fake for deterministic queue scheduling tests."""

    def __init__(self) -> None:
        self.operations: list[tuple[str, int | float]] = []
        self.close_calls = 0
        self.audio_error: BaseException | None = None
        self.close_error: BaseException | None = None

    def append_audio(self, command: AudioWriteCommand) -> None:
        self.operations.append(("audio", command.source_start_sample))
        if self.audio_error is not None:
            raise self.audio_error

    def append_transcript(self, record: TranscriptRecord) -> None:
        self.operations.append(("transcript", record.sequence))

    def replace_discussion_state(
        self,
        previous_revision: int,
        state: DiscussionState,
    ) -> None:
        del state
        self.operations.append(("discussion", previous_revision))

    def append_event(self, record: EventRecord) -> None:
        self.operations.append(("event", record.sequence))

    def sync_if_due(self, now_monotonic: float) -> bool:
        self.operations.append(("sync", now_monotonic))
        return True

    def force_sync(self) -> None:
        self.operations.append(("force_sync", 0))

    def finalize(self, command: WriterFinalize) -> SessionManifest:
        del command
        self.operations.append(("finalize", 0))
        return make_manifest()

    def close_incomplete(self) -> None:
        self.close_calls += 1
        self.operations.append(("close", self.close_calls))
        if self.close_error is not None:
            raise self.close_error


class _StorageError(OSError):
    """Storage error carrying state that must not enter the response payload."""

    def __init__(self) -> None:
        super().__init__("disk full")
        self.nonserializable = object()


class _BrokenAudioQueue:
    """Audio queue boundary that fails while the worker polls it."""

    def get_nowait(self) -> object:
        raise EOFError("audio queue closed")


class _DelayedFenceQueue:
    """Audio queue whose fence appears only after one empty observation."""

    def __init__(self) -> None:
        self.fence_observed = False

    def get_nowait(self) -> object:
        raise Empty

    def get(self, timeout: float | None = None) -> object:
        del timeout
        self.fence_observed = True
        return AudioDrainFence()


class _CrossQueueLagAudioQueue:
    """Hide queued audio until the Writer performs one bounded wait."""

    def __init__(self, *items: object) -> None:
        self._items = deque(items)
        self._released = False

    def get_nowait(self) -> object:
        if not self._released or not self._items:
            raise Empty
        return self._items.popleft()

    def get(self, timeout: float | None = None) -> object:
        del timeout
        self._released = True
        if not self._items:
            raise Empty
        return self._items.popleft()


class _StopDuringFenceWaitQueue:
    """Audio queue that requests lifecycle stop while a fence is missing."""

    def __init__(self, stop: ThreadEvent) -> None:
        self._stop = stop

    def get_nowait(self) -> object:
        raise Empty

    def get(self, timeout: float | None = None) -> object:
        del timeout
        self._stop.set()
        raise Empty


class _ScriptedClock:
    """Deterministic monotonic clock with a stable exhausted value."""

    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)
        self._last = values[-1]

    def __call__(self) -> float:
        try:
            self._last = next(self._values)
        except StopIteration:
            pass
        return self._last


def _as_process_queue(value: object) -> Queue[object]:
    return cast(Queue[object], value)


def _as_process_event(value: object) -> Event:
    return cast(Event, value)


def _install_fake_writer(
    monkeypatch: pytest.MonkeyPatch,
    fake: _FakeSessionWriter,
) -> None:
    def open_fake(command: WriterOpenSession) -> SessionWriter:
        del command
        return cast(SessionWriter, fake)

    monkeypatch.setattr(writer_module, "_open_session", open_fake)


def control_envelope(
    sequence: int,
    message_type: MessageType,
    payload: object,
) -> MessageEnvelope[object]:
    """Build one GUI-sourced Writer control envelope."""

    return MessageEnvelope(
        1,
        SESSION_ID,
        message_type,
        sequence,
        ProcessSource.GUI,
        sequence * 100,
        payload,
    )


def make_open_envelope(
    session_dir: Path,
    sequence: int,
) -> MessageEnvelope[object]:
    """Build the session-open control."""

    return control_envelope(
        sequence,
        MessageType.WRITER_OPEN_SESSION,
        WriterOpenSession(session_dir, make_manifest(), make_discussion_state()),
    )


def make_audio_command(
    source_start_sample: int,
    *,
    source: AudioSource = AudioSource.ME,
) -> AudioWriteCommand:
    """Build one 800 ms microphone-audio write."""

    return AudioWriteCommand(
        source,
        b"\x00\x00" * 12_800,
        source_start_sample,
        source_start_sample + 12_800,
        source_start_sample // 16,
        1_000 + source_start_sample // 16,
    )


def make_transcript_envelope(sequence: int) -> MessageEnvelope[object]:
    """Build the first committed-transcript control."""

    return control_envelope(
        sequence,
        MessageType.TRANSCRIPT_COMMITTED,
        TranscriptCommitted(make_transcript_record(1)),
    )


def make_finalize_envelope(sequence: int) -> MessageEnvelope[object]:
    """Build the normal-finalization control."""

    return control_envelope(
        sequence,
        MessageType.WRITER_FINALIZE,
        make_finalize_command(),
    )


def make_shutdown_envelope(sequence: int) -> MessageEnvelope[object]:
    """Build the Writer shutdown control."""

    return control_envelope(
        sequence,
        MessageType.WRITER_SHUTDOWN,
        WriterShutdown(),
    )


def assert_ack(
    responses: Queue[object],
    acknowledged_sequence: int,
) -> None:
    """Read and validate one bounded Writer acknowledgement."""

    response = responses.get(timeout=5)
    assert isinstance(response, MessageEnvelope)
    assert response.message_type is MessageType.WRITER_ACK
    assert response.source is ProcessSource.WRITER
    assert isinstance(response.payload, WriterAck)
    assert response.payload.acknowledged_sequence == acknowledged_sequence
    assert response.payload.latest_successful_save_at.utcoffset() is not None
