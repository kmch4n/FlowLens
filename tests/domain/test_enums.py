from flowlens.domain.enums import (
    AudioSource,
    EventType,
    MessageType,
    ProcessSource,
    SessionMode,
    SessionStatus,
)


def test_session_mode_wire_values_are_spec_values() -> None:
    assert [mode.value for mode in SessionMode] == [
        "MEETING",
        "INTERVIEW",
        "GENERAL",
    ]


def test_audio_source_wire_values_are_spec_values() -> None:
    assert [source.value for source in AudioSource] == ["ME", "OTHERS"]


def test_session_status_wire_values_are_spec_values() -> None:
    assert [status.value for status in SessionStatus] == [
        "incomplete",
        "completed",
        "recovered",
    ]


def test_process_source_wire_values_are_spec_values() -> None:
    assert [source.value for source in ProcessSource] == [
        "GUI",
        "AUDIO",
        "ASR",
        "DISCUSSION",
        "WRITER",
    ]


def test_event_type_wire_values_are_spec_values() -> None:
    assert [event_type.value for event_type in EventType] == [
        "SESSION_START",
        "PAUSE_START",
        "PAUSE_END",
        "STOP_REQUESTED",
        "SESSION_COMPLETED",
        "SOURCE_DISCONNECTED",
        "SOURCE_RECONNECTED",
        "ASR_LAG_STARTED",
        "ASR_LAG_ENDED",
        "ANALYSIS_PAUSED",
        "ANALYSIS_RESUMED",
        "ANALYSIS_FAILED",
        "WORKER_EXITED",
        "WORKER_RESTARTED",
        "STORAGE_FAILED",
        "FORCE_CLOSE_REQUESTED",
        "SESSION_RECOVERED",
    ]


def test_message_type_wire_values_are_spec_values() -> None:
    assert [message_type.value for message_type in MessageType] == [
        "WORKER_START",
        "WORKER_READY",
        "WORKER_PAUSE",
        "WORKER_RESUME",
        "WORKER_STOP",
        "WORKER_STOPPED",
        "WORKER_ERROR",
        "AUDIO_LEVEL",
        "SOURCE_DISCONNECTED",
        "SOURCE_RECONNECTED",
        "TRANSCRIPT_PARTIAL",
        "TRANSCRIPT_COMMITTED",
        "ASR_STATUS",
        "DISCUSSION_ANALYZE",
        "DISCUSSION_STATE_REPLACED",
        "DISCUSSION_STATUS",
        "WRITER_OPEN_SESSION",
        "EVENT_APPENDED",
        "WRITER_FLUSH",
        "WRITER_FINALIZE",
        "WRITER_SHUTDOWN",
        "WRITER_ACK",
        "WRITER_FATAL",
    ]
