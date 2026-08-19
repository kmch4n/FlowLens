# FlowLens Foundation and Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Python 3.12 project foundation, strict typed domain and IPC contracts, local configuration, single-owner Writer Worker, crash-safe session persistence, and offline recovery required by the FlowLens MVP.

**Architecture:** Keep shared immutable records in `flowlens.domain`, configuration in `flowlens.config`, and file-format mechanics in `flowlens.persistence`. The Writer Worker is the only runtime component allowed to mutate session artifacts; it consumes small typed control envelopes and a separate typed audio queue. Recovery runs synchronously at launch before a new session starts, repairs only provably recoverable tails and WAV headers, appends an auditable recovery event, and changes an incomplete manifest to `recovered` as its final mutation.

**Tech Stack:** Python 3.12, PySide6, llama-cpp-python, `python-ulid`, standard-library `dataclasses`, `enum`, `json`, `wave`, `multiprocessing`, `pathlib`, and `os`; pytest, pytest-cov, pytest-qt, PyInstaller, Black, Ruff, and mypy for development checks.

**Spec:** `docs/mvp-spec.md` sections 5, 6.3, 15, 19, 19.1, 21, 22, 23, 24, 25, 27.4, and 28.

## Global Constraints

- Target Python is exactly 3.12 on Windows 10 Pro, 64-bit; the existing `.venv` is Python 3.11.9 and must not be used for implementation verification.
- Runtime session work is fully local: no HTTP server, WebSocket server, telemetry, update check, cloud inference, or network fallback.
- IPC uses `multiprocessing.Queue` and typed Python messages; PCM bytes travel only on the dedicated audio queue.
- JSON uses UTF-8 without BOM, LF, four-space indentation for snapshot files, compact one-record-per-line encoding for JSONL, and `ensure_ascii=False`.
- CSV is not part of this subsystem.
- Every touched Python function and method has complete type annotations; indentation is four spaces and strings use double quotes.
- Shared contracts are owned by exactly `src/flowlens/domain/enums.py`, `src/flowlens/domain/ids.py`, `src/flowlens/domain/messages.py`, `src/flowlens/domain/session.py`, and `src/flowlens/domain/discussion.py`. Audio-specific `AudioFrame` and ASR stream types are owned by the audio/ASR implementation plan.
- `MessageEnvelope` must carry `schema_version`, `session_id`, `message_type`, sender-local `sequence`, `source`, `created_monotonic_ms`, and a typed `payload` exactly as required by specification section 21.
- `session.json` is persisted as `incomplete` before capture can start; changing it to `completed` is the final persistent mutation of normal finalization.
- Writer ownership is fail-closed: a PID other than the PID that opened a `SessionWriter` cannot mutate session artifacts.
- Each JSONL record is flushed before its append call returns; all open session files are synchronized with `os.fsync` when the one-second durability deadline expires and during finalization.
- No session is deleted automatically. Recovery may remove only an invalid, non-newline-terminated final JSONL fragment; valid preceding bytes and PCM payload bytes are preserved.
- Recovery changes status from `incomplete` to `recovered`, never to `completed`, and performs no network operation.
- Do not run `git add`, `git commit`, `git push`, or any GitHub write command while executing this plan.
- Run all commands from the repository root: `C:\Users\kmch4n\OneDrive - 同志社大学\dev\FlowLens`.

## Locked File Map

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata and exact Black, Ruff, mypy, pytest configuration |
| `requirements.txt` | Exact shared production dependency pins owned by the foundation plan |
| `requirements-dev.txt` | Exact test, format, lint, and type-check tool pins |
| `src/flowlens/domain/enums.py` | Shared wire-value enums |
| `src/flowlens/domain/ids.py` | Shared `new_ulid() -> str` wrapper over the pinned ULID dependency |
| `src/flowlens/domain/discussion.py` | Immutable discussion snapshot and state-history record |
| `src/flowlens/domain/session.py` | Session manifest, devices, models, pauses, and lifecycle persistence values |
| `src/flowlens/domain/messages.py` | Transcript/event records, section-21 envelope, sequence validation, Writer control payloads, dedicated audio write command |
| `src/flowlens/config/model.py` | Exact non-session preference schema and pure geometry clamping |
| `src/flowlens/config/store.py` | Strict UTF-8 config loading and atomic saving |
| `src/flowlens/persistence/paths.py` | `%LOCALAPPDATA%\FlowLens` path resolution and session folder naming |
| `src/flowlens/persistence/json_files.py` | Strict JSONL append/tail validation and atomic JSON replacement |
| `src/flowlens/persistence/wav_sink.py` | Canonical 16 kHz, 16-bit, mono WAV append/finalize/sync and header repair |
| `src/flowlens/persistence/session_writer.py` | Seven-artifact creation, invariants, append APIs, one-second sync, ordered finalization |
| `src/flowlens/workers/writer.py` | Queue-driven Writer process and fatal persistence reporting |
| `src/flowlens/persistence/recovery.py` | Incomplete-session scan and deterministic recovery transaction |

All package directories also receive empty `__init__.py` files. Tests mirror these responsibilities under `tests/`.

## Dependency Pin Evidence

Live-verified against official PyPI release pages on 2026-08-19:

| Pin | Evidence | Python 3.12 / Windows finding |
| --- | --- | --- |
| `PySide6==6.11.2` | [PyPI 6.11.2 release](https://pypi.org/project/PySide6/6.11.2/) | Exists; released 2026-08-18; requires Python `>=3.10,<3.15`; publishes `cp310-abi3-win_amd64` |
| `llama-cpp-python==0.3.35` | [PyPI 0.3.35 release](https://pypi.org/project/llama-cpp-python/0.3.35/) | Exists; released 2026-08-17; requires Python `>=3.8`; PyPI publishes a source distribution, so the install step must build with `GGML_CUDA=on` for the MVP rather than accept a default CPU build |
| `PyInstaller==6.21.0` | [PyPI 6.21.0 release](https://pypi.org/project/PyInstaller/6.21.0/) | Exists; released 2026-06-13; requires Python `>=3.8,<3.16`; publishes `py3-none-win_amd64` |
| `pytest-qt==4.5.0` | [PyPI 4.5.0 release](https://pypi.org/project/pytest-qt/4.5.0/) | Exists; released 2025-07-01; requires Python `>=3.9`; publishes `py3-none-any` |
| `python-ulid==3.1.0` | [PyPI 3.1.0 release](https://pypi.org/project/python-ulid/3.1.0/) | Exists; released 2025-08-18; requires Python `>=3.9`; publishes `py3-none-any`; its documented API supports `str(ULID())` |
| `PyAudioWPatch==0.2.12.8` | [PyPI 0.2.12.8 release](https://pypi.org/project/PyAudioWPatch/0.2.12.8/) | Exists; Windows x86-64 and CPython 3.12 support verified for WASAPI loopback capture |
| `numpy==2.5.2` | [PyPI 2.5.2 release](https://pypi.org/project/numpy/2.5.2/) | Exists; Windows x86-64 CPython 3.12 wheel support verified |
| `soxr==1.1.0` | [PyPI 1.1.0 release](https://pypi.org/project/soxr/1.1.0/) | Exists; Windows x86-64 CPython 3.12 wheel support verified for resampling |
| `webrtcvad-wheels==2.0.14` | [PyPI 2.0.14 release](https://pypi.org/project/webrtcvad-wheels/2.0.14/) | Exists; Windows x86-64 CPython 3.12 wheel support verified; runtime import name is `webrtcvad` |
| `faster-whisper==1.2.1` | [PyPI 1.2.1 release](https://pypi.org/project/faster-whisper/1.2.1/) | Exists; Python 3.12 installation support verified; GPU execution is provided through the pinned CTranslate2 runtime |
| `ctranslate2==4.7.2` | [PyPI 4.7.2 release](https://pypi.org/project/ctranslate2/4.7.2/) | Exists; Windows x86-64 CPython 3.12 support verified; CUDA 12 and cuDNN runtime DLL directories must be discoverable through `PATH` |

Pin existence and declared interpreter compatibility are confirmed. Install `webrtcvad-wheels` and import it as `webrtcvad`; the legacy `webrtcvad` distribution must not appear in either requirements file. Actual installation and import on the designated PC remain mandatory gates; PyPI metadata alone does not prove that the local CUDA compiler/toolkit can build `llama-cpp-python`, that CTranslate2 can load the required CUDA 12/cuDNN DLLs, or that the designated GPU can execute float16 inference.

## Shared Interface Catalog

These names and signatures are fixed for downstream audio/ASR, analysis/UI, and session-controller plans:

```python
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

PayloadT = TypeVar("PayloadT")
type JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]

@dataclass(frozen=True, slots=True)
class MessageEnvelope(Generic[PayloadT]):
    schema_version: int
    session_id: str
    message_type: MessageType
    sequence: int
    source: ProcessSource
    created_monotonic_ms: int
    payload: PayloadT

@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    schema_version: int
    segment_id: str
    sequence: int
    source: AudioSource
    text: str
    session_start_ms: int
    session_end_ms: int
    source_start_sample: int
    source_end_sample: int
    committed_at: datetime

@dataclass(frozen=True, slots=True)
class TranscriptCommitted:
    record: TranscriptRecord


@dataclass(frozen=True, slots=True)
class EventRecord:
    schema_version: int
    session_id: str
    sequence: int
    event_type: EventType
    source: ProcessSource
    session_time_ms: int
    created_at: datetime
    details: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class DiscussionStateReplaced:
    previous_revision: int
    state: DiscussionState

@dataclass(frozen=True, slots=True)
class AudioWriteCommand:
    source: AudioSource
    pcm_s16le: bytes
    source_start_sample: int
    source_end_sample: int
    session_start_ms: int
    captured_monotonic_ms: int
```

The persistence-facing API is fixed to these callable signatures:

```text
ConfigStore.__init__(path: Path) -> None
ConfigStore.load() -> AppConfig
ConfigStore.save(config: AppConfig) -> None
SessionWriter.open(session_dir: Path, manifest: SessionManifest, initial_state: DiscussionState, *, sync_interval_seconds: float = 1.0) -> SessionWriter
SessionWriter.append_audio(command: AudioWriteCommand) -> None
SessionWriter.append_transcript(record: TranscriptRecord) -> None
SessionWriter.replace_discussion_state(previous_revision: int, state: DiscussionState) -> None
SessionWriter.append_event(record: EventRecord) -> None
SessionWriter.sync_if_due(now_monotonic: float) -> bool
SessionWriter.finalize(command: WriterFinalize) -> SessionManifest
SessionWriter.close_incomplete() -> None
recover_incomplete_sessions(sessions_root: Path, recovered_at: datetime) -> tuple[RecoveryReport, ...]
```

---

### Task 1: Python 3.12 Package and Quality Gate

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `requirements-dev.txt`
- Modify: `.gitignore`
- Create: `src/flowlens/__init__.py`
- Create: `src/flowlens/domain/__init__.py`
- Create: `src/flowlens/config/__init__.py`
- Create: `src/flowlens/persistence/__init__.py`
- Create: `src/flowlens/workers/__init__.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: Python 3.12 selected with `py -3.12`.
- Produces: importable `flowlens` package with `__version__: str = "0.1.0"`; one reproducible verification command used by every later task.

- [ ] **Step 1: Create a Python 3.12 environment without reusing the current Python 3.11 environment**

Run:

```powershell
Rename-Item -LiteralPath ".venv" -NewName ".venv-py311-backup"
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe --version
```

Expected: `Python 3.12.x`. The backup is retained and is not deleted by this plan.

- [ ] **Step 2: Write the failing package test**

```python
from flowlens import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"
```

- [ ] **Step 3: Add package and tool configuration**

Write `.gitignore` with these exact workspace-local and build artifacts:

```gitignore
.env
.venv/
.venv-py311-backup/
__pycache__/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
build/
dist/
*.spec
```

`requirements.txt` contains the exact shared production pins coordinated with the UI/analysis plan:

```text
python-ulid==3.1.0
PySide6==6.11.2
llama-cpp-python==0.3.35
PyAudioWPatch==0.2.12.8
numpy==2.5.2
soxr==1.1.0
webrtcvad-wheels==2.0.14
faster-whisper==1.2.1
ctranslate2==4.7.2
```

`requirements-dev.txt` contains exact pins:

```text
-r requirements.txt
black==25.1.0
mypy==1.17.1
PyInstaller==6.21.0
pytest==8.4.1
pytest-cov==6.2.1
pytest-qt==4.5.0
ruff==0.12.9
```

Configure `pyproject.toml` with `requires-python = ">=3.12,<3.13"`, setuptools `src` discovery, Black line length 88, Ruff target `py312` with `E`, `F`, `I`, `B`, `UP`, and `RUF`, mypy `python_version = "3.12"` plus `strict = true`, and pytest `testpaths = ["tests"]`, `addopts = "--strict-markers --strict-config"`.

- [ ] **Step 4: Install and run the focused test**

Run:

```powershell
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
try {
    .\.venv\Scripts\python.exe -m pip install --requirement requirements-dev.txt
} finally {
    Remove-Item -LiteralPath "Env:CMAKE_ARGS" -ErrorAction SilentlyContinue
}
.\.venv\Scripts\python.exe -m pip install --editable .
.\.venv\Scripts\python.exe -c "import pyaudiowpatch, numpy, soxr, webrtcvad, faster_whisper, ctranslate2"
.\.venv\Scripts\python.exe -m pytest tests\test_package.py -v
```

Expected: dependency installation completes without falling back to a CPU-only `llama-cpp-python` build, `python -c "import pyaudiowpatch, numpy, soxr, webrtcvad, faster_whisper, ctranslate2"` exits 0, and one test passes. If the CUDA source build fails or CTranslate2 cannot load CUDA 12/cuDNN DLLs from `PATH`, stop with the complete pip/CMake/DLL error; do not remove a pin, add the legacy `webrtcvad` distribution, or install an unpinned substitute.

Before audio/ASR implementation is accepted on the designated PC, run a focused GPU smoke with `faster_whisper.WhisperModel` configured exactly as the specification requires: local Kotoba-Whisper model path, `device="cuda"`, `compute_type="float16"`, `language="ja"`, `beam_size=1`, `temperature=0`, `condition_on_previous_text=False`, and `word_timestamps=True`. The smoke must transcribe a short local Japanese fixture, return at least one segment and at least one word timestamp, and show no CPU fallback or missing CUDA/cuDNN DLL error.

- [ ] **Step 5: Run the initial static gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m black --check src tests
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src tests
```

Expected: all three commands exit 0.

### Task 2: Shared Enums and Strict Record Validation

**Files:**
- Create: `src/flowlens/domain/enums.py`
- Create: `src/flowlens/domain/_validation.py`
- Test: `tests/domain/test_enums.py`
- Test: `tests/domain/test_validation.py`

**Interfaces:**
- Consumes: no application types.
- Produces: `SessionMode`, `AudioSource`, `SessionStatus`, `ProcessSource`, `EventType`, `MessageType`, `ContractValidationError`, `require_exact_keys()`, `parse_timezone_datetime()`, and `json_dumps()`.

- [ ] **Step 1: Write failing enum and validation tests**

```python
from datetime import datetime

import pytest

from flowlens.domain._validation import (
    ContractValidationError,
    json_dumps,
    parse_timezone_datetime,
    require_exact_keys,
)
from flowlens.domain.enums import AudioSource, EventType, SessionMode


def test_wire_values_are_spec_values() -> None:
    assert [mode.value for mode in SessionMode] == ["MEETING", "INTERVIEW", "GENERAL"]
    assert [source.value for source in AudioSource] == ["ME", "OTHERS"]
    assert EventType.SESSION_RECOVERED.value == "SESSION_RECOVERED"


def test_require_exact_keys_rejects_unknown_and_missing_keys() -> None:
    with pytest.raises(ContractValidationError, match="missing=.*b.*unknown=.*c"):
        require_exact_keys({"a": 1, "c": 2}, frozenset({"a", "b"}), "Record")


def test_wall_clock_requires_timezone() -> None:
    with pytest.raises(ContractValidationError, match="timezone"):
        parse_timezone_datetime("2026-08-19T12:00:00", "created_at")
    assert parse_timezone_datetime(
        "2026-08-19T12:00:00.123+09:00", "created_at"
    ) == datetime.fromisoformat("2026-08-19T12:00:00.123+09:00")


def test_json_encoding_preserves_japanese_and_uses_four_spaces() -> None:
    encoded = json_dumps({"text": "方針"})
    assert "方針" in encoded
    assert "\\u65b9" not in encoded
    assert encoded == '{\n    "text": "方針"\n}\n'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\domain\test_enums.py tests\domain\test_validation.py -v`

Expected: collection fails because the modules do not exist.

- [ ] **Step 3: Implement the exact wire enums**

Use `class Name(str, Enum)` and include every value below:

```python
class SessionMode(str, Enum):
    MEETING = "MEETING"
    INTERVIEW = "INTERVIEW"
    GENERAL = "GENERAL"


class AudioSource(str, Enum):
    ME = "ME"
    OTHERS = "OTHERS"


class SessionStatus(str, Enum):
    INCOMPLETE = "incomplete"
    COMPLETED = "completed"
    RECOVERED = "recovered"


class ProcessSource(str, Enum):
    GUI = "GUI"
    AUDIO = "AUDIO"
    ASR = "ASR"
    DISCUSSION = "DISCUSSION"
    WRITER = "WRITER"


class EventType(str, Enum):
    SESSION_START = "SESSION_START"
    PAUSE_START = "PAUSE_START"
    PAUSE_END = "PAUSE_END"
    STOP_REQUESTED = "STOP_REQUESTED"
    SESSION_COMPLETED = "SESSION_COMPLETED"
    SOURCE_DISCONNECTED = "SOURCE_DISCONNECTED"
    SOURCE_RECONNECTED = "SOURCE_RECONNECTED"
    ASR_LAG_STARTED = "ASR_LAG_STARTED"
    ASR_LAG_ENDED = "ASR_LAG_ENDED"
    ANALYSIS_PAUSED = "ANALYSIS_PAUSED"
    ANALYSIS_RESUMED = "ANALYSIS_RESUMED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    WORKER_EXITED = "WORKER_EXITED"
    WORKER_RESTARTED = "WORKER_RESTARTED"
    STORAGE_FAILED = "STORAGE_FAILED"
    FORCE_CLOSE_REQUESTED = "FORCE_CLOSE_REQUESTED"
    SESSION_RECOVERED = "SESSION_RECOVERED"


class MessageType(str, Enum):
    WORKER_START = "WORKER_START"
    WORKER_READY = "WORKER_READY"
    WORKER_PAUSE = "WORKER_PAUSE"
    WORKER_RESUME = "WORKER_RESUME"
    WORKER_STOP = "WORKER_STOP"
    WORKER_STOPPED = "WORKER_STOPPED"
    WORKER_ERROR = "WORKER_ERROR"
    AUDIO_LEVEL = "AUDIO_LEVEL"
    SOURCE_DISCONNECTED = "SOURCE_DISCONNECTED"
    SOURCE_RECONNECTED = "SOURCE_RECONNECTED"
    TRANSCRIPT_PARTIAL = "TRANSCRIPT_PARTIAL"
    TRANSCRIPT_COMMITTED = "TRANSCRIPT_COMMITTED"
    ASR_STATUS = "ASR_STATUS"
    DISCUSSION_ANALYZE = "DISCUSSION_ANALYZE"
    DISCUSSION_STATE_REPLACED = "DISCUSSION_STATE_REPLACED"
    DISCUSSION_STATUS = "DISCUSSION_STATUS"
    WRITER_OPEN_SESSION = "WRITER_OPEN_SESSION"
    EVENT_APPENDED = "EVENT_APPENDED"
    WRITER_FLUSH = "WRITER_FLUSH"
    WRITER_FINALIZE = "WRITER_FINALIZE"
    WRITER_SHUTDOWN = "WRITER_SHUTDOWN"
    WRITER_ACK = "WRITER_ACK"
    WRITER_FATAL = "WRITER_FATAL"
```

Worker-specific payload dataclasses are owned by their subsystem plans; `MessageEnvelope` can carry them without changing the shared envelope. Drained-worker acknowledgement uses `WORKER_STOPPED` with a subsystem-owned payload containing the worker identity and `drained=True`. Runtime analysis failures use `DISCUSSION_STATUS`; `ANALYSIS_FAILED` remains the persisted `EventType`.

- [ ] **Step 4: Implement strict validation helpers**

`require_exact_keys()` reports sorted missing and unknown sets. `parse_timezone_datetime()` accepts an ISO 8601 string only when `utcoffset()` is non-null. `json_dumps()` is exactly:

```python
def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=4) + "\n"
```

Also add typed `require_int()`, `require_non_negative_int()`, `require_str()`, `require_str_list()`, and `require_sha256()` helpers used by later record parsers. Reject `bool` where an integer is expected and require SHA-256 to match `[0-9a-f]{64}`.

- [ ] **Step 5: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\domain\test_enums.py tests\domain\test_validation.py -v
.\.venv\Scripts\python.exe -m black --check src\flowlens\domain tests\domain
.\.venv\Scripts\python.exe -m ruff check src\flowlens\domain tests\domain
.\.venv\Scripts\python.exe -m mypy src\flowlens\domain tests\domain
```

Expected: all commands pass.

### Task 3: Discussion and Session Persistence Models

**Files:**
- Create: `src/flowlens/domain/discussion.py`
- Create: `src/flowlens/domain/session.py`
- Test: `tests/domain/test_discussion.py`
- Test: `tests/domain/test_session.py`

**Interfaces:**
- Consumes: strict helpers and enums from Task 2.
- Produces: `DiscussionState`, `StateHistoryRecord`, `DeviceIdentity`, `ModelIdentity`, `PauseInterval`, and `SessionManifest`, each with strict `to_dict() -> dict[str, object]` and `from_dict(cls, value: object)` round trips.

- [ ] **Step 1: Write failing discussion contract tests**

```python
from datetime import datetime

from flowlens.domain.discussion import DiscussionState, StateHistoryRecord
from flowlens.domain.enums import SessionMode


NOW = datetime.fromisoformat("2026-08-19T12:35:02.125+09:00")


def test_state_history_matches_spec_shape() -> None:
    state = DiscussionState(
        revision=7,
        mode=SessionMode.INTERVIEW,
        current_focus="志望理由",
        key_points=("業務改善に関わった経験",),
        confirmed_outcomes=("完全ローカル動作をMVPの必須条件とする",),
        follow_up_items=("具体的な成果を確認する",),
        updated_at=NOW,
    )
    record = StateHistoryRecord(1, "01J00000000000000000000000", 6, 7, state)
    restored = StateHistoryRecord.from_dict(record.to_dict())
    assert restored == record
    assert restored.to_dict()["state"] == state.to_dict()
    assert "evidence_ids" not in str(restored.to_dict())


def test_initial_state_uses_revision_zero_and_empty_values() -> None:
    state = DiscussionState.initial(SessionMode.GENERAL, NOW)
    assert state.revision == 0
    assert state.current_focus == ""
    assert state.key_points == ()
```

- [ ] **Step 2: Write failing session manifest tests**

```python
from dataclasses import replace
from datetime import datetime

import pytest

from flowlens.domain.enums import SessionMode, SessionStatus
from flowlens.domain.session import (
    DeviceIdentity,
    ModelIdentity,
    PauseInterval,
    SessionManifest,
)


START = datetime.fromisoformat("2026-08-19T12:00:00+09:00")


def make_manifest() -> SessionManifest:
    return SessionManifest(
        schema_version=1,
        session_id="01J00000000000000000000000",
        status=SessionStatus.INCOMPLETE,
        mode=SessionMode.MEETING,
        started_at=START,
        ended_at=None,
        active_duration_ms=0,
        pause_intervals=(),
        microphone=DeviceIdentity("mic-1", "USB Microphone"),
        loopback_output=DeviceIdentity("out-1", "Speakers"),
        asr_model=ModelIdentity("kotoba-tech/kotoba-whisper-v2.0-faster", "rev-a", "a" * 64),
        discussion_model=ModelIdentity("Qwen/Qwen3-4B-Instruct-2507", "rev-b", "b" * 64),
        application_version="0.1.0",
        transcript_entry_count=0,
        final_discussion_state_revision=0,
        recovery_notes=(),
    )


def test_manifest_round_trip_contains_every_required_field() -> None:
    manifest = make_manifest()
    assert SessionManifest.from_dict(manifest.to_dict()) == manifest
    assert set(manifest.to_dict()) == {
        "schema_version", "session_id", "status", "mode", "started_at", "ended_at",
        "active_duration_ms", "pause_intervals", "microphone", "loopback_output",
        "asr_model", "discussion_model", "application_version",
        "transcript_entry_count", "final_discussion_state_revision", "recovery_notes",
    }


def test_completed_manifest_requires_end_time() -> None:
    with pytest.raises(ValueError, match="ended_at"):
        replace(make_manifest(), status=SessionStatus.COMPLETED)


def test_pause_interval_cannot_end_before_it_starts() -> None:
    with pytest.raises(ValueError, match="ended_ms"):
        PauseInterval(started_ms=900, ended_ms=800)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\domain\test_discussion.py tests\domain\test_session.py -v`

Expected: collection fails because both modules are absent.

- [ ] **Step 4: Implement immutable models and invariants**

Use `@dataclass(frozen=True, slots=True)` throughout. Tuple fields serialize as JSON arrays. Every wall-clock `to_dict()` value uses `datetime.isoformat(timespec="milliseconds")`, preserving its UTC offset. Enforce: schema version 1; non-negative revisions, counters, and monotonic millisecond values; `new_revision == state.revision`; `previous_revision + 1 == new_revision`; non-empty IDs/names/repositories/revisions/application version; timezone-aware wall clocks; lowercase 64-character checksums; `session_id` is 26 Crockford characters; and `COMPLETED`/`RECOVERED` manifests have a non-null `ended_at`.

Use this exact session manifest field order:

```python
schema_version, session_id, status, mode, started_at, ended_at,
active_duration_ms, pause_intervals, microphone, loopback_output,
asr_model, discussion_model, application_version, transcript_entry_count,
final_discussion_state_revision, recovery_notes
```

- [ ] **Step 5: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\domain\test_discussion.py tests\domain\test_session.py -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\domain tests\domain
.\.venv\Scripts\python.exe -m mypy src\flowlens\domain tests\domain
```

Expected: all commands pass.

### Task 4: Transcript, Event, and Section-21 IPC Contracts

**Files:**
- Create: `src/flowlens/domain/messages.py`
- Test: `tests/domain/test_messages.py`

**Interfaces:**
- Consumes: `AudioSource`, `ProcessSource`, `EventType`, `MessageType`, `DiscussionState`, `SessionManifest`, and `PauseInterval`.
- Produces: `TranscriptRecord`, `EventRecord`, `TranscriptCommitted`, `DiscussionStateReplaced`, `WriterOpenSession`, `WriterAppendEvent`, `WriterFlush`, `WriterFinalize`, `WriterShutdown`, `WriterAck`, `WriterFatal`, `AudioWriteCommand`, `MessageEnvelope[PayloadT]`, `SequenceTracker`, `SequenceResult`, `UnknownSchemaVersionError`, and `MessageSequenceError`.

- [ ] **Step 1: Write failing record-shape tests**

```python
from datetime import datetime

from flowlens.domain.enums import AudioSource, EventType, ProcessSource
from flowlens.domain.messages import EventRecord, TranscriptRecord


NOW = datetime.fromisoformat("2026-08-19T12:34:56.789+09:00")


def test_transcript_record_matches_spec_shape() -> None:
    record = TranscriptRecord(1, "01J00000000000000000000001", 42, AudioSource.ME,
        "今回の方針を確認します。", 12480, 15820, 182400, 235840, NOW)
    assert TranscriptRecord.from_dict(record.to_dict()) == record
    assert record.to_dict()["source"] == "ME"


def test_event_details_are_operational_json_values() -> None:
    event = EventRecord(1, "01J00000000000000000000000", 18,
        EventType.ASR_LAG_STARTED, ProcessSource.ASR, 125400, NOW, {"backlog_ms": 5200})
    assert EventRecord.from_dict(event.to_dict()) == event
    assert event.to_dict()["event_type"] == "ASR_LAG_STARTED"
```

- [ ] **Step 2: Write failing envelope and sequencing tests**

```python
import pytest

from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.domain.messages import (
    MessageEnvelope,
    SequenceResult,
    SequenceTracker,
    TranscriptCommitted,
    UnknownSchemaVersionError,
)


def envelope(sequence: int) -> MessageEnvelope[TranscriptCommitted]:
    return MessageEnvelope(1, "01J00000000000000000000000",
        MessageType.TRANSCRIPT_COMMITTED, sequence, ProcessSource.ASR, 15540,
        TranscriptCommitted(record=make_transcript_record(sequence)))


def test_sequence_tracker_detects_duplicate_and_gap_per_sender() -> None:
    tracker = SequenceTracker()
    assert tracker.observe(envelope(1)) is SequenceResult.ACCEPTED
    assert tracker.observe(envelope(1)) is SequenceResult.DUPLICATE
    assert tracker.observe(envelope(3)) is SequenceResult.GAP
    assert tracker.expected(ProcessSource.ASR, "01J00000000000000000000000") == 4


def test_unknown_schema_is_rejected_before_payload_dispatch() -> None:
    invalid = MessageEnvelope(2, "01J00000000000000000000000",
        MessageType.TRANSCRIPT_COMMITTED, 1, ProcessSource.ASR, 100,
        TranscriptCommitted(record=make_transcript_record(1)))
    with pytest.raises(UnknownSchemaVersionError, match="2"):
        SequenceTracker().observe(invalid)
```

Define the helper in the same test file:

```python
def make_transcript_record(sequence: int) -> TranscriptRecord:
    return TranscriptRecord(
        schema_version=1,
        segment_id="01J00000000000000000000001",
        sequence=sequence,
        source=AudioSource.ME,
        text="今回の方針を確認します。",
        session_start_ms=12480,
        session_end_ms=15820,
        source_start_sample=182400,
        source_end_sample=235840,
        committed_at=NOW,
    )
```

- [ ] **Step 3: Write failing dedicated-audio boundary tests**

```python
import pickle

import pytest

from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import AudioWriteCommand


def test_audio_command_is_picklable_and_not_an_envelope_payload() -> None:
    command = AudioWriteCommand(AudioSource.ME, b"\x00\x00" * 320, 0, 320, 0, 15540)
    assert pickle.loads(pickle.dumps(command)) == command


def test_audio_command_requires_complete_16_bit_samples() -> None:
    with pytest.raises(ValueError, match="even number of bytes"):
        AudioWriteCommand(AudioSource.ME, b"\x00", 0, 0, 0, 15540)


def test_audio_command_sample_range_matches_pcm_length() -> None:
    with pytest.raises(ValueError, match="source sample range"):
        AudioWriteCommand(AudioSource.OTHERS, b"\x00\x00" * 320, 0, 319, 0, 15540)
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\domain\test_messages.py -v`

Expected: collection fails because `messages.py` is absent.

- [ ] **Step 5: Implement the records, typed payloads, envelope, and tracker**

`MessageEnvelope` uses the field order in the Shared Interface Catalog. It rejects non-1 schema versions only when `validate_schema()` or `SequenceTracker.observe()` is called so a receiver can log the original envelope. `SequenceTracker` keys state by `(session_id, source)`; the first accepted sender sequence is 1, a lower sequence is `DUPLICATE`, and a higher-than-expected sequence is `GAP` while advancing the expected value to `sequence + 1`.

`TranscriptRecord` rejects empty/whitespace text, `session_end_ms < session_start_ms`, `source_end_sample < source_start_sample`, invalid source values, nonpositive sequence numbers, naive `committed_at`, unknown keys, and unknown schema versions. `EventRecord` performs the corresponding session/sequence/time checks and recursively validates `details` as finite JSON values. Both serialize wall clocks with millisecond precision and timezone. `AudioWriteCommand` uses exactly the six fields in the Shared Interface Catalog and requires `source_end_sample - source_start_sample == len(pcm_s16le) // 2`.

Writer payloads have these exact fields:

```python
@dataclass(frozen=True, slots=True)
class WriterOpenSession:
    session_dir: Path
    manifest: SessionManifest
    initial_state: DiscussionState

@dataclass(frozen=True, slots=True)
class WriterFinalize:
    ended_at: datetime
    active_duration_ms: int
    pause_intervals: tuple[PauseInterval, ...]
    final_state: DiscussionState
    completion_event: EventRecord

@dataclass(frozen=True, slots=True)
class WriterAck:
    acknowledged_sequence: int
    latest_successful_save_at: datetime

@dataclass(frozen=True, slots=True)
class WriterFatal:
    failed_sequence: int
    error_type: str
    message: str
```

`WriterAppendEvent` wraps one `EventRecord`; `WriterFlush` and `WriterShutdown` have no fields. `TranscriptCommitted` and `DiscussionStateReplaced` are defined exactly as in the Shared Interface Catalog. Reject partial transcript types from this persistence contract by not defining one here.

- [ ] **Step 6: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\domain\test_messages.py -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\domain tests\domain
.\.venv\Scripts\python.exe -m mypy src\flowlens\domain tests\domain
```

Expected: all commands pass.

### Task 5: Exact Local Configuration and Geometry Clamping

**Files:**
- Create: `src/flowlens/config/model.py`
- Create: `src/flowlens/config/store.py`
- Test: `tests/config/test_model.py`
- Test: `tests/config/test_store.py`

**Interfaces:**
- Consumes: `SessionMode`, strict JSON helpers.
- Produces: `Rect`, `WindowPreferences`, `DevicePreferences`, `AppConfig.default()`, `clamp_window(window, displays) -> WindowPreferences`, `ConfigStore`, and `ConfigLoadError`.

- [ ] **Step 1: Write failing exact-schema and clamp tests**

```python
from flowlens.config.model import AppConfig, Rect, clamp_window


def test_default_config_has_only_specified_preferences() -> None:
    value = AppConfig.default().to_dict()
    assert value == {
        "schema_version": 1,
        "window": {"x": 100, "y": 100, "width": 1280, "height": 800,
            "maximized": False, "always_on_top": False},
        "devices": {"microphone_id": "", "loopback_output_id": ""},
        "last_mode": "MEETING",
    }


def test_window_is_clamped_to_display_with_largest_intersection() -> None:
    saved = AppConfig.default().window.with_geometry(x=1800, y=900, width=1280, height=800)
    clamped = clamp_window(saved, (Rect(0, 0, 1920, 1080), Rect(1920, 0, 1920, 1080)))
    assert (clamped.x, clamped.y, clamped.width, clamped.height) == (1920, 280, 1280, 800)


def test_offscreen_window_moves_to_primary_display() -> None:
    saved = AppConfig.default().window.with_geometry(x=9000, y=9000, width=1280, height=800)
    clamped = clamp_window(saved, (Rect(0, 0, 1920, 1080),))
    assert (clamped.x, clamped.y) == (640, 280)
```

- [ ] **Step 2: Write failing config store tests**

```python
import pytest

from flowlens.config.model import AppConfig
from flowlens.config.store import ConfigLoadError, ConfigStore


def test_missing_config_returns_defaults_without_writing(tmp_path) -> None:
    path = tmp_path / "config.json"
    assert ConfigStore(path).load() == AppConfig.default()
    assert not path.exists()


def test_save_is_utf8_without_bom_and_round_trips(tmp_path) -> None:
    path = tmp_path / "config.json"
    store = ConfigStore(path)
    store.save(AppConfig.default())
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw
    assert store.load() == AppConfig.default()


def test_invalid_config_is_reported_instead_of_silently_overwritten(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text('{"schema_version": 2}', encoding="utf-8")
    with pytest.raises(ConfigLoadError, match="schema_version"):
        ConfigStore(path).load()
    assert path.read_text(encoding="utf-8") == '{"schema_version": 2}'
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\config -v`

Expected: collection fails because config modules are absent.

- [ ] **Step 4: Implement the immutable config schema and deterministic clamp**

Use frozen slotted dataclasses and reject unknown or missing keys. Device IDs are strings; `""` means no saved selection and must never be resolved to an arbitrary device. Clamp width to `min(max(saved.width, 900), display.width)` and height to `min(max(saved.height, 600), display.height)`. Select the display with the greatest intersection area, breaking ties by input order; if every intersection is zero, use the first display. Clamp x and y so the entire adjusted window is inside that display. Preserve `maximized` and `always_on_top`.

- [ ] **Step 5: Implement strict atomic config loading and saving**

`ConfigStore.save()` creates the parent directory, writes `config.json.tmp` with `json_dumps()`, flushes and calls `os.fsync()`, then calls `os.replace(temp_path, path)`. On any parse/schema/encoding failure, `load()` raises `ConfigLoadError` with the path and original reason and leaves bytes unchanged. In a `finally` block, remove only the store's own temp file if it still exists.

- [ ] **Step 6: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\config -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\config tests\config
.\.venv\Scripts\python.exe -m mypy src\flowlens\config tests\config
```

Expected: all commands pass.

### Task 6: Shared Session IDs, Local Paths, and Folder Naming

**Files:**
- Create: `src/flowlens/domain/ids.py`
- Create: `src/flowlens/persistence/paths.py`
- Test: `tests/domain/test_ids.py`
- Test: `tests/persistence/test_paths.py`

**Interfaces:**
- Consumes: timezone-aware `datetime`.
- Produces: shared `new_ulid() -> str`, `AppPaths.from_environment(environment: Mapping[str, str]) -> AppPaths`, `session_directory_name(started_at: datetime, session_id: str) -> str`, and `new_session_directory(sessions_root: Path, started_at: datetime, id_factory: Callable[[], str] = new_ulid) -> Path`.

- [ ] **Step 1: Write failing deterministic ID and path tests**

```python
from datetime import datetime

import pytest

from flowlens.domain.ids import new_ulid
from flowlens.persistence.paths import AppPaths, new_session_directory, session_directory_name


def test_new_ulid_returns_uppercase_26_character_wire_id() -> None:
    value = new_ulid()
    assert len(value) == 26
    assert set(value) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


def test_paths_are_rooted_at_local_appdata(tmp_path) -> None:
    paths = AppPaths.from_environment({"LOCALAPPDATA": str(tmp_path)})
    assert paths.config == tmp_path / "FlowLens" / "config.json"
    assert paths.models == tmp_path / "FlowLens" / "models"
    assert paths.sessions == tmp_path / "FlowLens" / "sessions"


def test_missing_localappdata_is_explicit() -> None:
    with pytest.raises(RuntimeError, match="LOCALAPPDATA"):
        AppPaths.from_environment({})


def test_session_folder_name_is_windows_safe() -> None:
    started = datetime.fromisoformat("2026-08-19T12:34:56+09:00")
    name = session_directory_name(started, "01J00000000000000000000000")
    assert name == "20260819T123456+0900_01J00000000000000000000000"
    assert ":" not in name


def test_session_directory_generation_accepts_deterministic_id_factory(tmp_path) -> None:
    started = datetime.fromisoformat("2026-08-19T12:34:56+09:00")
    path = new_session_directory(
        tmp_path,
        started,
        id_factory=lambda: "01J00000000000000000000000",
    )
    assert path == tmp_path / "20260819T123456+0900_01J00000000000000000000000"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\domain\test_ids.py tests\persistence\test_paths.py -v`

Expected: collection fails because persistence modules are absent.

- [ ] **Step 3: Implement the shared pinned-ULID wrapper**

Import `ULID` from the pinned `python-ulid` distribution and implement the exact shared signature `def new_ulid() -> str: return str(ULID()).upper()`. Keep time/entropy injection out of this public function. Consumers that require deterministic tests, including `new_session_directory()`, receive a `Callable[[], str]` factory whose default is `new_ulid`.

- [ ] **Step 4: Implement injected environment path resolution**

`AppPaths` is a frozen slotted dataclass with `root`, `config`, `models`, and `sessions`. `from_environment()` performs no directory creation. `session_directory_name()` rejects a naive `datetime` and invalid session IDs, then uses `%Y%m%dT%H%M%S%z_<session-id>`. `new_session_directory()` calls its injected factory exactly once, validates the returned ID, and returns the path without creating it.

- [ ] **Step 5: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\domain\test_ids.py tests\persistence\test_paths.py -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\persistence tests\persistence
.\.venv\Scripts\python.exe -m mypy src\flowlens\persistence tests\persistence
```

Expected: all commands pass.

### Task 7: Crash-Safe JSON and JSONL Primitives

**Files:**
- Create: `src/flowlens/persistence/json_files.py`
- Test: `tests/persistence/test_json_files.py`

**Interfaces:**
- Consumes: `json_dumps()` and record `to_dict()` results.
- Produces: `AtomicJsonFile.path: Path`, `AtomicJsonFile.replace(value: object) -> None`, `JsonlAppender.append(value: object) -> None`, `JsonlAppender.sync() -> None`, `JsonlAppender.close() -> None`, read-only `inspect_jsonl_tail(path: Path) -> JsonlRepairPlan`, `apply_jsonl_tail_repair(path: Path, plan: JsonlRepairPlan) -> JsonlRepairResult`, convenience `validate_and_repair_jsonl_tail(path: Path) -> JsonlRepairResult`, and `JsonlValidationError`.

- [ ] **Step 1: Write failing durability and encoding tests**

```python
import json

from flowlens.persistence.json_files import AtomicJsonFile, JsonlAppender


def test_jsonl_flushes_each_record_and_uses_one_compact_lf_line(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    appender = JsonlAppender.open(path)
    appender.append({"schema_version": 1, "text": "保存"})
    assert path.read_bytes() == '{"schema_version":1,"text":"保存"}\n'.encode("utf-8")
    appender.close()


def test_atomic_json_is_indented_utf8_and_has_no_leftover_temp(tmp_path) -> None:
    path = tmp_path / "discussion-state.json"
    AtomicJsonFile(path).replace({"revision": 1, "current_focus": "方針"})
    assert json.loads(path.read_text(encoding="utf-8"))["revision"] == 1
    assert b"\r\n" not in path.read_bytes()
    assert not path.with_name("discussion-state.json.tmp").exists()
```

- [ ] **Step 2: Write failing tail-repair tests**

```python
import pytest

from flowlens.persistence.json_files import (
    JsonlValidationError,
    validate_and_repair_jsonl_tail,
)


def test_truncated_final_fragment_is_the_only_discarded_bytes(tmp_path) -> None:
    path = tmp_path / "transcript.jsonl"
    good = b'{"sequence":1}\n{"sequence":2}\n'
    path.write_bytes(good + b'{"sequence":')
    result = validate_and_repair_jsonl_tail(path)
    assert path.read_bytes() == good
    assert result.valid_record_count == 2
    assert result.discarded_tail_bytes == len(b'{"sequence":')


def test_valid_final_record_without_lf_is_preserved_and_terminated(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_bytes(b'{"sequence":1}')
    result = validate_and_repair_jsonl_tail(path)
    assert path.read_bytes() == b'{"sequence":1}\n'
    assert result.discarded_tail_bytes == 0


def test_invalid_complete_middle_line_is_not_silently_removed(tmp_path) -> None:
    path = tmp_path / "state-history.jsonl"
    original = b'{"revision":1}\nnot-json\n{"revision":2}\n'
    path.write_bytes(original)
    with pytest.raises(JsonlValidationError, match="line 2"):
        validate_and_repair_jsonl_tail(path)
    assert path.read_bytes() == original
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\persistence\test_json_files.py -v`

Expected: collection fails because `json_files.py` is absent.

- [ ] **Step 4: Implement append and atomic replace**

Open JSONL in binary append mode. Encode records with `json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"`; flush before returning from every append. `sync()` flushes then calls `os.fsync(file.fileno())`. Atomic JSON uses a sibling `<name>.tmp`, writes `json_dumps()`, flushes, fsyncs, closes, and calls `os.replace()`.

- [ ] **Step 5: Implement conservative tail validation**

`inspect_jsonl_tail()` reads bytes once, splits with `keepends=True`, validates UTF-8 and JSON object shape line by line, and returns a plan without writing. It never proposes modification for an invalid newline-terminated line and instead raises. For a non-newline final fragment, the plan appends LF if it is a valid JSON object; otherwise it truncates exactly that fragment. `apply_jsonl_tail_repair()` verifies the current file size and SHA-256 still match the inspected plan before changing bytes, then returns `JsonlRepairResult(valid_record_count: int, discarded_tail_bytes: int, appended_final_lf: bool)` and fsyncs. `validate_and_repair_jsonl_tail()` composes these two functions for focused callers.

- [ ] **Step 6: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\persistence\test_json_files.py -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\persistence tests\persistence
.\.venv\Scripts\python.exe -m mypy src\flowlens\persistence tests\persistence
```

Expected: all commands pass.

### Task 8: Canonical WAV Sink and Header Repair

**Files:**
- Create: `src/flowlens/persistence/wav_sink.py`
- Test: `tests/persistence/test_wav_sink.py`

**Interfaces:**
- Consumes: even-length `pcm_s16le` bytes and explicit source sample range from `AudioWriteCommand`.
- Produces: `WavSink.open(path: Path) -> WavSink`, `append(pcm: bytes) -> int`, `sync()`, `finalize()`, `close_incomplete()`, and `repair_wav_header(path: Path) -> WavRepairResult`.

- [ ] **Step 1: Write failing canonical-format and offset tests**

```python
import wave

from flowlens.persistence.wav_sink import WavSink


def test_sink_writes_canonical_wav_and_returns_sample_offsets(tmp_path) -> None:
    path = tmp_path / "mic.wav"
    sink = WavSink.open(path)
    assert sink.append(b"\x00\x00" * 320) == 0
    assert sink.append(b"\x01\x00" * 160) == 320
    sink.finalize()
    with wave.open(str(path), "rb") as reader:
        assert (reader.getframerate(), reader.getsampwidth(), reader.getnchannels()) == (16000, 2, 1)
        assert reader.getnframes() == 480
```

- [ ] **Step 2: Write failing abnormal-close and repair tests**

```python
import struct
import wave

from flowlens.persistence.wav_sink import WavSink, repair_wav_header


def test_repair_uses_file_size_without_changing_pcm_payload(tmp_path) -> None:
    path = tmp_path / "loopback.wav"
    sink = WavSink.open(path)
    pcm = b"\x02\x00" * 320
    sink.append(pcm)
    sink.close_incomplete()
    with path.open("r+b") as file:
        file.seek(4); file.write(struct.pack("<I", 0))
        file.seek(40); file.write(struct.pack("<I", 0))
    before_payload = path.read_bytes()[44:]
    result = repair_wav_header(path)
    assert path.read_bytes()[44:] == before_payload
    assert result.valid_pcm_bytes == len(pcm)
    with wave.open(str(path), "rb") as reader:
        assert reader.getnframes() == 320
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\persistence\test_wav_sink.py -v`

Expected: collection fails because `wav_sink.py` is absent.

- [ ] **Step 4: Implement WAV writing with an owned binary handle**

Open one owned `w+b` binary file, write the canonical 44-byte PCM header with RIFF and data sizes initially zero, and append PCM directly after byte 44. `append()` rejects odd byte counts and calls no fsync. Track `sample_count`; return the first sample offset for each append. `sync()` flushes and fsyncs the binary handle. `finalize()` rewrites the two size fields from `sample_count`, flushes, fsyncs, and closes once. `close_incomplete()` flushes, fsyncs, and closes without rewriting either size field, deterministically exercising the recovery contract after abnormal termination. The standard `wave` module is used only to verify produced files in tests.

- [ ] **Step 5: Implement fixed-format header repair**

Require a file of at least 44 bytes and verify the `RIFF`, `WAVE`, `fmt `, PCM format 1, mono, 16,000 Hz, 16-bit, and `data` markers. Compute `valid_pcm_bytes = ((file_size - 44) // 2) * 2`, write little-endian `36 + valid_pcm_bytes` at byte 4 and `valid_pcm_bytes` at byte 40, fsync, and leave every byte from offset 44 onward unchanged. Return original and valid byte counts plus whether header values changed.

- [ ] **Step 6: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\persistence\test_wav_sink.py -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\persistence tests\persistence
.\.venv\Scripts\python.exe -m mypy src\flowlens\persistence tests\persistence
```

Expected: all commands pass.

### Task 9: Seven-Artifact Session Bootstrap and Append Invariants

**Files:**
- Create: `src/flowlens/persistence/session_writer.py`
- Create: `tests/__init__.py`
- Create: `tests/factories.py`
- Create: `tests/persistence/conftest.py`
- Test: `tests/persistence/test_session_writer_open.py`
- Test: `tests/persistence/test_session_writer_append.py`

**Interfaces:**
- Consumes: all domain persistence records, `AtomicJsonFile`, `JsonlAppender`, and `WavSink`.
- Produces: `SessionWriter.open()`, append methods from the Shared Interface Catalog, `PersistenceInvariantError`, and `WriterOwnershipError`.

- [ ] **Step 1: Write failing bootstrap test**

```python
import json
import wave

from flowlens.domain.discussion import DiscussionState
from flowlens.domain.enums import SessionStatus
from flowlens.persistence.session_writer import SessionWriter


def test_open_creates_exactly_seven_required_artifacts(tmp_path) -> None:
    manifest = make_manifest(status=SessionStatus.INCOMPLETE)
    state = DiscussionState.initial(manifest.mode, manifest.started_at)
    writer = SessionWriter.open(tmp_path / "session", manifest, state)
    assert {path.name for path in (tmp_path / "session").iterdir()} == {
        "session.json", "mic.wav", "loopback.wav", "transcript.jsonl",
        "discussion-state.json", "state-history.jsonl", "events.jsonl",
    }
    assert json.loads((tmp_path / "session" / "session.json").read_text(encoding="utf-8"))["status"] == "incomplete"
    for name in ("mic.wav", "loopback.wav"):
        with wave.open(str(tmp_path / "session" / name), "rb") as reader:
            assert reader.getnframes() == 0
    writer.close_incomplete()
```

Create reusable deterministic factories in `tests/factories.py`. `make_manifest()` uses the Task 3 constructor with these exact values and parameters:

```python
def make_manifest(
    *,
    status: SessionStatus = SessionStatus.INCOMPLETE,
    session_id: str = "01J00000000000000000000000",
) -> SessionManifest:
    started_at = datetime.fromisoformat("2026-08-19T12:00:00+09:00")
    ended_at = None
    if status is not SessionStatus.INCOMPLETE:
        ended_at = datetime.fromisoformat("2026-08-19T12:30:00+09:00")
    return SessionManifest(
        schema_version=1,
        session_id=session_id,
        status=status,
        mode=SessionMode.MEETING,
        started_at=started_at,
        ended_at=ended_at,
        active_duration_ms=0 if ended_at is None else 1_800_000,
        pause_intervals=(),
        microphone=DeviceIdentity("mic-1", "USB Microphone"),
        loopback_output=DeviceIdentity("out-1", "Speakers"),
        asr_model=ModelIdentity(
            "kotoba-tech/kotoba-whisper-v2.0-faster", "rev-a", "a" * 64
        ),
        discussion_model=ModelIdentity(
            "Qwen/Qwen3-4B-Instruct-2507", "rev-b", "b" * 64
        ),
        application_version="0.1.0",
        transcript_entry_count=0,
        final_discussion_state_revision=0,
        recovery_notes=(),
    )


def make_discussion_state(revision: int = 0) -> DiscussionState:
    return DiscussionState(
        revision=revision,
        mode=SessionMode.MEETING,
        current_focus="方針" if revision else "",
        key_points=("ローカル保存",) if revision else (),
        confirmed_outcomes=(),
        follow_up_items=(),
        updated_at=datetime.fromisoformat("2026-08-19T12:05:00+09:00"),
    )


def make_transcript_record(sequence: int = 1) -> TranscriptRecord:
    segment_ids = {
        1: "01J00000000000000000000001",
        2: "01J00000000000000000000002",
    }
    return TranscriptRecord(
        1,
        segment_ids[sequence],
        sequence,
        AudioSource.ME,
        "今回の方針を確認します。",
        sequence * 1_000,
        sequence * 1_000 + 800,
        (sequence - 1) * 12_800,
        sequence * 12_800,
        datetime.fromisoformat("2026-08-19T12:05:00+09:00"),
    )
```

Import every referenced domain type explicitly at the top of that file. `tests/persistence/conftest.py` owns resource cleanup:

```python
@pytest.fixture
def open_writer(tmp_path: Path) -> Iterator[SessionWriter]:
    manifest = make_manifest()
    writer = SessionWriter.open(
        tmp_path / "session",
        manifest,
        make_discussion_state(),
    )
    try:
        yield writer
    finally:
        writer.close_incomplete()
```

- [ ] **Step 2: Write failing append and ordering tests**

```python
import json

import pytest

from flowlens.domain.enums import AudioSource
from flowlens.domain.messages import AudioWriteCommand
from flowlens.persistence.session_writer import PersistenceInvariantError, SessionWriter


def test_append_routes_audio_and_enforces_source_sample_contiguity(open_writer) -> None:
    open_writer.append_audio(AudioWriteCommand(AudioSource.ME, b"\x00\x00" * 320, 0, 320, 0, 15540))
    with pytest.raises(PersistenceInvariantError, match="expected source_start_sample 320"):
        open_writer.append_audio(AudioWriteCommand(AudioSource.ME, b"\x00\x00" * 320, 640, 960, 40, 15580))


def test_transcript_sequence_is_strictly_monotonic(open_writer) -> None:
    open_writer.append_audio(
        AudioWriteCommand(AudioSource.ME, b"\x00\x00" * 12_800, 0, 12_800, 0, 1_800)
    )
    first = make_transcript_record(sequence=1)
    open_writer.append_transcript(first)
    with pytest.raises(PersistenceInvariantError, match="expected transcript sequence 2"):
        open_writer.append_transcript(first)
    lines = (open_writer.session_dir / "transcript.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["sequence"] for line in lines] == [1]


def test_discussion_replacement_writes_snapshot_then_history(open_writer) -> None:
    state = make_discussion_state(revision=1)
    open_writer.replace_discussion_state(previous_revision=0, state=state)
    snapshot = json.loads((open_writer.session_dir / "discussion-state.json").read_text(encoding="utf-8"))
    history = json.loads((open_writer.session_dir / "state-history.jsonl").read_text(encoding="utf-8"))
    assert snapshot["revision"] == 1
    assert history["previous_revision"] == 0
    assert history["new_revision"] == 1
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\persistence\test_session_writer_open.py tests\persistence\test_session_writer_append.py -v`

Expected: collection fails because `session_writer.py` is absent.

- [ ] **Step 4: Implement fail-closed session opening**

Reject pre-existing non-empty session directories and manifests whose status is not `INCOMPLETE`. Create the directory, atomically persist `session.json`, create both canonical WAVs, create all three empty JSONL files, and atomically persist the initial discussion state. If any creation fails, close every opened handle, keep created artifacts for diagnosis/recovery, and re-raise. Capture `os.getpid()` as the sole owner PID.

- [ ] **Step 5: Implement append invariants and ordering**

Every public mutation first checks owner PID and open state. Session identity for the dedicated audio queue is established by the queue instance owned by the active Writer; `AudioWriteCommand` intentionally has no session ID. Maintain independent ME/OTHERS sample cursors, next transcript sequence starting at 1, next event sequence starting at 1, and discussion revision starting from `initial_state.revision`. Validate `source_end_sample - source_start_sample == len(pcm_s16le) // 2`, non-negative capture/session times, and exact source cursor continuity before writing. A committed transcript record is accepted only when its `source_end_sample` is no greater than the already-persisted cursor for that record's source. For state replacement, append and flush `StateHistoryRecord` first, then atomically write the new live snapshot; update the in-memory revision only after both succeed. If snapshot replacement fails, the Writer fails closed and recovery rebuilds it from the durable final history record. Raise `PersistenceInvariantError` before writing on a mismatch.

- [ ] **Step 6: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\persistence\test_session_writer_open.py tests\persistence\test_session_writer_append.py -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\persistence tests\persistence
.\.venv\Scripts\python.exe -m mypy src\flowlens\persistence tests\persistence
```

Expected: all commands pass.

### Task 10: One-Second Synchronization and Ordered Finalization

**Files:**
- Modify: `src/flowlens/persistence/session_writer.py`
- Test: `tests/persistence/test_session_writer_finalize.py`

**Interfaces:**
- Consumes: open `SessionWriter`, `WriterFinalize`, and all underlying sync/finalize methods.
- Produces: `sync_if_due(now_monotonic) -> bool`, `finalize(command) -> SessionManifest`, and `close_incomplete()` with idempotent resource closure.

- [ ] **Step 1: Write failing sync-deadline test**

```python
def test_sync_occurs_only_when_one_second_deadline_is_due(open_writer, monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(open_writer, "_sync_all", lambda: calls.append("sync"))
    assert open_writer.sync_if_due(open_writer.opened_monotonic + 0.999) is False
    assert open_writer.sync_if_due(open_writer.opened_monotonic + 1.000) is True
    assert open_writer.sync_if_due(open_writer.opened_monotonic + 1.500) is False
    assert calls == ["sync"]
```

- [ ] **Step 2: Write failing final-mutation test**

```python
import json

from flowlens.domain.enums import SessionStatus


def test_finalize_marks_completed_only_after_every_other_persistent_step(open_writer, monkeypatch) -> None:
    order: list[str] = []
    monkeypatch.setattr(open_writer, "_sync_jsonl", lambda: order.append("flush-jsonl"))
    monkeypatch.setattr(open_writer, "_finalize_wavs", lambda: order.append("finalize-wavs"))
    monkeypatch.setattr(open_writer, "_replace_state", lambda state: order.append("final-state"))
    monkeypatch.setattr(open_writer, "_append_completion_event", lambda event: order.append("completion-event"))
    original_manifest_write = open_writer._write_manifest
    def record_manifest(manifest):
        order.append(f"manifest-{manifest.status.value}")
        original_manifest_write(manifest)
    monkeypatch.setattr(open_writer, "_write_manifest", record_manifest)
    result = open_writer.finalize(make_finalize_command())
    assert result.status is SessionStatus.COMPLETED
    assert order == ["completion-event", "flush-jsonl", "finalize-wavs", "final-state", "manifest-completed"]
    persisted = json.loads((open_writer.session_dir / "session.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "completed"
```

Add this factory to `tests/factories.py`; its `final_state` revision exactly equals an unchanged Writer's persisted discussion revision:

```python
def make_finalize_command(
    event_sequence: int = 1,
    session_id: str = "01J00000000000000000000000",
) -> WriterFinalize:
    return WriterFinalize(
        ended_at=datetime.fromisoformat("2026-08-19T12:30:00+09:00"),
        active_duration_ms=1_800_000,
        pause_intervals=(),
        final_state=make_discussion_state(revision=0),
        completion_event=EventRecord(
            schema_version=1,
            session_id=session_id,
            sequence=event_sequence,
            event_type=EventType.SESSION_COMPLETED,
            source=ProcessSource.GUI,
            session_time_ms=1_800_000,
            created_at=datetime.fromisoformat("2026-08-19T12:30:00+09:00"),
            details={},
        ),
    )
```

Import `WriterFinalize`, `EventRecord`, `EventType`, and `ProcessSource` in the factory module. Add a separate assertion that a newer or older revision raises `PersistenceInvariantError` before the completion event is appended; final discussion updates travel through `replace_discussion_state()` before finalization.

- [ ] **Step 3: Write failing incomplete-close test**

```python
import json


def test_force_close_path_never_marks_session_completed(open_writer) -> None:
    open_writer.close_incomplete()
    persisted = json.loads((open_writer.session_dir / "session.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "incomplete"
    open_writer.close_incomplete()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\persistence\test_session_writer_finalize.py -v`

Expected: failures report absent synchronization and finalization methods.

- [ ] **Step 5: Implement synchronization and finalization order**

Set `opened_monotonic = time.monotonic()` and next deadline to `opened + sync_interval`. A due sync flushes/fsyncs both WAV handles and all JSONL appenders, then advances the deadline by whole intervals until it is strictly after `now_monotonic`. Finalization validates the completion event is `SESSION_COMPLETED`, matches the session, and has the next event sequence. Execute exactly: append completion event; flush and fsync JSONL; finalize and fsync both WAVs; atomically replace the final discussion state; build a new manifest with completion timestamps, duration, pauses, transcript count, and final state revision; atomically replace `session.json` last. Return that manifest and reject every later mutation.

- [ ] **Step 6: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\persistence\test_session_writer_finalize.py -v
.\.venv\Scripts\python.exe -m pytest tests\persistence -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\persistence tests\persistence
.\.venv\Scripts\python.exe -m mypy src\flowlens\persistence tests\persistence
```

Expected: all commands pass.

### Task 11: Writer Worker Queue Dispatch and Fatal Reporting

**Files:**
- Create: `src/flowlens/workers/writer.py`
- Test: `tests/workers/test_writer.py`

**Interfaces:**
- Consumes: `MessageEnvelope` with Writer control payloads, dedicated `AudioWriteCommand`, `SequenceTracker`, and `SessionWriter`.
- Produces: `run_writer_worker(control_queue: Queue[object], audio_queue: Queue[object], response_queue: Queue[object], stop_event: Event) -> None` and one `WriterAck` or `WriterFatal` response envelope per control mutation.

- [ ] **Step 1: Write failing dispatch test with spawn-safe queues**

```python
from multiprocessing import get_context
from queue import Empty

from flowlens.domain.enums import MessageType, ProcessSource
from flowlens.domain.messages import MessageEnvelope, WriterAck
from flowlens.workers.writer import run_writer_worker


def test_writer_process_opens_appends_and_finalizes(tmp_path) -> None:
    context = get_context("spawn")
    control = context.Queue()
    audio = context.Queue()
    responses = context.Queue()
    stop = context.Event()
    process = context.Process(target=run_writer_worker, args=(control, audio, responses, stop))
    process.start()
    control.put(make_open_envelope(tmp_path, sequence=1))
    assert_ack(responses, acknowledged_sequence=1)
    audio.put(make_audio_command(source_start_sample=0))
    control.put(make_transcript_envelope(sequence=2))
    assert_ack(responses, acknowledged_sequence=2)
    control.put(make_finalize_envelope(sequence=3))
    assert_ack(responses, acknowledged_sequence=3)
    control.put(make_shutdown_envelope(sequence=4))
    process.join(timeout=10)
    assert process.exitcode == 0
```

Define the queue test helpers in the same module:

```python
SESSION_ID = "01J00000000000000000000000"


def control_envelope(sequence: int, message_type: MessageType, payload: object) -> MessageEnvelope[object]:
    return MessageEnvelope(
        1, SESSION_ID, message_type, sequence, ProcessSource.GUI, sequence * 100, payload
    )


def make_open_envelope(session_dir: Path, sequence: int) -> MessageEnvelope[object]:
    return control_envelope(
        sequence,
        MessageType.WRITER_OPEN_SESSION,
        WriterOpenSession(session_dir, make_manifest(), make_discussion_state()),
    )


def make_audio_command(source_start_sample: int) -> AudioWriteCommand:
    return AudioWriteCommand(
        AudioSource.ME,
        b"\x00\x00" * 12_800,
        source_start_sample,
        source_start_sample + 12_800,
        source_start_sample // 16,
        1_000 + source_start_sample // 16,
    )


def make_transcript_envelope(sequence: int) -> MessageEnvelope[object]:
    return control_envelope(
        sequence,
        MessageType.TRANSCRIPT_COMMITTED,
        TranscriptCommitted(make_transcript_record(1)),
    )


def make_finalize_envelope(sequence: int) -> MessageEnvelope[object]:
    return control_envelope(sequence, MessageType.WRITER_FINALIZE, make_finalize_command())


def make_shutdown_envelope(sequence: int) -> MessageEnvelope[object]:
    return control_envelope(sequence, MessageType.WRITER_SHUTDOWN, WriterShutdown())


def assert_ack(responses: Queue[object], acknowledged_sequence: int) -> None:
    response = responses.get(timeout=5)
    assert response.message_type is MessageType.WRITER_ACK
    assert isinstance(response.payload, WriterAck)
    assert response.payload.acknowledged_sequence == acknowledged_sequence
```

Import the named Task 4 contracts and `tests.factories` functions explicitly. Each queue read uses `get(timeout=5)` so failures terminate rather than hang.

- [ ] **Step 2: Write failing fatal-path test**

```python
def test_writer_reports_fatal_and_leaves_incomplete_on_audio_gap(tmp_path) -> None:
    context = get_context("spawn")
    control = context.Queue()
    audio = context.Queue()
    responses = context.Queue()
    stop = context.Event()
    session_dir = tmp_path / "session"
    process = context.Process(
        target=run_writer_worker,
        args=(control, audio, responses, stop),
    )
    process.start()
    control.put(make_open_envelope(session_dir, sequence=1))
    assert_ack(responses, 1)
    audio.put(make_audio_command(source_start_sample=640))
    fatal = responses.get(timeout=5)
    assert fatal.message_type is MessageType.WRITER_FATAL
    assert fatal.payload.failed_sequence == 0
    process.join(timeout=10)
    assert process.exitcode != 0
    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\workers\test_writer.py -v`

Expected: collection fails because `workers/writer.py` is absent.

- [ ] **Step 4: Implement bounded-fair queue dispatch**

Before open, accept only a valid sequence-1 `WRITER_OPEN_SESSION` control envelope. After open, drain at most 64 immediately available audio commands, then process at most one control envelope, call `sync_if_due(time.monotonic())`, and repeat. Use queue timeouts no greater than 100 ms so the one-second fsync deadline and stop event remain observable. Reject objects of any unrecognized runtime type. On `stop_event` or dead `multiprocessing.parent_process()`, call `close_incomplete()` and exit.

- [ ] **Step 5: Implement acknowledgements and fail-closed errors**

Successful control mutations emit a Writer-sourced `MessageEnvelope[WriterAck]` with writer-local response sequence and timezone-aware latest-save timestamp. Duplicate control messages emit an ACK without repeating the mutation; sequence gaps and unknown schemas emit `WriterFatal`. Any `OSError`, `PersistenceInvariantError`, or unexpected exception closes incomplete, emits a `WriterFatal` containing exception class and message but no transcript/audio content, calls `response_queue.close()` and `response_queue.join_thread()` so the fatal envelope reaches the controller, then exits nonzero by re-raising.

- [ ] **Step 6: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\workers\test_writer.py -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\workers tests\workers
.\.venv\Scripts\python.exe -m mypy src\flowlens\workers tests\workers
```

Expected: all commands pass and no test process remains running.

### Task 12: Recovery Scan and Typed Artifact Validation

**Files:**
- Create: `src/flowlens/persistence/recovery.py`
- Create: `tests/persistence/recovery_support.py`
- Test: `tests/persistence/test_recovery_scan.py`

**Interfaces:**
- Consumes: strict `SessionManifest.from_dict()`, read-only `inspect_jsonl_tail()`, record parsers, and WAV header inspection rules.
- Produces: `find_incomplete_sessions(sessions_root: Path) -> tuple[Path, ...]`, `RecoveryReport`, `RecoveryError`, and private typed validators used by the coordinator.

`RecoveryReport` is a frozen slotted dataclass with the exact fields `session_id: str`, `session_dir: Path`, `discarded_jsonl_tail_bytes: dict[str, int]`, `repaired_wav_headers: tuple[str, ...]`, `transcript_entry_count: int`, `final_discussion_state_revision: int`, and `active_duration_ms: int`. `RecoveryError` stores the failing artifact path and reason in its message.

- [ ] **Step 1: Write failing deterministic scan test**

```python
from flowlens.persistence.recovery import find_incomplete_sessions


def test_scan_returns_only_valid_incomplete_sessions_in_name_order(tmp_path) -> None:
    create_session_fixture(tmp_path / "20260819T120000+0900_01J00000000000000000000000", "incomplete")
    create_session_fixture(tmp_path / "20260819T130000+0900_01J00000000000000000000001", "completed")
    create_session_fixture(tmp_path / "20260819T110000+0900_01J00000000000000000000002", "incomplete")
    assert [path.name for path in find_incomplete_sessions(tmp_path)] == [
        "20260819T110000+0900_01J00000000000000000000002",
        "20260819T120000+0900_01J00000000000000000000000",
    ]
```

- [ ] **Step 2: Write failing record-validation test**

```python
import pytest

from flowlens.persistence.recovery import RecoveryError, inspect_incomplete_session


def test_complete_but_schema_invalid_jsonl_record_blocks_recovery(tmp_path) -> None:
    session_dir = create_session_fixture(tmp_path / "session", "incomplete")
    (session_dir / "events.jsonl").write_text('{"schema_version":1,"unexpected":true}\n', encoding="utf-8", newline="\n")
    original = (session_dir / "events.jsonl").read_bytes()
    with pytest.raises(RecoveryError, match="events.jsonl.*line 1"):
        inspect_incomplete_session(session_dir)
    assert (session_dir / "events.jsonl").read_bytes() == original
```

Create the shared recovery fixture builder in `tests/persistence/recovery_support.py`:

```python
def create_session_fixture(session_dir: Path, status: str) -> Path:
    session_id = session_dir.name[-26:]
    if len(session_id) != 26:
        session_id = "01J00000000000000000000000"
    writer = SessionWriter.open(
        session_dir,
        make_manifest(session_id=session_id),
        make_discussion_state(),
    )
    if status == "completed":
        writer.finalize(make_finalize_command(session_id=session_id))
    elif status == "incomplete":
        writer.close_incomplete()
    else:
        raise ValueError(f"unsupported fixture status: {status}")
    return session_dir
```

Import `Path`, `SessionWriter`, and the three `tests.factories` functions explicitly.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\persistence\test_recovery_scan.py -v`

Expected: collection fails because `recovery.py` is absent.

- [ ] **Step 4: Implement read-only scan and inspection**

Ignore non-directories and directories without `session.json`. Strictly parse manifests and select only status `INCOMPLETE`; malformed manifests raise `RecoveryError` naming the path rather than being silently skipped. Inspect all required artifact names, call read-only `inspect_jsonl_tail()` for all three logs, then parse every retained transcript/state-history/event line through its strict domain `from_dict()`. Verify transcript and event sequences are contiguous from 1, state revisions are contiguous, every record session ID matches the manifest, and both WAVs satisfy the fixed-format preconditions. Compare `discussion-state.json` to the final state-history state; if history is newer, include an atomic snapshot rebuild in the repair plan, while a snapshot revision newer than history is a recovery error because no successful state replacement record proves it. Return a private `RecoveryInspection` with next event sequence, counts, final state revision, and planned repair operations; do not modify any file or manifest status during inspection.

- [ ] **Step 5: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\persistence\test_recovery_scan.py -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\persistence tests\persistence
.\.venv\Scripts\python.exe -m mypy src\flowlens\persistence tests\persistence
```

Expected: all commands pass.

### Task 13: End-to-End Incomplete Session Recovery

**Files:**
- Modify: `src/flowlens/persistence/recovery.py`
- Modify: `tests/persistence/recovery_support.py`
- Test: `tests/persistence/test_recovery.py`

**Interfaces:**
- Consumes: recovery inspection, `EventRecord`, `AtomicJsonFile`, `JsonlAppender`, and WAV repair.
- Produces: `recover_incomplete_session(session_dir: Path, recovered_at: datetime) -> RecoveryReport` and `recover_incomplete_sessions(sessions_root: Path, recovered_at: datetime) -> tuple[RecoveryReport, ...]`.

- [ ] **Step 1: Write failing complete recovery test**

```python
import json
import wave
from datetime import datetime

import pytest

from flowlens.persistence.recovery import recover_incomplete_session


def test_recovery_repairs_tails_appends_event_and_marks_recovered_last(tmp_path) -> None:
    session_dir = create_interrupted_session(tmp_path)
    transcript_prefix = (session_dir / "transcript.jsonl").read_bytes()
    (session_dir / "transcript.jsonl").write_bytes(transcript_prefix + b'{"schema_version":')
    corrupt_wav_sizes(session_dir / "mic.wav")
    recovered_at = datetime.fromisoformat("2026-08-19T13:00:00+09:00")
    report = recover_incomplete_session(session_dir, recovered_at)
    manifest = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (session_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["status"] == "recovered"
    assert manifest["ended_at"] == recovered_at.isoformat(timespec="milliseconds")
    assert manifest["transcript_entry_count"] == 1
    assert events[-1]["event_type"] == "SESSION_RECOVERED"
    assert events[-1]["details"]["discarded_jsonl_tail_bytes"]["transcript.jsonl"] > 0
    assert report.session_id == manifest["session_id"]
    with wave.open(str(session_dir / "mic.wav"), "rb") as reader:
        assert reader.getnframes() > 0
```

- [ ] **Step 2: Write failing preservation and idempotency tests**

```python
def test_recovery_preserves_pcm_payload_and_existing_valid_records(tmp_path) -> None:
    session_dir = create_interrupted_session(tmp_path)
    before_pcm = (session_dir / "loopback.wav").read_bytes()[44:]
    before_transcript = (session_dir / "transcript.jsonl").read_bytes()
    recover_incomplete_session(session_dir, aware_recovery_time())
    assert (session_dir / "loopback.wav").read_bytes()[44:] == before_pcm
    assert (session_dir / "transcript.jsonl").read_bytes().startswith(before_transcript)


def test_global_recovery_does_not_recover_already_recovered_session_twice(tmp_path) -> None:
    session_dir = create_interrupted_session(tmp_path)
    first = recover_incomplete_sessions(tmp_path, aware_recovery_time())
    second = recover_incomplete_sessions(tmp_path, aware_recovery_time())
    assert [report.session_dir for report in first] == [session_dir]
    assert second == ()
    events = (session_dir / "events.jsonl").read_text(encoding="utf-8")
    assert events.count("SESSION_RECOVERED") == 1
```

- [ ] **Step 3: Write failing final-mutation test**

```python
def test_manifest_remains_incomplete_when_recovery_event_append_fails(tmp_path, monkeypatch) -> None:
    session_dir = create_interrupted_session(tmp_path)
    def fail_append(self, value):
        raise OSError("disk full")
    monkeypatch.setattr("flowlens.persistence.recovery.JsonlAppender.append", fail_append)
    with pytest.raises(OSError, match="disk full"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert load_manifest_status(session_dir) == "incomplete"
```

Add a retry test for a crash after the recovery event is durable but before the final manifest replacement:

```python
def test_retry_reuses_durable_recovery_event_without_duplicate(tmp_path, monkeypatch) -> None:
    session_dir = create_interrupted_session(tmp_path)
    original_replace = AtomicJsonFile.replace

    def fail_recovered_manifest_once(self, value):
        is_recovered = isinstance(value, dict) and value.get("status") == "recovered"
        if self.path.name == "session.json" and is_recovered:
            raise OSError("manifest replace failed")
        original_replace(self, value)

    monkeypatch.setattr(AtomicJsonFile, "replace", fail_recovered_manifest_once)
    with pytest.raises(OSError, match="manifest replace failed"):
        recover_incomplete_session(session_dir, aware_recovery_time())
    assert load_manifest_status(session_dir) == "incomplete"

    monkeypatch.setattr(AtomicJsonFile, "replace", original_replace)
    recover_incomplete_session(session_dir, aware_recovery_time())
    events = (session_dir / "events.jsonl").read_text(encoding="utf-8")
    assert events.count("SESSION_RECOVERED") == 1
    assert load_manifest_status(session_dir) == "recovered"
```

Append these exact helpers to `tests/persistence/recovery_support.py`:

```python
def aware_recovery_time() -> datetime:
    return datetime.fromisoformat("2026-08-19T13:00:00+09:00")


def create_interrupted_session(sessions_root: Path) -> Path:
    session_dir = sessions_root / "20260819T120000+0900_01J00000000000000000000000"
    writer = SessionWriter.open(
        session_dir,
        make_manifest(),
        make_discussion_state(),
    )
    writer.append_audio(
        AudioWriteCommand(AudioSource.ME, b"\x00\x00" * 12_800, 0, 12_800, 0, 1_800)
    )
    writer.append_audio(
        AudioWriteCommand(
            AudioSource.OTHERS,
            b"\x01\x00" * 12_800,
            0,
            12_800,
            0,
            1_800,
        )
    )
    writer.append_transcript(make_transcript_record(1))
    writer.append_event(
        EventRecord(
            1,
            "01J00000000000000000000000",
            1,
            EventType.SESSION_START,
            ProcessSource.GUI,
            0,
            datetime.fromisoformat("2026-08-19T12:00:00+09:00"),
            {},
        )
    )
    writer.close_incomplete()
    return session_dir


def corrupt_wav_sizes(path: Path) -> None:
    with path.open("r+b") as file:
        file.seek(4)
        file.write(struct.pack("<I", 0))
        file.seek(40)
        file.write(struct.pack("<I", 0))


def load_manifest_status(session_dir: Path) -> str:
    value = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    status = value["status"]
    if not isinstance(status, str):
        raise TypeError("manifest status must be a string")
    return status
```

Import `json`, `struct`, `datetime`, domain message/enum types, and test factories explicitly. The recovery tests import these helpers rather than redefining them.

- [ ] **Step 4: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests\persistence\test_recovery.py -v`

Expected: failures report absent recovery coordinator behavior.

- [ ] **Step 5: Implement the recovery transaction**

Require an aware `recovered_at`. In deterministic filename order: inspect all artifacts without mutation; apply each approved JSONL tail repair; repair both WAV headers; atomically rebuild `discussion-state.json` only when final state history is newer; re-parse all retained records; append and fsync one `SESSION_RECOVERED` event sourced from `GUI` at the next event sequence; build recovery notes naming discarded byte counts and repaired WAV header byte counts; atomically replace `session.json` last with status `RECOVERED`, `ended_at=recovered_at`, recovered transcript count/final state revision, pause intervals reconstructed from valid `PAUSE_START`/`PAUSE_END` pairs, and `active_duration_ms` derived from the larger valid source sample count at 16,000 Hz. Event details are exactly `{"discarded_jsonl_tail_bytes": {name: count}, "repaired_wav_headers": [sorted names]}` and contain no transcript text or PCM. An unmatched final `PAUSE_START` becomes a pause ending at the recovered active-duration boundary and is named in recovery notes. If inspection finds a valid final `SESSION_RECOVERED` event while the manifest remains `incomplete`, treat it as an interrupted recovery transaction: reuse that event's timestamp/details, do not append a second event, and retry only the final recovered-manifest replacement.

- [ ] **Step 6: Run focused and static checks**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\persistence\test_recovery.py -v
.\.venv\Scripts\python.exe -m pytest tests\persistence -v
.\.venv\Scripts\python.exe -m ruff check src\flowlens\persistence tests\persistence
.\.venv\Scripts\python.exe -m mypy src\flowlens\persistence tests\persistence
```

Expected: all commands pass.

### Task 14: Persistence Boundary Integration and Full Foundation Gate

**Files:**
- Test: `tests/integration/test_persistence_boundary.py`

**Interfaces:**
- Consumes: all contracts, Writer process, config store, and recovery entrypoint from Tasks 1-13.
- Produces: a regression proof that a spawned Writer creates/finalizes all seven artifacts and a separate interrupted session is recoverable offline.

- [ ] **Step 1: Write the failing integration test**

```python
import json
import socket
import wave
from multiprocessing import get_context

from flowlens.persistence.recovery import recover_incomplete_sessions
from flowlens.workers.writer import run_writer_worker


def test_writer_and_recovery_boundary_is_complete_and_offline(tmp_path, monkeypatch) -> None:
    def network_forbidden(*args, **kwargs):
        raise AssertionError("foundation persistence attempted network access")
    monkeypatch.setattr(socket, "socket", network_forbidden)

    completed_dir = run_scripted_writer_session(tmp_path / "sessions", finalize=True)
    assert required_artifact_names(completed_dir) == {
        "session.json", "mic.wav", "loopback.wav", "transcript.jsonl",
        "discussion-state.json", "state-history.jsonl", "events.jsonl",
    }
    assert load_manifest_status(completed_dir) == "completed"
    assert load_transcript_sequences(completed_dir) == [1, 2]
    for wav_name in ("mic.wav", "loopback.wav"):
        with wave.open(str(completed_dir / wav_name), "rb") as reader:
            assert (reader.getframerate(), reader.getsampwidth(), reader.getnchannels()) == (16000, 2, 1)

    interrupted_dir = run_scripted_writer_session(tmp_path / "sessions", finalize=False)
    damage_only_final_jsonl_fragment_and_wav_headers(interrupted_dir)
    reports = recover_incomplete_sessions(tmp_path / "sessions", aware_recovery_time())
    assert [report.session_dir for report in reports] == [interrupted_dir]
    assert load_manifest_status(interrupted_dir) == "recovered"
```

Keep all deterministic helper implementations in this integration test file. Use queue operations with five-second timeouts and `process.join(timeout=10)`; terminate a process in test cleanup only when it failed to exit, then fail the test with its PID.

- [ ] **Step 2: Run the integration test before adding any fixes**

Run: `.\.venv\Scripts\python.exe -m pytest tests\integration\test_persistence_boundary.py -v`

Expected: the test exposes any remaining contract, spawn, flush, ordering, or recovery mismatch. If it already passes, continue without altering production behavior.

- [ ] **Step 3: Make only the minimal cross-boundary corrections demonstrated by the failing assertion**

Allowed production files are the exact files introduced by Tasks 1-13. Preserve all public signatures from the Shared Interface Catalog. Add a focused regression assertion to the owning unit test whenever a correction changes observable behavior.

- [ ] **Step 4: Run the complete automated gate**

Run:

```powershell
.\.venv\Scripts\python.exe -m black --check src tests
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m mypy src tests
.\.venv\Scripts\python.exe -m pytest --cov=flowlens --cov-report=term-missing --cov-fail-under=90 -v
```

Expected: all four commands exit 0, statement coverage is at least 90%, all spawned processes exit, and the test run makes no network request.

- [ ] **Step 5: Verify repository text formats and scope**

Run:

```powershell
$files = rg --files pyproject.toml requirements.txt requirements-dev.txt src tests
$badBom = $files | Where-Object { $bytes = [IO.File]::ReadAllBytes($_); $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF }
$badCrLf = $files | Where-Object { [IO.File]::ReadAllText($_).Contains("`r`n") }
if ($badBom) { throw "UTF-8 BOM found: $badBom" }
if ($badCrLf) { throw "CRLF found: $badCrLf" }
git status --short
```

Expected: no BOM or CRLF exception. Git status lists only the files prescribed by this plan plus changes explicitly owned by other concurrently executed plans; no Git staging or commit occurs.

## Implementation Completion Evidence

The implementer records these exact outputs in the task handoff:

- Python version from `.\.venv\Scripts\python.exe --version`.
- Passing Black, Ruff, mypy, and pytest command summaries from Task 14.
- The final integration test's completed and recovered session paths.
- `git status --short`, confirming no staged changes and no files outside the assigned plan scope.
