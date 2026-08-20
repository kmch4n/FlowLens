# FlowLens Analysis and UI Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the local Qwen discussion worker, deterministic session lifecycle, accessible Hallmark PySide6 UI, five-process integration, folder-based Windows package, and executable smoke/acceptance harness for the FlowLens MVP.

**Architecture:** Keep discussion semantics and lifecycle decisions in pure Python cores, with `llama-cpp-python`, Qt, Windows devices, filesystem, clocks, processes, and folder opening behind typed ports. The GUI process owns `SessionController` and routes typed `flowlens.domain.messages.MessageEnvelope` values among Audio, ASR, Discussion, and Writer workers over `multiprocessing.Queue`; large PCM remains on the two dedicated audio queues. Qt Widgets render immutable controller snapshots and never call a model, worker, or filesystem directly.

**Tech Stack:** Python 3.12, PySide6 6.11.2 with Qt Widgets, llama-cpp-python 0.3.35 with CUDA, multiprocessing `spawn`, PyInstaller 6.21.0 `onedir`, pytest/pytest-qt, Black, Ruff, mypy.

**Spec:** `docs/mvp-spec.md`

## Global Constraints

- Target one designated Windows 10 Pro 64-bit PC: Ryzen 7 3800X, approximately 64 GB RAM, GeForce RTX 4060 8 GB, observed NVIDIA driver 596.36, Python 3.12.
- A live session performs no network request; no telemetry, update check, cloud inference, remote logging, local HTTP server, or WebSocket server is permitted.
- Load both models only from `%LOCALAPPDATA%\FlowLens\models\`; a missing file or SHA-256 mismatch blocks start and names the invalid model.
- Discussion model is exactly Qwen/Qwen3-4B-Instruct-2507, pinned official Safetensors converted with a pinned `llama.cpp` revision to GGUF Q4_K_M, context 8192, all GPU layers, temperature 0, maximum output 512; record the resulting SHA-256 in the model manifest.
- Five OS processes are required: GUI / Session Controller, Audio Worker, ASR Worker, Discussion Worker, and Writer Worker.
- Runtime priority is saved audio, committed transcript, partial transcript, then discussion state. Discussion work must never block audio dispatch.
- Lifecycle is `IDLE -> PREFLIGHT -> STARTING -> RECORDING <-> PAUSED -> STOPPING -> COMPLETED`; `ERROR` is only for fatal session failures.
- All workers, devices, models, and storage acknowledge readiness before recording. Worker readiness times out at 60 seconds. GUI worker-health polling is at most 500 ms apart.
- Preflight needs 500 MB free and blocks on absent/unconfirmed microphone, absent/unconfirmed loopback output, invalid model, or unavailable session directory, with a specific adjacent reason.
- Default window is 1280x800 logical px; minimum is 900x600. At widths at least 1000, transcript/discussion are approximately 62/38; below 1000, transcript precedes discussion vertically.
- Hallmark contract is fixed: Atmospheric genre, Midnight theme, Workbench macrostructure, technical/austere/distraction-free tone, no enrichment, dark only, no pure black/white, gradients, glassmorphism, decorative glow, nested cards, or equal-card grids.
- Use the specification color tokens, IBM Plex Sans JP for UI/transcript, IBM Plex Mono for timer/latency/status, one accent below approximately 3% of visible area, red only for recording-critical errors or destructive recording actions.
- Body text meets WCAG 2.1 AA; focus indication is at least 3:1; state is never color-only; primary hit targets are at least 44x44 logical px; keyboard focus is always visible.
- Only opacity transitions for partial-to-committed and discussion-state replacement plus at most one logical pixel of button press movement are allowed. Never animate layout. Reduced motion makes changes instant or opacity-only at no more than 150 ms.
- The UI supports `Ctrl+Enter`, `Space`, `Ctrl+Shift+S`, `Ctrl+T`, and `Escape` exactly as specified.
- Stop finalization order is fixed. At 30 seconds, show the slow-finalization choice; never force close automatically. Force close leaves the session `incomplete`.
- All dependency versions are exact pins. Production and tests use type annotations, four-space indentation, double quotes, Black, Ruff, and mypy. Files stay UTF-8 without BOM and LF, except CSV artifacts, which use UTF-8 with BOM and CRLF.
- Do not add history/review, playback, editing, search, export, retranscription, translation, advice, pro/con generation, evidence links, speaker attribution on discussion items, tray behavior, compact mode, installer, updater, or light theme.
- This plan must be implemented without `git add`, `git commit`, `git push`, or GitHub write actions.

## Shared Contract Prerequisites and Ownership

The foundation/persistence plan owns these modules. Do not redefine or fork them:

- `flowlens.domain.enums`: `SessionMode`, `AudioSource`, `SessionStatus`, `ProcessSource`, `EventType`, `MessageType`.
- `flowlens.domain.session`: `SessionManifest`, `PauseInterval`.
- `flowlens.domain.discussion`: immutable `DiscussionState` and `StateHistoryRecord`.
- `flowlens.domain.messages`: generic immutable `MessageEnvelope`, `TranscriptRecord`, `TranscriptCommitted`, `EventRecord`, `DiscussionStateReplaced`, `AudioWriteCommand`, `AudioDrainFence`, `WriterOpenSession(session_dir, manifest, initial_state)`, `WriterAppendEvent`, `WriterFlush`, `WriterFinalize`, `WriterShutdown`, `WriterAck`, and `WriterFatal`.
- `flowlens.config.model`: `AppConfig`, `WindowPreferences`, `DevicePreferences`; `ConfigStore.load()` and `ConfigStore.save()`.
- Foundation-owned dependency files provide exact runtime pins `PySide6==6.11.2` and `llama-cpp-python==0.3.35`, plus exact development pins `PyInstaller==6.21.0` and `pytest-qt==4.5.0`. Tasks below verify these pins and do not introduce a second dependency file.

The Audio/ASR plan owns:

```python
def run_audio_worker(
    config: AudioWorkerConfig,
    control_in: Queue[MessageEnvelope[object]],
    control_out: Queue[MessageEnvelope[object]],
    writer_audio_out: Queue[AudioWriteCommand | AudioDrainFence],
    asr_audio_out: Queue[AudioFrame],
) -> None: ...

def run_asr_worker(
    config: AsrWorkerConfig,
    audio_in: Queue[AudioFrame],
    control_in: Queue[MessageEnvelope[object]],
    control_out: Queue[MessageEnvelope[object]],
) -> None: ...
```

This plan consumes `AudioWorkerConfig`, `AsrWorkerConfig`, `AudioFrame`, Audio readiness/level/disconnect/drain messages, ASR partial/committed/status/drain messages, and their exact `MessageType` members. A committed ASR payload contains the shared `TranscriptRecord`; its order field is `sequence`.

Control messages use shared `WORKER_START`, `WORKER_PAUSE`, `WORKER_RESUME`, and `WORKER_STOP` with a typed payload naming the target worker. Drain acknowledgements use `WORKER_STOPPED`: Audio reports `writer_frames` and `asr_frames`, ASR reports `committed_count`, and Discussion reports `final_revision` and `pending_count`, with every payload also carrying `worker` and `drained=true`. Audio `WORKER_STOPPED/drained=true` specifically means Audio has stopped capture, put every final `AudioWriteCommand`, and then put exactly one shared `AudioDrainFence` on the dedicated Writer audio queue. Runtime status uses shared `AUDIO_LEVEL`, `SOURCE_DISCONNECTED`, `SOURCE_RECONNECTED`, `ASR_STATUS`, and `DISCUSSION_STATUS`. `ASR_STATUS` is authoritative and carries `state`, `backlog_ms`, and `analysis_paused`. A discussion failure is a `DISCUSSION_STATUS` runtime message and a persisted `EventType.ANALYSIS_FAILED`; there is no separate failure-only message type.

## File Map

Create or modify only the following implementation surfaces when executing this plan:

- `src/flowlens/discussion/contracts.py` — model request, backend protocol, worker configuration, typed status/failure payloads.
- `src/flowlens/discussion/schema.py` — closed JSON Schema and strict snapshot parser.
- `src/flowlens/discussion/prompt.py` — mode framing and field semantics, with explicit anti-advice language.
- `src/flowlens/discussion/context.py` — newest-first selection capped at 60 records and 6,000 transcript tokens.
- `src/flowlens/discussion/scheduler.py` — meaningful-commit filtering, 500 ms coalescing, one-in-flight rule, retry-on-next-commit behavior.
- `src/flowlens/discussion/llama_cpp_adapter.py` — local Qwen GGUF adapter and tokenizer.
- `src/flowlens/discussion/worker.py` — queue loop and parent-liveness handling.
- `scripts/prepare_qwen_model.ps1` — explicit pinned initial-setup conversion into local Q4_K_M plus manifest SHA-256.
- `src/flowlens/controller/models.py` — lifecycle/UI snapshots, preflight selection/report, completion summary.
- `src/flowlens/controller/ports.py` — device, model, storage, clock, worker runtime, folder opener, and accessibility ports.
- `src/flowlens/controller/preflight.py` — exact blocking logic.
- `src/flowlens/controller/routing.py` — sender sequence/gap detection and message fan-out decisions.
- `src/flowlens/controller/supervision.py` — health cadence and single-restart policy.
- `src/flowlens/controller/finalization.py` — ordered acknowledgement-driven stop sequence.
- `src/flowlens/controller/session_controller.py` — pure lifecycle coordinator.
- `src/flowlens/adapters/local_models.py` — manifest/path/checksum probe with no remote fallback.
- `src/flowlens/adapters/storage.py` — 500 MB and writability probe.
- `src/flowlens/adapters/windows_devices.py` — device discovery/meter bridge to the Audio adapter.
- `src/flowlens/adapters/windows_shell.py` — absolute folder opener and reduced-motion lookup.
- `src/flowlens/integration/worker_runtime.py` — `spawn` processes, queues, polling, restart, shutdown.
- `src/flowlens/integration/composition.py` — production dependency graph.
- `src/flowlens/ui/design.py`, `assets/styles/flowlens.qss`, `assets/fonts/*` — Hallmark tokens, fonts, QSS, contrast/motion utilities.
- `src/flowlens/ui/widgets.py` — input meter, status indicator, accessible button/select helpers.
- `src/flowlens/ui/preflight_page.py` — mode/device/model/storage screen.
- `src/flowlens/ui/transcript_model.py`, `src/flowlens/ui/transcript_view.py` — merged immutable transcript and partial rows.
- `src/flowlens/ui/discussion_panel.py`, `src/flowlens/ui/status_strip.py`, `src/flowlens/ui/live_page.py` — live Workbench surface.
- `src/flowlens/ui/dialogs.py`, `src/flowlens/ui/completion_page.py` — stop/finalization/completion UI.
- `src/flowlens/ui/presenter.py`, `src/flowlens/ui/main_window.py` — controller binding, shortcuts, geometry, close path.
- `src/flowlens/app.py`, `src/flowlens/__main__.py` — CLI options, early `freeze_support()`, application entry.
- `packaging/FlowLens.spec`, `packaging/hooks/*`, `scripts/build_windows.ps1` — folder build.
- `scripts/smoke_discussion.py`, `scripts/smoke_integration.py`, `scripts/validate_session.py`, `scripts/collect_acceptance.py`, `scripts/run_acceptance.ps1` — real-model, real-hardware, offline, performance, recovery evidence.
- Matching tests under `tests/discussion/`, `tests/controller/`, `tests/integration/`, `tests/ui/`, `tests/packaging/`, and `tests/smoke/`.

---

### Task 1: Discussion Output Contract, Prompt, and Bounded Context

**Files:**
- Create: `src/flowlens/discussion/contracts.py`
- Create: `src/flowlens/discussion/schema.py`
- Create: `src/flowlens/discussion/prompt.py`
- Create: `src/flowlens/discussion/context.py`
- Create: `tests/discussion/factories.py`
- Create: `tests/discussion/test_schema.py`
- Create: `tests/discussion/test_prompt.py`
- Create: `tests/discussion/test_context.py`

**Interfaces:**
- Consumes: `SessionMode`, `DiscussionState`, `TranscriptRecord` from the shared domain modules.
- Produces: `ChatMessage(role: Literal["system", "user"], content: str)`, `DiscussionRequest(current_state, records, requested_revision, updated_at)`, `DiscussionStatusPayload(state, revision, pending_count, error_code)`, `DiscussionStoppedPayload(worker, drained, final_revision, pending_count)`, `DiscussionBackend.count_tokens(text) -> int`, `DiscussionBackend.generate(messages, response_schema) -> str`, `DiscussionOutputError`, `DiscussionGenerationError`, `DiscussionContextError`, `discussion_state_schema(request) -> dict[str, object]`, `parse_discussion_state(raw, request) -> DiscussionState`, `build_messages(request) -> tuple[ChatMessage, ...]`, `select_recent_records(records, count_tokens) -> tuple[TranscriptRecord, ...]`.

- [ ] **Step 1: Write closed-schema and strict-parser tests**

```python
def test_schema_locks_revision_mode_timestamp_and_extra_fields() -> None:
    request = make_request(mode=SessionMode.INTERVIEW, revision=7)
    schema = discussion_state_schema(request)
    properties = schema["properties"]
    assert schema["additionalProperties"] is False
    assert properties["revision"] == {"const": 7}
    assert properties["mode"] == {"const": "INTERVIEW"}
    assert properties["updated_at"] == {"const": "2026-08-19T12:35:02.125+09:00"}
    assert schema["required"] == [
        "revision",
        "mode",
        "current_focus",
        "key_points",
        "confirmed_outcomes",
        "follow_up_items",
        "updated_at",
    ]


def test_parser_rejects_wrong_revision_and_keeps_previous_state_external() -> None:
    request = make_request(mode=SessionMode.MEETING, revision=4)
    raw = json.dumps(
        {
            "revision": 5,
            "mode": "MEETING",
            "current_focus": "Scope",
            "key_points": [],
            "confirmed_outcomes": [],
            "follow_up_items": [],
            "updated_at": "2026-08-19T12:35:02.125+09:00",
        },
        ensure_ascii=False,
    )
    with pytest.raises(DiscussionOutputError, match="revision"):
        parse_discussion_state(raw, request)
```

- [ ] **Step 2: Run schema tests and confirm the missing-module failure**

Run: `python -m pytest tests/discussion/test_schema.py -q`

Expected: collection fails with `ModuleNotFoundError: No module named 'flowlens.discussion.schema'`.

- [ ] **Step 3: Implement the immutable request/protocol and schema/parser**

```python
@dataclass(frozen=True, slots=True)
class DiscussionRequest:
    current_state: DiscussionState
    records: tuple[TranscriptRecord, ...]
    requested_revision: int
    updated_at: datetime


class DiscussionBackend(Protocol):
    def count_tokens(self, text: str) -> int:
        """Return tokenizer tokens without loading or contacting a remote model."""

    def generate(
        self,
        messages: tuple[ChatMessage, ...],
        response_schema: dict[str, object],
    ) -> str:
        """Return one grammar-constrained JSON object."""
```

Implement the seven required properties as closed JSON Schema properties. Parse with `json.loads`, reject non-objects, extra/missing fields, wrong constants, non-string items, and non-timezone timestamps, then construct a new immutable `DiscussionState`. Do not mutate or pass the old state into the parser as a fallback; callers retain it on exceptions.

- [ ] **Step 4: Write prompt semantic tests for all three modes**

```python
@pytest.mark.parametrize(
    ("mode", "required_label", "forbidden_label"),
    [
        (SessionMode.MEETING, "Current focus", "Current question / topic"),
        (SessionMode.INTERVIEW, "Current question / topic", "unresolved issues"),
        (SessionMode.GENERAL, "Current topic", "Decisions / confirmations"),
    ],
)
def test_prompt_uses_mode_framing_and_prohibits_advice(
    mode: SessionMode,
    required_label: str,
    forbidden_label: str,
) -> None:
    prompt = "\n".join(message.content for message in build_messages(make_request(mode=mode)))
    assert required_label in prompt
    assert forbidden_label not in prompt
    assert "Return a complete replacement snapshot" in prompt
    assert "Do not invent facts, decisions, questions, or next actions" in prompt
    assert "Do not suggest what anyone should say" in prompt
    for field in (
        "current_focus",
        "key_points",
        "confirmed_outcomes",
        "follow_up_items",
        "updated_at",
    ):
        assert field in prompt
```

- [ ] **Step 5: Implement exact mode text and full field semantics**

Use one system message for immutable safety/semantic rules and one user message containing the current full state plus selected new records. Serialize JSON with `ensure_ascii=False`, `sort_keys=True`, and four-space indentation. Do not add advice, evaluation, pros/cons, speaker attribution, or transcript evidence IDs to prompt examples.

- [ ] **Step 6: Write context-cap tests**

```python
def test_context_keeps_newest_60_in_chronological_order() -> None:
    records = tuple(make_record(sequence=index, text="一") for index in range(1, 66))
    selected = select_recent_records(records, lambda text: 1)
    assert [record.sequence for record in selected] == list(range(6, 66))


def test_context_drops_oldest_until_transcript_tokens_fit() -> None:
    records = (
        make_record(sequence=1, text="a" * 3000),
        make_record(sequence=2, text="b" * 3000),
        make_record(sequence=3, text="c" * 3000),
    )
    selected = select_recent_records(records, len)
    assert [record.sequence for record in selected] == [2, 3]


def test_context_rejects_one_record_larger_than_cap_without_truncating_it() -> None:
    with pytest.raises(DiscussionContextError, match="6000"):
        select_recent_records((make_record(sequence=1, text="x" * 6001),), len)
```

- [ ] **Step 7: Implement context selection and run the task gate**

Sort by `sequence`, retain at most the newest 60, calculate tokens on each record's exact serialized transcript representation, and remove oldest whole records until at most 6,000 transcript tokens remain. Never trim transcript text or the current state.

Run: `python -m pytest tests/discussion/test_schema.py tests/discussion/test_prompt.py tests/discussion/test_context.py -q`

Expected: all tests pass.

### Task 2: Coalescing and Failure-Safe Discussion Scheduler

**Files:**
- Create: `src/flowlens/discussion/scheduler.py`
- Create: `tests/discussion/test_scheduler.py`

**Interfaces:**
- Consumes: `TranscriptRecord`, `DiscussionState`, `DiscussionRequest`.
- Produces: `DiscussionScheduler.add(record, now_ms) -> None`, `set_paused(paused) -> None`, `next_request(now_ms, updated_at) -> DiscussionRequest | None`, `succeed(request, new_state) -> None`, `fail(request) -> None`, `final_request(updated_at) -> DiscussionRequest | None`, `has_pending: bool`.

- [ ] **Step 1: Write meaningful-entry filter tests**

```python
@pytest.mark.parametrize("text", ["", "   ", "[音楽]", "(無音)", "えー", "えっと。", "あー……"])
def test_non_meaningful_text_does_not_schedule(text: str) -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text=text), now_ms=0)
    assert scheduler.has_pending is False


@pytest.mark.parametrize("text", ["はい", "いいえ", "そうです", "えっと、結論は進めます"])
def test_short_answers_and_meaningful_sentences_schedule(text: str) -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text=text), now_ms=0)
    assert scheduler.has_pending is True
```

- [ ] **Step 2: Run the filter tests and confirm failure**

Run: `python -m pytest tests/discussion/test_scheduler.py -q`

Expected: collection fails because `DiscussionScheduler` does not exist.

- [ ] **Step 3: Implement a conservative hesitation filter**

Normalize whitespace and Japanese punctuation, discard known non-speech markers, and treat only a full normalized match in `{"えー", "えっと", "あー", "あの"}` as hesitation-only. Never substring-filter a meaningful sentence and never filter `はい`, `いいえ`, or equivalents.

- [ ] **Step 4: Write timing, in-flight, and retry tests**

```python
def test_coalesces_for_500_ms_and_allows_only_one_request() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text="方針を確認します"), now_ms=100)
    scheduler.add(make_record(sequence=2, text="はい"), now_ms=450)
    assert scheduler.next_request(now_ms=949, updated_at=NOW) is None
    request = scheduler.next_request(now_ms=950, updated_at=NOW)
    assert request is not None
    assert [record.sequence for record in request.records] == [1, 2]
    scheduler.add(make_record(sequence=3, text="次へ進みます"), now_ms=1000)
    assert scheduler.next_request(now_ms=2000, updated_at=NOW) is None


def test_failure_retains_batch_but_waits_for_next_meaningful_commit() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text="論点です"), now_ms=0)
    failed = scheduler.next_request(now_ms=500, updated_at=NOW)
    assert failed is not None
    scheduler.fail(failed)
    assert scheduler.next_request(now_ms=10_000, updated_at=NOW) is None
    scheduler.add(make_record(sequence=2, text="補足です"), now_ms=10_001)
    retry = scheduler.next_request(now_ms=10_501, updated_at=NOW)
    assert retry is not None
    assert [record.sequence for record in retry.records] == [1, 2]


def test_success_replaces_state_and_removes_only_completed_batch() -> None:
    scheduler = make_scheduler()
    scheduler.add(make_record(sequence=1, text="一件目"), now_ms=0)
    request = scheduler.next_request(now_ms=500, updated_at=NOW)
    assert request is not None
    scheduler.add(make_record(sequence=2, text="二件目"), now_ms=501)
    scheduler.succeed(request, make_state(revision=1))
    assert scheduler.current_state.revision == 1
    assert scheduler.has_pending is True
```

- [ ] **Step 5: Implement scheduler state transitions**

Maintain `pending`, `in_flight`, `paused`, `coalesce_deadline_ms`, and `needs_new_commit_after_failure`. Never launch while paused or in flight. On success remove only records in that request; on invalid/failure retain them and require another meaningful commit before retry. `final_request` bypasses the coalescing delay, launches at most once, and leaves the previous state valid if it fails.

- [ ] **Step 6: Run the scheduler gate**

Run: `python -m pytest tests/discussion/test_scheduler.py -q`

Expected: all tests pass.

### Task 3: Local llama.cpp Qwen Adapter

**Files:**
- Create: `src/flowlens/discussion/llama_cpp_adapter.py`
- Create: `tests/discussion/test_llama_cpp_adapter.py`
- Create: `scripts/prepare_qwen_model.ps1`
- Create: `tests/discussion/test_prepare_qwen_model.py`

**Interfaces:**
- Consumes: `DiscussionBackend`, `ChatMessage`.
- Produces: checked `DiscussionModelConfig(model_path: Path, sha256: str, n_ctx: int = 8192, n_gpu_layers: int = -1, temperature: float = 0.0, max_tokens: int = 512)`, `LlamaCppDiscussionBackend(config, cl)`, `load_llama_cpp_backend(config) -> LlamaCppDiscussionBackend`.

- [ ] **Step 1: Verify the foundation-owned runtime pin**

Assert `requirements.txt` contains exactly `llama-cpp-python==0.3.35`. The designated-PC setup builds/installs its CUDA wheel with the approved pinned `llama.cpp` revision; this adapter never invokes `from_pretrained`, Hugging Face APIs, or any URL.

- [ ] **Step 2: Write a reproducible model-preparation test**

```python
def test_qwen_preparation_pins_source_converter_and_quantization() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
    assert requirements.count("llama-cpp-python==0.3.35") == 1
    script = Path("scripts/prepare_qwen_model.ps1").read_text(encoding="utf-8")
    assert "cdbee75f17c01a7cc42f958dc650907174af0554" in script
    assert "2e92ecd0247d25f09797f8fdb044a166522fc05d" in script
    assert "Q4_K_M" in script
    assert "Get-FileHash" in script
    assert "manifest.json" in script
    assert "Qwen3-4B-Instruct-2507-Q4_K_M.gguf" in script


def test_qwen_preparation_uses_staging_then_atomic_move() -> None:
    script = Path("scripts/prepare_qwen_model.ps1").read_text(encoding="utf-8")
    assert ".staging-qwen-" in script
    assert "Move-Item -LiteralPath" in script
    assert "from_pretrained" not in script


def test_qwen_preparation_preserves_installed_asr_manifest_entry() -> None:
    result = update_manifest(
        {"models": {"kotoba-whisper-v2.0-faster": {"sha256": "asr-hash"}}},
        prepared_qwen_entry(sha256="qwen-hash"),
    )
    assert result["models"]["kotoba-whisper-v2.0-faster"]["sha256"] == "asr-hash"
    assert result["models"]["qwen3-4b-instruct-2507"]["sha256"] == "qwen-hash"
```

- [ ] **Step 3: Implement the explicit initial-setup conversion script**

`prepare_qwen_model.ps1` runs only on explicit initial setup, never at session runtime. It validates `%LOCALAPPDATA%`, Git LFS, CMake, and MSVC; clones Qwen/Qwen3-4B-Instruct-2507 at exact revision `cdbee75f17c01a7cc42f958dc650907174af0554`; clones `ggml-org/llama.cpp` at exact revision `2e92ecd0247d25f09797f8fdb044a166522fc05d`; builds the CUDA converter/quantizer; converts official Safetensors to F16 GGUF; quantizes to `Q4_K_M`; computes SHA-256; then atomically moves `Qwen3-4B-Instruct-2507-Q4_K_M.gguf` into `%LOCALAPPDATA%\FlowLens\models\qwen3-4b-instruct-2507\`. It reads and validates an existing manifest, preserves the installed Kotoba entry, updates only the Qwen entry, and atomically replaces `manifest.json` after conversion/checksum succeed, recording repository, both pinned revisions, runtime format, relative path, SHA-256, and Apache-2.0 license. A failed step resolves and verifies its PID-scoped staging directory is inside the model root before removing only that staging directory; it leaves any installed valid model/manifest unchanged.

- [ ] **Step 4: Write spy-backend tests for exact model and generation arguments**

```python
def test_load_uses_local_path_full_gpu_and_fixed_context(tmp_path: Path) -> None:
    model_path = tmp_path / "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
    model_path.write_bytes(b"gguf")
    factory = SpyLlamaFactory()
    load_llama_cpp_backend(make_config(model_path), factory=factory)
    assert factory.kwargs == {
        "model_path": str(model_path),
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "verbose": False,
        "use_mmap": True,
    }


def test_generate_uses_json_schema_grammar_and_fixed_sampling() -> None:
    cl = SpyLlama()
    backend = LlamaCppDiscussionBackend(make_config(Path("model.gguf")), cl)
    raw = backend.generate((ChatMessage("user", "input"),), {"type": "object"})
    assert raw == '{"revision":1}'
    assert cl.call_kwargs["response_format"] == {
        "type": "json_object",
        "schema": {"type": "object"},
    }
    assert cl.call_kwargs["temperature"] == 0.0
    assert cl.call_kwargs["max_tokens"] == 512
    assert cl.call_kwargs["stream"] is False
```

- [ ] **Step 5: Run adapter tests and confirm failure**

Run: `python -m pytest tests/discussion/test_prepare_qwen_model.py tests/discussion/test_llama_cpp_adapter.py -q`

Expected: collection fails because the adapter does not exist.

- [ ] **Step 6: Implement local-only loading and response extraction**

```python
response = self._cl.create_chat_completion(
    messages=[{"role": item.role, "content": item.content} for item in messages],
    response_format={"type": "json_object", "schema": response_schema},
    temperature=self._config.temperature,
    max_tokens=self._config.max_tokens,
    stream=False,
)
content = response["choices"][0]["message"]["content"]
if not isinstance(content, str):
    raise DiscussionGenerationError("llama.cpp returned no text content")
return content
```

Name the model instance `cl`. Refuse a non-absolute or absent path. Tokenize via `cl.tokenize(text.encode("utf-8"), add_bos=False, special=True)`. Convert llama exceptions into `DiscussionGenerationError` without logging prompt or transcript text.

- [ ] **Step 7: Run adapter and static gates**

Run: `python -m pytest tests/discussion/test_llama_cpp_adapter.py -q`

Run: `python -m mypy src/flowlens/discussion`

Expected: tests and type checking pass without a real model or GPU.

### Task 4: Discussion Worker Queue Loop

**Files:**
- Create: `src/flowlens/discussion/worker.py`
- Create: `tests/discussion/test_worker.py`

**Interfaces:**
- Consumes: shared `MessageEnvelope`, `MessageType`, `TranscriptCommitted`, `DiscussionStateReplaced`; `DiscussionScheduler`; `DiscussionBackend`.
- Produces: `DiscussionWorkerConfig(session_id: str, model: DiscussionModelConfig, initial_state: DiscussionState, coalesce_ms: int = 500)`, `run_discussion_worker(config, control_in, control_out) -> None`; `DiscussionWorkerCore.handle(envelope) -> tuple[MessageEnvelope[object], ...]`; `DiscussionWorkerCore.tick(now_ms, now) -> tuple[MessageEnvelope[object], ...]`.
- Message contract: controller sends `WORKER_START`, `TRANSCRIPT_COMMITTED`, `WORKER_PAUSE`, `WORKER_RESUME`, and `WORKER_STOP` with a payload naming `DISCUSSION`; stop also carries `"finalize": true`. Worker emits `WORKER_READY`, `DISCUSSION_STATUS`, `DISCUSSION_STATE_REPLACED`, `WORKER_STOPPED` with `{"worker": "DISCUSSION", "drained": true, "final_revision": int, "pending_count": int}`, and fatal `WORKER_ERROR`.

- [ ] **Step 1: Write pure-core message tests**

```python
def test_committed_message_generates_typed_replacement_after_coalesce() -> None:
    core = make_core(raw=valid_output(revision=1))
    core.handle(committed_envelope(sequence=1, text="方針を確認します", at_ms=0))
    outgoing = core.tick(now_ms=499, now=NOW)
    assert outgoing == ()
    outgoing = core.tick(now_ms=500, now=NOW)
    replacement = outgoing[0]
    assert replacement.message_type is MessageType.DISCUSSION_STATE_REPLACED
    assert isinstance(replacement.payload, DiscussionStateReplaced)
    assert replacement.payload.previous_revision == 0
    assert replacement.payload.state.revision == 1


def test_invalid_output_emits_metadata_only_failure_and_keeps_state() -> None:
    core = make_core(raw="not json")
    core.handle(committed_envelope(sequence=1, text="入力", at_ms=0))
    outgoing = core.tick(now_ms=500, now=NOW)
    assert outgoing[0].message_type is MessageType.DISCUSSION_STATUS
    assert outgoing[0].payload == DiscussionStatusPayload(
        state="FAILED",
        revision=0,
        pending_count=1,
        error_code="INVALID_OUTPUT",
    )
    assert core.state.revision == 0
```

- [ ] **Step 2: Run core tests and confirm failure**

Run: `python -m pytest tests/discussion/test_worker.py -q`

Expected: collection fails because `DiscussionWorkerCore` is absent.

- [ ] **Step 3: Implement core routing, sequence allocation, and safe failures**

Each outgoing envelope has schema version 1, the same session ID, `ProcessSource.DISCUSSION`, a monotonically increasing sender sequence, and the injected monotonic time. Use frozen dataclass payloads for discussion status/stopped messages. Never include prompt text or transcript text in failure payloads.

- [ ] **Step 4: Write pause, final drain, and parent-death loop tests**

```python
def test_pause_defers_generation_and_resume_preserves_pending() -> None:
    core = make_core(raw=valid_output(revision=1))
    core.handle(control_envelope(MessageType.WORKER_PAUSE, {"worker": "DISCUSSION"}))
    core.handle(committed_envelope(sequence=1, text="保留", at_ms=0))
    assert core.tick(now_ms=5_000, now=NOW) == ()
    core.handle(control_envelope(MessageType.WORKER_RESUME, {"worker": "DISCUSSION"}))
    assert core.tick(now_ms=5_500, now=NOW)[0].message_type is MessageType.DISCUSSION_STATE_REPLACED


def test_stop_runs_one_final_request_then_reports_drained() -> None:
    core = make_core(raw=valid_output(revision=1))
    core.handle(committed_envelope(sequence=1, text="最終入力", at_ms=0))
    outgoing = core.handle(
        control_envelope(MessageType.WORKER_STOP, {"worker": "DISCUSSION", "finalize": True})
    )
    assert [message.message_type for message in outgoing] == [
        MessageType.DISCUSSION_STATE_REPLACED,
        MessageType.WORKER_STOPPED,
    ]
    assert outgoing[-1].payload == DiscussionStoppedPayload(
        worker="DISCUSSION",
        drained=True,
        final_revision=1,
        pending_count=0,
    )
```

- [ ] **Step 5: Implement the blocking queue loop around the core**

Poll the control queue no slower than 50 ms while active. Load Qwen once after `WORKER_START` with `{"worker": "DISCUSSION"}`, emit readiness only after successful load, and check `multiprocessing.parent_process().is_alive()` every loop. When the GUI parent disappears, finish an already-created replacement envelope, emit no new generation, and exit. A GPU allocation/load failure becomes `WORKER_ERROR` with code `MODEL_LOAD_FAILED` or `GPU_OOM`; controller policy decides whether to restart or disable analysis.

- [ ] **Step 6: Run worker gate**

Run: `python -m pytest tests/discussion/test_worker.py -q`

Expected: all tests pass with fake queues, clock, parent-liveness probe, and backend.

### Task 5: Preflight Ports, Local Model Probe, and Exact Blocking Reasons

**Files:**
- Create: `src/flowlens/controller/models.py`
- Create: `src/flowlens/controller/ports.py`
- Create: `src/flowlens/controller/preflight.py`
- Create: `src/flowlens/adapters/local_models.py`
- Create: `src/flowlens/adapters/storage.py`
- Create: `src/flowlens/adapters/windows_devices.py`
- Create: `tests/controller/test_preflight.py`
- Create: `tests/adapters/test_local_models.py`

**Interfaces:**
- Produces `DeviceOption(id, display_name, loopback_capable)`, `ModelCheck(model_id, path, ready, reason)`, `StorageCheck(root, free_bytes, writable, reason)`, `PreflightSelection(mode, microphone_id, loopback_output_id)`, `BlockingIssue(control_id, message)`, `PreflightReport(selection, microphones, loopbacks, mic_level, loopback_level, models, storage, destination, issues, can_start)`.
- Runtime-checkable ports: `DeviceCatalog.list_microphones()`, `list_loopback_outputs()`, `read_level(source, device_id)`, `ModelReadiness.check_required()`, `StorageReadiness.check(root, required_bytes)`, `Clock.monotonic_ms()/now()`, `WorkerRuntime`, `FolderOpener`, `MotionPreferences`, `AccessibilityAnnouncer`. Decorate every protocol used by composition tests with `@runtime_checkable`.
- Adapters never expose a vendor object to controller/UI code.

- [ ] **Step 1: Write all preflight blocker tests**

```python
@pytest.mark.parametrize(
    ("setup", "control_id", "message"),
    [
        ("no_mic", "microphone", "Select an available microphone."),
        ("no_loopback", "loopback", "Select a loopback-capable Windows output device."),
        ("asr_missing", "asr_model", "Kotoba-Whisper model files are missing."),
        ("discussion_checksum", "discussion_model", "Qwen discussion model checksum does not match."),
        ("storage_unwritable", "storage", "FlowLens cannot create the session folder."),
        ("storage_small", "storage", "At least 500 MB of free space is required."),
    ],
)
def test_preflight_names_each_blocker(setup: str, control_id: str, message: str) -> None:
    report = build_service(setup).evaluate(default_selection())
    assert BlockingIssue(control_id, message) in report.issues
    assert report.can_start is False


def test_missing_saved_device_is_not_replaced_implicitly() -> None:
    selection = PreflightSelection(SessionMode.MEETING, "gone-mic", "gone-output")
    report = build_service("ready").evaluate(selection)
    assert report.selection.microphone_id is None
    assert report.selection.loopback_output_id is None
    assert report.can_start is False
```

- [ ] **Step 2: Run tests and confirm missing modules**

Run: `python -m pytest tests/controller/test_preflight.py -q`

Expected: collection fails on `flowlens.controller.preflight`.

- [ ] **Step 3: Implement pure preflight evaluation**

Preserve prior IDs only when still present. Filter output options to `loopback_capable=True`. Always return both meter values and all blockers in stable control order. A zero level is visible but is not an additional start blocker because the specification lists device availability, model, directory, and capacity blockers.

- [ ] **Step 4: Write checksum/no-network adapter tests**

```python
def test_model_probe_hashes_only_manifested_local_file(tmp_path: Path) -> None:
    model = tmp_path / "qwen.gguf"
    model.write_bytes(b"local-model")
    manifest = write_manifest(tmp_path, model.name, hashlib.sha256(b"local-model").hexdigest())
    probe = LocalModelReadiness(tmp_path, manifest)
    result = probe.check_required()
    assert result["discussion"].ready is True
    assert result["discussion"].path == model.resolve()


def test_model_probe_never_imports_http_clients(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "requests", FailingModule())
    monkeypatch.setitem(sys.modules, "urllib.request", FailingModule())
    result = LocalModelReadiness(tmp_path, tmp_path / "manifest.json").check_required()
    assert result["asr"].ready is False
    assert result["discussion"].ready is False
```

- [ ] **Step 5: Implement local adapters**

Manifest path is `%LOCALAPPDATA%\FlowLens\models\manifest.json`. Require entries for `kotoba-whisper-v2.0-faster` and `qwen3-4b-instruct-2507`; the Qwen runtime relative path is exactly `qwen3-4b-instruct-2507/Qwen3-4B-Instruct-2507-Q4_K_M.gguf`. Each entry carries repository, pinned source revision, runtime format, relative path, SHA-256, and license. Resolve paths and reject traversal outside the model root. Hash in chunks. Storage probe creates, fsyncs, closes, and removes one uniquely named probe file under the intended sessions root and uses `shutil.disk_usage`; it returns structured failure rather than raising into Qt.

- [ ] **Step 6: Run preflight gate**

Run: `python -m pytest tests/controller/test_preflight.py tests/adapters/test_local_models.py -q`

Expected: all tests pass without audio hardware, models, or `%LOCALAPPDATA%` access.

### Task 6: Session Lifecycle, Routing, Supervision, and Degradation

**Files:**
- Create: `src/flowlens/controller/routing.py`
- Create: `src/flowlens/controller/supervision.py`
- Create: `src/flowlens/controller/session_controller.py`
- Create: `tests/controller/test_routing.py`
- Create: `tests/controller/test_session_controller.py`
- Create: `tests/controller/test_supervision.py`

**Interfaces:**
- Produces `SessionState` with the eight exact lifecycle values, `SessionController.enter_preflight()`, `refresh_preflight()`, `start()`, `pause()`, `resume()`, `request_stop()`, `cancel_stop()`, `confirm_stop()`, `keep_waiting()`, `force_close()`, `handle_message()`, `tick()`, `snapshot()`.
- `SessionLaunch(session_id, session_dir, manifest, initial_state, audio_config, asr_config, discussion_config)` is the complete process launch value. `WorkerRuntime.start_all(launch)`, `send(target, envelope)`, `poll()`, `restart(target)`, `health()`, `shutdown()` is the sole process boundary.
- `SequenceTracker.accept(envelope) -> SequenceResult(accepted, duplicate, gap)` is per sender/session.

- [ ] **Step 1: Write legal-transition and readiness-barrier tests**

```python
def test_recording_waits_for_all_four_worker_readiness_acknowledgements() -> None:
    controller = make_controller()
    controller.enter_preflight()
    controller.start(valid_selection())
    assert controller.state is SessionState.STARTING
    assert controller.runtime.sent[0].message_type is MessageType.WRITER_OPEN_SESSION
    controller.handle_message(writer_open_acknowledgement())
    for worker in (ProcessSource.AUDIO, ProcessSource.ASR):
        controller.handle_message(ready_envelope(worker))
        assert controller.state is SessionState.STARTING
    controller.handle_message(ready_envelope(ProcessSource.DISCUSSION))
    assert controller.state is SessionState.RECORDING


def test_illegal_transition_has_no_side_effect() -> None:
    controller = make_controller()
    with pytest.raises(InvalidTransition, match="IDLE -> PAUSED"):
        controller.pause()
    assert controller.runtime.sent == []
```

- [ ] **Step 2: Run lifecycle tests and confirm failure**

Run: `python -m pytest tests/controller/test_session_controller.py -q`

Expected: collection fails because `SessionController` does not exist.

- [ ] **Step 3: Implement lifecycle state and 60-second readiness timeout**

Start Writer first with `WRITER_OPEN_SESSION` carrying `WriterOpenSession(session_dir, manifest, initial_state)` and require the corresponding `WRITER_ACK` before considering Writer ready; that acknowledgement means `session.json` already exists as `incomplete`. Only then send start controls to Audio/ASR/Discussion. At `started_ms + 60_000`, stop every started worker, return to `PREFLIGHT`, and set a visible issue such as `"ASR worker did not become ready within 60 seconds."`. Never enter partial recording.

- [ ] **Step 4: Write sequence and routing tests**

```python
def test_sequence_tracker_rejects_duplicate_and_reports_gap() -> None:
    tracker = SequenceTracker()
    assert tracker.accept(envelope(source=ProcessSource.ASR, sequence=1)).accepted is True
    assert tracker.accept(envelope(source=ProcessSource.ASR, sequence=1)).duplicate is True
    result = tracker.accept(envelope(source=ProcessSource.ASR, sequence=3))
    assert result.accepted is True
    assert result.gap == (2, 2)


def test_writer_rewrap_uses_one_gui_local_sequence_after_open() -> None:
    controller = recording_controller()
    incoming_transcript = committed_envelope(sequence=10, text="確認します")
    incoming_state = discussion_state_envelope(sequence=7, revision=1)
    controller.handle_message(incoming_transcript)
    controller.handle_message(incoming_state)
    controller.persist_event(make_event_record())
    writer_messages = controller.runtime.sent_to(RouteTarget.WRITER)
    assert [message.message_type for message in writer_messages] == [
        MessageType.WRITER_OPEN_SESSION,
        MessageType.TRANSCRIPT_COMMITTED,
        MessageType.DISCUSSION_STATE_REPLACED,
        MessageType.EVENT_APPENDED,
    ]
    assert [message.sequence for message in writer_messages] == [1, 2, 3, 4]
    assert all(message.source is ProcessSource.GUI for message in writer_messages)
    assert writer_messages[1] is not incoming_transcript
    assert writer_messages[1].payload == incoming_transcript.payload
    assert writer_messages[2] is not incoming_state
    assert writer_messages[2].payload == incoming_state.payload
```

- [ ] **Step 5: Implement message routing and operational-only gap events**

Route `AudioWriteCommand` and `AudioDrainFence` only on the dedicated
Writer-audio queue; neither is a general control envelope. General envelopes
stay small. The controller validates each sender-local input sequence, but it
never forwards a raw ASR or Discussion envelope to Writer. For every
Writer-bound transcript, discussion-state replacement, and persisted event, it
creates a fresh envelope that preserves the typed payload and its embedded
origin semantics while assigning `source=ProcessSource.GUI` and the next value
from one dedicated contiguous Writer sequence. That sequence starts with
`WRITER_OPEN_SESSION=1`; the first subsequent mutation is 2 regardless of the
ASR or Discussion sender sequence. Persist committed transcript before
discussion fan-out. Unknown schema versions are rejected and logged as an
operational event without payload text.

- [ ] **Step 6: Write restart/degradation threshold tests**

```python
def test_asr_and_discussion_restart_once_only() -> None:
    supervisor = WorkerSupervisor()
    assert supervisor.on_exit(ProcessSource.ASR).action is RecoveryAction.RESTART
    assert supervisor.on_exit(ProcessSource.ASR).action is RecoveryAction.SAFE_STOP
    assert supervisor.on_exit(ProcessSource.DISCUSSION).action is RecoveryAction.RESTART
    assert supervisor.on_exit(ProcessSource.DISCUSSION).action is RecoveryAction.DISABLE_ANALYSIS


def test_backlog_pauses_analysis_above_five_seconds_and_resumes_below_two() -> None:
    controller = recording_controller()
    controller.handle_message(asr_status(backlog_ms=2_000))
    assert controller.snapshot().asr_status == "Running"
    controller.handle_message(asr_status(backlog_ms=2_001))
    assert controller.snapshot().asr_status == "Delayed"
    controller.handle_message(asr_status(backlog_ms=5_000))
    assert controller.snapshot().analysis_status == "Running"
    assert not any(
        envelope.message_type is MessageType.WORKER_PAUSE
        for envelope in controller.runtime.sent
    )
    controller.handle_message(asr_status(backlog_ms=5_001))
    assert controller.snapshot().analysis_status == "Paused for ASR delay"
    assert last_type(controller.runtime.sent) is MessageType.WORKER_PAUSE
    pause_count = sum(
        envelope.message_type is MessageType.WORKER_PAUSE
        for envelope in controller.runtime.sent
    )
    controller.handle_message(asr_status(backlog_ms=2_000))
    assert sum(
        envelope.message_type is MessageType.WORKER_PAUSE
        for envelope in controller.runtime.sent
    ) == pause_count
    controller.handle_message(asr_status(backlog_ms=1_999))
    assert last_type(controller.runtime.sent) is MessageType.WORKER_RESUME


def test_writer_exit_stops_immediately_and_audio_exit_safely_stops() -> None:
    controller = recording_controller()
    controller.handle_worker_exit(ProcessSource.WRITER)
    assert controller.state is SessionState.ERROR
    assert controller.snapshot().fatal_error.startswith("Session storage is unsafe")
```

- [ ] **Step 7: Implement 500 ms health ceiling and explicit degraded state**

`tick` polls health every 250 ms. Record and announce disconnects, reconnects, restarts, analysis pause/resume, and worker exits. Retry Audio source connections remains Audio-worker-owned at 2,000 ms. `GPU_OOM` from Discussion disables analysis while recording/ASR continue; writer queue exhaustion and storage failure start fatal safe-stop. Direct controller button methods update the snapshot before returning so UI feedback stays below 100 ms.

- [ ] **Step 8: Run lifecycle/supervision gate**

Run: `python -m pytest tests/controller/test_routing.py tests/controller/test_supervision.py tests/controller/test_session_controller.py -q`

Expected: all tests pass with deterministic fake runtime and clock.

### Task 7: Ordered Finalization and Force-Close Boundary

**Files:**
- Create: `src/flowlens/controller/finalization.py`
- Create: `tests/controller/test_finalization.py`
- Modify: `src/flowlens/controller/session_controller.py`

**Interfaces:**
- Produces `FinalizationStep(STOP_AUDIO, DRAIN_AUDIO, FINALIZE_ASR, FINAL_ANALYSIS, FINALIZE_WRITER, COMPLETE)`, `FinalizationCoordinator.begin()`, `acknowledge(message_type)`, `tick(now_ms)`, `keep_waiting()`, `force_close()`.
- Consumes shared `WriterAppendEvent`, `WriterFlush`, `WriterFinalize`, `WriterShutdown` and worker drain messages.

- [ ] **Step 1: Write an exact command-order test**

```python
def test_finalization_advances_after_post_fence_audio_stop_then_finalizes_later() -> None:
    coordinator = make_finalizer()
    first = coordinator.begin(now_ms=0)
    assert types(first) == [MessageType.WORKER_STOP]
    assert first[0].payload["worker"] == "AUDIO"
    second = coordinator.acknowledge(
        MessageType.WORKER_STOPPED, {"worker": "AUDIO", "drained": True}
    )
    assert types(second) == [MessageType.WORKER_STOP]
    assert second[0].payload["worker"] == "ASR"
    assert MessageType.WRITER_FINALIZE not in types(second)
    third = coordinator.acknowledge(
        MessageType.WORKER_STOPPED, {"worker": "ASR", "drained": True}
    )
    assert types(third) == [MessageType.WORKER_STOP]
    assert third[0].payload["worker"] == "DISCUSSION"
    assert MessageType.WRITER_FINALIZE not in types(third)
    fourth = coordinator.acknowledge(
        MessageType.WORKER_STOPPED, {"worker": "DISCUSSION", "drained": True}
    )
    assert types(fourth) == [MessageType.WRITER_FINALIZE]
    finalize_sequence = fourth[0].sequence
    assert coordinator.completed is False
    assert types(
        coordinator.acknowledge(
            MessageType.WRITER_ACK,
            WriterAck(acknowledged_sequence=finalize_sequence, latest_successful_save_at=NOW),
        )
    ) == []
    assert coordinator.completed is True
```

- [ ] **Step 2: Run test and confirm failure**

Run: `python -m pytest tests/controller/test_finalization.py -q`

Expected: collection fails because `FinalizationCoordinator` is absent.

- [ ] **Step 3: Implement acknowledgement-driven finalization**

Audio stop means no new capture and drain to both Writer and ASR. Audio
`WORKER_STOPPED/drained=true` means its `AudioDrainFence` was already enqueued
after every Writer audio command. ASR stop uses `{"worker": "ASR", "finalize":
true}`. Discussion stop uses `{"worker": "DISCUSSION", "finalize": true}` only
after the ASR `WORKER_STOPPED` drain acknowledgement. The controller may send
`WriterFinalize` later in its dedicated GUI sequence; the Writer keeps that
terminal control pending until it consumes the explicit fence and never guesses
drain completion from queue emptiness. It then performs JSONL flushes, WAV
header finalization, final state write, completed event, and the final
`session.json` completed mutation. Controller stores the finalize envelope
sequence and enters `COMPLETED` only after a `WriterAck.acknowledged_sequence`
matches it; only then does the UI show completion.

- [ ] **Step 4: Write slow-finalization and force-close tests**

```python
def test_thirty_seconds_only_offers_choices_and_never_force_closes() -> None:
    coordinator = make_finalizer()
    coordinator.begin(now_ms=100)
    result = coordinator.tick(now_ms=30_099)
    assert result.show_slow_message is False
    result = coordinator.tick(now_ms=30_100)
    assert result.show_slow_message is True
    assert result.commands == ()
    assert coordinator.completed is False


def test_force_close_flushes_incomplete_session_without_completed_mutation() -> None:
    coordinator = make_finalizer()
    coordinator.begin(now_ms=0)
    commands = coordinator.force_close(now=NOW)
    assert [type(command.payload) for command in commands] == [
        WriterAppendEvent,
        WriterFlush,
        WriterShutdown,
    ]
    assert all(not isinstance(command.payload, WriterFinalize) for command in commands)
```

- [ ] **Step 5: Connect stop confirmation and window close to one path**

`request_stop` opens confirmation without changing capture. `confirm_stop` enters `STOPPING`, snapshot text becomes `Finalizing`, and starts the coordinator. `keep_waiting` hides the slow dialog and continues polling. An active/paused `QCloseEvent` calls the same request path. Only explicit `force_close` requests incomplete shutdown.

- [ ] **Step 6: Run finalization gate**

Run: `python -m pytest tests/controller/test_finalization.py tests/controller/test_session_controller.py -q`

Expected: all tests pass.

### Task 8: Hallmark Midnight Design Tokens, Fonts, and Stateful Widgets

**Files:**
- Create: `src/flowlens/ui/design.py`
- Create: `src/flowlens/ui/widgets.py`
- Create: `assets/styles/flowlens.qss`
- Create: `assets/fonts/IBMPlexSansJP-Regular.ttf`
- Create: `assets/fonts/IBMPlexSansJP-SemiBold.ttf`
- Create: `assets/fonts/IBMPlexMono-Regular.ttf`
- Create: `assets/fonts/OFL.txt`
- Create: `tests/ui/test_design_contract.py`
- Create: `tests/ui/test_widgets.py`

**Interfaces:**
- Produces immutable `DesignTokens`, `contrast_ratio(foreground, background)`, `build_stylesheet(tokens, reduced_motion) -> str`, `load_bundled_fonts(resource_root) -> FontFamilies`, `StatefulButton`, `StatusIndicator`, and `InputMeter`.
- Hallmark mapping: genre Atmospheric, theme Midnight, macrostructure Workbench, enrichment none. For this live desktop tool, Workbench means one operational canvas with fixed control/status rails and two content work areas, not marketing screenshot frames or fake browser chrome.

- [ ] **Step 1: Verify the PySide6/pytest-qt pins and write token/contrast tests**

Assert `requirements.txt` contains exactly `PySide6==6.11.2` and `requirements-dev.txt` contains exactly `pytest-qt==4.5.0`.

```python
def test_tokens_match_approved_midnight_contract() -> None:
    assert Path("requirements.txt").read_text(encoding="utf-8").splitlines().count(
        "PySide6==6.11.2"
    ) == 1
    assert Path("requirements-dev.txt").read_text(encoding="utf-8").splitlines().count(
        "pytest-qt==4.5.0"
    ) == 1
    tokens = DesignTokens.approved()
    assert tokens.background == "#0D1117"
    assert tokens.surface == "#131922"
    assert tokens.elevated_surface == "#19212C"
    assert tokens.rule == "#2A3441"
    assert tokens.primary_text == "#E6EDF3"
    assert tokens.muted_text == "#9AA7B5"
    assert tokens.accent == "#D6A13D"
    assert tokens.focus == "#78A9FF"
    assert tokens.error == "#E16A6A"
    assert tokens.success == "#55B982"
    assert "#000000" not in tokens.values()
    assert "#FFFFFF" not in tokens.values()
    assert contrast_ratio(tokens.primary_text, tokens.background) >= 4.5
    assert contrast_ratio(tokens.focus, tokens.background) >= 3.0


def test_stylesheet_has_hallmark_stamp_and_bans() -> None:
    stylesheet = build_stylesheet(DesignTokens.approved(), reduced_motion=False)
    assert stylesheet.startswith(
        "/* Hallmark · genre: atmospheric · macrostructure: Workbench · "
        "theme: Midnight · tone: technical-austere · enrichment: none */"
    )
    lowered = stylesheet.lower()
    for banned in ("gradient", "qgraphicsdropshadoweffect", "border-radius: 999", "transition-all"):
        assert banned not in lowered
```

- [ ] **Step 2: Run design tests and confirm failure**

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/ui/test_design_contract.py -q`

Expected: collection fails because `flowlens.ui.design` is absent.

- [ ] **Step 3: Implement semantic tokens, 4 px spacing, and bundled font loading**

The QSS template contains semantic format fields only; `build_stylesheet` substitutes approved tokens once. QSS font families are `"IBM Plex Sans JP"` and `"IBM Plex Mono"`. `QFontDatabase.addApplicationFont` must return non-negative IDs and `applicationFontFamilies` must include both names; otherwise preflight shows a fatal app-resource error. Keep rules/spacing instead of nested bordered cards.

- [ ] **Step 4: Write full-state widget tests**

```python
def test_stateful_button_exposes_all_eight_states(qtbot: QtBot) -> None:
    button = StatefulButton("Start session")
    qtbot.addWidget(button)
    assert set(button.supported_states) == {
        "default", "hover", "focus", "active", "disabled", "loading", "error", "success"
    }
    button.set_ui_state("loading", "Checking models")
    assert button.isEnabled() is False
    assert button.accessibleDescription() == "Checking models"
    button.set_ui_state("error", "Model checksum failed")
    assert button.property("uiState") == "error"
    assert "Model checksum failed" in button.accessibleDescription()


def test_primary_controls_have_44_by_44_minimum(qtbot: QtBot) -> None:
    button = StatefulButton("Stop")
    qtbot.addWidget(button)
    assert button.minimumWidth() >= 44
    assert button.minimumHeight() >= 44
```

- [ ] **Step 5: Implement explicit widget states and motion boundary**

Use dynamic `uiState` properties with QSS selectors for default, hover, focus, active, disabled, loading, error, and success. Pair error/success with text and icon, never color alone. Focus rings appear immediately. Buttons move at most one logical pixel when pressed. Partial-commit and discussion replacement use `QGraphicsOpacityEffect` only; reduced motion sets duration to 0, otherwise 120 ms. No other animation is created.

- [ ] **Step 6: Run design/widget gate**

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/ui/test_design_contract.py tests/ui/test_widgets.py -q`

Expected: all tests pass.

### Task 9: Preflight Qt Page

**Files:**
- Create: `src/flowlens/ui/preflight_page.py`
- Create: `tests/ui/test_preflight_page.py`

**Interfaces:**
- Consumes immutable `PreflightReport` only.
- Produces signals `selection_changed(PreflightSelection)` and `start_requested()`; method `render(report) -> None`.

- [ ] **Step 1: Write semantic control and blocker tests**

```python
def test_preflight_renders_every_required_control(qtbot: QtBot) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)
    page.render(ready_report())
    assert page.meeting_radio.isChecked() is True
    assert page.microphone_combo.count() == 2
    assert page.loopback_combo.count() == 1
    assert page.mic_meter.accessibleName() == "Microphone activity"
    assert page.loopback_meter.accessibleName() == "PC audio activity"
    assert page.model_status.text() == "Local models ready"
    assert page.storage_status.text() == "Storage ready: 500 MB minimum satisfied"
    assert page.destination_summary.text().startswith("Sessions are saved to ")
    assert page.start_button.isEnabled() is True


def test_blocking_reason_is_adjacent_and_start_is_disabled(qtbot: QtBot) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)
    page.show()
    page.render(report_with_issue("microphone", "Select an available microphone."))
    assert page.microphone_error.text() == "Select an available microphone."
    assert page.microphone_error.isVisibleTo(page) is True
    assert page.start_button.isEnabled() is False
```

- [ ] **Step 2: Run page tests and confirm failure**

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/ui/test_preflight_page.py -q`

Expected: collection fails because `PreflightPage` is absent.

- [ ] **Step 3: Implement the Workbench preflight page**

Use visible labels above mode/device controls, stable one-line helper/error areas, two live meters, separate model/storage rows, absolute destination summary, and one primary start button. Restore available IDs; render unavailable saved IDs as unselected and never select index zero as a silent replacement. Device changes emit a complete selection. Tooltips show immediately on keyboard focus and after 800 ms hover via the shared tooltip helper.

- [ ] **Step 4: Write keyboard-start and tab-order tests**

```python
def test_ctrl_enter_emits_start_only_when_valid(qtbot: QtBot) -> None:
    page = PreflightPage()
    qtbot.addWidget(page)
    page.render(ready_report())
    with qtbot.waitSignal(page.start_requested, timeout=500):
        QTest.keyClick(page, Qt.Key_Enter, Qt.ControlModifier)
    page.render(report_with_issue("storage", "At least 500 MB of free space is required."))
    with qtbot.assertNotEmitted(page.start_requested):
        QTest.keyClick(page, Qt.Key_Enter, Qt.ControlModifier)


def test_tab_order_reaches_mode_devices_and_start(qtbot: QtBot) -> None:
    page = PreflightPage()
    assert page.focus_chain() == [
        page.meeting_radio,
        page.interview_radio,
        page.general_radio,
        page.microphone_combo,
        page.loopback_combo,
        page.start_button,
    ]
```

- [ ] **Step 5: Run preflight UI gate**

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/ui/test_preflight_page.py -q`

Expected: all tests pass.

### Task 10: Live Transcript, Discussion State, Status, Dialog, and Completion UI

**Files:**
- Create: `src/flowlens/ui/transcript_model.py`
- Create: `src/flowlens/ui/transcript_view.py`
- Create: `src/flowlens/ui/discussion_panel.py`
- Create: `src/flowlens/ui/status_strip.py`
- Create: `src/flowlens/ui/live_page.py`
- Create: `src/flowlens/ui/dialogs.py`
- Create: `src/flowlens/ui/completion_page.py`
- Create: `tests/ui/test_transcript.py`
- Create: `tests/ui/test_live_page.py`
- Create: `tests/ui/test_dialogs_completion.py`
- Create: `tests/ui/test_rendered_visual_contract.py`

**Interfaces:**
- `TranscriptListModel.set_partial(source, partial)`, `commit(record)`, `clear_partial(source)`; committed rows cannot be edited or replaced.
- `TranscriptView.return_to_latest()` and `auto_scroll_enabled`.
- `DiscussionPanel.render(state, labels)`, `LivePage.render(ControllerSnapshot)`, `CompletionPage.render(CompletionSummary)`.
- Signals: `pause_requested`, `resume_requested`, `stop_requested`, `always_on_top_changed`, `force_close_requested`, `keep_waiting_requested`, `open_folder_requested`, `start_another_requested`, `close_requested`.

- [ ] **Step 1: Write chronological/immutable transcript tests**

```python
def test_commits_sort_by_start_then_me_before_others() -> None:
    model = TranscriptListModel()
    model.commit(make_record(sequence=2, source=AudioSource.OTHERS, start_ms=1000))
    model.commit(make_record(sequence=1, source=AudioSource.ME, start_ms=1000))
    assert [model.row(index).source for index in range(model.rowCount())] == [
        AudioSource.ME,
        AudioSource.OTHERS,
    ]


def test_partial_is_ephemeral_and_commit_is_immutable() -> None:
    model = TranscriptListModel()
    model.set_partial(AudioSource.ME, make_partial("途中", start_ms=0))
    assert model.partial(AudioSource.ME).text == "途中"
    record = make_record(sequence=1, source=AudioSource.ME, text="確定", start_ms=0)
    model.commit(record)
    assert model.partial(AudioSource.ME) is None
    with pytest.raises(ImmutableTranscriptError):
        model.commit(dataclasses.replace(record, text="改変"))
```

- [ ] **Step 2: Run transcript tests and confirm failure**

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/ui/test_transcript.py -q`

Expected: collection fails because `TranscriptListModel` is absent.

- [ ] **Step 3: Implement model/delegate and auto-scroll behavior**

Render source text labels `ME`/`OTHERS` plus distinct shapes/icons so color is not the sole signal. Partial text uses muted emphasis and an explicit `Partial` accessibility description. Do not display timestamps continuously. A scrollbar movement away from maximum disables auto-scroll and reveals `Return to latest`; new rows do not steal position until the user invokes it.

- [ ] **Step 4: Write layout, labels, empty-state, and status tests**

```python
@pytest.mark.parametrize(
    ("mode", "labels"),
    [
        (SessionMode.MEETING, ("Current focus", "Key points", "Decisions / confirmations", "Unresolved / next actions")),
        (SessionMode.INTERVIEW, ("Current question / topic", "Answer highlights", "Confirmed content", "Follow-ups / points to clarify")),
        (SessionMode.GENERAL, ("Current topic", "Key points", "Confirmed items", "Items to revisit")),
    ],
)
def test_discussion_sections_have_fixed_order_and_mode_labels(
    qtbot: QtBot,
    mode: SessionMode,
    labels: tuple[str, str, str, str],
) -> None:
    panel = DiscussionPanel()
    qtbot.addWidget(panel)
    panel.render(empty_state(mode), labels_for(mode))
    assert panel.section_titles() == labels
    assert all(text != "" for text in panel.empty_explanations())


def test_live_layout_switches_at_exact_breakpoint(qtbot: QtBot) -> None:
    page = LivePage()
    qtbot.addWidget(page)
    page.show()
    page.resize(1000, 700)
    page.reflow()
    assert page.main_splitter.orientation() is Qt.Horizontal
    assert page.main_splitter.sizes()[0] / sum(page.main_splitter.sizes()) == pytest.approx(0.62, abs=0.03)
    page.resize(999, 700)
    page.reflow()
    assert page.main_splitter.orientation() is Qt.Vertical
    assert page.main_splitter.indexOf(page.transcript_view) < page.main_splitter.indexOf(page.discussion_panel)


def test_statuses_are_separate_not_one_aggregate_string(qtbot: QtBot) -> None:
    strip = StatusStrip()
    qtbot.addWidget(strip)
    strip.render(status_snapshot(asr="Delayed", delay_ms=2100, analysis="Paused", saved="12:35:02"))
    assert strip.microphone_status.text() != strip.asr_status.text()
    assert "Delayed" in strip.asr_status.text()
    assert "Paused" in strip.analysis_status.text()
    assert "12:35:02" in strip.save_status.text()
```

- [ ] **Step 5: Implement live Workbench layout and constrained motion**

Top bar contains FlowLens, mode, recording state, monospace elapsed time, pause/resume, red destructive stop, and always-on-top. Reserve a fixed-height error/status banner row even when empty. Main regions use rules and spacing rather than card wrappers. At 999 px and below, place the vertical transcript-first/discussion-second content inside a keyboard-focusable `QScrollArea` with horizontal scrolling disabled so both regions remain reachable at 900x600. Bottom strip has microphone, PC audio, ASR plus measured delay, analysis, and latest save as five separately accessible statuses. Opacity-animation only on partial commit/discussion change.

- [ ] **Step 6: Write stop/finalization/completion tests**

```python
def test_slow_finalization_dialog_never_auto_selects_force_close(qtbot: QtBot) -> None:
    dialog = SlowFinalizationDialog()
    qtbot.addWidget(dialog)
    assert dialog.text() == "Finalization is taking longer than expected"
    assert dialog.keep_waiting_button.isDefault() is True
    assert dialog.force_close_button.isDefault() is False


def test_completion_contains_only_mvp_actions(qtbot: QtBot, tmp_path: Path) -> None:
    page = CompletionPage()
    qtbot.addWidget(page)
    page.render(CompletionSummary(1_800_000, 42, tmp_path.resolve()))
    assert page.duration_value.text() == "30:00"
    assert page.transcript_count_value.text() == "42"
    assert page.path_value.text() == str(tmp_path.resolve())
    assert page.action_labels() == ["Open folder", "Start another session", "Close"]
    assert not hasattr(page, "play_button")
    assert not hasattr(page, "search_box")
```

- [ ] **Step 7: Run live/completion gate**

Add a rendered-image contract test after all three pages exist:

```python
@pytest.mark.parametrize("page", [make_preflight_page, make_live_page, make_completion_page])
def test_rendered_page_uses_no_pure_extremes_and_limits_exact_accent_pixels(
    qtbot: QtBot,
    page: Callable[[], QWidget],
) -> None:
    widget = page()
    qtbot.addWidget(widget)
    widget.resize(1280, 800)
    widget.show()
    image = widget.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    pixels = [QColor(image.pixel(x, y)).name().upper() for y in range(image.height()) for x in range(image.width())]
    assert "#000000" not in pixels
    assert "#FFFFFF" not in pixels
    assert pixels.count("#D6A13D") / len(pixels) < 0.03
```

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/ui/test_transcript.py tests/ui/test_live_page.py tests/ui/test_dialogs_completion.py tests/ui/test_rendered_visual_contract.py -q`

Expected: all tests pass at 1280x800, 1000x700, 999x700, and 900x600.

### Task 11: Qt Presenter, Main Window, Shortcuts, Geometry, and Accessibility Announcements

**Files:**
- Create: `src/flowlens/ui/presenter.py`
- Create: `src/flowlens/ui/main_window.py`
- Create: `src/flowlens/adapters/windows_shell.py`
- Create: `tests/ui/test_presenter.py`
- Create: `tests/ui/test_main_window.py`
- Create: `tests/adapters/test_windows_shell.py`

**Interfaces:**
- `QtSessionPresenter(controller, window, announcer)` runs a 50 ms controller-message/UI timer and delegates no business rule to widgets.
- `MainWindow.show_preflight()`, `show_live()`, `show_completion()`, `set_always_on_top(enabled)`, and `closeEvent(event)`.
- `QtAccessibilityAnnouncer.announce(widget, message, assertive=False)` uses `QAccessibleAnnouncementEvent` and `QAccessible.updateAccessibility`.
- `WindowsFolderOpener.open(path)` accepts an existing absolute session directory only.
- Presenter consumes the shared `ConfigStore`; it restores/clamps `WindowPreferences`, restores only still-available `DevicePreferences`, restores `last_mode`, and saves only the exact approved preference schema.

- [ ] **Step 1: Write presenter feedback/shortcut tests**

```python
def test_pause_feedback_is_rendered_in_same_event_turn(qtbot: QtBot) -> None:
    controller = recording_controller()
    window = MainWindow()
    presenter = QtSessionPresenter(controller, window, FakeAnnouncer())
    qtbot.addWidget(window)
    started = time.perf_counter()
    QTest.keyClick(window, Qt.Key_Space)
    qtbot.waitUntil(lambda: window.live_page.recording_state.text() == "Paused", timeout=100)
    assert (time.perf_counter() - started) < 0.1


def test_space_does_nothing_when_text_or_menu_has_focus(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(recording=True)
    line_edit = QLineEdit(window)
    line_edit.setFocus()
    QTest.keyClick(line_edit, Qt.Key_Space)
    assert controller.state is SessionState.RECORDING


def test_global_shortcuts_match_spec(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(recording=True)
    QTest.keyClick(window, Qt.Key_T, Qt.ControlModifier)
    assert window.is_always_on_top() is True
    QTest.keyClick(window, Qt.Key_S, Qt.ControlModifier | Qt.ShiftModifier)
    assert window.stop_dialog.isVisible() is True


def test_start_another_returns_to_fresh_preflight(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(completed=True)
    QTest.mouseClick(window.completion_page.start_another_button, Qt.LeftButton)
    assert controller.state is SessionState.PREFLIGHT
    assert window.current_page() is window.preflight_page
    assert window.live_page.transcript_view.model().rowCount() == 0
```

- [ ] **Step 2: Run presenter tests and confirm failure**

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/ui/test_presenter.py -q`

Expected: collection fails because presenter/main window are absent.

- [ ] **Step 3: Implement timer binding and exact shortcuts**

The 50 ms Qt timer drains controller messages, calls `controller.tick`, renders a new snapshot only when changed, and announces polite status changes. Fatal recording/storage errors are assertive. `Escape` closes only active menu/dialog. `Ctrl+Enter` is active only on valid preflight. `Space` checks focused widget ancestry for text inputs and menus before pause/resume.

- [ ] **Step 4: Write geometry/close/folder safety tests**

```python
def test_saved_geometry_is_clamped_to_available_screens() -> None:
    screens = [QRect(0, 0, 1920, 1080), QRect(1920, 0, 1920, 1080)]
    saved = WindowPreferences(-9000, 9000, 4000, 3000, False, False)
    actual = clamp_geometry(saved, screens, minimum=QSize(900, 600))
    assert any(screen.contains(actual.center()) for screen in screens)
    assert actual.width() >= 900
    assert actual.height() >= 600


def test_active_close_uses_stop_confirmation(qtbot: QtBot) -> None:
    presenter, window, controller = make_presenter(recording=True)
    window.close()
    assert window.isVisible() is True
    assert window.stop_dialog.isVisible() is True
    assert controller.state is SessionState.RECORDING


def test_folder_opener_rejects_relative_or_missing_path(tmp_path: Path) -> None:
    opener = WindowsFolderOpener(shell_execute=spy_shell_execute)
    with pytest.raises(ValueError):
        opener.open(Path("relative"))
    with pytest.raises(FileNotFoundError):
        opener.open(tmp_path / "missing")


def test_presenter_persists_only_approved_non_session_preferences() -> None:
    presenter, window, controller, store = make_presenter_with_config()
    window.setGeometry(100, 100, 1280, 800)
    window.set_always_on_top(True)
    presenter.on_selection_changed(
        PreflightSelection(SessionMode.INTERVIEW, "mic-1", "output-1")
    )
    presenter.save_preferences()
    saved = store.saved.to_dict()
    assert set(saved) == {"schema_version", "window", "devices", "last_mode"}
    assert "transcript" not in json.dumps(saved, ensure_ascii=False).lower()
    assert "prompt" not in json.dumps(saved, ensure_ascii=False).lower()
```

- [ ] **Step 5: Implement Qt accessibility announcements and resource-safe shell adapter**

Create `QAccessibleAnnouncementEvent(widget, message)`, set polite/assertive priority, then call `QAccessible.updateAccessibility(event)`. Open folders using `os.startfile(str(path.resolve()))` only after directory validation; never build a shell command string. Save geometry on an orderly window close or completed-session close and save mode/device/always-on-top changes through `ConfigStore`; never place session text, model prompts, credentials, or session paths in config.

- [ ] **Step 6: Run presenter/window gate**

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/ui/test_presenter.py tests/ui/test_main_window.py tests/adapters/test_windows_shell.py -q`

Expected: all tests pass.

### Task 12: Five-Process Runtime and Composition Root

**Files:**
- Create: `src/flowlens/integration/worker_runtime.py`
- Create: `src/flowlens/integration/composition.py`
- Create: `src/flowlens/app.py`
- Create: `src/flowlens/__main__.py`
- Create: `tests/integration/test_worker_runtime.py`
- Create: `tests/integration/test_composition.py`

**Interfaces:**
- `MultiprocessingWorkerRuntime(context, worker_targets)` implements controller `WorkerRuntime`.
- `AudioQueueBindings(writer_audio_out, asr_audio_out)` exposes the dedicated `Queue[AudioWriteCommand | AudioDrainFence]` Writer path and `Queue[AudioFrame]` ASR path passed only to Audio, Writer, and ASR process arguments.
- Production targets are `run_audio_worker`, `run_asr_worker`, `run_discussion_worker`, and foundation `run_writer_worker(control_queue, audio_queue, response_queue, stop_event)`.
- Queue graph: GUI control-out per worker; one shared worker-to-GUI event queue; bounded Audio-to-Writer queue; bounded Audio-to-ASR queue. There is no socket or server.
- `build_application(paths, options) -> ApplicationGraph` is the only production composition function.

- [ ] **Step 1: Write spawn topology and queue-isolation tests**

```python
def test_runtime_starts_exactly_four_children_with_spawn_context() -> None:
    context = FakeMultiprocessingContext(start_method="spawn")
    runtime = MultiprocessingWorkerRuntime(context, fake_targets())
    runtime.start_all(make_launch())
    assert [process.name for process in context.processes] == [
        "FlowLens-Writer",
        "FlowLens-Audio",
        "FlowLens-ASR",
        "FlowLens-Discussion",
    ]
    assert all(process.daemon is False for process in context.processes)


def test_audio_payload_and_writer_fence_never_enter_general_control_queue() -> None:
    runtime = make_runtime()
    frame = make_audio_frame()
    bindings = runtime.audio_bindings()
    bindings.asr_audio_out.put(frame)
    bindings.writer_audio_out.put(make_audio_write_command(frame))
    bindings.writer_audio_out.put(AudioDrainFence())
    assert runtime.asr_audio_queue.items == [frame]
    assert runtime.writer_audio_queue.items[0].pcm_s16le == frame.pcm_s16le
    assert isinstance(runtime.writer_audio_queue.items[1], AudioDrainFence)
    assert runtime.general_queues_contain_bytes() is False
    assert runtime.general_queues_contain(AudioDrainFence) is False
```

- [ ] **Step 2: Run integration tests and confirm failure**

Run: `python -m pytest tests/integration/test_worker_runtime.py -q`

Expected: collection fails because runtime is absent.

- [ ] **Step 3: Implement process creation, routing, health, restart, and shutdown**

Create queues before processes. `audio_bindings() -> AudioQueueBindings(writer_audio_out, asr_audio_out)` exposes only the two dedicated queue objects used to construct Audio worker arguments; the GUI/controller never places PCM or `AudioDrainFence` on them. Audio alone appends the foundation-owned fence to the Writer path after its final command; the fence never enters the ASR or general queues. Use `multiprocessing.get_context("spawn")`. Do not make children daemons because they must flush on parent loss. `poll()` uses non-blocking drains with a fixed per-tick budget so Qt remains responsive. `restart(ASR|DISCUSSION)` closes and joins the old process, creates a fresh process with the same local model config and queues, and relies on controller restart limits. `shutdown()` sends typed controls, joins with bounded waits, and reports any process requiring termination; it never marks a session completed.

- [ ] **Step 4: Write composition/local-only tests**

```python
def test_composition_has_no_http_or_websocket_dependency() -> None:
    graph = build_application(fake_paths(), AppOptions())
    import_names = graph.production_adapter_modules()
    assert all("requests" not in name for name in import_names)
    assert all("httpx" not in name for name in import_names)
    assert all("websocket" not in name for name in import_names)


def test_composition_injects_hardware_adapters_behind_ports() -> None:
    graph = build_application(fake_paths(), AppOptions())
    assert isinstance(graph.controller.preflight.device_catalog, DeviceCatalog)
    assert isinstance(graph.controller.preflight.model_readiness, ModelReadiness)
    assert isinstance(graph.controller.runtime, WorkerRuntime)
```

- [ ] **Step 5: Implement early frozen-process diversion and lightweight entry**

```python
if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()

    from flowlens.app import main

    raise SystemExit(main())
```

Keep PySide6, CUDA, and model imports after `freeze_support()`. `app.main()` supports `--help`, `--package-self-check`, and `--acceptance-report PATH` in addition to normal launch. The self-check loads packaged native dependencies without workers/models, and the acceptance option records local measurements; neither changes session semantics.

- [ ] **Step 6: Run runtime/composition gate**

Run: `python -m pytest tests/integration/test_worker_runtime.py tests/integration/test_composition.py -q`

Expected: all tests pass with fake process/queue targets.

### Task 13: Folder-Based Windows Packaging and License/Resource Audit

**Files:**
- Create: `packaging/FlowLens.spec`
- Create: `packaging/hooks/hook-llama_cpp.py`
- Create: `packaging/hooks/hook-ctranslate2.py`
- Create: `packaging/hooks/hook-pyaudiowpatch.py`
- Create: `scripts/build_windows.ps1`
- Create: `scripts/check_package.py`
- Create: `licenses/PySide6-LGPL-3.0-only.txt`
- Create: `licenses/llama-cpp-python-MIT.txt`
- Create: `licenses/IBM-Plex-OFL.txt`
- Create: `licenses/Qwen3-4B-Instruct-2507-Apache-2.0.txt`
- Create: `licenses/kotoba-whisper-v2.0-license.txt`
- Create: `tests/packaging/test_package_spec.py`
- Create: `tests/packaging/test_package_audit.py`

**Interfaces:**
- PyInstaller output is exactly `dist/FlowLens/FlowLens.exe`, `dist/FlowLens/runtime/`, and `dist/FlowLens/licenses/`.
- Models remain only in `%LOCALAPPDATA%\FlowLens\models\`.
- `check_package(package_root) -> PackageAudit` verifies structure, imports/DLLs, fonts, licenses, absent models, and executable launch probe.

- [ ] **Step 1: Verify the foundation-owned packaging pin and write spec-structure tests**

Assert `requirements-dev.txt` contains exactly `PyInstaller==6.21.0`.

```python
def test_spec_is_onedir_with_runtime_contents_directory() -> None:
    assert Path("requirements-dev.txt").read_text(encoding="utf-8").splitlines().count(
        "PyInstaller==6.21.0"
    ) == 1
    text = Path("packaging/FlowLens.spec").read_text(encoding="utf-8")
    assert 'name="FlowLens"' in text
    assert 'contents_directory="runtime"' in text
    assert "exclude_binaries=True" in text
    assert "COLLECT(" in text
    assert "onefile" not in text.lower()


def test_spec_never_collects_localappdata_models() -> None:
    text = Path("packaging/FlowLens.spec").read_text(encoding="utf-8").lower()
    assert "localappdata" not in text
    assert "models\\" not in text
    assert "models/" not in text
```

- [ ] **Step 2: Run packaging tests and confirm missing spec**

Run: `python -m pytest tests/packaging/test_package_spec.py -q`

Expected: tests fail because `packaging/FlowLens.spec` is absent.

- [ ] **Step 3: Implement explicit onedir spec and native hooks**

Use `Analysis` plus hooks that call PyInstaller helpers `collect_dynamic_libs` and `collect_submodules` for `llama_cpp`, `ctranslate2`, and `pyaudiowpatch`. Include the three fonts and QSS as application data. Configure `EXE(..., exclude_binaries=True, console=False, contents_directory="runtime")` and `COLLECT`. Build script removes only the validated `build/FlowLens` and `dist/FlowLens` targets, invokes `python -m PyInstaller --clean --noconfirm --additional-hooks-dir packaging/hooks packaging/FlowLens.spec`, then copies `licenses/` beside the executable.

- [ ] **Step 4: Write folder audit tests**

```python
def test_package_audit_requires_structure_and_rejects_models(tmp_path: Path) -> None:
    package = make_fake_package(tmp_path)
    audit = check_package(package)
    assert audit.errors == ()
    (package / "runtime" / "qwen.gguf").write_bytes(b"model")
    audit = check_package(package)
    assert audit.errors == ("Runtime package must not contain model artifacts: runtime/qwen.gguf",)


def test_package_audit_requires_every_license_and_font(tmp_path: Path) -> None:
    package = make_fake_package(tmp_path)
    (package / "licenses" / "IBM-Plex-OFL.txt").unlink()
    assert "Missing license: IBM-Plex-OFL.txt" in check_package(package).errors
```

- [ ] **Step 5: Implement package audit and executable probe**

Audit `.gguf`, `.bin`, and known model directories recursively; require the two IBM families to be loadable from packaged resources; require dependency/model/font license texts; require `FlowLens.exe --help` to exit 0 within 10 seconds without spawning workers or loading models. Inspect runtime imports by launching `FlowLens.exe --package-self-check`, which loads Qt platform plugins, llama_cpp DLLs, CTranslate2, and PyAudioWPatch but performs no model generation/device capture.

- [ ] **Step 6: Build and audit on the designated PC**

Run: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_windows.ps1`

Run: `python scripts/check_package.py --package dist/FlowLens`

Expected: audit reports `PASS`, the folder has the exact three top-level entries, and no model file is duplicated.

### Task 14: Discussion, Integration, Offline, Performance, and Recovery Smoke Harness

**Files:**
- Create: `scripts/smoke_discussion.py`
- Create: `scripts/smoke_integration.py`
- Create: `scripts/validate_session.py`
- Create: `scripts/collect_acceptance.py`
- Create: `scripts/run_acceptance.ps1`
- Create: `tests/fixtures/discussion/meeting.json`
- Create: `tests/fixtures/discussion/interview.json`
- Create: `tests/fixtures/discussion/general.json`
- Create: `tests/smoke/test_discussion_smoke.py`
- Create: `tests/smoke/test_validate_session.py`
- Create: `tests/smoke/test_acceptance_metrics.py`

**Interfaces:**
- `smoke_discussion.py --model-manifest PATH --report PATH` runs one real local prompt for each mode and validates schema/mode/no-advice rules.
- `smoke_integration.py --microphone-id ID --loopback-output-id ID --duration-seconds 300 --pause-at-seconds 120 --pause-duration-seconds 5 --report PATH` exercises the real five-process path headlessly.
- `validate_session.py SESSION_DIR --minimum-active-seconds N --require-completed|--require-recovered` validates all seven artifacts, WAV format/duration, JSONL completeness/order, state revisions, and event rules.
- `collect_acceptance.py` computes latency p95, overflows, WAV error, memory growth, GPU OOM, artifacts, and offline evidence into deterministic JSON.

- [ ] **Step 1: Write fixture-driven discussion smoke tests**

```python
@pytest.mark.parametrize("mode", ["MEETING", "INTERVIEW", "GENERAL"])
def test_discussion_fixture_contract(mode: str) -> None:
    fixture = load_fixture(Path(f"tests/fixtures/discussion/{mode.lower()}.json"))
    state = validate_discussion_smoke(fixture["output"], SessionMode(mode))
    assert state.mode.value == mode
    combined = " ".join(
        (state.current_focus, *state.key_points, *state.confirmed_outcomes, *state.follow_up_items)
    )
    for prohibited in fixture["prohibited_phrases"]:
        assert prohibited not in combined
```

Fixtures contain short Japanese transcripts and conservative expected properties: meeting confirmation, interview labels without `decisions`/`unresolved issues`, general neutral labels, and prohibited advice phrases. They do not assert exact model prose.

- [ ] **Step 2: Implement and run the real discussion smoke**

Run: `python scripts/smoke_discussion.py --model-manifest "$env:LOCALAPPDATA\FlowLens\models\manifest.json" --report build/reports/discussion-smoke.json`

Expected: three schema-valid snapshots, correct modes/revisions, no extra fields, no advice/pro-con text, and report `passed: true`.

- [ ] **Step 3: Write artifact validator tests**

```python
def test_validator_requires_exact_seven_session_artifacts(tmp_path: Path) -> None:
    session = make_valid_session(tmp_path)
    result = validate_session(session, minimum_active_seconds=300, expected_status="completed")
    assert result.errors == ()
    (session / "events.jsonl").unlink()
    result = validate_session(session, minimum_active_seconds=300, expected_status="completed")
    assert result.errors == ("Missing required artifact: events.jsonl",)


def test_validator_checks_wav_format_and_pause_excluded_duration(tmp_path: Path) -> None:
    session = make_valid_session(tmp_path, active_ms=300_000, wav_ms=298_800)
    result = validate_session(session, minimum_active_seconds=300, expected_status="completed")
    assert result.wav_error_percent == pytest.approx(0.4)
    assert result.wav_error_percent < 0.5
    assert result.mic_format == (1, 2, 16_000)
    assert result.loopback_format == (1, 2, 16_000)
```

- [ ] **Step 4: Implement the five-minute real integration smoke**

The script constructs production ports directly, starts the same four workers, automatically pauses at 120 seconds for five seconds, resumes, stops at 300 active seconds, waits for writer completion, and invokes `validate_session`. It records Audio queue overflow count, both source presence, committed labels, discussion revision, and pause events. It does not bypass worker/controller semantics or use network access.

Run: `python scripts/smoke_integration.py --microphone-id "$env:FLOWLENS_MIC_ID" --loopback-output-id "$env:FLOWLENS_LOOPBACK_ID" --duration-seconds 300 --pause-at-seconds 120 --pause-duration-seconds 5 --report build/reports/integration-smoke.json`

Expected: report `passed: true`, zero Audio queue overflow, separate ME/OTHERS audio, at least one committed row per source, pause start/end, all seven artifacts, completed status.

- [ ] **Step 5: Write acceptance metric tests**

```python
def test_acceptance_thresholds_are_exact() -> None:
    metrics = make_metrics(
        partial_p95_ms=2000,
        commit_p95_ms=3000,
        discussion_p95_ms=5000,
        max_ui_feedback_ms=100,
        queue_overflows=0,
        wav_error_percent=0.49,
        memory_growth_mb=499,
        gpu_oom_count=0,
        network_blocked=True,
    )
    assert evaluate_acceptance(metrics).errors == ()


def test_acceptance_rejects_analysis_that_pushes_commit_latency_over_limit() -> None:
    metrics = make_metrics(commit_p95_ms=3001, discussion_enabled=True)
    assert "Committed ASR p95 exceeds 3000 ms" in evaluate_acceptance(metrics).errors
```

- [ ] **Step 6: Implement the scoped offline 30-minute harness**

`run_acceptance.ps1` resolves `dist\FlowLens\FlowLens.exe`, creates one outbound-block Windows Firewall rule whose exact program is that executable and whose name includes the script PID, launches the packaged app with `--acceptance-report`, samples process RSS and `nvidia-smi` every five seconds, and removes that exact rule in `finally`. It never changes another rule. The operator selects the two devices, starts once, leaves the app hands-free for at least 30 active minutes, performs one pause/resume, then stops normally.

Run in an elevated PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_acceptance.ps1 `
    -Executable dist\FlowLens\FlowLens.exe `
    -MinimumActiveMinutes 30 `
    -Report build\reports\acceptance-30m.json
```

Expected: firewall evidence shows outbound block active throughout session; completion succeeds; partial p95 <=2,000 ms, commit p95 <=3,000 ms, discussion p95 <=5,000 ms, direct UI feedback <=100 ms, zero Audio overflows, WAV error <0.5%, minute-5-to-minute-30 RSS growth <500 MB, no GPU OOM, and all seven artifacts.

- [ ] **Step 7: Run the separate forced-termination recovery check**

Use the same harness with `-RecoveryCheck`. It launches a fresh session, waits until both WAV files and session.json exist, terminates only the recorded FlowLens process tree, relaunches FlowLens for offline recovery, and passes the recovered folder to `validate_session.py --require-recovered`. The result must retain complete JSONL lines, repair usable WAV headers, append `SESSION_RECOVERED`, preserve recovered data, and set `recovered`, never `completed`.

Run in an elevated PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_acceptance.ps1 `
    -Executable dist\FlowLens\FlowLens.exe `
    -RecoveryCheck `
    -Report build\reports\acceptance-recovery.json
```

Expected: recovery report `passed: true` with one recovered session and no network access.

### Task 15: Full Verification and Spec Traceability Gate

**Files:**
- Create: `docs/verification/analysis-ui-traceability.md`
- Create: `tests/test_no_network_dependencies.py`
- Create: `tests/test_text_encoding.py`

**Interfaces:**
- Traceability rows map each applicable `docs/mvp-spec.md` section to implementation files, automated test command, and designated-PC evidence report.

- [ ] **Step 1: Add a static offline-dependency guard**

```python
def test_runtime_source_has_no_network_client_or_server_imports() -> None:
    banned = {
        "requests", "httpx", "aiohttp", "websocket", "websockets", "fastapi", "flask",
        "socket", "urllib", "http",
    }
    violations: list[str] = []
    for path in Path("src/flowlens").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                names = set()
            if names & banned:
                violations.append(f"{path}:{node.lineno}:{sorted(names & banned)}")
    assert violations == []
```

- [ ] **Step 2: Run focused suites in dependency order**

Run: `python -m pytest tests/discussion tests/controller tests/integration -q`

Run: `$env:QT_QPA_PLATFORM = "offscreen"; python -m pytest tests/ui -q`

Run: `python -m pytest tests/packaging tests/smoke tests/test_no_network_dependencies.py tests/test_text_encoding.py -q`

Expected: every suite passes.

- [ ] **Step 3: Run formatting, lint, and type gates**

Run: `python -m black --check src tests scripts`

Run: `python -m ruff check src tests scripts`

Run: `python -m mypy src tests`

Expected: all commands exit 0.

- [ ] **Step 4: Populate traceability with evidence paths**

Cover specification sections 6–16, 19–21, 24–29, and the applicable completion criteria. Each row points to at least one test. Hardware/model/performance rows point to `build/reports/discussion-smoke.json`, `integration-smoke.json`, `acceptance-30m.json`, or `acceptance-recovery.json`; unit-test rows do not claim physical hardware evidence.

- [ ] **Step 5: Run the final packaged acceptance gate**

Run: `python scripts/check_package.py --package dist/FlowLens`

Run: `python scripts/validate_session.py "$env:FLOWLENS_ACCEPTANCE_SESSION" --minimum-active-seconds 1800 --require-completed`

Run: `python scripts/collect_acceptance.py --session "$env:FLOWLENS_ACCEPTANCE_SESSION" --samples build/reports/acceptance-samples.jsonl --offline-evidence build/reports/firewall-evidence.json --output build/reports/acceptance-30m.json`

Expected: package, session, performance, offline, and recovery evidence all pass. Only after this gate may the implementation be described as MVP-complete.

## Implementation References

- `llama-cpp-python` JSON Schema mode: `https://github.com/abetlen/llama-cpp-python#json-and-json-schema-mode`
- Qt accessibility announcements: `https://doc.qt.io/qtforpython-6/PySide6/QtGui/QAccessibleAnnouncementEvent.html`
- Qt bundled font loading: `https://doc.qt.io/qtforpython-6/PySide6/QtGui/QFontDatabase.html`
- PyInstaller multiprocessing `freeze_support`: `https://pyinstaller.org/en/stable/common-issues-and-pitfalls.html#multi-processing`
