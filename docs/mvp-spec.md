# FlowLens MVP Specification

Status: Approved for implementation

Version: 1.0

Last updated: 2026-08-19

Target platform: One designated Windows PC

## 1. Purpose

FlowLens is a fully local Windows desktop application that transcribes and
organizes online conversations in real time.

The application is intended for:

- Online meetings and discussions.
- Online interviews.
- General online conversations that benefit from live organization.

The application must remain useful while the conversation is in progress. Its
primary product qualities are low latency, reliable recording, readable live
transcription, and conservative organization of what was actually said.

FlowLens is not a group-discussion-specific product. Labels and behavior must
remain appropriate for meetings, interviews, and general conversations.

## 2. Source of Truth

This document is the source of truth for the MVP.

Implementation choices must follow this document. A developer must not add a
feature, replace a selected model, change a persistence format, or reinterpret
an ambiguous requirement without first updating and approving this document.

Unknown performance characteristics must be resolved through the acceptance
criteria in this document, not by inventing new product requirements.

## 3. Product Principles

1. Audio persistence has the highest runtime priority.
2. Committed transcription has priority over partial transcription.
3. Transcription has priority over discussion analysis.
4. Discussion analysis organizes the conversation but does not advise the user.
5. The application must not silently discard audio or hide degraded operation.
6. A live session must work without an internet connection.
7. The interface must be readable at a glance during another online activity.
8. Saved session data must remain recoverable after an abnormal termination.

## 4. MVP Scope

### 4.1 Included

- Select one microphone device as `ME`.
- Select one Windows output device and capture its WASAPI loopback as `OTHERS`.
- Capture both sources simultaneously.
- Save each source to a separate audio file.
- Display partial and committed Japanese transcription in real time.
- Merge ME and OTHERS into a chronological transcript.
- Display the current topic or question.
- Display key points for the current topic.
- Display confirmed outcomes or confirmed content.
- Display unresolved or follow-up items.
- Support meeting/discussion, interview, and general modes.
- Support start, pause, resume, and stop.
- Support sessions of at least 30 minutes.
- Save session artifacts locally and retain them indefinitely.
- Recover usable data from an interrupted session where technically possible.
- Provide a folder-based Windows executable for the designated PC.

### 4.2 Explicitly Excluded

- Cloud APIs or cloud inference.
- Telemetry, analytics, crash uploads, or remote logging.
- In-person conversation capture.
- Per-person diarization within OTHERS.
- Application-specific Windows audio capture.
- Automatic language detection.
- Translation.
- Languages other than Japanese, except embedded English technical terms.
- AI suggestions, rebuttals, speaking advice, or answer generation.
- Pro/con material generation.
- Editing committed transcripts.
- Editing organized discussion state.
- Evidence links from organized items to transcript segments.
- Speaker attribution on organized items.
- Session history or review UI.
- Audio playback.
- Search, export, or retranscription.
- Automatic log deletion.
- Manual session naming.
- Tray integration.
- Compact mode.
- Installer and automatic updates.
- Light theme.

## 5. Target Environment

The MVP only needs to run on the designated development PC.

| Component | Target |
| --- | --- |
| Operating system | Windows 10 Pro, 64-bit |
| CPU | AMD Ryzen 7 3800X, 8 cores / 16 threads |
| Memory | Approximately 64 GB RAM |
| GPU | NVIDIA GeForce RTX 4060, 8 GB VRAM |
| NVIDIA driver observed during design | 596.36 |
| Application runtime | Python 3.12 |

Support for other machines, operating systems, GPUs, or Windows versions is not
an MVP requirement.

## 6. Privacy and Offline Contract

### 6.1 Session Runtime

After model installation, a session must perform no network request.

The following must remain local:

- Audio capture.
- Transcription.
- Discussion analysis.
- Session storage.
- Error logs.
- Configuration.

The application must not include telemetry or update checks.

### 6.2 Initial Model Installation

An explicit model download during initial setup is allowed. Model installation
must complete before a session can start.

After installation:

- Models are loaded only from local paths.
- No runtime fallback to a remote model is allowed.
- Missing files block session start.
- A checksum mismatch blocks session start.
- The application must explain which model is missing or invalid.

Models are stored under:

```text
%LOCALAPPDATA%\FlowLens\models\
```

### 6.3 Local Configuration

Non-session preferences are stored at:

```text
%LOCALAPPDATA%\FlowLens\config.json
```

The configuration contains exactly these user preferences:

```json
{
    "schema_version": 1,
    "window": {
        "x": 100,
        "y": 100,
        "width": 1280,
        "height": 800,
        "maximized": false,
        "always_on_top": false
    },
    "devices": {
        "microphone_id": "device-id",
        "loopback_output_id": "device-id"
    },
    "last_mode": "MEETING"
}
```

Window geometry must be clamped to the currently available displays on launch.
A missing saved device does not silently select an arbitrary replacement; the
preflight screen requires the user to confirm another available device.

Configuration must not contain transcript text, audio, model prompts, or any
credential.

## 7. User Modes

The user manually selects one mode before starting. Automatic mode detection is
not allowed.

All modes use the same internal state schema. Only labels and analysis framing
change.

| Internal field | Meeting / Discussion | Interview | General |
| --- | --- | --- | --- |
| `current_focus` | Current focus | Current question / topic | Current topic |
| `key_points` | Key points | Answer highlights | Key points |
| `confirmed_outcomes` | Decisions / confirmations | Confirmed content | Confirmed items |
| `follow_up_items` | Unresolved / next actions | Follow-ups / points to clarify | Items to revisit |

The interview mode must never present the unnatural labels "decisions" or
"unresolved issues" to the user.

## 8. Primary User Flow

1. Launch FlowLens.
2. Complete initial model setup if required.
3. Select a mode.
4. Confirm or change the microphone.
5. Confirm or change the Windows output device.
6. Verify both input meters.
7. Start the session.
8. Leave the application hands-free while the conversation continues.
9. Optionally pause and resume.
10. Stop the session.
11. Wait for finalization.
12. Open the saved folder or start another session.

## 9. Preflight Screen

The preflight screen contains:

- Mode selection.
- Microphone selection.
- Windows output device selection.
- Live input meter for each source.
- Local model readiness.
- Storage readiness.
- A short destination summary.
- `Start session`.

The previous device selections are restored when still available.

Start is blocked when:

- No microphone is selected or available.
- No loopback-capable output device is selected or available.
- Either required model is missing or invalid.
- The session directory cannot be created.
- Available storage is insufficient for a 30-minute session.

The UI must state the specific blocking reason next to the affected control.
The storage check requires at least 500 MB of available space before start.

## 10. Live Screen

### 10.1 Layout

The default window size is 1280 by 800 logical pixels. The minimum supported
window size is 900 by 600 logical pixels.

At widths of 1000 logical pixels or greater:

- Transcript uses approximately 62% of the main width.
- Discussion state uses approximately 38%.

Below 1000 logical pixels:

- Transcript appears first.
- Discussion state moves below it.
- Both regions remain reachable by keyboard and scrolling.

The screen contains:

1. A top control bar.
2. A reserved error/status banner row.
3. The transcript region.
4. The discussion-state region.
5. A bottom status strip.

### 10.2 Top Control Bar

The top bar shows:

- FlowLens.
- Selected mode.
- Recording state.
- Elapsed session time.
- Pause or resume.
- Stop.
- Always-on-top toggle.

### 10.3 Transcript Region

- ME and OTHERS appear in one chronological list.
- Each entry has a visible source label.
- Source distinction must not rely on color alone.
- Partial text uses reduced visual emphasis.
- Committed text uses normal emphasis.
- Committed text is immutable.
- Timestamps are stored but are not continuously shown.
- The list auto-scrolls while the user remains at the latest entry.
- Manual upward scrolling disables auto-scroll.
- A visible `Return to latest` control restores auto-scroll.
- Overlapping speech remains as separate entries.

Entries are ordered by speech start time. If start times are equal, ME appears
before OTHERS.

### 10.4 Discussion-State Region

The vertical order is fixed:

1. Current focus.
2. Key points.
3. Confirmed outcomes.
4. Follow-up items.

These sections use rules and spacing rather than nested cards. Empty sections
remain visible with a short mode-appropriate explanation.

### 10.5 Bottom Status Strip

The status strip shows:

- Microphone activity.
- PC audio activity.
- ASR status and measured delay.
- Discussion analysis status.
- Latest successful save time.

Recording, ASR delay, and analysis availability must be represented as separate
statuses.

## 11. Visual Design Contract

The approved Hallmark direction is:

- Genre: Atmospheric.
- Theme: Midnight.
- Macrostructure: Workbench.
- Tone: Technical, austere, and distraction-free.
- Enrichment: None.

### 11.1 Visual Rules

- Dark theme only.
- Do not use pure black or pure white.
- Do not use gradients.
- Do not use glassmorphism.
- Do not use decorative glow.
- Do not use card-in-card layouts.
- Do not use a uniform grid of equal cards.
- Use one accent color sparingly, below approximately 3% of visible area.
- Reserve red for recording-critical errors or destructive recording actions.
- Pair every color state with text, shape, or iconography.
- Body text must meet WCAG 2.1 AA contrast.
- Focus indicators must meet at least 3:1 contrast.

### 11.2 Color Tokens

The implementation must derive and verify final colors from these starting
tokens. Minor adjustments are allowed only to meet contrast requirements.

| Token | Value |
| --- | --- |
| Background | `#0D1117` |
| Surface | `#131922` |
| Elevated surface | `#19212C` |
| Rule | `#2A3441` |
| Primary text | `#E6EDF3` |
| Muted text | `#9AA7B5` |
| Accent | `#D6A13D` |
| Focus | `#78A9FF` |
| Error | `#E16A6A` |
| Success | `#55B982` |

### 11.3 Typography

- Bundle IBM Plex Sans JP for interface and transcript text.
- Bundle IBM Plex Mono for timers, latency, and technical status values.
- Do not depend on a web font service.
- Do not replace the selected typefaces with generic defaults.

### 11.4 Motion

Only the following motion is allowed:

- A short opacity transition when partial text becomes committed.
- A short opacity transition when a discussion-state value changes.
- A pressed-state movement of at most one logical pixel for buttons.

No transition may animate layout dimensions. Focus indicators appear instantly.
Reduced-motion settings collapse transitions to instant changes or a maximum
150 ms opacity change.

## 12. Controls and Accessibility

Keyboard shortcuts:

| Shortcut | Action |
| --- | --- |
| `Ctrl+Enter` | Start from a valid preflight screen |
| `Space` | Pause or resume when no text input or menu has focus |
| `Ctrl+Shift+S` | Request session stop |
| `Ctrl+T` | Toggle always on top |
| `Escape` | Close the active menu or dialog |

Requirements:

- Every interactive control has default, hover, focus, active, disabled,
  loading, error, and success behavior where applicable.
- Keyboard focus is always visible.
- Primary hit targets are at least 44 by 44 logical pixels.
- Tooltips appear immediately for keyboard focus and after a delay for hover.
- No function is available only through hover.
- Error messages state what failed, why it matters, and what the user can do.
- Status changes use polite accessible announcements.

## 13. Pause and Stop Behavior

### 13.1 Pause

While paused:

- Audio capture stops.
- Audio persistence stops.
- ASR stops receiving new audio.
- Discussion analysis stops receiving new transcript entries.
- Existing transcript and discussion state remain visible.
- A pause event is appended to `events.jsonl`.
- No synthetic silence is inserted into either WAV file.

Resume appends a corresponding event and continues the same session.

### 13.2 Stop

Stop requires confirmation because it ends live capture.

Finalization order is fixed:

1. Stop accepting new audio.
2. Drain captured audio to Writer and ASR.
3. Finalize all uncommitted ASR text.
4. Run one final discussion analysis if committed text is pending.
5. Flush JSONL files.
6. Finalize WAV headers.
7. Write the final discussion state.
8. Mark the session `completed`.
9. Display the completion screen.

The application must show `Finalizing` while this sequence runs. It must not
pretend that the session is complete before persistence finishes.

If finalization has not completed after 30 seconds, the UI changes to
`Finalization is taking longer than expected` and offers `Keep waiting` and
`Force close`. It must never force-close automatically. `Force close` leaves the
session as `incomplete` for recovery on the next launch.

Closing the application during an active or paused session uses the same stop
confirmation and finalization path.

## 14. Completion Screen

The completion screen shows:

- Session duration.
- Number of committed transcript entries.
- Absolute save path.
- `Open folder`.
- `Start another session`.
- `Close`.

It does not show playback, editing, search, or historical session browsing.

## 15. Runtime Architecture

The MVP uses five operating-system processes.

| Process | Responsibility |
| --- | --- |
| GUI / Session Controller | UI, lifecycle, IPC routing, worker monitoring |
| Audio Worker | Device capture, normalization, resampling, frame dispatch |
| ASR Worker | ME and OTHERS transcription using one shared model |
| Discussion Worker | Local structured discussion-state generation |
| Writer Worker | All persistent session writes |

No local HTTP or WebSocket server is permitted. Inter-process communication
uses `multiprocessing.Queue` and typed Python messages.

### 15.1 Technology Stack

| Concern | Selection |
| --- | --- |
| Language | Python 3.12 |
| GUI | PySide6 with Qt Widgets |
| Microphone and loopback | PyAudioWPatch / WASAPI |
| ASR runtime | faster-whisper with CTranslate2 and CUDA |
| Discussion runtime | llama-cpp-python with CUDA |
| Packaging | Folder-based Windows executable |
| Dependency management | pip and `requirements.txt` |
| Formatting | Black and Ruff |
| Type checking | mypy |

All dependency versions must be pinned before implementation is declared
complete.

## 16. Session Lifecycle

Valid lifecycle states are:

```text
IDLE
  -> PREFLIGHT
  -> STARTING
  -> RECORDING
  <-> PAUSED
  -> STOPPING
  -> COMPLETED
```

`ERROR` is reserved for fatal session failures.

All required workers, devices, models, and storage must acknowledge readiness
before `STARTING` can transition to `RECORDING`.

Worker readiness has a 60-second timeout. A timeout returns the application to
preflight with the responsible worker named. It does not start a partial
session.

The GUI checks worker health at least every 500 ms.

## 17. Audio Capture

### 17.1 Sources

- `ME`: selected microphone.
- `OTHERS`: WASAPI loopback of the selected Windows output device.

The loopback source includes every sound routed to that output device,
including notifications. App-specific filtering is not part of the MVP.

### 17.2 Processing Format

Each source is captured at its supported device format and converted to:

- 16,000 Hz.
- 16-bit signed PCM.
- Mono.
- 20 ms internal frames.

The converted frames are the canonical audio sent to both Writer and ASR.

### 17.3 Queue Policy

- Audio callback work must remain minimal.
- Audio persistence and ASR receive separate bounded queues.
- Writer queue exhaustion is fatal and safely stops the session.
- ASR queue delay is visible to the user.
- ASR delay must never silently delete audio from the saved WAV files.
- Discussion load must never block audio dispatch.

## 18. ASR Model and Behavior

### 18.1 Fixed Model

The MVP model is:

```text
Repository: kotoba-tech/kotoba-whisper-v2.0-faster
Runtime: faster-whisper / CTranslate2
Device: CUDA
Compute type: float16
Language: ja
Task: transcribe
Beam size: 1
Temperature: 0
condition_on_previous_text: false
```

The model must not be replaced by `large-v3-turbo`, Qwen3-ASR, another
Kotoba-Whisper version, or any other ASR model without a specification update.

One loaded model instance services two logical stream states. Work is scheduled
fairly between ME and OTHERS, with the oldest pending audio handled first.

### 18.2 Partial and Commit Rules

- Active speech is decoded approximately every 500 ms when capacity permits.
- Approximately 450 ms of silence marks a potential utterance end.
- A prefix may be committed when it is unchanged across two consecutive decodes
  and is at least 1.2 seconds old.
- An utterance-end decode commits the remaining stable text.
- Continuous speech is split no later than 12 seconds, preferably at a phrase
  or punctuation boundary.
- Committed text is immutable.
- Empty text and recognized non-speech markers are discarded.
- Filler-only text remains in the transcript.

## 19. Transcript Data Model

Partial transcript messages are ephemeral and are not written to
`transcript.jsonl`.

Each committed transcript record contains:

```json
{
    "schema_version": 1,
    "segment_id": "01J...",
    "sequence": 42,
    "source": "ME",
    "text": "今回の方針を確認します。",
    "session_start_ms": 12480,
    "session_end_ms": 15820,
    "source_start_sample": 182400,
    "source_end_sample": 235840,
    "committed_at": "2026-08-19T12:34:56.789+09:00"
}
```

Rules:

- `source` is exactly `ME` or `OTHERS`.
- `sequence` is monotonically increasing in merged display order.
- Session timestamps use a monotonic session clock.
- Wall-clock timestamps use ISO 8601 with timezone.
- Source sample offsets exclude pause gaps and map directly to the source WAV.

### 19.1 State History Records

Each successful discussion-state replacement appends one record to
`state-history.jsonl`:

```json
{
    "schema_version": 1,
    "session_id": "01J...",
    "previous_revision": 6,
    "new_revision": 7,
    "state": {
        "revision": 7,
        "mode": "INTERVIEW",
        "current_focus": "志望理由",
        "key_points": [],
        "confirmed_outcomes": [],
        "follow_up_items": [],
        "updated_at": "2026-08-19T12:35:02.125+09:00"
    }
}
```

State-history records do not store transcript evidence IDs. Ordering is derived
from the state revision and JSONL record order.

## 20. Discussion Analysis

### 20.1 Fixed Model

The MVP model is:

```text
Source repository: Qwen/Qwen3-4B-Instruct-2507
Source format: official Safetensors at a pinned revision
Runtime format: GGUF Q4_K_M
Runtime: llama-cpp-python
Context: 8192 tokens
GPU layers: all available layers
Temperature: 0
Maximum output: 512 tokens
Thinking mode: not supported by the selected model
```

The GGUF must be generated from the pinned official model using a pinned
`llama.cpp` revision. The resulting SHA-256 must be recorded in the model
manifest.

The model must not be replaced without a specification update.

### 20.2 Triggering

- Only committed transcript entries trigger analysis.
- A 500 ms coalescing window combines closely spaced commits.
- Only one analysis request may run at a time.
- New commits received during analysis remain queued for the next request.
- Empty and non-speech entries do not trigger analysis.
- A transcript containing only obvious hesitation such as `えー` or `えっと`
  does not trigger analysis.
- `はい`, `いいえ`, and equivalent short answers are not filtered.

### 20.3 Input Context

Each request includes:

1. Mode-specific instructions.
2. The current full discussion-state snapshot.
3. The newest committed transcript entries.

Recent transcript context is limited to:

- At most 60 committed entries.
- At most 6,000 input tokens.

Oldest transcript entries are removed first. The current discussion state is
never removed to make room for transcript context.

### 20.4 Output Contract

Output is constrained with a JSON Schema-derived grammar. The prompt must also
describe every field because the grammar alone does not communicate semantics.

The model returns a complete new snapshot, not a list of mutations.

```json
{
    "revision": 7,
    "mode": "INTERVIEW",
    "current_focus": "志望理由",
    "key_points": [
        "業務改善に関わった経験"
    ],
    "confirmed_outcomes": [
        "完全ローカル動作をMVPの必須条件とする"
    ],
    "follow_up_items": [
        "具体的な成果を確認する"
    ],
    "updated_at": "2026-08-19T12:35:02.125+09:00"
}
```

### 20.5 Semantic Rules

- `current_focus` is replaced when the active topic changes.
- `key_points` contains only points relevant to the current focus.
- `confirmed_outcomes` accumulates across the session.
- `follow_up_items` remains until the conversation resolves or supersedes it.
- Previous focus and key points leave the live snapshot but remain in history.
- The model must not invent facts, decisions, questions, or next actions.
- The model must not evaluate the user or other participants.
- The model must not suggest what to say.
- The model must not create pro/con arguments.
- State items do not include ME or OTHERS attribution.
- State items do not include transcript evidence IDs.

An invalid or failed output is discarded. The previous state remains visible,
and pending transcript entries are included in the next meaningful update.

## 21. IPC Message Contract

Every inter-process message uses this envelope:

```json
{
    "schema_version": 1,
    "session_id": "01J...",
    "message_type": "TRANSCRIPT_COMMITTED",
    "sequence": 42,
    "source": "ASR",
    "created_monotonic_ms": 15540,
    "payload": {}
}
```

Rules:

- Each sender maintains a monotonically increasing sequence.
- Receivers detect duplicates and gaps.
- Unknown schema versions are rejected and logged.
- Large audio payloads use dedicated audio queues, not general control queues.
- Control messages remain small and serializable.
- The empty immutable `AudioDrainFence` marker is used only on dedicated audio
  queues. Audio puts one fence after the final Writer audio command and one
  ordered ASR fence after every pause or stop boundary's preceding frames.
- ASR pause or stop completion is proven by consuming its corresponding fence
  from the same queue as the audio frames, never by observing queue emptiness.
- Every `ASR_STATUS` carries current `backlog_ms` and cumulative
  `maximum_backlog_ms`. Status emission remains transition-only, while the final
  `STOPPED` status reports the authoritative maximum observed on every ASR poll.

## 22. Persistence

### 22.1 Root Directory

Sessions are stored under:

```text
%LOCALAPPDATA%\FlowLens\sessions\<timestamp>_<session-id>\
```

The session ID uses a lexicographically sortable unique identifier. The folder
timestamp uses local time and is safe for Windows filenames.

### 22.2 Required Files

```text
session.json
mic.wav
loopback.wav
transcript.jsonl
discussion-state.json
state-history.jsonl
events.jsonl
```

No mixed audio file is produced.

### 22.3 `session.json`

This file contains at least:

- Schema version.
- Session ID.
- Status: `incomplete`, `completed`, or `recovered`.
- Mode.
- Start and end wall-clock timestamps.
- Active duration.
- Pause intervals.
- Selected device identifiers and display names.
- Model identifiers, revisions, and checksums.
- Application version.
- Transcript entry count.
- Final discussion-state revision.
- Recovery notes.

### 22.4 Write Discipline

- Writer Worker owns every session file write.
- JSONL records are appended and flushed individually.
- Files are synchronized to disk approximately once per second.
- `discussion-state.json` is written to a temporary sibling and atomically
  replaced.
- `session.json` is created as `incomplete` before capture starts.
- Session completion is the final persistent mutation.

### 22.5 Audio Size

Each 16 kHz, 16-bit, mono WAV uses approximately 57.6 MB per 30 minutes.
Two files use approximately 115.2 MB, excluding metadata.

The application does not automatically delete saved sessions.

### 22.6 Event Records

Each `events.jsonl` record contains:

```json
{
    "schema_version": 1,
    "session_id": "01J...",
    "sequence": 18,
    "event_type": "ASR_LAG_STARTED",
    "source": "ASR",
    "session_time_ms": 125400,
    "created_at": "2026-08-19T12:36:44.125+09:00",
    "details": {
        "backlog_ms": 5200
    }
}
```

Allowed MVP event types are:

- `SESSION_START`.
- `PAUSE_START`.
- `PAUSE_END`.
- `STOP_REQUESTED`.
- `SESSION_COMPLETED`.
- `SOURCE_DISCONNECTED`.
- `SOURCE_RECONNECTED`.
- `ASR_LAG_STARTED`.
- `ASR_LAG_ENDED`.
- `ANALYSIS_PAUSED`.
- `ANALYSIS_RESUMED`.
- `ANALYSIS_FAILED`.
- `WORKER_EXITED`.
- `WORKER_RESTARTED`.
- `STORAGE_FAILED`.
- `FORCE_CLOSE_REQUESTED`.
- `SESSION_RECOVERED`.

Event details contain operational metadata only. They do not duplicate audio or
full transcript content.

## 23. Recovery

On launch, FlowLens scans for `incomplete` sessions.

For each incomplete session:

- Validate JSONL records up to the final complete line.
- Discard only a truncated final JSONL line.
- Derive valid PCM length from the audio file size.
- Repair the WAV header when possible.
- Preserve all original recovered data.
- Append a recovery event.
- Change status to `recovered`, never `completed`.

Recovery must not require an internet connection.

## 24. Failure Behavior

| Failure | Required behavior |
| --- | --- |
| Missing device at start | Block start and identify the device problem |
| One source disconnects | Continue the other source, log the gap, retry connection |
| Audio Worker exits | Safely stop the session |
| Writer Worker exits | Safely stop immediately because persistence is unsafe |
| ASR Worker exits | Restart once; continue recording; mark the transcript gap |
| Discussion Worker exits | Restart once; continue recording and transcription |
| ASR backlog grows | Show delay and pause discussion analysis |
| Discussion generation fails | Keep the previous state and retry on the next trigger |
| GPU memory pressure | Preserve ASR and stop discussion analysis if necessary |
| Storage write fails | Show a prominent fatal error and safely stop |
| Parent GUI exits unexpectedly | Workers flush what they own and exit |

Reconnection or worker restart must never be silent. The UI and `events.jsonl`
must record the degraded interval.

A disconnected audio source is retried every two seconds until it reconnects or
the session stops. ASR and Discussion Workers receive at most one automatic
restart attempt per session.

## 25. Runtime Priority and Degradation

Priority is fixed:

1. Saved audio.
2. Committed transcript.
3. Partial transcript.
4. Discussion state.

When overloaded:

1. Coalesce discussion updates for longer.
2. Reduce partial transcription refresh frequency.
3. Stop discussion analysis.

ASR backlog above two seconds changes its status to `Delayed`. Backlog above
five seconds pauses new discussion analysis. Analysis resumes only after ASR
backlog falls below two seconds.

The selected ASR model is not replaced dynamically. Saved audio and committed
transcript are never intentionally discarded to preserve analysis throughput.

## 26. Performance Requirements

Latency is measured with monotonic timestamps in `events.jsonl`.

| Metric | Normal target | p95 target |
| --- | ---: | ---: |
| Partial transcript availability | 1.0 s | 2.0 s |
| Commit after utterance end | 2.0 s | 3.0 s |
| Discussion update after utterance end | 4.0 s | 5.0 s |

Additional requirements:

- Direct UI feedback appears within 100 ms.
- Audio queue overflow count is zero in the final acceptance session.
- Excluding pauses, WAV duration error is below 0.5%.
- Resident memory growth from minute 5 to minute 30 is below 500 MB.
- No GPU out-of-memory error occurs in the final acceptance session.
- Discussion analysis must not cause committed ASR p95 to exceed its limit.

These targets validate the selected implementation. They are not a model
comparison benchmark.

## 27. Minimal Verification Strategy

Model comparison and dataset benchmarking are explicitly not required.

### 27.1 Audio Smoke Test

Duration: approximately one minute.

Verify:

- ME and OTHERS are captured simultaneously.
- Each source is written to the correct WAV.
- Both WAV files are 16 kHz, 16-bit, mono.
- No queue overflow occurs.

### 27.2 ASR Smoke Test

Duration: approximately two minutes.

Verify:

- Japanese speech produces partial and committed text.
- ME and OTHERS remain correctly labeled.
- A short overlapping section produces separate entries.
- The selected Kotoba-Whisper model is used.

No CER dataset or competing model run is required.

### 27.3 Discussion Smoke Test

Use one short input for each mode.

Verify:

- Meeting labels and state are appropriate.
- Interview labels avoid decisions and unresolved-issue wording.
- General labels remain neutral.
- Output conforms to the schema.
- No advice is generated.

No competing LLM run is required.

### 27.4 Integration Smoke Test

Duration: approximately five minutes.

Verify the full path from audio capture through saved artifacts, including one
pause and resume.

### 27.5 Final Acceptance Session

Run one session of at least 30 minutes after implementation is otherwise
complete.

The session must:

- Complete without a crash.
- Meet the performance requirements.
- Produce all seven required artifacts.
- Contain both audio sources.
- Continue hands-free after start.
- Complete with networking blocked.
- Recover usable artifacts after a separate forced-termination check.

The 30-minute session is not a prerequisite for starting implementation.

## 28. MVP Completion Criteria

The MVP is complete only when all of the following are true:

- The application starts from the packaged executable.
- A user can select mode and devices and start a session.
- ME and OTHERS audio are captured separately.
- Partial and committed transcription are visibly distinct.
- Committed transcription is saved sequentially.
- Discussion state updates conservatively in all three modes.
- Start, pause, resume, and stop work.
- Required status and failure states are visible.
- All seven session files are saved.
- Interrupted session artifacts can be recovered.
- A live session requires no network access.
- The final 30-minute acceptance session passes.
- The folder-based executable runs on the designated PC.

## 29. Packaging

The release uses a folder-based build, not a single-file executable.

```text
FlowLens\
    FlowLens.exe
    runtime\
    licenses\
```

Models remain in `%LOCALAPPDATA%\FlowLens\models\` and are not duplicated in
the executable directory.

The build must include licenses for the application dependencies, bundled
fonts, and selected model artifacts.

No installer and no automatic updater are produced for the MVP.

## 30. Roadmap Items

The following ideas are retained but do not affect MVP implementation:

- Speaking suggestions during a conversation.
- Session history and review.
- Audio playback.
- Search and export.
- Transcript correction.
- Retranscription with another model.
- App-specific Windows audio capture.
- Per-person diarization for OTHERS.
- Manual session deletion and retention controls.
- Compact mode and tray controls.
- Additional languages.

## 31. Selected Model References

- [Kotoba-Whisper v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0)
- [Kotoba-Whisper v2.0 for CTranslate2](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster)
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- [llama.cpp JSON Schema grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch)

## 32. Approval Record

The following decisions were explicitly approved during product design:

- Fully local operation is mandatory for the MVP.
- Only the designated PC must be supported.
- The product is online-only in the MVP.
- Audio is retained, not automatically deleted.
- Raw microphone and loopback audio are saved separately.
- The live product organizes discussion but does not advise the user.
- Mode-specific labels share one internal discussion-state schema.
- Organized items do not carry speaker attribution or transcript evidence IDs.
- History, playback, editing, and export are postponed.
- The Hallmark Workbench / Midnight interface direction is approved.
- Worker-process architecture is approved.
- Model selection is delegated and fixed by documented research.
- Model comparison benchmarks are removed from the MVP plan.
