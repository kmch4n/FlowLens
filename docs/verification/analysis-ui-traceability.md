# Analysis and UI MVP Traceability

This matrix records what can be verified automatically in the repository and
what still requires the designated Windows PC. A passing automated test is not
physical-device, real-model, CUDA, packaged-executable, administrator, network
isolation, or 30-minute acceptance evidence.

## Verification commands

The command identifiers used below refer to these repository-root commands:

| ID | Command |
| --- | --- |
| `AUTO-DOMAIN` | `.\.venv\Scripts\python.exe -m pytest tests/domain tests/config tests/persistence tests/audio tests/asr -q` |
| `AUTO-ANALYSIS` | `.\.venv\Scripts\python.exe -m pytest tests/discussion tests/controller tests/integration -q` |
| `AUTO-UI` | `$env:QT_QPA_PLATFORM = "offscreen"; .\.venv\Scripts\python.exe -m pytest tests/ui -q` |
| `AUTO-GATES` | `.\.venv\Scripts\python.exe -m pytest tests/packaging tests/smoke tests/test_no_network_dependencies.py tests/test_text_encoding.py -q` |
| `AUTO-WORKERS` | `.\.venv\Scripts\python.exe -m pytest tests/workers tests/adapters -q` |
| `STATIC` | `.\.venv\Scripts\python.exe -m black --check src tests scripts`; `.\.venv\Scripts\python.exe -m ruff check src tests scripts`; `.\.venv\Scripts\python.exe -m mypy src tests` |

The designated-PC report paths are contracts for evidence collection. They do
not currently exist and are therefore `DEFERRED/BLOCKED`, not passing:

| Evidence | Required real gate | Current blocker |
| --- | --- | --- |
| `build/reports/discussion-smoke.json` | Three-mode inference with the pinned local Qwen GGUF | Model manifest and `llama_cpp` CUDA runtime unavailable |
| `build/reports/integration-smoke.json` | Five-process microphone/loopback integration run | Selected physical device IDs and native model runtimes unavailable |
| `build/reports/acceptance-30m.json` | Packaged, firewall-blocked 30-active-minute session | Package, administrator shell, device selection, model runtimes, and CUDA toolchain unavailable |
| `build/reports/acceptance-recovery.json` | Forced termination and packaged relaunch recovery | Same package/administrator/runtime prerequisites unavailable |

## Specification sections

| Spec | Implementation | Automated evidence | Designated-PC evidence/status |
| --- | --- | --- | --- |
| 6 Privacy and offline contract | `src/flowlens/offline_imports.py`, `src/flowlens/adapters/local_models.py`, `src/flowlens/config/`, `src/flowlens/integration/composition.py` | Static, aliased, literal dynamic, and guarded nonliteral import cases in `tests/test_no_network_dependencies.py`; isolated side-effect traps in `tests/smoke/test_cli_entrypoints.py`; local-model/config tests via `AUTO-DOMAIN`/`AUTO-GATES` | `acceptance-30m.json` — `DEFERRED/BLOCKED`; firewall isolation has not run |
| 7 User modes | `src/flowlens/domain/enums.py`, `src/flowlens/ui/discussion_panel.py`, `src/flowlens/discussion/prompt.py` | `tests/domain/test_enums.py`, `tests/ui/test_live_page.py`, `tests/discussion/test_prompt.py` via `AUTO-DOMAIN`/`AUTO-ANALYSIS`/`AUTO-UI` | `discussion-smoke.json` — `DEFERRED/BLOCKED`; real three-mode inference has not run |
| 8 Primary user flow | `src/flowlens/ui/main_window.py`, `src/flowlens/ui/presenter.py`, `src/flowlens/controller/session_controller.py` | `tests/ui/test_main_window.py`, `tests/ui/test_presenter.py`, `tests/controller/test_session_controller.py` via `AUTO-ANALYSIS`/`AUTO-UI` | `integration-smoke.json` — `DEFERRED/BLOCKED`; real end-to-end flow has not run |
| 9 Preflight screen | `src/flowlens/controller/preflight.py`, `src/flowlens/ui/preflight_page.py` | `tests/controller/test_preflight.py`, `tests/ui/test_preflight_page.py` via `AUTO-ANALYSIS`/`AUTO-UI` | `integration-smoke.json` — `DEFERRED/BLOCKED`; physical meters, devices, and model readiness have not run |
| 10 Live screen | `src/flowlens/ui/live_page.py`, `src/flowlens/ui/transcript_view.py`, `src/flowlens/ui/status_strip.py` | `tests/ui/test_live_page.py`, `tests/ui/test_transcript.py`, `tests/ui/test_rendered_visual_contract.py` via `AUTO-UI` | `acceptance-30m.json` — `DEFERRED/BLOCKED`; live usability and hands-free operation have not run |
| 11 Visual design contract | `src/flowlens/ui/design.py`, `assets/styles/flowlens.qss`, `assets/fonts/` | `tests/ui/test_design_contract.py`, `tests/ui/test_rendered_visual_contract.py`, `tests/test_text_encoding.py` via `AUTO-UI`/`AUTO-GATES` | No separate physical gate; offscreen rendering covers the defined static contract, not display-specific visual QA |
| 12 Controls and accessibility | `src/flowlens/ui/main_window.py`, `src/flowlens/ui/dialogs.py`, `src/flowlens/ui/widgets.py` | `tests/ui/test_main_window.py`, `tests/ui/test_dialogs_completion.py`, `tests/ui/test_widgets.py` via `AUTO-UI` | `acceptance-30m.json` — `DEFERRED/BLOCKED`; direct UI latency on the designated PC has not run |
| 13 Pause and stop | `src/flowlens/controller/finalization.py`, `src/flowlens/controller/session_controller.py`, `src/flowlens/workers/finalization_gate.py` | `tests/controller/test_finalization.py`, `tests/controller/test_session_controller.py`, `tests/workers/test_writer_force_close.py` via `AUTO-ANALYSIS`/`AUTO-WORKERS` | `integration-smoke.json` and `acceptance-recovery.json` — `DEFERRED/BLOCKED` |
| 14 Completion screen | `src/flowlens/ui/completion_page.py`, `src/flowlens/ui/presenter.py` | `tests/ui/test_dialogs_completion.py`, `tests/ui/test_presenter.py` via `AUTO-UI` | `acceptance-30m.json` — `DEFERRED/BLOCKED`; packaged completion has not run |
| 15 Runtime architecture | `src/flowlens/integration/composition.py`, `src/flowlens/integration/worker_runtime.py`, `src/flowlens/domain/messages.py` | `tests/integration/test_composition.py`, `tests/integration/test_worker_runtime.py`, `tests/domain/test_messages.py`, `tests/test_no_network_dependencies.py` via `AUTO-DOMAIN`/`AUTO-ANALYSIS`/`AUTO-GATES` | `integration-smoke.json` — `DEFERRED/BLOCKED`; real five-process native-runtime run has not completed |
| 16 Session lifecycle | `src/flowlens/controller/session_controller.py`, `src/flowlens/controller/supervision.py` | `tests/controller/test_session_controller.py`, `tests/controller/test_supervision.py`, `tests/integration/test_composition.py` via `AUTO-ANALYSIS` | `integration-smoke.json` — `DEFERRED/BLOCKED`; readiness and health monitoring with real workers have not run |
| 19 Transcript data model | `src/flowlens/domain/messages.py`, `src/flowlens/asr/commit.py`, `src/flowlens/persistence/session_writer.py` | `tests/domain/test_messages.py`, `tests/asr/test_commit.py`, `tests/persistence/test_session_writer_append.py` via `AUTO-DOMAIN` | `integration-smoke.json` — `DEFERRED/BLOCKED`; real-source transcript records have not run |
| 20 Discussion analysis | `src/flowlens/discussion/`, `src/flowlens/domain/discussion.py` | `tests/discussion/`, `tests/domain/test_discussion.py`, `tests/smoke/test_discussion_smoke.py` via `AUTO-DOMAIN`/`AUTO-ANALYSIS`/`AUTO-GATES` | `discussion-smoke.json` — `DEFERRED/BLOCKED`; real local GGUF inference has not run |
| 21 IPC message contract | `src/flowlens/domain/messages.py`, `src/flowlens/controller/routing.py`, `src/flowlens/audio/dispatch.py` | `tests/domain/test_messages.py`, `tests/controller/test_routing.py`, `tests/audio/test_dispatch.py`, `tests/integration/test_audio_asr_pipeline.py` via `AUTO-DOMAIN`/`AUTO-ANALYSIS` | `integration-smoke.json` — `DEFERRED/BLOCKED`; physical five-process delivery has not run |
| 24 Failure behavior | `src/flowlens/controller/supervision.py`, `src/flowlens/audio/worker.py`, `src/flowlens/workers/writer.py`, `src/flowlens/persistence/recovery.py` | `tests/controller/test_supervision.py`, `tests/audio/test_worker.py`, `tests/workers/test_writer_fatal.py`, `tests/persistence/test_recovery.py` via `AUTO-DOMAIN`/`AUTO-ANALYSIS`/`AUTO-WORKERS` | `acceptance-recovery.json` — `DEFERRED/BLOCKED`; forced termination of the package has not run |
| 25 Runtime priority and degradation | `src/flowlens/audio/dispatch.py`, `src/flowlens/asr/worker.py`, `src/flowlens/controller/routing.py`, `src/flowlens/discussion/scheduler.py` | `tests/audio/test_dispatch.py`, `tests/asr/test_worker.py`, `tests/controller/test_routing.py`, `tests/discussion/test_scheduler.py` via `AUTO-DOMAIN`/`AUTO-ANALYSIS` | `acceptance-30m.json` — `DEFERRED/BLOCKED`; sustained overload behavior has not run on the designated PC |
| 26 Performance requirements | `src/flowlens/app.py`, `scripts/collect_acceptance.py`, `scripts/run_acceptance.ps1` | `tests/smoke/test_acceptance_metrics.py`, `tests/smoke/test_acceptance_script_contract.py` via `AUTO-GATES` | `acceptance-30m.json` — `DEFERRED/BLOCKED`; no real latency, memory, overflow, WAV-error, or GPU-OOM result exists |
| 27 Minimal verification strategy | `src/flowlens/smoke/`, `scripts/smoke_audio.ps1`, `scripts/smoke_asr.ps1`, `scripts/smoke_discussion.py`, `scripts/smoke_integration.py`, `scripts/run_acceptance.ps1` | `tests/audio/test_smoke_script_contract.py`, `tests/smoke/` via `AUTO-DOMAIN`/`AUTO-GATES` | All four report paths above — `DEFERRED/BLOCKED`; harness contracts pass but real smoke/acceptance runs remain required |
| 28 MVP completion criteria | Application, workers, persistence, UI, packaging, and smoke harness files listed in this matrix | `AUTO-DOMAIN`, `AUTO-ANALYSIS`, `AUTO-UI`, `AUTO-GATES`, `AUTO-WORKERS`, and `STATIC` | `acceptance-30m.json` plus `acceptance-recovery.json` — `BLOCKED`; the MVP must not be described as complete |
| 29 Packaging | `packaging/FlowLens.spec`, `packaging/hooks/`, `scripts/build_windows.ps1`, `scripts/check_package.py` | `tests/packaging/` via `AUTO-GATES` | `dist/FlowLens/FlowLens.exe` and package audit — `DEFERRED/BLOCKED`; no package exists |

## Completion-criteria status

`Automated` below means the repository contract is covered; it does not
supersede the required real evidence in the last column.

| Section 28 criterion | Automated mapping | Required real evidence/status |
| --- | --- | --- |
| Packaged executable starts | `tests/packaging/` (`AUTO-GATES`) | Package self-check and designated-PC launch — `BLOCKED` |
| Mode/devices can be selected and a session starts | Controller/UI/composition tests (`AUTO-ANALYSIS`, `AUTO-UI`) | `integration-smoke.json` — `BLOCKED` |
| ME and OTHERS are captured separately | Audio dispatch/worker and validator tests (`AUTO-DOMAIN`, `AUTO-GATES`) | `integration-smoke.json` — `BLOCKED` |
| Partial and committed transcription are visibly distinct | ASR and transcript UI tests (`AUTO-DOMAIN`, `AUTO-UI`) | `integration-smoke.json` — `BLOCKED` |
| Committed transcription is saved sequentially | Message, writer, and session-validator tests (`AUTO-DOMAIN`, `AUTO-GATES`) | `integration-smoke.json` — `BLOCKED` |
| Discussion state updates conservatively in all modes | Discussion and discussion-smoke contract tests (`AUTO-ANALYSIS`, `AUTO-GATES`) | `discussion-smoke.json` — `BLOCKED` |
| Start, pause, resume, and stop work | Controller/finalization/presenter tests (`AUTO-ANALYSIS`, `AUTO-UI`) | `integration-smoke.json` — `BLOCKED` |
| Required status and failure states are visible | Supervision, live-page, status, and presenter tests (`AUTO-ANALYSIS`, `AUTO-UI`) | `acceptance-30m.json` — `BLOCKED` |
| All seven session files are saved | Persistence and session-validator tests (`AUTO-DOMAIN`, `AUTO-GATES`) | `integration-smoke.json` — `BLOCKED` |
| Interrupted artifacts can be recovered | Production recovery and validator tests (`AUTO-DOMAIN`, `AUTO-GATES`) | `acceptance-recovery.json` — `BLOCKED` |
| Live session requires no network | Static/dynamic AST import guard and runtime exact-name allowlist (`AUTO-GATES`) | Firewall evidence within `acceptance-30m.json` — `BLOCKED` |
| Final 30-minute acceptance passes | Acceptance collector boundary tests (`AUTO-GATES`) | `acceptance-30m.json` — `BLOCKED` |
| Folder-based executable runs on designated PC | Spec/audit/build-script tests (`AUTO-GATES`) | Real `dist/FlowLens/FlowLens.exe` audit and launch — `BLOCKED` |

## Release decision

The implementation and harness remain under verification. The MVP is **not
complete** because the required real package, model/device/CUDA smokes,
firewall-isolated 30-minute acceptance, and recovery acceptance evidence are
absent. Run those gates on the designated PC only after the native toolchain,
local model manifests, package, device IDs, and administrator prerequisites are
available.
