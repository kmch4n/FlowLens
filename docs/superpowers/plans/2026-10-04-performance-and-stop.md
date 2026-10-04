# Performance and Stop Experience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce avoidable launch delay, locate live transcription bottlenecks, and make ordered stop/finalization understandable without compromising saved data.

**Architecture:** Three independent tasks change model preflight, privacy-safe timing diagnostics, and the controller-to-Qt finalization view. Each task has its own red/green tests and commit. No task may infer live speed improvements solely from synthetic tests.

**Tech Stack:** Python 3, PySide6/Qt Widgets, pytest/pytest-qt, multiprocessing workers, local SHA-256, PowerShell packaging.

**Spec:** `docs/superpowers/specs/2026-10-04-performance-and-stop-design.md`; canonical behavior remains in `docs/mvp-spec.md`.

## Global Constraints

- Completely local; no network or cloud processing in the application.
- Preserve full manifest SHA-256 checks before enabling Start; do not add a persistent trust cache.
- Preserve Audio → ASR → Discussion → Writer finalization and durable completion acknowledgement.
- Never auto-force-close; retain the explicit 30-second choice.
- Keep raw audio, transcript, and prompts out of diagnostic reports.
- Build only to `dist/FlowLens`; avoid recording or speaker playback in automated checks.
- Use test-first changes, full suite, Black, Ruff, mypy, package audit and build before completion claims.

## Review Focus

- One model checksum mismatches while the other passes: Start remains blocked with the correct failing model.
- A model path is a link or disappears during checking: no readiness result is falsely marked ready.
- A source disconnects or ASR restarts: timing samples remain non-negative and no content leaks into reports.
- Stop occurs while the app is paused: the same stage progression and durable completion rules apply.
- Window close occurs during finalization: the window remains open with current progress and no implicit force-close.

---

### Task 1: Concurrent full-integrity preflight

**Files:** Modify `src/flowlens/adapters/local_models.py`; test `tests/adapters/test_local_models.py`; update `docs/mvp-spec.md` and `docs/verification/startup-2026-09-22.md`.

**Interfaces:** `LocalModelReadiness.check_required() -> dict[str, ModelCheck]` retains its shape and stable `asr`, `discussion` keys. `_check_entry(models, model_id) -> ModelCheck` remains the per-file verifier.

- [ ] **Step 1: Add failing deterministic concurrency test.** Use the existing `write_ready_manifest()` fixture and a barrier-wrapped `_hash_file` for the two large-model paths; each task must enter hashing before either releases. Assert both models are ready and result keys retain `asr`, `discussion` order. A serial implementation must fail by barrier timeout. Test shape:

```python
barrier = threading.Barrier(2, timeout=2)
original_hash = local_models._hash_file

def observed_hash(path: Path, chunk_size: int) -> str:
    if path.name in {"model.bin", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"}:
        barrier.wait()
    return original_hash(path, chunk_size)

monkeypatch.setattr(local_models, "_hash_file", observed_hash)
assert list(LocalModelReadiness(root, manifest).check_required()) == [
    "asr", "discussion"
]
```
- [ ] **Step 2: Run red.** `.venv/Scripts/python.exe -m pytest tests/adapters/test_local_models.py::test_model_probe_hashes_large_models_concurrently -q` must fail for serial checks, not fixture setup.
- [ ] **Step 3: Implement minimal change.** After the manifest is parsed, submit exactly the two `_check_entry` calls to `ThreadPoolExecutor(max_workers=2)` and collect futures in the existing key order. Keep every per-entry path, metadata, size, SHA-256 and sidecar validation unchanged. Ensure executor lifetime ends before returning. Use this return structure:

```python
with ThreadPoolExecutor(max_workers=2) as executor:
    asr = executor.submit(self._check_entry, models, _ASR_ID)
    discussion = executor.submit(self._check_entry, models, _DISCUSSION_ID)
    return {"asr": asr.result(), "discussion": discussion.result()}
```

- [ ] **Step 4: Run green and edge tests.** Run the full `tests/adapters/test_local_models.py` file. Add `test_parallel_probe_keeps_mixed_model_failure` by replacing one model's bytes after writing the manifest and checking only that model returns `checksum`. Add `test_parallel_probe_rejects_disappeared_file` by removing one model before `check_required()` and checking its `missing` result while the other remains ready.
- [ ] **Step 5: Measure and document.** Compare three read-only runs of sequential reference hashing and `check_required()` on the designated PC. Record median and range, along with disk-contention caveat; do not claim a universal speedup. Update the startup/spec notes.
- [ ] **Step 6: Commit this independent behavior** after verification using the required gitmoji format and user identity checks.

### Task 2: Content-free latency diagnostics and evidence-led ASR tuning

**Files:** Modify `src/flowlens/asr/engine.py` and `src/flowlens/asr/worker.py` for a bounded in-worker decode-duration summary; extend `src/flowlens/domain/enums.py`, `src/flowlens/controller/routing.py`, `src/flowlens/controller/session_controller.py`, and `src/flowlens/app.py` for an optional acceptance-only summary; tests in `tests/asr/test_worker.py`, `tests/controller/test_routing.py`, `tests/controller/test_session_controller.py`, and `tests/smoke/test_acceptance_metrics.py`; update `docs/verification/quality-2026-10-03.md`.

**Interfaces:** Keep existing `ControllerSnapshot.partial_latencies_ms`, `commit_latencies_ms`, `discussion_latencies_ms`, and report `latencies_ms` backward compatible. Add `decode_durations_ms: tuple[int, ...] = ()` to the snapshot and an optional integer list at `latencies_ms.decode`; cap retained samples at 256. `AsrEngine.process_ready(now_monotonic_ms: int) -> AsrBatch` and `finalize(...) -> AsrBatch` retain signatures; add `AsrEngine.take_decode_durations_ms() -> tuple[int, ...]` to drain pending measurements. Add `MessageType.ASR_DECODE_TIMING` with exact payload `{"duration_ms": int}` rather than changing strict `ASR_STATUS` fields.

- [ ] **Step 1: Establish a baseline without capture or playback.** Inspect existing acceptance samples and run a local public/synthetic PCM benchmark; identify whether capture-to-worker backlog, decoder runtime, or GUI delivery dominates. Record p50/p95 and model warm-up separately; do not attribute the result to a live meeting.
- [ ] **Step 2: Add a failing privacy and timing test.** Construct a snapshot with sentinel transcript text and `decode_durations_ms=(37, 42)`; require `_controller_measurements()` to return numeric `latencies_ms.decode == [37, 42]` and ensure serialized report JSON has no sentinel. Run `tests/smoke/test_acceptance_metrics.py` red because the decode field is absent.
- [ ] **Step 3: Implement bounded decode timing.** In `AsrEngine._decode`, measure `time.monotonic_ns()` around the decoder call and append the integer to a bounded queue returned by `take_decode_durations_ms()`. Add `MessageType.ASR_DECODE_TIMING`; after `process_ready()` and `finalize()`, emit `{"duration_ms": duration}` for each drained sample. Extend routing with exact-key non-negative integer validation and append at most the latest 256 samples to `ControllerSnapshot.decode_durations_ms` only when acceptance is enabled. Export the list through `app.py`. Do not log PCM, words, prompts, device names, or unbounded per-frame records. Add restart/disconnect tests that reject negative or stale samples. The shape is:

```python
elapsed_ms = max(0, (time.monotonic_ns() - started_ns) // 1_000_000)
emitter.emit(MessageType.ASR_DECODE_TIMING, {"duration_ms": elapsed_ms})
```
- [ ] **Step 4: Reproduce the dominant delay.** Run the existing public benchmark and a synthetic worker stress case with no physical output. Compare existing partial/commit lag metrics with the new decode summary. Record the slowest measured stage. If the dominant stage cannot be reproduced, keep the diagnostics but make no cadence/VAD/beam change; document that live evidence is needed.
- [ ] **Step 5: Gate any ASR tuning on a separate red test.** Write the test for the single measured bottleneck and specify the same-input latency/content threshold in that test before production changes. Keep final beam 5 and confidence gates unless quality data justify a change. If the measured data cannot support a safe tuning hypothesis, do not implement one in this plan.
- [ ] **Step 6: Document exact evidence and commit diagnostics separately.** Distinguish public/synthetic measurements from real two-source acceptance and leave unmet gates open.

### Task 3: Visible, safe stop progress

**Files:** Modify `src/flowlens/controller/session_controller.py`, `src/flowlens/ui/live_page.py`, and `src/flowlens/ui/dialogs.py`; tests `tests/controller/test_finalization.py`, `tests/ui/test_live_page.py`, `tests/ui/test_main_window.py`, `tests/ui/test_dialogs_completion.py`; update `docs/mvp-spec.md`.

**Interfaces:** Add `finalization_step: FinalizationStep | None = None` and `finalization_elapsed_ms: int = 0` to `ControllerSnapshot` so synthetic fixtures remain valid. The controller derives the stage from `FinalizationCoordinator.snapshot().step`; Qt renders it without commanding worker transitions.

- [ ] **Step 1: Add failing controller tests.** For Audio, ASR, Discussion and Writer acknowledgements, assert exact stage labels and non-decreasing wait time; test paused Stop and completion ACK. Existing `tests/controller/test_finalization.py` harness supplies ordered envelopes.

```python
controller.confirm_stop()
assert controller.snapshot().finalization_step is FinalizationStep.DRAIN_AUDIO
controller.handle_message(audio_stopped_ack)
assert controller.snapshot().finalization_step is FinalizationStep.FINALIZE_ASR
```
- [ ] **Step 2: Run red.** `.venv/Scripts/python.exe -m pytest tests/controller/test_finalization.py -q` must show the new stage expectation failing on the generic `Finalizing` snapshot.
- [ ] **Step 3: Implement only snapshot projection.** Map `DRAIN_AUDIO`, `FINALIZE_ASR`, `FINAL_ANALYSIS`, `FINALIZE_WRITER` to human-readable status. Take elapsed wait from the same monotonic controller clock, avoiding a fake progress percentage or ETA. Clear stage on completion/error. In `snapshot()`, project the coordinator state into a returned immutable copy rather than mutating persistent state on every read:

```python
finalization = self._finalization.snapshot() if self._finalization else None
return replace(
    self._snapshot,
    finalization_step=finalization.step if finalization else None,
    finalization_elapsed_ms=(
        max(0, self._last_clock_ms - finalization.started_ms)
        if finalization and finalization.started_ms is not None and self._last_clock_ms
        else 0
    ),
)
```
- [ ] **Step 4: Add failing offscreen Qt tests.** Render each stage and assert visible capture-stopped message, stage, elapsed wait, and saved-at value. Closing during STOPPING must retain the window and show current progress. At 30 seconds, Keep waiting stays default; Force close copy warns that final text/summary may be incomplete.
- [ ] **Step 5: Implement minimal UI.** Put the stage in the live header/status area, disable active capture controls while STOPPING, and keep the slow-choice dialog explicit. Do not move completion before durable Writer ACK or add automatic force close.
- [ ] **Step 6: Run focused tests and inspect offscreen 1280×800 and 900×700 screenshots.** Update `docs/mvp-spec.md` stop behavior and commit this UX change separately.

### Task 4: Integrated verification and handoff

**Files:** Update relevant `docs/verification/` note with fresh results; no runtime change unless a failing test identifies a defect.

**Interfaces:** Preserve the packaged `dist/FlowLens/FlowLens.exe` path and current main branch.

- [ ] **Step 1: Run full `.venv/Scripts/python.exe -m pytest -q` with `QT_QPA_PLATFORM=offscreen`.** Report pass, fail, skip counts exactly.
- [ ] **Step 2: Run Black check, Ruff and mypy using repository commands; run `git diff --check`.** Repair only task-related issues.
- [ ] **Step 3: Rebuild and audit `dist/FlowLens` with `scripts/build_windows.ps1`; run the bounded package self-check.** Do not launch a physical-audio session.
- [ ] **Step 4: Inspect the staged diff for unrelated `.claude/` or `.codex/` files; verify git identity and GitHub account, then push task commits to current `main`.** Report which latency claims remain unverified by live capture.
