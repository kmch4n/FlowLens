# Runtime Recovery Implementation Plan

**Goal:** Fix the three reproduced shutdown/recovery defects and investigate the
intermittent Qt event-processing abort without physical audio or model execution.

**Architecture:** Preserve Writer's durable-completion gate. Dispose of queues
after consumers stop, or when a failed launch is abandoned and cannot be reused.
Separate ASR generation-local validation from
session statistics, and preserve Discussion pause transitions across readiness.

**Tech Stack:** Python 3.12, multiprocessing, PySide6, pytest, Black, Ruff, mypy.

**Spec:** `docs/mvp-spec.md`; findings in
`docs/verification/debug-review-2026-09-19.md`.

## Constraints

- Work on the authorized main checkout; preserve unrelated personal settings.
- No playback, physical recording, model inference, or network-backed inference.
- Do not relax acceptance requirements or equate tests with physical acceptance.
- Review this plan independently before production edits. Execute with TDD.

## Task 1: Bounded abandoned-queue disposal

Files: `src/flowlens/integration/worker_runtime.py`,
`tests/integration/test_worker_runtime.py`, new
`tests/integration/test_queue_disposal.py`.

- [x] Reproduce `_close_queues((q,))` in a bounded subprocess with a real spawn
  Queue containing an unconsumed 1 MB payload. Assert subprocess returns zero
  within 10 seconds; run first against the current implementation.
- [x] After consumers stop (or failed-launch abandonment), call
  `cancel_join_thread()` and `close()` instead of
  an unbounded `join_thread()`. Preserve deduplication and error collection,
  including trying close if cancellation fails. No save/completion decisions
  change. Update fake queues to represent the new cleanup interface.
- [x] Cover shutdown/restart callers and cleanup errors. Run runtime and real
  queue tests; retain existing durable finalization tests.

## Task 2: ASR generation-local validation

Files: `src/flowlens/controller/session_controller.py`, new
`tests/controller/test_asr_recovery.py` (reuse existing controller fixtures).

- [x] Reproduce old maximum 7000 -> restart -> READY/RUNNING at 0 -> DELAYED at
  6000. Assert current backlog becomes 6000 and session maximum remains 7000.
  Also assert a later decrease in the same generation is rejected.
- [x] Reproduce ERROR -> preflight -> new session and assert maximum/current
  backlog reset to zero and READY(0,0) is accepted.
- [x] Add `_asr_generation_maximum_ms: int`, initialized/reset at session start
  and successful ASR replacement. Compare incoming maxima against this value;
  update it only for accepted status. Keep snapshot maximum as session aggregate.
- [x] Run controller tests, including existing protocol rejection cases.

## Task 3: Preserve analysis recovery transitions

Files: same controller and recovery test module.

- [x] Reproduce lag-induced Discussion PAUSE -> ASR restart -> READY -> RUNNING;
  assert exactly one Discussion RESUME and Running display. Repeat RUNNING to
  ensure no duplicate resume. Include manually paused and disabled analysis.
- [x] While recording, READY must not clear the lag pause reason before RUNNING
  can reconcile Discussion controls. While manually paused, clear the obsolete
  lag reason without resuming Discussion; user resume remains authoritative.
- [x] Verify restart while paused, lag hysteresis, and second-session behavior.

## Task 4: Isolate Qt abort and add a deterministic regression

Files: `src/flowlens/ui/discussion_panel.py` and UI tests only if evidence points
there; otherwise change only the demonstrated owner of the invalid lifecycle.

- [ ] Run offscreen UI tests with uncaptured diagnostics; isolate the failing
  test and capture the actual callback/error rather than treating abort as cause.
- [ ] Inspect animation ownership, interruption, completion and panel destruction.
  Force event-loop timing (not longer sleeps) to reproduce the invalid state in
  a subprocess if the failure aborts Python.
- [ ] Add a regression for the evidenced lifecycle error before changing code.
  Preserve 120 ms animation and reduced-motion behavior; no visual redesign.
- [ ] Run focused UI tests, then full offscreen tests with normal capture.

## Completion

- [x] Run `.venv/Scripts/python -m pytest -q` with `QT_QPA_PLATFORM=offscreen`.
- [x] Run Black/Ruff on `src tests scripts`, mypy on `src tests`, diff checks.
- [x] Update the verification record with actual red/green outcomes and any
  unresolved evidence. Do not claim completion if Qt remains unexplained.

## Plan review

The independent reviewer confirmed the three proposed fixes. The failed-launch
queue abandonment exception is now explicit; the real Queue regression checks
subprocess exit as well as method return. A separate race was identified:
pause/resume during ASR restart readiness can diverge from the launch-time
`start_paused` value because READY workers ignore these controls. This additional
finding is pending user approval and is not covered by the completed fixes.
