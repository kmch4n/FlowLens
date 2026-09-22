# Silent debugging review — 2026-09-19

## Scope

Reviewed session lifecycle/status handling, worker shutdown, local ASR loading,
finalization, integration smoke input validation, and session artifact validation.
No speaker playback, physical recording, or model inference was run.
This review does not close the outstanding designated-PC acceptance gates.

## Fixed locally

1. `scripts/validate_session.py`: a completed WAV with exactly 0.5% duration
   error was accepted despite the specification requiring error below 0.5%.
   Reject equality consistently with `collect_acceptance.py`. Regression cases
   cover both shorter and longer WAVs at the boundary and a larger error.
2. `scripts/smoke_integration.py`: NaN and positive infinity were accepted as
   timing arguments, invalidating stop/deadline comparisons. Reject all
   nonfinite values before constructing the application. Nine CLI regression
   cases check all three timing options without accessing real devices.

## Independent review findings (fixed in the September 20 follow-up)

1. **P1 — unbounded queue cleanup.** `worker_runtime.py` calls `close()` and
   `join_thread()` without a deadline. After consumers exit, a feeder with pending
   data can block shutdown/restart indefinitely. A real spawn-context Queue with
   an unconsumed 1 MB payload did not return from `_close_queues` within an outer
   four-second subprocess timeout. Separate durable-save confirmation from
   disposal of abandoned queues; add a real-Queue subprocess regression test.
2. **P2 — ASR generation/session maximum conflation.** Controller status
   validation compares a restarted worker's maximum backlog with the previous
   worker's session-wide maximum. Reproduction: DELAYED(7000,7000), worker exit,
   WORKER_READY, READY(0,0), RUNNING(0,0), DELAYED(6000,6000). The final update is
   rejected as inconsistent. Starting another session also retains the old
   maximum. Track generation-local validation independently of session metrics
   and reset session metrics at session start.
3. **P2 — discussion remains paused after ASR restart.** In the same sequence,
   READY clears the lag-pause flag before RUNNING can send DISCUSSION RESUME.
   The actual worker and displayed status remain paused. Preserve the transition
   until RUNNING or explicitly reconcile worker controls. Verify exactly one
   resume command after recovery, including manual-pause interactions.

The reviewer reproduced these with fake controller dependencies or a bounded
subprocess. The initial review left these unmodified; the follow-up below records
the subsequently authorized fixes.

## Verification

- Baseline: 1656 passed, 2 skipped.
- New focused regression run before fixes: 12 failed (including error-message
  assertions); exact WAV boundaries and nonfinite application startup reproduced.
- After fixes: both affected smoke test modules, 40 passed.
- Black: 164 files unchanged; Ruff passed; documented `mypy src tests`: 158 files
  passed. An exploratory `mypy src scripts` invocation failed on duplicate module
  resolution; it is not the documented check command.
- Full post-change suite: aborted in pytest-qt event processing, not a pass.
- UI-only quiet run: also aborted after 58 tests. Diagnostic `-vv -s` UI-only
  run: 74 passed. The event-processing failure is intermittent/unresolved;
  passing the diagnostic run does not establish its cause or resolution.

The Qt abort requires further isolation around animation/event cleanup. No
speculative UI change or claim of all-suite success has been made. At the time
of the initial review, changes had not been committed or pushed.

## September 20 follow-up

Plan: `docs/superpowers/plans/2026-09-19-runtime-recovery.md`, independently
reviewed before production edits.

### Implemented

- Queue disposal cancels the exit-time feeder join before closing abandoned
  queues. Writer's durable-completion gate is unchanged. Cleanup continues after
  individual cancellation/close failures. A subprocess containing an unconsumed
  1 MB real Queue now returns and exits normally instead of timing out.
- ASR status validation tracks a generation-local maximum separately from the
  session aggregate. Restart accepts the new worker's smaller maximum, while
  decreasing maxima within one generation remain rejected. New sessions reset
  both current and aggregate backlog statistics.
- READY while recording preserves the lag-pause reason until RUNNING sends
  exactly one Discussion RESUME. Manual pause remains authoritative and disabled
  analysis is not resumed.

### Fresh evidence

- Before implementation: 4 regression failures and 1 passing manual-pause case.
  Failures covered maximum rejection, missing resume, stale new-session metrics,
  and the real Queue subprocess exceeding its 10-second deadline.
- After implementation: controller/runtime subset, 188 passed; subsequently
  added cancellation/close error and disabled-analysis coverage.
- Full offscreen suite: **1680 passed, 2 skipped**, 67.06 seconds.
- Black: 166 files unchanged. Ruff: passed. `mypy src tests`: 160 files passed.

### Not resolved

- **Qt native abort:** UI-only diagnostic runs and the full suite passed, but
  the original abort remains unexplained. Independent bounded probes covered
  200 panels / 8000 renders with interruption, DeferredDelete and GC, plus 35
  panels / 280 renders with natural animation completion and GC. Both exited
  normally. No speculative UI fix was applied; this is not a resolved defect.
- **Additional plan-review finding:** manual pause/resume during ASR restart
  readiness can diverge from the launch-time `start_paused` value. The controller
  snapshots this flag in `_restart_launch`, while the READY ASR worker ignores
  PAUSE/RESUME until WORKER_START. Fixing this new finding awaits user approval.

No playback, physical recording, or model inference was run. Existing physical
acceptance and packaging gates remain unchanged. Follow-up changes are not yet
committed or pushed.
