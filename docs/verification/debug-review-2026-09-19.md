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

## Independent review findings awaiting approval to fix

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
subprocess; no runtime fix from this review has been applied yet.

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
