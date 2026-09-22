# Preflight usability verification — 2026-09-22

## Root cause and change

The presenter synchronously called `refresh_preflight` for each selection change.
`PreflightService.evaluate` enumerated devices and called LocalModelReadiness,
which hashes full model artifacts. This repeated expensive I/O on the GUI thread.

Selection now uses `update_preflight_selection` and the pure `reselect_preflight`
reducer. It retains environment blockers and only normalizes selections. Initial
inspection, explicit recheck and session start still perform full validation.
Dropdown models are retained when their device lists have not changed.

The frontend-design revision preserves bundled fonts and color tokens but
reorganizes setup into settings, local readiness and a persistent start footer.
Native combo popups, disclosure arrows, selected modes, input meters and headings
are styled consistently. The body scrolls at the supported minimum size.

## Evidence

- Three initial focused regressions failed before implementation; the resulting
  tests pass. Additional tests cover preserved model/storage blockers, rejected
  start after environmental changes, and minimum-size scrolling.
- Controller/UI subset: 233 passed before the final two added regressions.
- Full offscreen suite: **1687 passed, 2 skipped**, 64.36 seconds.
- Black: 167 files unchanged; Ruff passed; mypy: 161 files passed.
- Independent read-only review found no required correction in the scoped diff.
- Offscreen ready fixtures inspected at 1280 x 800 and 900 x 600, plus the native
  dropdown and blocked-state fixture. Images: `build/reports/ui-redesign/`.
- PyInstaller build and `check_package.py` passed for
  `dist/ui-refresh/FlowLens`. The dropdown SVG is included in package assets.

## Boundaries

The existing `dist/FlowLens/FlowLens.exe` was running and was not terminated or
overwritten. Use `dist/ui-refresh/FlowLens/FlowLens.exe` for the updated build.
Both builds use the same local data location; do not run two recording sessions.

No physical audio capture/playback or model inference was performed. Timing is
verified by excluding I/O from selection, not by a physical-device latency claim.
Initial inspection, explicit recheck and start-time verification remain
synchronous and can take time. The existing Qt intermittent-abort investigation,
ASR readiness race and physical MVP acceptance gates remain separate open work.
Existing uncommitted runtime recovery changes were preserved in the working tree
and are included in this locally built executable. No commits or pushes were
performed for this UI revision.
