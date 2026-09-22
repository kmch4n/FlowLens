# Preflight usability repair

## Approved direction

Use frontend-design for a native desktop setup workspace, not a web landing page.
Keep the bundled IBM Plex Sans JP family and accessible existing ink/surface/text
tokens (#0D1117, #131922, #19212C, #E6EDF3, #9AA7B5). Use amber #D6A13D only
for selection and the primary action. Establish a 28 px title, 18 px section
headings and 14 px control text. Avoid monospace for instructional headings.

Layout: left-aligned heading and explanation, then a generous settings column
alongside a narrower readiness panel, and a persistent footer with the start
action. Mode choices describe intent; microphone and PC output stay distinct.
Use inline actionable errors and an explicit refresh action for device changes.
At the supported 900 x 600 minimum, scroll the body rather than clip controls.

## Performance contract

Selection changes use the last complete preflight report, normalizing device IDs
and recomputing only selection blockers. They must not enumerate devices, hash
models, access storage or read audio levels. Retain model/storage blockers and
clear stale meter values. Initial inspection, explicit refresh and actual start
still perform the complete checks; a cached preview never authorizes capture.
Keep combo models unchanged when the device list is unchanged.

## Implementation and verification

1. Add failing tests for side-effect-free selection updates, blocker preservation
   and stable combo models. Implement a pure selection reducer and a dedicated
   controller method used by the presenter; keep full refresh/start separate.
2. Redesign `preflight_page.py` in place and extend `flowlens.qss` for native
   combo popups, radio choices, section typography and the primary action.
3. Add refresh wiring and preserve keyboard shortcuts, accessible names, error
   text, exact device IDs and unavailable-device behavior.
4. Render ready and blocked fixtures offscreen at 1280 x 800 and 900 x 600;
   visually inspect images and test layout bounds without real audio.
5. Run controller/UI regressions, full offscreen tests, Black/Ruff/mypy. Record
   unresolved physical/performance gates separately from deterministic tests.

Existing uncommitted runtime recovery work is preserved, not part of this UI
redesign. No microphone capture, speaker playback, model inference or deployment.
