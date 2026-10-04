# Device continuity and transcript paragraphs

## Behavior

At preflight, FlowLens prefers saved device IDs that are still available.
Missing IDs fall back to the first compatible device in the current WASAPI
catalog. The backend puts the WASAPI default input and output first when they
are present. The chosen IDs are saved once a session starts, so a later crash
does not erase the latest device choice. If a source has no compatible device,
Start stays blocked.

The live transcript combines consecutive ME fragments when the gap is at most
1.2 seconds, the previous fragment has no sentence-ending punctuation, and
the displayed paragraph stays within 180 characters. Different sources never
join. OTHERS remains separate because it can contain several speakers. The
underlying immutable records, timestamps and `transcript.jsonl` are unchanged.

## Verification

- Fake WASAPI catalog tests cover defaults enumerated after other devices.
- Presenter tests cover available saved IDs, disappeared IDs and saving at
  session start.
- Offscreen Qt tests cover joining, interruption, sentence boundaries and raw
  record preservation. A narrow rendered row expanded from 68 to 106 px when
  a fragment was appended.
- Black and Ruff passed; mypy passed for 178 source files.
- The full suite passed: 1767 passed, 2 skipped.
- `dist/FlowLens` was rebuilt on 2026-10-04; the package audit returned PASS.

No physical capture or speaker playback was performed. The Windows default
output can differ from a meeting app's own output route; the user can still
change the selected device before Start.
