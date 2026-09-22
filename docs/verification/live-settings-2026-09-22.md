# Live workspace and speech-noise protection

## Scope

- Keep native Windows move, resize, snap and caption buttons, requesting a dark
  caption through DWM. Unsupported systems keep their native caption.
- Use the existing frontend-design direction: readable sans-serif headings,
  restrained separators, compact controls and more space for conversation.
- Stop is an ordinary action; manually paused analysis is not an error.
- Settings are available from the menu or Ctrl+comma. Recognition changes apply
  to the next session; transcript text size changes immediately.

## Recognition contract

The canonical int16 PCM RMS gate is 400 / 200 / 80 for low / standard / high
sensitivity. Both microphone and PC audio use the selected preset. At least
160 ms of VAD-positive audio is required before decoding. This does not remove
the beginning of accepted utterances. Standard end-of-speech silence is 600 ms,
adjustable from 300 to 1200 ms. Transcript font size ranges from 14 to 22 px.

Decoder segments are rejected when supplied confidence metadata is invalid,
nonfinite, no_speech_prob >= 0.6, avg_logprob < -1.0 or compression_ratio > 2.4.
There is no text blacklist: confidently recognized repetitions remain valid.
These gates reduce a demonstrated false-positive path; they cannot guarantee
that a local speech model will never hallucinate or drop quiet/short speech.

Settings are atomically saved in `%LOCALAPPDATA%/FlowLens/settings.json`, separately
from existing window/device preferences. Cancel does not persist edits.

## Verification boundaries

Only synthetic PCM, fake decoders, offscreen Qt and package self-checks are used.
No speaker playback, physical recording or real model inference is performed.
Offscreen renders do not verify the native Windows caption appearance.

The complete offscreen regression suite passed: 1736 passed, 2 skipped.
Black, Ruff and mypy passed for all 169 source/test Python files.

Independent review found a corrupt-settings recovery gap: invalid settings can
leave preflight Ready but prevent Start. The original file is not overwritten.
Recovery behavior is awaiting approval; this is not a completed acceptance item.

## References

- [Microsoft DWM window attributes](https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute)
- [Microsoft native dark theme guidance](https://learn.microsoft.com/ja-jp/windows/apps/desktop/modernize/ui/apply-windows-themes)
- [faster-whisper transcription implementation](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)
