"""Closed content-free acceptance diagnostic names and retention limit."""

TIMING_SAMPLE_LIMIT = 256
STAGE_TIMING_NAMES = frozenset(
    f"{stage}_{source}"
    for stage in ("capture_to_asr", "backlog", "asr_to_ui")
    for source in ("me", "others")
)
