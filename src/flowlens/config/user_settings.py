"""Validated user controls stored separately from window/device preferences."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UserSettings:
    """Recognition controls apply to the next session, never mid-utterance."""

    sensitivity: str = "normal"
    silence_end_ms: int = 600
    text_size: int = 16

    def __post_init__(self) -> None:
        if self.sensitivity not in ("low", "normal", "high"):
            raise ValueError("Unsupported microphone sensitivity")
        if (
            type(self.silence_end_ms) is not int
            or not 300 <= self.silence_end_ms <= 1200
        ):
            raise ValueError("Speech end delay must be between 300 and 1200 ms")
        if type(self.text_size) is not int or not 14 <= self.text_size <= 22:
            raise ValueError("Text size must be between 14 and 22 px")

    @property
    def min_speech_rms(self) -> int:
        """Return the int16 RMS gate for this sensitivity preset."""
        return {"low": 400, "normal": 200, "high": 80}[self.sensitivity]


@dataclass(frozen=True, slots=True)
class SettingsStore:
    """Atomically save settings; reject corrupt files rather than overwrite them."""

    path: Path

    def load(self) -> UserSettings:
        """Read version one settings without creating a file on first launch."""
        if not self.path.exists():
            return UserSettings()
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or set(value)
            != {"schema_version", "sensitivity", "silence_end_ms", "text_size"}
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("Invalid settings file")
        return UserSettings(
            value["sensitivity"], value["silence_end_ms"], value["text_size"]
        )

    def save(self, settings: UserSettings) -> None:
        """Replace the settings file only after its new contents are flushed."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix="settings-", suffix=".tmp", dir=self.path.parent
        )
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                json.dump(
                    {"schema_version": 1, **asdict(settings)},
                    output,
                    ensure_ascii=False,
                    indent=4,
                    allow_nan=False,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
