"""Atomic persistence for the strict local configuration contract."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

from flowlens.config.model import AppConfig
from flowlens.domain._validation import ContractValidationError, json_dumps


class ConfigLoadError(RuntimeError):
    """Raised when the existing configuration cannot be read safely."""


@dataclass(frozen=True, slots=True)
class ConfigStore:
    """Load and atomically replace exactly one local configuration file."""

    path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))

    def load(self) -> AppConfig:
        """Load configuration or return defaults without creating a file."""

        if not self.path.exists():
            return AppConfig.default()
        try:
            encoded = self.path.read_text(encoding="utf-8")
            parsed = json.loads(encoded)
            return AppConfig.from_dict(parsed)
        except (
            ContractValidationError,
            json.JSONDecodeError,
            OSError,
            RecursionError,
            UnicodeError,
        ) as error:
            message = f"Failed to load config {self.path}: {error}"
            raise ConfigLoadError(message) from error

    def save(self, config: AppConfig) -> None:
        """Write configuration via a flushed temporary file and atomic replace."""

        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_name(f"{self.path.name}.tmp")
        try:
            with temp_path.open(encoding="utf-8", mode="w", newline="\n") as file:
                file.write(json_dumps(config.to_dict()))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self.path)
        finally:
            if temp_path.exists():
                temp_path.unlink()
