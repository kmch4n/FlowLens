"""Strict validation helpers for persisted and IPC records."""

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import cast

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ContractValidationError(ValueError):
    """Raised when a value violates a FlowLens wire contract."""


def require_exact_keys(
    value: Mapping[str, object],
    expected_keys: frozenset[str],
    record_name: str,
) -> None:
    """Require a record to contain exactly the expected keys.

    Args:
        value: Record to validate.
        expected_keys: Complete set of permitted keys.
        record_name: Human-readable record name for validation errors.

    Raises:
        ContractValidationError: If keys are missing or unknown.
    """

    actual_keys = frozenset(value)
    missing = sorted(expected_keys - actual_keys)
    unknown = sorted(actual_keys - expected_keys)
    if missing or unknown:
        raise ContractValidationError(
            f"{record_name}: missing={missing}, unknown={unknown}"
        )


def parse_timezone_datetime(value: object, field_name: str) -> datetime:
    """Parse an ISO 8601 datetime that includes a timezone.

    Args:
        value: Candidate datetime string.
        field_name: Field name for validation errors.

    Returns:
        Parsed timezone-aware datetime.

    Raises:
        ContractValidationError: If the value is not a valid aware datetime.
    """

    if not isinstance(value, str):
        raise ContractValidationError(
            f"{field_name} must be an ISO 8601 datetime string"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ContractValidationError(
            f"{field_name} must be an ISO 8601 datetime string"
        ) from error
    if parsed.utcoffset() is None:
        raise ContractValidationError(f"{field_name} must include a timezone")
    return parsed


def json_dumps(value: object) -> str:
    """Encode a value as indented UTF-8 JSON with a final newline."""

    return json.dumps(value, ensure_ascii=False, indent=4) + "\n"


def require_int(value: object, field_name: str) -> int:
    """Require an integer value while excluding booleans."""

    if not isinstance(value, int) or isinstance(value, bool):
        raise ContractValidationError(f"{field_name} must be an integer")
    return value


def require_non_negative_int(value: object, field_name: str) -> int:
    """Require an integer greater than or equal to zero."""

    parsed = require_int(value, field_name)
    if parsed < 0:
        raise ContractValidationError(f"{field_name} must be non-negative")
    return parsed


def require_str(value: object, field_name: str) -> str:
    """Require a string value."""

    if not isinstance(value, str):
        raise ContractValidationError(f"{field_name} must be a string")
    return value


def require_str_list(value: object, field_name: str) -> list[str]:
    """Require a list containing only strings."""

    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractValidationError(f"{field_name} must be a list of strings")
    return cast(list[str], value.copy())


def require_sha256(value: object, field_name: str) -> str:
    """Require a lowercase hexadecimal SHA-256 digest."""

    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ContractValidationError(
            f"{field_name} must be a lowercase SHA-256 hexadecimal digest"
        )
    return value
