"""Shared wire-format identifiers."""

from ulid import ULID


def new_ulid() -> str:
    """Create a new uppercase Crockford ULID wire identifier."""

    return str(ULID()).upper()
