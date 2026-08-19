from datetime import datetime

import pytest

from flowlens.domain._validation import (
    ContractValidationError,
    json_dumps,
    parse_timezone_datetime,
    require_exact_keys,
    require_int,
    require_non_negative_int,
    require_sha256,
    require_str,
    require_str_list,
)


def test_require_exact_keys_accepts_an_exact_match() -> None:
    require_exact_keys({"a": 1, "b": 2}, frozenset({"a", "b"}), "Record")


def test_require_exact_keys_reports_sorted_unknown_and_missing_keys() -> None:
    with pytest.raises(ContractValidationError) as error:
        require_exact_keys(
            {"a": 1, "z": 2, "c": 3},
            frozenset({"a", "b", "d"}),
            "Record",
        )

    message = str(error.value)
    assert "Record" in message
    assert "missing=['b', 'd']" in message
    assert "unknown=['c', 'z']" in message


@pytest.mark.parametrize(
    "value",
    [
        "2026-08-19T12:00:00",
        "2026-08-19T12:00:00.123",
    ],
)
def test_parse_timezone_datetime_rejects_naive_wall_clock(value: str) -> None:
    with pytest.raises(ContractValidationError, match="created_at.*timezone"):
        parse_timezone_datetime(value, "created_at")


@pytest.mark.parametrize("value", [None, 1, "not-a-datetime"])
def test_parse_timezone_datetime_rejects_non_iso_values(value: object) -> None:
    with pytest.raises(ContractValidationError, match="created_at"):
        parse_timezone_datetime(value, "created_at")


def test_parse_timezone_datetime_accepts_an_aware_iso_value() -> None:
    value = "2026-08-19T12:00:00.123+09:00"

    assert parse_timezone_datetime(value, "created_at") == datetime.fromisoformat(value)


def test_json_dumps_preserves_japanese_and_uses_four_spaces() -> None:
    encoded = json_dumps({"text": "方針"})

    assert "方針" in encoded
    assert "\\u65b9" not in encoded
    assert encoded == '{\n    "text": "方針"\n}\n'


def test_require_int_accepts_an_integer() -> None:
    assert require_int(42, "sequence") == 42


@pytest.mark.parametrize("value", [True, False, 1.0, "1", None])
def test_require_int_rejects_non_integer_values(value: object) -> None:
    with pytest.raises(ContractValidationError, match="sequence"):
        require_int(value, "sequence")


@pytest.mark.parametrize("value", [0, 12])
def test_require_non_negative_int_accepts_zero_or_positive_values(value: int) -> None:
    assert require_non_negative_int(value, "duration_ms") == value


@pytest.mark.parametrize("value", [-1, True, 1.0, "1", None])
def test_require_non_negative_int_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ContractValidationError, match="duration_ms"):
        require_non_negative_int(value, "duration_ms")


@pytest.mark.parametrize("value", ["", "方針"])
def test_require_str_accepts_strings(value: str) -> None:
    assert require_str(value, "text") == value


@pytest.mark.parametrize("value", [None, 1, ["text"]])
def test_require_str_rejects_non_strings(value: object) -> None:
    with pytest.raises(ContractValidationError, match="text"):
        require_str(value, "text")


def test_require_str_list_accepts_a_list_of_strings() -> None:
    value = ["first", "二番目"]

    assert require_str_list(value, "items") == value


def test_require_str_list_returns_an_independent_list() -> None:
    value = ["first"]

    validated = require_str_list(value, "items")
    value.append("second")

    assert validated == ["first"]


@pytest.mark.parametrize(
    "value",
    [None, "text", ("first", "second"), ["first", 2]],
)
def test_require_str_list_rejects_non_string_lists(value: object) -> None:
    with pytest.raises(ContractValidationError, match="items"):
        require_str_list(value, "items")


def test_require_sha256_accepts_lowercase_hexadecimal() -> None:
    value = "0123456789abcdef" * 4

    assert require_sha256(value, "sha256") == value


@pytest.mark.parametrize(
    "value",
    [
        None,
        "0123456789abcdef" * 3,
        "0123456789ABCDEF" * 4,
        "g" * 64,
    ],
)
def test_require_sha256_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ContractValidationError, match="sha256"):
        require_sha256(value, "sha256")
