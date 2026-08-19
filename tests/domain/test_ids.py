from flowlens.domain.ids import new_ulid


def test_new_ulid_returns_uppercase_26_character_wire_id() -> None:
    value = new_ulid()

    assert len(value) == 26
    assert set(value) <= set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
