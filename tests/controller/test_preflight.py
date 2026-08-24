import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from flowlens.controller.models import (
    BlockingIssue,
    DeviceOption,
    ModelCheck,
    PreflightSelection,
    StorageCheck,
)
from flowlens.controller.ports import DeviceCatalog, ModelReadiness, StorageReadiness
from flowlens.controller.preflight import REQUIRED_FREE_BYTES, PreflightService
from flowlens.domain.enums import AudioSource, SessionMode


class FakeDevices:
    def __init__(self) -> None:
        self.microphones: tuple[DeviceOption, ...] = (
            DeviceOption("input:1", "Mic", False),
        )
        self.loopbacks: tuple[DeviceOption, ...] = (
            DeviceOption("wasapi-output:2", "Speakers", True),
        )

    def list_microphones(self) -> tuple[DeviceOption, ...]:
        return self.microphones

    def list_loopback_outputs(self) -> tuple[DeviceOption, ...]:
        return self.loopbacks

    def read_level(self, source: AudioSource, device_id: str) -> float:
        del device_id
        return 0.0 if source is AudioSource.ME else 0.25


class FakeModels:
    def __init__(self) -> None:
        self.checks = {
            "asr": ModelCheck("kotoba-whisper-v2.0-faster", None, True, None),
            "discussion": ModelCheck("qwen3-4b-instruct-2507", None, True, None),
        }

    def check_required(self) -> dict[str, ModelCheck]:
        return self.checks.copy()


class FakeStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.check_value = StorageCheck(root, REQUIRED_FREE_BYTES, True, None)

    def check(self, root: Path, required_bytes: int) -> StorageCheck:
        assert root == self.root
        assert required_bytes == REQUIRED_FREE_BYTES
        return self.check_value


def make_service(
    tmp_path: Path,
) -> tuple[PreflightService, FakeDevices, FakeModels, FakeStorage]:
    root = tmp_path.resolve()
    devices = FakeDevices()
    models = FakeModels()
    storage = FakeStorage(root)
    return PreflightService(devices, models, storage, root), devices, models, storage


def selection(
    mic: str | None = "input:1", loopback: str | None = "wasapi-output:2"
) -> PreflightSelection:
    return PreflightSelection(SessionMode.MEETING, mic, loopback)


@pytest.mark.parametrize(
    ("mutation", "control_id", "message"),
    [
        ("no_mic", "microphone", "Select an available microphone."),
        (
            "no_loopback",
            "loopback",
            "Select a loopback-capable Windows output device.",
        ),
        ("asr_missing", "asr_model", "Kotoba-Whisper model files are missing."),
        (
            "discussion_checksum",
            "discussion_model",
            "Qwen discussion model checksum does not match.",
        ),
        (
            "storage_unwritable",
            "storage",
            "FlowLens cannot create the session folder.",
        ),
        ("storage_small", "storage", "At least 500 MB of free space is required."),
    ],
)
def test_preflight_names_each_blocker(
    tmp_path: Path,
    mutation: str,
    control_id: str,
    message: str,
) -> None:
    service, devices, models, storage = make_service(tmp_path)
    if mutation == "no_mic":
        devices.microphones = ()
    elif mutation == "no_loopback":
        devices.loopbacks = ()
    elif mutation == "asr_missing":
        models.checks["asr"] = ModelCheck(
            "kotoba-whisper-v2.0-faster", None, False, "missing"
        )
    elif mutation == "discussion_checksum":
        models.checks["discussion"] = ModelCheck(
            "qwen3-4b-instruct-2507", None, False, "checksum"
        )
    elif mutation == "storage_unwritable":
        storage.check_value = StorageCheck(tmp_path.resolve(), 0, False, "unwritable")
    elif mutation == "storage_small":
        storage.check_value = StorageCheck(tmp_path.resolve(), 1, True, None)

    report = service.evaluate(selection())

    assert BlockingIssue(control_id, message) in report.issues
    assert report.can_start is False


def test_missing_saved_device_is_not_replaced_and_zero_level_is_visible(
    tmp_path: Path,
) -> None:
    service, _, _, _ = make_service(tmp_path)

    report = service.evaluate(selection("gone-mic", "gone-output"))

    assert report.selection.microphone_id is None
    assert report.selection.loopback_output_id is None
    assert report.mic_level == 0.0
    assert report.loopback_level == 0.0
    assert [issue.control_id for issue in report.issues] == ["microphone", "loopback"]


def test_preflight_reports_all_checks_in_stable_control_order(tmp_path: Path) -> None:
    service, devices, models, storage = make_service(tmp_path)
    devices.microphones = ()
    devices.loopbacks = ()
    models.checks = {
        "discussion": ModelCheck("qwen3-4b-instruct-2507", None, False, "checksum"),
        "asr": ModelCheck("kotoba-whisper-v2.0-faster", None, False, "missing"),
    }
    storage.check_value = StorageCheck(tmp_path.resolve(), 1, False, "unwritable")

    report = service.evaluate(selection())

    assert [issue.control_id for issue in report.issues] == [
        "microphone",
        "loopback",
        "asr_model",
        "discussion_model",
        "storage",
        "storage",
    ]
    assert tuple(check.model_id for check in report.models) == (
        "kotoba-whisper-v2.0-faster",
        "qwen3-4b-instruct-2507",
    )


def test_value_objects_are_frozen_and_reject_bool_as_integer(tmp_path: Path) -> None:
    option = DeviceOption("input:1", "Mic", False)
    with pytest.raises(FrozenInstanceError):
        option.display_name = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        StorageCheck(tmp_path.resolve(), True, True, None)


def test_preflight_values_are_picklable_and_ports_are_runtime_checkable(
    tmp_path: Path,
) -> None:
    service, devices, models, storage = make_service(tmp_path)
    report = service.evaluate(selection())

    assert pickle.loads(pickle.dumps(report)) == report
    assert isinstance(devices, DeviceCatalog)
    assert isinstance(models, ModelReadiness)
    assert isinstance(storage, StorageReadiness)


def test_device_discovery_os_failure_becomes_device_blockers(tmp_path: Path) -> None:
    service, devices, _, _ = make_service(tmp_path)

    def fail_discovery() -> tuple[DeviceOption, ...]:
        raise OSError("vendor detail must not escape")

    devices.list_microphones = fail_discovery  # type: ignore[method-assign]
    devices.list_loopback_outputs = fail_discovery  # type: ignore[method-assign]

    report = service.evaluate(selection())

    assert report.selection.microphone_id is None
    assert report.selection.loopback_output_id is None
    assert tuple(issue.control_id for issue in report.issues[:2]) == (
        "microphone",
        "loopback",
    )


def test_adversarial_device_port_exception_is_sanitized_without_stringification(
    tmp_path: Path,
) -> None:
    service, devices, _, _ = make_service(tmp_path)

    class HostilePortError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("must not stringify")

    def fail_discovery() -> tuple[DeviceOption, ...]:
        raise HostilePortError()

    devices.list_microphones = fail_discovery  # type: ignore[method-assign]
    devices.list_loopback_outputs = fail_discovery  # type: ignore[method-assign]

    report = service.evaluate(selection())

    assert tuple(issue.control_id for issue in report.issues[:2]) == (
        "microphone",
        "loopback",
    )


def test_adversarial_meter_port_exception_becomes_zero(tmp_path: Path) -> None:
    service, devices, _, _ = make_service(tmp_path)

    class HostilePortError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("must not stringify")

    def fail_level(source: AudioSource, device_id: str) -> float:
        del source, device_id
        raise HostilePortError()

    devices.read_level = fail_level  # type: ignore[method-assign]

    report = service.evaluate(selection())

    assert report.mic_level == 0.0
    assert report.loopback_level == 0.0


def test_adversarial_model_and_storage_ports_fail_closed(tmp_path: Path) -> None:
    service, _, models, storage = make_service(tmp_path)

    class HostilePortError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("must not stringify")

    def fail_models() -> dict[str, ModelCheck]:
        raise HostilePortError()

    def fail_storage(root: Path, required_bytes: int) -> StorageCheck:
        del root, required_bytes
        raise HostilePortError()

    models.check_required = fail_models  # type: ignore[method-assign]
    storage.check = fail_storage  # type: ignore[method-assign]

    report = service.evaluate(selection())

    assert [issue.control_id for issue in report.issues] == [
        "asr_model",
        "discussion_model",
        "storage",
        "storage",
    ]
    assert report.can_start is False


def test_value_objects_reject_subclasses_of_builtin_scalar_types(
    tmp_path: Path,
) -> None:
    class StringSubclass(str):
        pass

    class IntegerSubclass(int):
        pass

    with pytest.raises(ValueError):
        DeviceOption(StringSubclass("input:1"), "Mic", False)
    with pytest.raises(ValueError):
        StorageCheck(tmp_path.resolve(), IntegerSubclass(1), True, None)
