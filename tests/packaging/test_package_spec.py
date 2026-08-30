"""Static contracts for the Windows onedir PyInstaller configuration."""

from pathlib import Path


def test_spec_is_onedir_with_runtime_contents_directory() -> None:
    """Package contents must be separate from the executable."""

    assert (
        Path("requirements-dev.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        .count("PyInstaller==6.21.0")
        == 1
    )
    text = Path("packaging/FlowLens.spec").read_text(encoding="utf-8")
    assert 'name="FlowLens"' in text
    assert 'contents_directory="runtime"' in text
    assert "exclude_binaries=True" in text
    assert "COLLECT(" in text
    assert "onefile" not in text.lower()


def test_spec_never_collects_localappdata_models() -> None:
    """Models are installed beside user data, never in a distributable bundle."""

    text = Path("packaging/FlowLens.spec").read_text(encoding="utf-8").lower()
    assert "localappdata" not in text
    assert "models\\" not in text
    assert "models/" not in text


def test_spec_collects_application_resources_and_worker_modules() -> None:
    """The frozen process can locate fonts, QSS, and late-bound workers."""

    text = Path("packaging/FlowLens.spec").read_text(encoding="utf-8")
    for resource in (
        "IBMPlexSansJP-Regular.ttf",
        "IBMPlexSansJP-SemiBold.ttf",
        "IBMPlexMono-Regular.ttf",
        "flowlens.qss",
    ):
        assert resource in text
    for module in (
        "flowlens.workers.writer",
        "flowlens.audio.worker",
        "flowlens.asr.worker",
        "flowlens.discussion.worker",
        "PySide6.QtWidgets",
        "pyaudiowpatch",
        "faster_whisper",
        "ctranslate2",
        "llama_cpp",
    ):
        assert module in text


def test_native_hooks_collect_dynamic_libraries_and_submodules() -> None:
    """Native packages have explicit PyInstaller hook coverage."""

    for package in ("llama_cpp", "ctranslate2", "pyaudiowpatch"):
        path = Path(f"packaging/hooks/hook-{package}.py")
        text = path.read_text(encoding="utf-8")
        assert "collect_dynamic_libs" in text
        assert "collect_submodules" in text
        assert f'"{package}"' in text
    pyaudio_hook = Path("packaging/hooks/hook-pyaudiowpatch.py").read_text(
        encoding="utf-8"
    )
    assert '"_portaudiowpatch"' in pyaudio_hook

    webrtcvad_hook = Path("packaging/hooks/hook-webrtcvad.py").read_text(
        encoding="utf-8"
    )
    assert 'copy_metadata("webrtcvad-wheels")' in webrtcvad_hook
    assert '"_webrtcvad"' in webrtcvad_hook


def test_spec_build_uses_hook_path_only_inside_the_spec() -> None:
    """PyInstaller forbids makespec-only hook options with a spec file."""

    spec = Path("packaging/FlowLens.spec").read_text(encoding="utf-8")
    build_script = Path("scripts/build_windows.ps1").read_text(encoding="utf-8")

    assert "hookspath=" in spec
    assert "--additional-hooks-dir" not in build_script


def test_spec_excludes_path_sourced_icu_binaries() -> None:
    """Qt must resolve Windows ICU instead of an unrelated PATH copy."""

    text = Path("packaging/FlowLens.spec").read_text(encoding="utf-8")

    assert '"icuuc.dll"' in text
    assert '"icudt"' in text
    assert "a.binaries =" in text


def test_spec_excludes_test_only_packages_from_runtime() -> None:
    """Development-only test stacks must not become product dependencies."""

    text = Path("packaging/FlowLens.spec").read_text(encoding="utf-8")

    for package in ('"pytest"', '"_pytest"', '"pygments"'):
        assert package in text
