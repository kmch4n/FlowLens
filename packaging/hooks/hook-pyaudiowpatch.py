"""Collect PyAudioWPatch's extension modules and PortAudio DLLs."""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

binaries = collect_dynamic_libs("pyaudiowpatch") + collect_dynamic_libs(
    "_portaudiowpatch"
)
hiddenimports = [*collect_submodules("pyaudiowpatch"), "_portaudiowpatch"]
