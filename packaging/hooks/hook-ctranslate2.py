"""Collect CTranslate2's extension modules and runtime DLLs."""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

binaries = collect_dynamic_libs("ctranslate2")
hiddenimports = collect_submodules("ctranslate2")
