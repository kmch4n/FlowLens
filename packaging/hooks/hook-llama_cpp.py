"""Collect llama-cpp-python's extension modules and runtime DLLs."""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

binaries = collect_dynamic_libs("llama_cpp")
hiddenimports = collect_submodules("llama_cpp")
