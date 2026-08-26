"""PyInstaller 6.21 onedir definition for the local FlowLens desktop app."""

from pathlib import Path


PROJECT_ROOT = Path(SPEC).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
ASSETS_ROOT = PROJECT_ROOT / "assets"

APPLICATION_DATA = [
    (
        str(ASSETS_ROOT / "fonts" / "IBMPlexSansJP-Regular.ttf"),
        "assets/fonts",
    ),
    (
        str(ASSETS_ROOT / "fonts" / "IBMPlexSansJP-SemiBold.ttf"),
        "assets/fonts",
    ),
    (
        str(ASSETS_ROOT / "fonts" / "IBMPlexMono-Regular.ttf"),
        "assets/fonts",
    ),
    (str(ASSETS_ROOT / "styles" / "flowlens.qss"), "assets/styles"),
]

APPLICATION_HIDDEN_IMPORTS = [
    "flowlens.app",
    "flowlens.integration.composition",
    "flowlens.integration.worker_runtime",
    "flowlens.workers.writer",
    "flowlens.audio.worker",
    "flowlens.asr.worker",
    "flowlens.discussion.worker",
    "flowlens.audio.pyaudiowpatch_backend",
    "flowlens.asr.kotoba_whisper",
    "flowlens.discussion.llama_cpp_adapter",
]

a = Analysis(
    [str(SOURCE_ROOT / "flowlens" / "__main__.py")],
    pathex=[str(SOURCE_ROOT)],
    binaries=[],
    datas=APPLICATION_DATA,
    hiddenimports=APPLICATION_HIDDEN_IMPORTS,
    hookspath=[str(PROJECT_ROOT / "packaging" / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FlowLens",
    console=False,
    contents_directory="runtime",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="FlowLens",
)
