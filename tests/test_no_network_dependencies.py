import ast
import os
from pathlib import Path, PurePosixPath

import pytest

RUNTIME_ROOT = Path("src/flowlens")
BANNED_NETWORK_MODULES = {
    "aiohttp",
    "fastapi",
    "flask",
    "http",
    "httpx",
    "requests",
    "socket",
    "urllib",
    "websocket",
    "websockets",
}
ALLOWED_NONLITERAL_DYNAMIC_IMPORTS = {
    (
        PurePosixPath("src/flowlens/offline_imports.py"),
        "import_local_module",
    ): "One runtime-guarded gateway restricts names to an immutable local allowlist.",
}


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, path: PurePosixPath) -> None:
        self._path = path
        self._importlib_aliases = {"importlib"}
        self._import_module_aliases: set[str] = set()
        self._function_stack: list[str] = []
        self.violations: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.partition(".")[0]
            if root in BANNED_NETWORK_MODULES:
                self._add(node, root)
            if alias.name == "importlib":
                self._importlib_aliases.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module is not None:
            root = node.module.partition(".")[0]
            if root in BANNED_NETWORK_MODULES:
                self._add(node, root)
            if node.module == "importlib":
                for alias in node.names:
                    if alias.name == "import_module":
                        self._import_module_aliases.add(alias.asname or alias.name)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._function_stack.append(node.name)
        self.generic_visit(node)
        self._function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_dynamic_import_call(node.func):
            self._check_dynamic_import(node)
        self.generic_visit(node)

    def _is_dynamic_import_call(self, function: ast.expr) -> bool:
        if isinstance(function, ast.Name):
            return function.id == "__import__" or (
                function.id in self._import_module_aliases
            )
        return (
            isinstance(function, ast.Attribute)
            and function.attr == "import_module"
            and isinstance(function.value, ast.Name)
            and function.value.id in self._importlib_aliases
        )

    def _check_dynamic_import(self, node: ast.Call) -> None:
        module = (
            node.args[0].value
            if node.args and isinstance(node.args[0], ast.Constant)
            else None
        )
        if isinstance(module, str):
            root = module.partition(".")[0]
            if root in BANNED_NETWORK_MODULES:
                self._add(node, root)
            return
        function = self._function_stack[-1] if self._function_stack else "<module>"
        if (self._path, function) not in ALLOWED_NONLITERAL_DYNAMIC_IMPORTS:
            self._add(node, "nonliteral-dynamic-import")

    def _add(self, node: ast.AST, module: str) -> None:
        self.violations.append(f"{self._path}:{getattr(node, 'lineno', 0)}:{module}")


def _network_import_violations(source: str, path: str) -> list[str]:
    tree = ast.parse(source, filename=path)
    visitor = _ImportVisitor(PurePosixPath(path))
    visitor.visit(tree)
    return visitor.violations


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            'import importlib as loader\nloader.import_module("requests.api")\n',
            "requests",
        ),
        (
            'from importlib import import_module as load\nload("http.client")\n',
            "http",
        ),
        ('__import__("socket")\n', "socket"),
        (
            'import importlib\nimportlib.import_module("".join(("req", "uests")))\n',
            "nonliteral-dynamic-import",
        ),
    ],
)
def test_dynamic_network_import_forms_are_rejected(source: str, expected: str) -> None:
    violations = _network_import_violations(source, "src/flowlens/example.py")

    assert len(violations) == 1
    assert violations[0].endswith(f":{expected}")


def test_literal_local_dynamic_import_is_allowed() -> None:
    source = 'import importlib\nimportlib.import_module("flowlens.audio.worker")\n'

    assert _network_import_violations(source, "src/flowlens/example.py") == []


def test_guarded_gateway_is_the_only_nonliteral_dynamic_import_allowance() -> None:
    source = (
        "import importlib\n"
        "def import_local_module(name):\n"
        "    return importlib.import_module(name)\n"
    )

    assert (
        _network_import_violations(
            source,
            "src/flowlens/offline_imports.py",
        )
        == []
    )
    assert _network_import_violations(source, "src/flowlens/copy.py") == [
        "src/flowlens/copy.py:3:nonliteral-dynamic-import"
    ]


def test_local_import_gateway_rejects_constructed_network_and_unknown_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flowlens.offline_imports import import_local_module

    loaded: list[str] = []

    def fake_import_module(name: str) -> object:
        loaded.append(name)
        return object()

    monkeypatch.setattr(
        "flowlens.offline_imports.importlib.import_module",
        fake_import_module,
    )

    import_local_module("flowlens.audio.worker")
    with pytest.raises(ValueError, match="not in the offline import allowlist"):
        import_local_module("req" + "uests")
    with pytest.raises(ValueError, match="not in the offline import allowlist"):
        import_local_module("flowlens.unapproved")

    assert loaded == ["flowlens.audio.worker"]


def test_local_import_gateway_registers_cuda_12_before_asr_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flowlens import offline_imports

    cuda_root = tmp_path / "CUDA" / "v12.8"
    cuda_bin = cuda_root / "bin"
    cuda_bin.mkdir(parents=True)
    (cuda_bin / "cublas64_12.dll").write_bytes(b"cublas")
    (cuda_bin / "cudart64_12.dll").write_bytes(b"cudart")
    registered: list[str] = []
    loaded: list[str] = []

    def register_directory(path: str) -> object:
        registered.append(path)
        return object()

    def import_module(name: str) -> object:
        loaded.append(name)
        return object()

    monkeypatch.setenv("CUDA_PATH_V12_8", str(cuda_root))
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setattr(
        os,
        "add_dll_directory",
        register_directory,
    )
    monkeypatch.setattr(
        "flowlens.offline_imports.importlib.import_module",
        import_module,
    )
    monkeypatch.setattr(
        offline_imports,
        "_DLL_DIRECTORY_HANDLES",
        [],
        raising=False,
    )
    monkeypatch.setattr(
        offline_imports,
        "_REGISTERED_DLL_DIRECTORIES",
        set(),
        raising=False,
    )

    offline_imports.import_local_module("faster_whisper")

    assert registered == [os.fspath(cuda_bin)]
    assert loaded == ["faster_whisper"]


def test_runtime_source_has_no_static_or_unguarded_dynamic_network_imports() -> None:
    violations: list[str] = []

    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        violations.extend(
            _network_import_violations(
                path.read_text(encoding="utf-8"),
                path.as_posix(),
            )
        )

    assert violations == []
