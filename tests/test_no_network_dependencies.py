import ast
from pathlib import Path

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


def _imported_roots(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Import):
        return {alias.name.partition(".")[0] for alias in node.names}
    if isinstance(node, ast.ImportFrom) and node.module is not None:
        return {node.module.partition(".")[0]}
    return set()


def test_runtime_source_has_no_network_client_or_server_imports() -> None:
    violations: list[str] = []

    for path in sorted(RUNTIME_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            banned = sorted(_imported_roots(node) & BANNED_NETWORK_MODULES)
            if banned:
                violations.append(f"{path}:{getattr(node, 'lineno', 0)}:{banned}")

    assert violations == []
