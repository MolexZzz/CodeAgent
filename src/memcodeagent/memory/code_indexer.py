from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class CodeSymbol:
    """A function or class extracted from Python source via AST parsing."""

    kind: str  # "function" or "class"
    name: str
    signature: str  # e.g., "def foo(a: int, b: str) -> bool:"
    docstring: str
    path: Path
    line: int

    def to_text(self) -> str:
        """Return a plain-text representation for indexing and display."""
        parts = [f"{self.kind} {self.name}", self.signature]
        if self.docstring:
            parts.append(self.docstring)
        return "\n".join(parts)


def extract_symbols(source_path: Path) -> list[CodeSymbol]:
    """Parse a Python file and extract top-level function/class definitions.

    Returns an empty list if the file cannot be parsed or is not valid Python.
    """
    try:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    symbols: list[CodeSymbol] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            sig = _build_function_signature(node)
            doc = ast.get_docstring(node) or ""
            symbols.append(
                CodeSymbol(
                    kind="function",
                    name=node.name,
                    signature=sig,
                    docstring=doc,
                    path=source_path,
                    line=node.lineno,
                )
            )
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            symbols.append(
                CodeSymbol(
                    kind="class",
                    name=node.name,
                    signature=f"class {node.name}:",
                    docstring=doc,
                    path=source_path,
                    line=node.lineno,
                )
            )
    return symbols


def _build_function_signature(node: ast.FunctionDef) -> str:
    """Reconstruct a readable signature from a FunctionDef AST node."""
    args_parts = []
    for arg in node.args.args:
        arg_str = arg.arg
        if arg.annotation:
            arg_str += f": {ast.unparse(arg.annotation)}"
        args_parts.append(arg_str)
    args_str = ", ".join(args_parts)
    ret_str = ""
    if node.returns:
        ret_str = f" -> {ast.unparse(node.returns)}"
    return f"def {node.name}({args_str}){ret_str}:"
