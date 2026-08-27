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
    calls: list[str] = None  # Function names this symbol calls
    inherits: list[str] = None  # Parent class names (for classes)
    imports: list[str] = None  # Imported symbols/modules used in this file

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []
        if self.inherits is None:
            self.inherits = []
        if self.imports is None:
            self.imports = []

    def to_text(self) -> str:
        """Return a plain-text representation for indexing and display."""
        parts = [f"{self.kind} {self.name}", self.signature]
        if self.docstring:
            parts.append(self.docstring)
        return "\n".join(parts)


def extract_symbols(source_path: Path) -> list[CodeSymbol]:
    """Parse a Python file and extract top-level function/class definitions with relationships.

    Returns an empty list if the file cannot be parsed or is not valid Python.
    """
    try:
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    # Extract imports at file level (shared by all symbols in this file)
    file_imports = _extract_imports(tree)

    symbols: list[CodeSymbol] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            sig = _build_function_signature(node)
            doc = ast.get_docstring(node) or ""
            calls = _extract_function_calls(node)
            symbols.append(
                CodeSymbol(
                    kind="function",
                    name=node.name,
                    signature=sig,
                    docstring=doc,
                    path=source_path,
                    line=node.lineno,
                    calls=calls,
                    imports=file_imports,
                )
            )
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            inherits = _extract_base_classes(node)
            symbols.append(
                CodeSymbol(
                    kind="class",
                    name=node.name,
                    signature=f"class {node.name}:",
                    docstring=doc,
                    path=source_path,
                    line=node.lineno,
                    inherits=inherits,
                    imports=file_imports,
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


def _extract_function_calls(node: ast.FunctionDef) -> list[str]:
    """Extract names of functions called within a FunctionDef node."""
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            # Handle simple function calls: foo()
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            # Handle method calls: obj.method() - extract method name
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
    return calls


def _extract_base_classes(node: ast.ClassDef) -> list[str]:
    """Extract parent class names from a ClassDef node."""
    bases = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            bases.append(base.id)
        elif isinstance(base, ast.Attribute):
            # Handle module.ClassName -> extract ClassName
            bases.append(base.attr)
    return bases


def _extract_imports(tree: ast.AST) -> list[str]:
    """Extract imported symbols from a module's AST.

    Captures both 'import foo' and 'from foo import bar' statements.
    Returns a list of imported names that can be cross-referenced.
    """
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # import foo, bar
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # from foo import bar, baz
            if node.module:
                imports.append(node.module)
            for alias in node.names:
                if alias.name != "*":
                    imports.append(alias.name)
    return imports
