from __future__ import annotations

import ast
import re
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tree_sitter import Language, Parser
import tree_sitter_java


SUPPORTED_CODE_SUFFIXES = {".py", ".java"}


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
    """Extract code symbols from supported source files."""
    suffix = source_path.suffix.lower()
    if suffix == ".py":
        return _extract_python_symbols(source_path)
    if suffix == ".java":
        return _extract_java_symbols(source_path)
    return []


def _extract_python_symbols(source_path: Path) -> list[CodeSymbol]:
    """Parse a Python file and extract top-level function/class definitions with relationships."""
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


def _extract_java_symbols(source_path: Path) -> list[CodeSymbol]:
    """Parse a Java file and extract class/method symbols using tree-sitter."""
    try:
        source_bytes = source_path.read_bytes()
    except OSError:
        return []

    tree = _java_parser().parse(source_bytes)
    root = tree.root_node
    file_imports = _extract_java_imports(root, source_bytes)

    symbols: list[CodeSymbol] = []

    def visit(node: Any, enclosing_class: str | None = None) -> None:
        if node.type in {"class_declaration", "interface_declaration", "enum_declaration", "record_declaration"}:
            class_name = _java_node_name(node, source_bytes)
            if class_name:
                symbols.append(
                    CodeSymbol(
                        kind="class",
                        name=class_name,
                        signature=_java_signature(node, source_bytes),
                        docstring="",
                        path=source_path,
                        line=node.start_point.row + 1,
                        inherits=_java_inherits(node, source_bytes),
                        imports=file_imports,
                    )
                )
                enclosing_class = class_name

        if node.type in {"method_declaration", "constructor_declaration"}:
            method_name = _java_node_name(node, source_bytes)
            if method_name:
                if node.type == "constructor_declaration":
                    display_name = f"{enclosing_class}.<init>" if enclosing_class else "<init>"
                else:
                    display_name = f"{enclosing_class}.{method_name}" if enclosing_class else method_name
                symbols.append(
                    CodeSymbol(
                        kind="function",
                        name=display_name,
                        signature=_java_signature(node, source_bytes),
                        docstring="",
                        path=source_path,
                        line=node.start_point.row + 1,
                        calls=_java_calls(node, source_bytes),
                        imports=file_imports,
                    )
                )

        for child in node.children:
            if child.is_named:
                visit(child, enclosing_class=enclosing_class)

    visit(root)
    return symbols


@lru_cache(maxsize=1)
def _java_parser() -> Parser:
    parser = Parser()
    parser.language = Language(tree_sitter_java.language())
    return parser


def _java_text(source_bytes: bytes, node: Any) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _java_first_line(source_bytes: bytes, node: Any) -> str:
    text = _java_text(source_bytes, node).strip()
    return text.splitlines()[0].strip() if text else ""


def _java_node_name(node: Any, source_bytes: bytes) -> str:
    for child in node.children:
        if child.is_named and child.type == "identifier":
            return _java_text(source_bytes, child).strip()
    text = _java_text(source_bytes, node)
    match = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:\(|\{|$)", text)
    return match.group(1) if match else ""


def _java_signature(node: Any, source_bytes: bytes) -> str:
    first_line = _java_first_line(source_bytes, node)
    if first_line:
        return first_line
    text = _java_text(source_bytes, node).strip()
    return text.splitlines()[0] if text else ""


def _java_imports_text(source_bytes: bytes, node: Any) -> str:
    text = _java_first_line(source_bytes, node)
    text = text.rstrip(";")
    return text.split(" ", 1)[1].strip() if " " in text else ""


def _extract_java_imports(root: Any, source_bytes: bytes) -> list[str]:
    imports: list[str] = []
    for child in root.children:
        if not child.is_named:
            continue
        if child.type == "package_declaration":
            package_name = _java_imports_text(source_bytes, child)
            if package_name:
                imports.append(f"package {package_name}")
        elif child.type == "import_declaration":
            import_name = _java_imports_text(source_bytes, child)
            if import_name:
                imports.append(import_name)
    return imports


def _java_inherits(node: Any, source_bytes: bytes) -> list[str]:
    inherits: list[str] = []
    for child in node.children:
        if not child.is_named:
            continue
        if child.type in {"modifiers", "identifier", "class_body", "type_parameters"}:
            continue
        text = _java_text(source_bytes, child).strip()
        if text:
            inherits.append(re.sub(r"^(extends|implements)\s+", "", text).strip())
    return inherits


def _java_calls(node: Any, source_bytes: bytes) -> list[str]:
    calls: list[str] = []
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in {"method_invocation", "object_creation_expression"}:
            text = _java_text(source_bytes, current).strip()
            if current.type == "object_creation_expression":
                match = re.search(r"\bnew\s+([A-Za-z_][A-Za-z0-9_.]*)", text)
            else:
                match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", text)
            if match:
                calls.append(match.group(1).split(".")[-1])
        for child in reversed(current.children):
            if child.is_named:
                stack.append(child)
    return calls


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
