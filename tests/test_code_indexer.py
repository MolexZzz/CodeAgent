from pathlib import Path
from tempfile import TemporaryDirectory

from memcodeagent.memory.code_indexer import extract_symbols


def test_extract_symbols_from_valid_python() -> None:
    source = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

class Calculator:
    """A simple calculator."""
    pass
'''
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sample.py"
        path.write_text(source)
        symbols = extract_symbols(path)

    assert len(symbols) == 2
    func = next(s for s in symbols if s.kind == "function")
    assert func.name == "add"
    assert "int" in func.signature
    assert "Add two numbers" in func.docstring
    assert func.line == 2

    cls = next(s for s in symbols if s.kind == "class")
    assert cls.name == "Calculator"
    assert "simple calculator" in cls.docstring
    assert cls.line == 6


def test_extract_symbols_returns_empty_for_invalid_syntax() -> None:
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "broken.py"
        path.write_text("def foo( invalid syntax")
        symbols = extract_symbols(path)
    assert symbols == []


def test_extract_symbols_to_text() -> None:
    source = '''
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
'''
    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "greet.py"
        path.write_text(source)
        symbols = extract_symbols(path)

    assert len(symbols) == 1
    text = symbols[0].to_text()
    assert "function greet" in text
    assert "def greet(name: str) -> str:" in text
    assert "Say hello." in text
