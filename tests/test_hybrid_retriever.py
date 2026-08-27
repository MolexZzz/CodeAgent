from pathlib import Path
from tempfile import TemporaryDirectory

from memcodeagent.memory.hybrid_retriever import HybridRetriever
from memcodeagent.workspace import Workspace


def test_index_and_retrieve_code_symbols() -> None:
    """Test that HybridRetriever can index Python files and retrieve code via hybrid search."""
    with TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir)

        # Create sample Python files
        (workspace_root / "math_utils.py").write_text('''
def calculate_sum(numbers: list[int]) -> int:
    """Calculate the sum of a list of numbers."""
    return sum(numbers)

def calculate_average(numbers: list[float]) -> float:
    """Calculate the average of a list of numbers."""
    return sum(numbers) / len(numbers) if numbers else 0.0
''')

        (workspace_root / "string_utils.py").write_text('''
def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

class TextFormatter:
    """Format text in various ways."""
    pass
''')

        workspace = Workspace(workspace_root)
        retriever = HybridRetriever(workspace)

        # Index workspace
        msg = retriever.index_workspace()
        assert "4 code symbols" in msg

        # Retrieve using keyword query (should match BM25)
        context = retriever.retrieve("calculate sum numbers")
        assert len(context.items) > 0
        code_items = [item for item in context.items if item.kind.startswith("code_")]
        assert any("calculate_sum" in item.text.lower() for item in code_items)

        # Retrieve using semantic query (should match vector similarity)
        context2 = retriever.retrieve("compute average of values")
        code_items2 = [item for item in context2.items if item.kind.startswith("code_")]
        assert any("average" in item.text.lower() for item in code_items2)


def test_index_persistence() -> None:
    """Test that index is saved and can be loaded back."""
    with TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir)
        (workspace_root / "sample.py").write_text('''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}"
''')

        workspace = Workspace(workspace_root)
        retriever1 = HybridRetriever(workspace)
        retriever1.index_workspace()

        # Create new retriever instance (should load from disk)
        retriever2 = HybridRetriever(workspace)
        context = retriever2.retrieve("hello greeting")
        code_items = [item for item in context.items if item.kind.startswith("code_")]
        assert any("hello" in item.text.lower() for item in code_items)


def test_task_history_retrieval() -> None:
    """Test that task history is indexed and retrieved by keyword overlap."""
    with TemporaryDirectory() as tmpdir:
        workspace_root = Path(tmpdir)
        workspace = Workspace(workspace_root)
        retriever = HybridRetriever(workspace)

        # Record some task history
        retriever.remember_task(
            "Add unit tests for authentication module",
            "Created test_auth.py with 5 test cases covering login, logout, and token validation."
        )
        retriever.remember_task(
            "Fix bug in payment processing",
            "Fixed null pointer exception in PaymentService.process_payment() method."
        )

        # Retrieve by keyword
        context = retriever.retrieve("authentication tests")
        task_items = [item for item in context.items if item.kind == "task_history"]
        assert len(task_items) > 0
        assert any("authentication" in item.text.lower() for item in task_items)

        # Retrieve different task
        context2 = retriever.retrieve("payment bug fix")
        task_items2 = [item for item in context2.items if item.kind == "task_history"]
        assert any("payment" in item.text.lower() for item in task_items2)
