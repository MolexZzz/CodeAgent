"""Tests for CodeGraph Phase 2: reverse relationships, multi-hop expansion, and cross-file tracking."""

from pathlib import Path
from tempfile import TemporaryDirectory

from memcodeagent.memory.hybrid_retriever import HybridRetriever
from memcodeagent.workspace import Workspace


def test_reverse_indexes_built_during_indexing():
    """Verify that called_by and subclasses reverse indexes are built during indexing."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create a simple file with function calls and class inheritance
        (root / "module.py").write_text("""
def helper():
    pass

def main():
    helper()

class Base:
    pass

class Child(Base):
    pass
""")

        workspace = Workspace(root)
        retriever = HybridRetriever(workspace)
        retriever.index_workspace()

        # Verify reverse indexes were built
        assert "helper" in retriever._called_by
        assert "Base" in retriever._subclasses

        # helper is called by main
        helper_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "helper")
        main_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "main")
        assert main_idx in retriever._called_by["helper"]

        # Child inherits from Base
        base_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "Base")
        child_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "Child")
        assert child_idx in retriever._subclasses["Base"]


def test_multi_hop_expansion_with_depth_2():
    """Verify that graph expansion traverses 2 hops when graph_depth=2."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create a 3-level call chain: A -> B -> C
        (root / "chain.py").write_text("""
def c():
    pass

def b():
    c()

def a():
    b()
""")

        workspace = Workspace(root)
        retriever = HybridRetriever(workspace, graph_depth=2)
        retriever.index_workspace()

        # Find indices
        a_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "a")
        b_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "b")
        c_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "c")

        # Expand from 'a' with depth=2 should reach both 'b' and 'c'
        expanded = retriever._expand_graph({a_idx})
        assert a_idx in expanded
        assert expanded[a_idx] == 0  # initial match
        assert b_idx in expanded  # 1-hop
        assert expanded[b_idx] == 1
        assert c_idx in expanded  # 2-hop
        assert expanded[c_idx] == 2


def test_multi_hop_expansion_with_depth_1():
    """Verify that graph expansion traverses only 1 hop when graph_depth=1."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create a 3-level call chain: A -> B -> C
        (root / "chain.py").write_text("""
def c():
    pass

def b():
    c()

def a():
    b()
""")

        workspace = Workspace(root)
        retriever = HybridRetriever(workspace, graph_depth=1)
        retriever.index_workspace()

        # Find indices
        a_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "a")
        b_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "b")
        c_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "c")

        # Expand from 'a' with depth=1 should reach 'b' but not 'c'
        expanded = retriever._expand_graph({a_idx})
        assert a_idx in expanded
        assert expanded[a_idx] == 0  # initial match
        assert b_idx in expanded  # 1-hop
        assert expanded[b_idx] == 1
        assert c_idx not in expanded  # 2-hop, should not be reached


def test_reverse_relationship_expansion():
    """Verify that reverse relationships (called_by, subclasses) are expanded."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        (root / "impact.py").write_text("""
def target():
    pass

def caller1():
    target()

def caller2():
    target()
""")

        workspace = Workspace(root)
        retriever = HybridRetriever(workspace, graph_depth=1)
        retriever.index_workspace()

        # Find indices
        target_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "target")
        caller1_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "caller1")
        caller2_idx = next(i for i, doc in enumerate(retriever._code_docs) if doc.name == "caller2")

        # Expand from 'target' should find both callers via reverse relationship
        expanded = retriever._expand_graph({target_idx})
        assert target_idx in expanded
        assert caller1_idx in expanded
        assert caller2_idx in expanded


def test_imports_tracked_in_code_documents():
    """Verify that imports are extracted and stored in CodeDocuments."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        (root / "module.py").write_text("""
import os
from pathlib import Path
from typing import List, Dict

def process():
    pass
""")

        workspace = Workspace(root)
        retriever = HybridRetriever(workspace)
        retriever.index_workspace()

        # All symbols should have the same file-level imports
        for doc in retriever._code_docs:
            assert "os" in doc.imports
            assert "pathlib" in doc.imports
            assert "typing" in doc.imports


def test_imports_persisted_and_loaded():
    """Verify that imports are saved to disk and restored on load."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        (root / "module.py").write_text("""
import json

def load():
    pass
""")

        workspace = Workspace(root)

        # Index and save
        retriever1 = HybridRetriever(workspace)
        retriever1.index_workspace()

        # Load in a new retriever
        retriever2 = HybridRetriever(workspace)
        loaded = retriever2._load_index()
        assert loaded

        # Verify imports were persisted
        for doc in retriever2._code_docs:
            assert "json" in doc.imports


def test_reverse_indexes_rebuilt_after_load():
    """Verify that reverse indexes are rebuilt when loading from disk."""
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        (root / "module.py").write_text("""
def helper():
    pass

def main():
    helper()
""")

        workspace = Workspace(root)

        # Index and save
        retriever1 = HybridRetriever(workspace)
        retriever1.index_workspace()

        # Load in a new retriever
        retriever2 = HybridRetriever(workspace)
        loaded = retriever2._load_index()
        assert loaded

        # Verify reverse indexes were rebuilt
        assert "helper" in retriever2._called_by
        main_idx = next(i for i, doc in enumerate(retriever2._code_docs) if doc.name == "main")
        assert main_idx in retriever2._called_by["helper"]
