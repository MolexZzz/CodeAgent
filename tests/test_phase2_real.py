"""Real-world test of Phase 2 features on this project's codebase."""

from pathlib import Path
from memcodeagent.memory.hybrid_retriever import HybridRetriever
from memcodeagent.workspace import Workspace


def test_reverse_relationships_real():
    """Test: Find all callers of _tokenize function (should find retrieve methods)."""
    workspace = Workspace(Path("D:/Coding Agent"))
    retriever = HybridRetriever(workspace, graph_depth=1)

    # Index the actual codebase
    result = retriever.index_workspace()
    print(f"\n{result}\n")

    # Search for "_tokenize" - this is a utility function
    results = retriever.retrieve("_tokenize function")

    print("=" * 60)
    print("TEST 1: Reverse relationships - Who calls _tokenize?")
    print("=" * 60)
    for item in results.items[:10]:
        print(f"\n{item.kind} - score={item.score:.3f}")
        print(f"  {item.text[:200]}...")

    # Check if we found callers via reverse index
    tokenize_callers = [item for item in results.items if "retrieve" in item.text.lower()]
    print(f"\n✓ Found {len(tokenize_callers)} methods that call _tokenize")


def test_multi_hop_expansion_real():
    """Test: Multi-hop traversal from HybridRetriever.retrieve() method."""
    workspace = Workspace(Path("D:/Coding Agent"))
    retriever = HybridRetriever(workspace, graph_depth=2)
    retriever.index_workspace()

    # Search for the main retrieve method
    results = retriever.retrieve("HybridRetriever retrieve method")

    print("\n" + "=" * 60)
    print("TEST 2: Multi-hop expansion (depth=2) from retrieve()")
    print("=" * 60)

    # retrieve() calls _retrieve_code() and _retrieve_task_history()
    # _retrieve_code() calls _tokenize(), _ensure_model(), _expand_graph()
    # With depth=2, we should find all of these

    method_names = set()
    for item in results.items[:15]:
        if "function" in item.kind:
            name = item.text.split("\n")[0].split()[-1]
            method_names.add(name)
            print(f"  - {name}")

    expected = {"retrieve", "_retrieve_code", "_retrieve_task_history",
                "_tokenize", "_expand_graph", "_ensure_model"}
    found = expected & method_names
    print(f"\n✓ Found {len(found)}/{len(expected)} expected methods via 2-hop expansion")
    print(f"  Found: {found}")


def test_import_tracking_real():
    """Test: Cross-file import tracking."""
    workspace = Workspace(Path("D:/Coding Agent"))
    retriever = HybridRetriever(workspace, graph_depth=1)
    retriever.index_workspace()

    print("\n" + "=" * 60)
    print("TEST 3: Import tracking - Check if imports are captured")
    print("=" * 60)

    # Check a few documents to see if imports were captured
    sample_docs = retriever._code_docs[:5]
    for doc in sample_docs:
        if doc.imports:
            print(f"\n{doc.name} ({doc.kind}):")
            print(f"  imports: {', '.join(doc.imports[:5])}{'...' if len(doc.imports) > 5 else ''}")

    # Count how many symbols have imports tracked
    with_imports = sum(1 for doc in retriever._code_docs if doc.imports)
    total = len(retriever._code_docs)
    print(f"\n✓ {with_imports}/{total} symbols have imports tracked ({with_imports/total*100:.1f}%)")


if __name__ == "__main__":
    test_reverse_relationships_real()
    test_multi_hop_expansion_real()
    test_import_tracking_real()
    print("\n" + "=" * 60)
    print("All real-world tests completed!")
    print("=" * 60)
