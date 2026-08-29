from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

from memcodeagent.memory.code_indexer import extract_symbols
from memcodeagent.memory.schema import MemoryItem
from memcodeagent.memory.task_store import TaskMemoryStore
from memcodeagent.workspace import Workspace

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are",
    "add", "please", "task", "with", "that", "this", "it", "be", "as",
}
_MAX_RECORDS = 200
_MAX_RETRIEVED_TASK = 3
_MAX_RETRIEVED_CODE = 5
_FALLBACK_VECTOR_DIM = 64


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 (returns list to preserve frequency)."""
    words = re.findall(r"[A-Za-z0-9_./\\-]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


class _FallbackEmbeddingModel:
    """Tiny local embedding fallback used when sentence-transformers is unavailable."""

    def encode(
        self,
        texts: list[str] | str,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray | list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        vectors = np.zeros((len(texts), _FALLBACK_VECTOR_DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _tokenize(text):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % _FALLBACK_VECTOR_DIM
                vectors[row, index] += 1.0
        if convert_to_numpy:
            return vectors
        return vectors.tolist()


@dataclass(slots=True)
class RetrievalContext:
    items: list[MemoryItem] = field(default_factory=list)

    def to_prompt(self) -> str:
        if not self.items:
            return "(no retrieved context yet)"
        return "\n\n".join(f"[{item.kind}] {item.text}" for item in self.items)

    def to_display(self) -> str:
        if not self.items:
            return "(no retrieved context yet)"
        lines = []
        for item in self.items:
            location = f" {item.path}" if item.path else ""
            lines.append(f"- {item.kind}{location} score={item.score:.2f} reason={item.reason}\n  {item.text[:120]}...")
        return "\n".join(lines)


@dataclass(slots=True)
class CodeDocument:
    """A single indexed code symbol with its text and metadata."""
    text: str
    path: Path
    kind: str
    name: str
    line: int
    calls: list[str] = field(default_factory=list)
    inherits: list[str] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)


class HybridRetriever:
    """Code retrieval using BM25, vectors, and graph expansion."""

    def __init__(
        self,
        workspace: Workspace,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        graph_depth: int = 1,
        task_memory_store: TaskMemoryStore | None = None,
    ) -> None:
        self.workspace = workspace
        self.model_name = model_name
        self.graph_depth = graph_depth  # How many hops to traverse in the code graph
        self._model: "SentenceTransformer | None" = None
        self._code_docs: list[CodeDocument] = []
        self._code_bm25: BM25Okapi | None = None
        self._code_vectors: np.ndarray | None = None
        self._index_path = self.workspace.root / ".memcode" / "code_index.json"
        self._vectors_path = self.workspace.root / ".memcode" / "code_vectors.npy"
        self._file_hashes: dict[str, str] = {}
        self.task_memory = task_memory_store or TaskMemoryStore(self.workspace.root)
        # Reverse indexes for impact analysis
        self._called_by: dict[str, set[int]] = {}  # symbol_name -> set of doc indices that call it
        self._subclasses: dict[str, set[int]] = {}  # class_name -> set of doc indices that inherit from it

    def _ensure_model(self) -> "SentenceTransformer":
        if self._model is None:
            try:
                # Lazy import: only load sentence_transformers when actually needed.
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = _FallbackEmbeddingModel()  # type: ignore[assignment]
        return self._model

    # -- indexing --------------------------------------------------------------

    def index_workspace(self) -> str:
        """Scan workspace Python files, extract code symbols, build BM25 + vector index."""
        previous_docs: dict[str, list[CodeDocument]] = {}
        previous_hashes = dict(self._file_hashes)
        if self._load_index():
            previous_hashes = dict(self._file_hashes)
            for doc in self._code_docs:
                key = doc.path.relative_to(self.workspace.root).as_posix()
                previous_docs.setdefault(key, []).append(doc)

        python_files = [
            p for p in self.workspace.root.glob("**/*.py")
            if p.is_file() and not self.workspace.should_ignore(p)
        ]

        self._code_docs = []
        self._file_hashes = {}
        reused_files = 0
        for py_file in python_files:
            relative = py_file.relative_to(self.workspace.root).as_posix()
            digest = hashlib.sha256(py_file.read_bytes()).hexdigest()
            self._file_hashes[relative] = digest
            if previous_hashes.get(relative) == digest and relative in previous_docs:
                documents = previous_docs[relative]
                reused_files += 1
            else:
                documents = [
                    CodeDocument(
                        text=sym.to_text(),
                        path=py_file,
                        kind=sym.kind,
                        name=sym.name,
                        line=sym.line,
                        calls=sym.calls,
                        inherits=sym.inherits,
                        imports=sym.imports,
                    )
                    for sym in extract_symbols(py_file)
                ]
            for doc in documents:
                self._code_docs.append(
                    doc
                )

        if not self._code_docs:
            return f"Indexed {len(python_files)} Python files, found 0 code symbols."

        # Build BM25 index
        tokenized = [_tokenize(doc.text) for doc in self._code_docs]
        self._code_bm25 = BM25Okapi(tokenized)

        # Build vector index
        model = self._ensure_model()
        texts = [doc.text for doc in self._code_docs]
        self._code_vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Build reverse indexes for impact analysis
        self._build_reverse_indexes()

        # Persist to disk
        self._save_index()

        return (
            f"Indexed {len(python_files)} Python files, extracted {len(self._code_docs)} "
            f"code symbols ({reused_files} unchanged files reused; BM25 + vectors + reverse indexes)."
        )

    def _build_reverse_indexes(self) -> None:
        """Build reverse indexes: called_by and subclasses for impact analysis."""
        self._called_by = {}
        self._subclasses = {}

        for idx, doc in enumerate(self._code_docs):
            # Build called_by: for each function this doc calls, record that this doc calls it
            for called_name in doc.calls:
                if called_name not in self._called_by:
                    self._called_by[called_name] = set()
                self._called_by[called_name].add(idx)

            # Build subclasses: for each parent class, record that this doc inherits from it
            for parent_name in doc.inherits:
                if parent_name not in self._subclasses:
                    self._subclasses[parent_name] = set()
                self._subclasses[parent_name].add(idx)

    def _save_index(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "file_hashes": self._file_hashes,
            "docs": [
                {
                    "text": doc.text,
                    "path": str(doc.path.relative_to(self.workspace.root)),
                    "kind": doc.kind,
                    "name": doc.name,
                    "line": doc.line,
                    "calls": doc.calls,
                    "inherits": doc.inherits,
                    "imports": doc.imports,
                }
                for doc in self._code_docs
            ],
        }
        self._index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if self._code_vectors is not None:
            np.save(self._vectors_path, self._code_vectors)

    def _load_index(self) -> bool:
        """Load persisted index if it exists and matches the current model."""
        if not self._index_path.exists() or not self._vectors_path.exists():
            return False
        try:
            payload = json.loads(self._index_path.read_text(encoding="utf-8"))
            if payload.get("model_name") != self.model_name:
                return False
            self._code_docs = [
                CodeDocument(
                    text=d["text"],
                    path=self.workspace.root / d["path"],
                    kind=d["kind"],
                    name=d["name"],
                    line=d["line"],
                    calls=d.get("calls", []),
                    inherits=d.get("inherits", []),
                    imports=d.get("imports", []),
                )
                for d in payload.get("docs", [])
            ]
            self._file_hashes = {
                str(path): str(digest)
                for path, digest in payload.get("file_hashes", {}).items()
            }
            if not self._code_docs:
                return False
            tokenized = [_tokenize(doc.text) for doc in self._code_docs]
            self._code_bm25 = BM25Okapi(tokenized)
            self._code_vectors = np.load(self._vectors_path)
            # Rebuild reverse indexes after loading
            self._build_reverse_indexes()
            return True
        except (json.JSONDecodeError, OSError, KeyError):
            return False

    # -- retrieval -------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalContext:
        """Compatibility method returning code and task-history results."""
        return RetrievalContext(
            items=self.retrieve_code(query) + self.retrieve_task_history(query)
        )

    def retrieve_code(self, query: str) -> list[MemoryItem]:
        """Retrieve code symbols using BM25, embeddings, and graph expansion."""
        # Load index if not in memory yet
        if not self._code_docs:
            self._load_index()

        if self._code_docs and self._code_bm25 and self._code_vectors is not None:
            return self._retrieve_code(query)
        return []

    def retrieve_task_history(self, query: str) -> list[MemoryItem]:
        """Retrieve only persisted task memories."""
        return self.task_memory.retrieve(query)

    def _retrieve_code(self, query: str) -> list[MemoryItem]:
        """Hybrid code retrieval: BM25 + cosine similarity, graph expansion, then rerank top-K."""
        if not self._code_docs or not self._code_bm25 or self._code_vectors is None:
            return []

        # BM25 scores
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        bm25_scores = self._code_bm25.get_scores(query_tokens)
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        # Vector similarity (query encoding - reused for reranking)
        model = self._ensure_model()
        query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        cosine_scores = np.dot(self._code_vectors, query_vec) / (
            np.linalg.norm(self._code_vectors, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        cosine_norm = (cosine_scores + 1) / 2  # map [-1, 1] to [0, 1]

        # Combine: 50% BM25 + 50% vector
        combined = 0.5 * bm25_norm + 0.5 * cosine_norm

        # Top-K initial matches
        top_indices = np.argsort(combined)[::-1][:_MAX_RETRIEVED_CODE]
        initial_matches = set()
        for idx in top_indices:
            if combined[idx] < 0.1:  # skip very low scores
                continue
            initial_matches.add(idx)

        # Graph expansion: multi-hop traversal to find related symbols (returns hop distances)
        expanded_with_hops = self._expand_graph(initial_matches)

        # Rerank: score all expanded symbols by vector similarity, decay by hop distance
        scored_items: list[tuple[float, int, str]] = []
        for idx, hop_distance in expanded_with_hops.items():
            if idx in initial_matches:
                # Keep original hybrid score for initial matches (hop=0)
                score = float(combined[idx])
                reason = f"BM25={bm25_norm[idx]:.2f} vector={cosine_norm[idx]:.2f}"
            else:
                # Rerank expanded symbols: vector similarity with hop-based decay
                # 1-hop: ×0.7, 2-hop: ×0.5, 3-hop: ×0.3
                decay_factor = max(0.1, 0.9 - hop_distance * 0.2)
                score = float(cosine_norm[idx]) * decay_factor
                reason = f"graph-expanded hop={hop_distance} vector={cosine_norm[idx]:.2f}"
            scored_items.append((score, idx, reason))

        # Sort by score and take top-K (limit total results)
        scored_items.sort(key=lambda x: x[0], reverse=True)
        top_k = scored_items[:_MAX_RETRIEVED_CODE * 2]  # 2x initial limit after expansion

        # Build result items
        items: list[MemoryItem] = []
        for score, idx, reason in top_k:
            doc = self._code_docs[idx]
            rel_path = doc.path.relative_to(self.workspace.root)
            items.append(
                MemoryItem(
                    kind=f"code_{doc.kind}",
                    text=f"{doc.name} at {rel_path}:{doc.line}\n{doc.text}",
                    path=doc.path,
                    score=score,
                    reason=reason,
                )
            )
        return items

    def _expand_graph(self, initial_indices: set[int]) -> dict[int, int]:
        """Multi-hop graph expansion: traverse relationships to configured depth.

        Expands both forward relationships (calls, inherits, imports) and
        reverse relationships (called_by, subclasses) for impact analysis.

        Returns a dict mapping each reached index to its hop distance from the
        nearest initial match (0 for initial matches themselves), so callers can
        apply distance-based score decay.
        """
        # hop_distance[idx] = shortest number of hops from any initial match
        hop_distance: dict[int, int] = {idx: 0 for idx in initial_indices}

        if self.graph_depth < 1:
            return hop_distance

        name_to_idx = {doc.name: i for i, doc in enumerate(self._code_docs)}

        # Multi-hop traversal: each iteration adds one hop
        frontier = set(initial_indices)
        for hop in range(1, self.graph_depth + 1):
            next_frontier = set()

            for idx in frontier:
                doc = self._code_docs[idx]

                # Forward relationships: symbols this one depends on
                for called_name in doc.calls:
                    if called_name in name_to_idx:
                        target_idx = name_to_idx[called_name]
                        if target_idx not in hop_distance:
                            next_frontier.add(target_idx)
                            hop_distance[target_idx] = hop

                for parent_name in doc.inherits:
                    if parent_name in name_to_idx:
                        target_idx = name_to_idx[parent_name]
                        if target_idx not in hop_distance:
                            next_frontier.add(target_idx)
                            hop_distance[target_idx] = hop

                for import_name in doc.imports:
                    if import_name in name_to_idx:
                        target_idx = name_to_idx[import_name]
                        if target_idx not in hop_distance:
                            next_frontier.add(target_idx)
                            hop_distance[target_idx] = hop

                # Reverse relationships: symbols that depend on this one (impact analysis)
                if doc.name in self._called_by:
                    for caller_idx in self._called_by[doc.name]:
                        if caller_idx not in hop_distance:
                            next_frontier.add(caller_idx)
                            hop_distance[caller_idx] = hop

                if doc.name in self._subclasses:
                    for subclass_idx in self._subclasses[doc.name]:
                        if subclass_idx not in hop_distance:
                            next_frontier.add(subclass_idx)
                            hop_distance[subclass_idx] = hop

            # Move to next hop
            frontier = next_frontier
            if not frontier:
                break  # No more symbols to expand

        return hop_distance

    def _retrieve_task_history(self, query: str) -> list[MemoryItem]:
        return self.task_memory.retrieve(query)

    # -- task history persistence ----------------------------------------------

    def record_tool_result(self, tool_name: str, ok: bool, args: dict, content: str) -> None:
        self.task_memory.record_tool_result(tool_name, ok, args, content)

    def remember_task(self, task: str, summary: str) -> None:
        self.task_memory.remember_task(task, summary)
