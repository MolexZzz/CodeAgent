from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from memcodeagent.memory.code_indexer import extract_symbols
from memcodeagent.memory.schema import MemoryItem, TaskRecord
from memcodeagent.workspace import Workspace

_STOPWORDS = {
    "the", "a", "an", "to", "of", "in", "on", "for", "and", "or", "is", "are",
    "add", "please", "task", "with", "that", "this", "it", "be", "as",
}
_MAX_RECORDS = 200
_MAX_RETRIEVED_TASK = 3
_MAX_RETRIEVED_CODE = 5


def _tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 (returns list to preserve frequency)."""
    words = re.findall(r"[A-Za-z0-9_./\\-]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


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


class HybridRetriever:
    """Hybrid retrieval combining BM25 keyword search, vector semantic search,
    and task history. Builds code index from workspace Python files on demand.
    """

    def __init__(self, workspace: Workspace, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.workspace = workspace
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._code_docs: list[CodeDocument] = []
        self._code_bm25: BM25Okapi | None = None
        self._code_vectors: np.ndarray | None = None
        self._index_path = self.workspace.root / ".memcode" / "code_index.json"
        self._vectors_path = self.workspace.root / ".memcode" / "code_vectors.npy"
        self._memory_path = self.workspace.root / ".memcode" / "memory.json"
        self._changed_files: list[str] = []
        self._failed_commands: list[str] = []

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # -- indexing --------------------------------------------------------------

    def index_workspace(self) -> str:
        """Scan workspace Python files, extract code symbols, build BM25 + vector index."""
        python_files = [
            p for p in self.workspace.root.glob("**/*.py")
            if p.is_file() and not self.workspace.should_ignore(p)
        ]

        self._code_docs = []
        for py_file in python_files:
            symbols = extract_symbols(py_file)
            for sym in symbols:
                self._code_docs.append(
                    CodeDocument(
                        text=sym.to_text(),
                        path=py_file,
                        kind=sym.kind,
                        name=sym.name,
                        line=sym.line,
                    )
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

        # Persist to disk
        self._save_index()

        return f"Indexed {len(python_files)} Python files, extracted {len(self._code_docs)} code symbols (BM25 + vectors)."

    def _save_index(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_name": self.model_name,
            "docs": [
                {
                    "text": doc.text,
                    "path": str(doc.path.relative_to(self.workspace.root)),
                    "kind": doc.kind,
                    "name": doc.name,
                    "line": doc.line,
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
                )
                for d in payload.get("docs", [])
            ]
            if not self._code_docs:
                return False
            tokenized = [_tokenize(doc.text) for doc in self._code_docs]
            self._code_bm25 = BM25Okapi(tokenized)
            self._code_vectors = np.load(self._vectors_path)
            return True
        except (json.JSONDecodeError, OSError, KeyError):
            return False

    # -- retrieval -------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalContext:
        """Retrieve relevant code symbols and task history via hybrid search."""
        # Load index if not in memory yet
        if not self._code_docs:
            self._load_index()

        items: list[MemoryItem] = []

        # 1. Code retrieval (BM25 + vector)
        if self._code_docs and self._code_bm25 and self._code_vectors is not None:
            code_items = self._retrieve_code(query)
            items.extend(code_items)

        # 2. Task history retrieval (keyword overlap, same as SimpleRetriever)
        task_items = self._retrieve_task_history(query)
        items.extend(task_items)

        return RetrievalContext(items=items)

    def _retrieve_code(self, query: str) -> list[MemoryItem]:
        """Hybrid code retrieval: BM25 + cosine similarity, combine scores."""
        if not self._code_docs or not self._code_bm25 or self._code_vectors is None:
            return []

        # BM25 scores
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        bm25_scores = self._code_bm25.get_scores(query_tokens)
        bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_norm = bm25_scores / bm25_max

        # Vector similarity
        model = self._ensure_model()
        query_vec = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0]
        cosine_scores = np.dot(self._code_vectors, query_vec) / (
            np.linalg.norm(self._code_vectors, axis=1) * np.linalg.norm(query_vec) + 1e-8
        )
        cosine_norm = (cosine_scores + 1) / 2  # map [-1, 1] to [0, 1]

        # Combine: 50% BM25 + 50% vector
        combined = 0.5 * bm25_norm + 0.5 * cosine_norm

        # Top-K
        top_indices = np.argsort(combined)[::-1][:_MAX_RETRIEVED_CODE]
        items: list[MemoryItem] = []
        for idx in top_indices:
            if combined[idx] < 0.1:  # skip very low scores
                continue
            doc = self._code_docs[idx]
            rel_path = doc.path.relative_to(self.workspace.root)
            items.append(
                MemoryItem(
                    kind=f"code_{doc.kind}",
                    text=f"{doc.name} at {rel_path}:{doc.line}\n{doc.text}",
                    path=doc.path,
                    score=float(combined[idx]),
                    reason=f"BM25={bm25_norm[idx]:.2f} vector={cosine_norm[idx]:.2f}",
                )
            )
        return items

    def _retrieve_task_history(self, query: str) -> list[MemoryItem]:
        """Keyword-based task history retrieval (same logic as SimpleRetriever)."""
        records = self._load_task_records()
        if not records:
            return []

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, TaskRecord]] = []
        for record in records:
            record_tokens = set(_tokenize(record.task) + _tokenize(record.summary))
            for path in record.changed_files:
                record_tokens.update(_tokenize(path))
            for cmd in record.failed_commands:
                record_tokens.update(_tokenize(cmd))
            overlap = query_tokens & record_tokens
            if not overlap:
                continue
            score = len(overlap) / max(len(query_tokens), 1)
            scored.append((score, record))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        items: list[MemoryItem] = []
        for score, record in scored[:_MAX_RETRIEVED_TASK]:
            text = f"Previous task: {record.task}\nOutcome: {record.summary}"
            if record.changed_files:
                text += f"\nChanged files: {', '.join(record.changed_files)}"
            if record.failed_commands:
                text += f"\nFailed commands: {', '.join(record.failed_commands)}"
            items.append(
                MemoryItem(
                    kind="task_history",
                    text=text,
                    score=score,
                    reason="keyword overlap",
                )
            )
        return items

    # -- task history persistence ----------------------------------------------

    def record_tool_result(self, tool_name: str, ok: bool, args: dict, content: str) -> None:
        """Track file changes and failed commands for task memory."""
        if ok and tool_name in {"write_file", "apply_patch"}:
            path = args.get("path")
            if path and path not in self._changed_files:
                self._changed_files.append(path)
        if tool_name == "run_command" and not ok:
            command = args.get("command", "")
            if command:
                self._failed_commands.append(command)
        if tool_name == "run_command" and ok and "exit_code=0" not in content:
            command = args.get("command", "")
            if command:
                self._failed_commands.append(command)

    def remember_task(self, task: str, summary: str) -> None:
        """Persist completed task to memory.json."""
        records = self._load_task_records()
        records.append(
            TaskRecord(
                task=task,
                summary=summary,
                changed_files=list(self._changed_files),
                failed_commands=list(self._failed_commands),
                timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        )
        self._save_task_records(records)
        self._changed_files = []
        self._failed_commands = []

    def _load_task_records(self) -> list[TaskRecord]:
        if not self._memory_path.exists():
            return []
        try:
            raw = json.loads(self._memory_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        return [TaskRecord.from_dict(item) for item in raw.get("records", [])]

    def _save_task_records(self, records: list[TaskRecord]) -> None:
        self._memory_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [record.to_dict() for record in records[-_MAX_RECORDS:]]}
        self._memory_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
