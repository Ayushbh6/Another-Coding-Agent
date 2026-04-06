from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from aca.task_workspace import CANONICAL_REPO_DOCS


@dataclass(slots=True)
class RepoSummaryCache:
    generated_at: datetime
    payload: dict[str, Any]


class RepoSummaryService:
    def __init__(self, workspace_root: Path) -> None:
        self._workspace_root = workspace_root.resolve()
        self._cache: RepoSummaryCache | None = None

    def get(self, *, force_refresh: bool = False) -> dict[str, Any]:
        now = datetime.utcnow()
        if (
            not force_refresh
            and self._cache is not None
            and now - self._cache.generated_at < timedelta(minutes=15)
        ):
            cached = dict(self._cache.payload)
            cached["cache_status"] = "hit"
            return cached

        top_level_entries = sorted(
            entry.name for entry in self._workspace_root.iterdir() if entry.name not in {".git", ".venv", "__pycache__"}
        )
        top_level_dirs = [name for name in top_level_entries if (self._workspace_root / name).is_dir()][:12]
        top_level_files = [name for name in top_level_entries if (self._workspace_root / name).is_file()][:12]

        extension_counts: dict[str, int] = {}
        file_count = 0
        approx_chars = 0
        ignored = {".git", ".venv", "venv", "__pycache__", "node_modules", "dist", "build", ".pytest_cache", ".mypy_cache"}
        for path in self._workspace_root.rglob("*"):
            if not path.is_file():
                continue
            rel_parts = path.relative_to(self._workspace_root).parts
            if any(part in ignored for part in rel_parts):
                continue
            file_count += 1
            suffix = path.suffix.lower() or "<none>"
            extension_counts[suffix] = extension_counts.get(suffix, 0) + 1
            try:
                approx_chars += min(path.stat().st_size, 20000)
            except OSError:
                continue

        likely_stack: list[str] = []
        if (self._workspace_root / "pyproject.toml").exists() or (self._workspace_root / "requirements.txt").exists():
            likely_stack.append("Python")
        if (self._workspace_root / "package.json").exists():
            likely_stack.append("Node.js")
        if any(name.endswith(".db") for name in top_level_files):
            likely_stack.append("SQLite")

        important_entrypoints = [
            name
            for name in ("main.py", "aca/cli/app.py", "aca/orchestration/orchestrator.py", "aca/runtime.py")
            if (self._workspace_root / name).exists()
        ]
        top_extensions = [suffix for suffix, _ in sorted(extension_counts.items(), key=lambda item: (-item[1], item[0]))[:5]]
        payload = {
            "workspace_root": str(self._workspace_root),
            "top_level_count": len(top_level_entries),
            "top_level_dirs": top_level_dirs,
            "top_level_files": top_level_files,
            "languages": top_extensions,
            "likely_stack": likely_stack,
            "canonical_docs_present": sorted(doc for doc in CANONICAL_REPO_DOCS if (self._workspace_root / doc).exists()),
            "important_entrypoints": important_entrypoints,
            "file_count_estimate": file_count,
            "approx_token_estimate": max(1, approx_chars // 4),
            "where_to_start": important_entrypoints[:2] or top_level_dirs[:2] or top_level_files[:2],
            "cache_status": "miss",
        }
        self._cache = RepoSummaryCache(generated_at=now, payload=payload)
        return payload

