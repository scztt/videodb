"""Query-side protocol shared by every backend.

Today only `SidecarBackend` exists. Tomorrow a `ServerBackend` will
sit alongside it talking to an indexed remote DB. The CLI uses this
protocol and nothing else — no direct sidecar access from query
code paths.

The API is intentionally streaming/iterator-shaped so a remote
backend can paginate without making the local one awkward.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class FileAnalyses:
    """All passes' results for a single file, as the backend returns them.

    `analyses` maps **prompt text → entry dict** (each entry has
    `settings`, `output`, and optionally `parse_error`).
    `file_mtime` is the recorded video mtime at last analysis (None
    if the backend can't supply it).
    """
    file: Path
    file_mtime: float | None
    analyses: dict[str, dict[str, Any]]


@runtime_checkable
class Backend(Protocol):
    """Read-only query interface. Implementations may be local or remote."""

    def get_analyses(
        self,
        paths: list[Path],
        *,
        prompts: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Iterator[FileAnalyses]:
        """Yield analyses for every file resolved from *paths*.

        - A path that's a video file is included directly.
        - A path that's a directory is walked recursively for videos.
        - *prompts*, if given, restricts results to entries whose key
          (the prompt text) is in the list. Other entries are dropped
          from the returned dict.
        - *exclude* is a list of glob patterns (`fnmatch`-style)
          tested against each video's POSIX path; matches are skipped.
        - Files with no sidecar are still yielded, with an empty
          `analyses` dict — callers decide how to render them.
        """
        ...
