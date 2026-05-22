"""Local backend that reads `.videodb.json` sidecars off disk."""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import Iterator
from pathlib import Path

from ..config import find_videos, is_video
from ..settings import resolve_sidecar_mapping
from ..sidecar import find_sidecar, read_sidecar
from .base import FileAnalyses

log = logging.getLogger(__name__)


class SidecarBackend:
    """Reads sidecars directly from disk. Stateless."""

    def get_analyses(
        self,
        paths: list[Path],
        *,
        prompts: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> Iterator[FileAnalyses]:
        videos = _expand_paths(paths)
        if exclude:
            videos = _apply_exclude(videos, exclude)

        wanted = set(prompts) if prompts else None
        for video in videos:
            mapping = resolve_sidecar_mapping(video)
            sc = find_sidecar(video, mapping)
            if sc is None:
                yield FileAnalyses(file=video, file_mtime=None, analyses={})
                continue
            try:
                data = read_sidecar(sc)
            except (OSError, ValueError) as e:
                log.warning("Unreadable sidecar %s: %s", sc, e)
                yield FileAnalyses(file=video, file_mtime=None, analyses={})
                continue

            analyses = data.get("analyses") or {}
            if wanted is not None:
                analyses = {k: v for k, v in analyses.items() if k in wanted}
            yield FileAnalyses(
                file=video,
                file_mtime=data.get("file_mtime"),
                analyses=analyses,
            )


# --- helpers ------------------------------------------------------------


def _expand_paths(paths: list[Path]) -> list[Path]:
    """Resolve a mix of file and directory paths to a flat video list."""
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        p = raw.resolve()
        if not p.exists():
            log.warning("Path does not exist: %s", p)
            continue
        if p.is_dir():
            for v in find_videos(p):
                rv = v.resolve()
                if rv not in seen:
                    seen.add(rv)
                    out.append(rv)
        elif is_video(p):
            if p not in seen:
                seen.add(p)
                out.append(p)
        else:
            log.warning("Skipping non-video file: %s", p)
    return out


def _apply_exclude(videos: list[Path], patterns: list[str]) -> list[Path]:
    """Filter videos whose POSIX path matches any of *patterns* (fnmatch)."""
    if not patterns:
        return videos
    out: list[Path] = []
    for v in videos:
        s = v.as_posix()
        if any(fnmatch.fnmatch(s, pat) for pat in patterns):
            continue
        out.append(v)
    return out
