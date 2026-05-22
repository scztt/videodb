"""Constants, settings-string formatting, and file discovery helpers."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

# File extensions we will analyze. Lowercased.
VIDEO_EXTS: frozenset[str] = frozenset({
    ".mov", ".mp4", ".m4v", ".mkv", ".webm",
    ".avi", ".mpg", ".mpeg", ".mts", ".m2ts",
})

# Sidecar default mapping — can be overridden in settings.
DEFAULT_SIDECAR_MAPPING = "{fullpath}.videodb.json"

# Default Qwen model. Override via settings.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# Per-pass parser version. Bumped together with the parsing code in
# `passes/pass_def.py`. A bump invalidates every cached pass result
# because `version=N` is part of the settings string. Bump history:
#   1 — initial.
#   2 — switched fps to nframes for duration-independent sampling.
PARSER_VERSION: int = 2


def format_settings(items: dict[str, object]) -> str:
    """Format a pass's settings dict as `key=value&key=value`, keys sorted.

    Not URL-encoded — this is a human-readable cache key. Values are
    coerced via str(); separators must not appear in values (callers
    should pass primitive types only).
    """
    return "&".join(f"{k}={items[k]}" for k in sorted(items))


def parse_settings(raw: str) -> dict[str, str]:
    """Inverse of `format_settings`. Tolerant — bad pairs are skipped."""
    out: dict[str, str] = {}
    if not raw:
        return out
    for chunk in raw.split("&"):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        out[k] = v
    return out


def file_mtime(path: Path) -> float:
    return os.path.getmtime(path)


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


SortMode = Literal["path", "size", "size_desc", "mtime", "mtime_desc", "random"]
SORT_MODES: tuple[SortMode, ...] = (
    "path", "size", "size_desc", "mtime", "mtime_desc", "random",
)


def find_videos(
    root: Path, *, sort: SortMode = "path",
) -> list[Path]:
    """Walk *root* recursively and return video files in *sort* order.

    Sort modes:
      - `path`        : alphabetical by absolute path (default; stable)
      - `size`        : smallest first (great for "warm up on quick wins")
      - `size_desc`   : largest first
      - `mtime`       : oldest first
      - `mtime_desc`  : newest first
      - `random`      : shuffled (uses the system RNG)

    Files we can't stat (e.g. broken symlinks) are dropped with a log
    warning rather than crashing the walk.
    """
    paths: list[Path] = []
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        for fn in files:
            if fn.startswith("."):
                continue
            if Path(fn).suffix.lower() in VIDEO_EXTS:
                paths.append(Path(dirpath) / fn)

    if sort == "path":
        paths.sort()
        return paths
    if sort == "random":
        random.shuffle(paths)
        return paths

    # For size / mtime sorts we need to stat every file. Cache the
    # stat result in a tuple so we don't restat during comparisons.
    keyed: list[tuple[float, Path]] = []
    for p in paths:
        try:
            st = p.stat()
        except OSError as e:
            log.warning("Skipping unstattable file %s: %s", p, e)
            continue
        if sort == "size":
            keyed.append((st.st_size, p))
        elif sort == "size_desc":
            keyed.append((-st.st_size, p))
        elif sort == "mtime":
            keyed.append((st.st_mtime, p))
        elif sort == "mtime_desc":
            keyed.append((-st.st_mtime, p))
        else:
            # Caught upstream by the Literal type, but defend anyway.
            raise ValueError(f"Unknown sort mode: {sort}")
    # Tie-break on path so the order is fully deterministic.
    keyed.sort(key=lambda t: (t[0], t[1]))
    return [p for _, p in keyed]
