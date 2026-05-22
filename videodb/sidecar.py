"""Sidecar file I/O — JSON format.

Sidecars are plain JSON so other apps can read them without a Python
runtime. They live next to each video file by default; the mapping is
configurable via settings.

Sidecar shape (envelope version = SIDECAR_VERSION):

    {
        "version": 2,
        "file": "/abs/path/to/clip.mov",
        "file_mtime": 1707000000.123,
        "analyses": {
            "<prompt text>": {
                "settings": "fps=1.0&max_tokens=256&model=Qwen/...&version=1",
                "output": <string | parsed JSON value>,
                "parse_error": <optional bool>
            },
            ...
        }
    }

The prompt text is the analysis key. A cache hit requires both the
file's mtime to match AND the pass's `settings` string to match the
stored one — see `should_analyze`.

Bumping SIDECAR_VERSION (here in code) discards every existing
sidecar on next read. Use sparingly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DEFAULT_SIDECAR_MAPPING, file_mtime
from .settings import expand_sidecar_path

log = logging.getLogger(__name__)

# Bump to force re-analysis of every pass on every file.
SIDECAR_VERSION: int = 2


# --- types ---------------------------------------------------------------


@dataclass(frozen=True)
class PassResult:
    """Output of a single pass for a single file.

    Stored verbatim under `analyses[prompt]` in the sidecar.

    *skipped* records a permanent reason the pass cannot succeed for
    this file (e.g. video too short to sample frames). Skipped
    entries still count as fresh — same hash → cache hit on rerun.
    """
    prompt: str
    settings: str
    output: Any
    parse_error: bool = False
    skipped: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "settings": self.settings,
            "output": self.output,
        }
        if self.parse_error:
            d["parse_error"] = True
        if self.skipped is not None:
            d["skipped"] = self.skipped
        return d


# --- path helpers --------------------------------------------------------


def sidecar_path(video_path: Path, mapping: str | None = None) -> Path:
    """Sidecar path for *video_path* (where we would WRITE it)."""
    return expand_sidecar_path(video_path, mapping or DEFAULT_SIDECAR_MAPPING)


def find_sidecar(video_path: Path, mapping: str | None = None) -> Path | None:
    p = sidecar_path(video_path, mapping)
    return p if p.exists() else None


# --- read / write --------------------------------------------------------


def read_sidecar(path: Path) -> dict[str, Any]:
    """Read a sidecar. Raises on bad version or corruption."""
    with open(path) as f:
        data = json.load(f)
    if data.get("version") != SIDECAR_VERSION:
        raise ValueError(
            f"Sidecar version mismatch: got {data.get('version')}, "
            f"expected {SIDECAR_VERSION} ({path})"
        )
    return data


def _read_sidecar_tolerant(path: Path) -> dict[str, Any] | None:
    """Read a sidecar, returning None on any error (incl. version)."""
    try:
        return read_sidecar(path)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        log.warning("Unreadable sidecar %s: %s", path, e)
        return None


def write_sidecar(path: Path, data: dict[str, Any]) -> None:
    data = dict(data)
    data["version"] = SIDECAR_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        # `default=str` so unexpected types (e.g. Path) serialize
        # rather than crash; indent=2 keeps it readable.
        json.dump(data, f, indent=2, default=str)


def update_sidecar(
    video_path: Path,
    result: PassResult,
    mapping: str | None = None,
) -> Path:
    """Merge *result* into the sidecar for *video_path*. Returns the path."""
    out_path = sidecar_path(video_path, mapping)
    existing = _read_sidecar_tolerant(out_path) if out_path.exists() else None

    if existing is None:
        data: dict[str, Any] = {
            "version": SIDECAR_VERSION,
            "file": str(video_path.resolve()),
            "file_mtime": file_mtime(video_path),
            "analyses": {},
        }
    else:
        data = existing

    data["file"] = str(video_path.resolve())
    data["file_mtime"] = file_mtime(video_path)
    analyses = data.setdefault("analyses", {})
    analyses[result.prompt] = result.to_dict()

    write_sidecar(out_path, data)
    return out_path


# --- analysis check ------------------------------------------------------


def should_analyze(
    video_path: Path,
    prompt: str,
    expected_settings: str,
    mapping: str | None = None,
) -> bool:
    """True if the pass identified by *prompt* needs (re)analysis.

    Re-analyze when any is true:
      - no sidecar
      - sidecar unreadable / wrong version
      - prompt missing from `analyses`
      - stored `settings` string differs from *expected_settings*
      - video mtime changed since last analysis
    """
    sc = find_sidecar(video_path, mapping)
    if sc is None:
        return True

    data = _read_sidecar_tolerant(sc)
    if data is None:
        return True

    entry = (data.get("analyses") or {}).get(prompt)
    if entry is None:
        return True
    if entry.get("settings") != expected_settings:
        return True
    if data.get("file_mtime") != file_mtime(video_path):
        return True
    return False


def read_analyses(
    video_path: Path,
    mapping: str | None = None,
) -> dict[str, Any] | None:
    """Return the `analyses` dict from this video's sidecar, or None."""
    sc = find_sidecar(video_path, mapping)
    if sc is None:
        return None
    data = _read_sidecar_tolerant(sc)
    if data is None:
        return None
    return data.get("analyses") or {}


# --- factory -------------------------------------------------------------


def make_result(
    *,
    prompt: str,
    settings: str,
    output: Any,
    parse_error: bool = False,
    skipped: str | None = None,
) -> PassResult:
    return PassResult(
        prompt=prompt,
        settings=settings,
        output=output,
        parse_error=parse_error,
        skipped=skipped,
    )
