"""Analyze loop — runs every needed pass for every video file.

Strict serial execution: one file at a time, one pass at a time. The
Qwen model is process-global and ~16GB — parallelism doesn't pay here.
The shape (iterator yielding events) is deliberately queue-friendly
so we can swap in a worker pool later without touching the CLI.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .config import SortMode, find_videos
from .model import VideoTooShortError
from .passes import Pass, build_passes
from .settings import resolve_settings
from .sidecar import make_result, should_analyze, update_sidecar

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyzeEvent:
    """One unit of progress for the caller.

    *elapsed* is wall-clock seconds the pass took. For skipped / error
    events it's the time spent in the should_analyze check (≈ 0).
    *output* carries the pass output for "analyzed" events so the
    caller can render it live (None for skipped/error).
    """
    video: Path
    prompt: str
    status: str          # "analyzed" | "skipped" | "error"
    elapsed: float = 0.0
    detail: str = ""     # error message, when status == "error"
    output: object = None


@dataclass
class AnalyzeStats:
    total_files: int = 0
    total_passes: int = 0
    analyzed: int = 0
    skipped: int = 0
    errors: int = 0
    analyze_seconds: float = 0.0
    # Keyed by prompt — same key shape the rest of the system uses.
    pass_seconds: dict[str, float] = field(
        default_factory=lambda: dict[str, float]()
    )
    pass_counts: dict[str, int] = field(
        default_factory=lambda: dict[str, int]()
    )


@dataclass(frozen=True)
class PlannedWork:
    """Snapshot of what `analyze` would do right now.

    Lets the CLI show an accurate progress total before any model work
    begins.
    """
    videos: list[Path]
    todo: list[tuple[Path, str]]    # (video, prompt) pairs still to run
    skip_count: int                  # already-current passes
    total_passes: int                # todo + skips, across all files


def _resolve_sort(root: Path, override: SortMode | None) -> SortMode:
    """Pick a sort mode: explicit override > root's resolved settings."""
    if override is not None:
        return override
    s = resolve_settings(root)
    return s.get("analyze", {}).get("sort", "path")


def plan(
    root: Path,
    *,
    prompt_filter: list[str] | None = None,
    sort: SortMode | None = None,
) -> PlannedWork:
    """Pre-walk the tree to compute the real work to do.

    Cheap — loads sidecars (small JSON) and compares settings strings.
    No model touched, no video decoded.

    *prompt_filter*, when given, restricts to passes whose prompt is
    *exactly* in the list.
    *sort*, when given, overrides `analyze.sort` from settings.
    """
    sort_mode = _resolve_sort(root, sort)
    videos = (
        find_videos(root, sort=sort_mode) if root.is_dir() else [root]
    )
    todo: list[tuple[Path, str]] = []
    skip = 0
    total = 0
    for video in videos:
        settings = resolve_settings(video)
        mapping = settings["analyze"]["sidecar_mapping"]
        passes: list[Pass] = build_passes(settings)
        if prompt_filter is not None:
            wanted = set(prompt_filter)
            passes = [p for p in passes if p.prompt in wanted]
        for p in passes:
            total += 1
            try:
                stale = should_analyze(
                    video, p.prompt, p.settings, mapping=mapping,
                )
                if stale:
                    todo.append((video, p.prompt))
                else:
                    skip += 1
            except OSError:
                todo.append((video, p.prompt))
    return PlannedWork(
        videos=videos, todo=todo, skip_count=skip, total_passes=total,
    )


def iter_analyze(
    root: Path,
    *,
    prompt_filter: list[str] | None = None,
    sort: SortMode | None = None,
) -> Iterator[AnalyzeEvent]:
    """Yield an `AnalyzeEvent` for every (video, pass) pair under *root*.

    Settings are re-resolved per-video so that folder-level overrides
    (`.videodb_settings.toml`) actually apply. *sort* (or
    `analyze.sort` in settings) controls the visit order.
    """
    sort_mode = _resolve_sort(root, sort)
    videos = (
        find_videos(root, sort=sort_mode) if root.is_dir() else [root]
    )
    log.info(
        "Found %d video file(s) under %s (sort=%s)",
        len(videos), root, sort_mode,
    )

    for video in videos:
        settings = resolve_settings(video)
        mapping = settings["analyze"]["sidecar_mapping"]
        passes: list[Pass] = build_passes(settings)
        if prompt_filter is not None:
            wanted = set(prompt_filter)
            passes = [p for p in passes if p.prompt in wanted]

        for p in passes:
            try:
                fresh = not should_analyze(
                    video, p.prompt, p.settings, mapping=mapping,
                )
                if fresh:
                    yield AnalyzeEvent(
                        video=video, prompt=p.prompt, status="skipped",
                    )
                    continue
            except OSError as e:
                yield AnalyzeEvent(
                    video=video, prompt=p.prompt,
                    status="error", detail=f"sidecar check: {e}",
                )
                continue

            t0 = time.monotonic()
            try:
                output, parse_error = p.run(video)
            except VideoTooShortError as e:
                # Permanent — won't change between runs. Write a
                # cacheable "skipped" sentinel so future runs short-
                # circuit without reprobing.
                elapsed = time.monotonic() - t0
                result = make_result(
                    prompt=p.prompt,
                    settings=p.settings,
                    output=None,
                    skipped=f"too_short: {e}",
                )
                update_sidecar(video, result, mapping=mapping)
                yield AnalyzeEvent(
                    video=video, prompt=p.prompt, status="skipped",
                    elapsed=elapsed, detail=str(e),
                )
                continue
            except Exception as e:  # noqa: BLE001 — model errors vary
                log.exception("Pass failed on %s (%r)", video, p.prompt[:60])
                yield AnalyzeEvent(
                    video=video, prompt=p.prompt, status="error",
                    elapsed=time.monotonic() - t0, detail=str(e),
                )
                continue

            elapsed = time.monotonic() - t0
            result = make_result(
                prompt=p.prompt,
                settings=p.settings,
                output=output,
                parse_error=parse_error,
            )
            update_sidecar(video, result, mapping=mapping)
            yield AnalyzeEvent(
                video=video, prompt=p.prompt, status="analyzed",
                elapsed=elapsed, output=output,
            )


EventCallback = Callable[[AnalyzeEvent, AnalyzeStats], None]


def analyze(
    root: Path,
    *,
    prompt_filter: list[str] | None = None,
    sort: SortMode | None = None,
    on_event: EventCallback | None = None,
) -> AnalyzeStats:
    """Run `iter_analyze` and accumulate stats. Returns the stats."""
    stats = AnalyzeStats()
    seen_files: set[Path] = set()

    for ev in iter_analyze(root, prompt_filter=prompt_filter, sort=sort):
        if ev.video not in seen_files:
            stats.total_files += 1
            seen_files.add(ev.video)
        stats.total_passes += 1
        if ev.status == "analyzed":
            stats.analyzed += 1
            stats.analyze_seconds += ev.elapsed
            stats.pass_seconds[ev.prompt] = (
                stats.pass_seconds.get(ev.prompt, 0.0) + ev.elapsed
            )
            stats.pass_counts[ev.prompt] = (
                stats.pass_counts.get(ev.prompt, 0) + 1
            )
        elif ev.status == "skipped":
            stats.skipped += 1
        else:
            stats.errors += 1
        if on_event is not None:
            on_event(ev, stats)

    return stats
