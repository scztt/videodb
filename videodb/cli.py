"""videodb CLI — typer + rich.

Commands:
  analyze        run analysis passes over a folder (writes sidecars)
  query          dump analysis results for a file or folder
  passes         list which passes would run for a given path
  settings ...   inspect / edit settings (user or per-folder)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter, deque
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.json import JSON
from rich.logging import RichHandler
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text

from .analyze import (
    AnalyzeEvent,
    AnalyzeStats,
    analyze as run_analyze,
    plan as plan_analyze,
)
from .backend import FileAnalyses, SidecarBackend
from .passes import build_passes
from .settings import (
    get_settings,
    list_settings,
    resolve_settings,
    set_setting,
)

app = typer.Typer(
    name="videodb",
    help="Per-file video analysis with sidecar caching (Qwen2.5-VL).",
    add_completion=False,
    no_args_is_help=True,
)
settings_app = typer.Typer(
    name="settings",
    help="View and edit videodb settings.",
    no_args_is_help=True,
)
app.add_typer(settings_app)

console = Console()
err_console = Console(stderr=True)


class OutputFormat(str, Enum):
    """`query` output formats."""
    table = "table"
    json = "json"
    jsonl = "jsonl"


class SortOrder(str, Enum):
    """Order in which `analyze` visits videos. Matches `config.SortMode`."""
    path = "path"
    size = "size"
    size_desc = "size_desc"
    mtime = "mtime"
    mtime_desc = "mtime_desc"
    random = "random"


def _setup_logging(verbose: bool) -> None:
    """Configure logging.

    Normal runs (no `--verbose`) keep `videodb` at INFO but silence
    chatty third-party loggers that fire on every decode/model call.
    `--verbose` flips everything to DEBUG.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=err_console,
                rich_tracebacks=True,
                show_path=False,
            )
        ],
    )
    if not verbose:
        # Quiet the noise: torchcodec announces TORCHCODEC_NUM_THREADS
        # and prints decode stats per file; qwen_vl_utils logs every
        # backend pick; httpx / urllib3 log model fetches.
        for name in (
            "qwen_vl_utils",
            "qwen_vl_utils.vision_process",
            "torchcodec",
            "httpx",
            "httpcore",
            "urllib3",
            "transformers",
            "accelerate",
            "filelock",
        ):
            logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command("analyze")
def analyze_cmd(
    folder: Annotated[
        Path,
        typer.Argument(help="Folder of video files (recursive)"),
    ],
    prompt: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt", "-p",
            help=(
                "Restrict to passes whose prompt EXACTLY matches one "
                "of these strings. Repeatable."
            ),
        ),
    ] = None,
    sort: Annotated[
        SortOrder | None,
        typer.Option(
            "--sort", "-s",
            help=(
                "Order to visit files in. Overrides `analyze.sort` "
                "from settings. `size` = smallest first (great for "
                "long batches you might ctrl-C early)."
            ),
        ),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet", "-q",
            help="Suppress the live progress bar (still logs to stderr).",
        ),
    ] = False,
    window: Annotated[
        int,
        typer.Option(
            "--window",
            help=(
                "Cloud counts only the last N tag-producing events "
                "(rolling window). Keeps the cloud reactive on long "
                "runs instead of asymptoting to averages. 0 = unlimited."
            ),
        ),
    ] = 100,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run analysis passes, writing/updating sidecars.

    Each video is checked per-pass against its sidecar; already-current
    passes are skipped. Settings cascade per-file, so folder-local
    `.videodb_settings.toml` files take effect.

    Every state change also gets one plain-text line on stderr so
    `tee` / `tail | grep` works (`rich`'s progress bar uses ANSI
    overwrites and is invisible to line-oriented monitors).
    """
    _setup_logging(verbose)
    folder = folder.resolve()
    if not folder.exists():
        err_console.print(f"[red]Path not found: {folder}[/red]")
        raise typer.Exit(1)

    sort_mode = sort.value if sort is not None else None
    planned = plan_analyze(folder, prompt_filter=prompt, sort=sort_mode)
    if not planned.videos:
        err_console.print(f"[yellow]No videos found under {folder}[/yellow]")
        raise typer.Exit(0)

    n_todo = len(planned.todo)
    console.print(
        f"[bold]videodb analyze[/bold] — "
        f"{len(planned.videos)} files, "
        f"{planned.total_passes} pass(es) total, "
        f"[green]{n_todo} to run[/green], "
        f"[dim]{planned.skip_count} up-to-date[/dim]"
    )
    if n_todo == 0:
        console.print("[dim]Nothing to do.[/dim]")
        return

    # Rolling-average ETA: weighted average of per-prompt times,
    # weighted by how many of each prompt remain.
    remaining_by_prompt: dict[str, int] = {}
    for _, prom in planned.todo:
        remaining_by_prompt[prom] = remaining_by_prompt.get(prom, 0) + 1

    def fmt_eta(stats: AnalyzeStats, ev: AnalyzeEvent) -> str:
        if ev.status == "analyzed" and ev.prompt in remaining_by_prompt:
            remaining_by_prompt[ev.prompt] -= 1
            if remaining_by_prompt[ev.prompt] <= 0:
                remaining_by_prompt.pop(ev.prompt)
        if stats.analyzed == 0 or not remaining_by_prompt:
            return "—"
        secs = 0.0
        for prom, n_left in remaining_by_prompt.items():
            n_done = stats.pass_counts.get(prom, 0)
            if n_done == 0:
                avg = stats.analyze_seconds / stats.analyzed
            else:
                avg = stats.pass_seconds[prom] / n_done
            secs += avg * n_left
        return _fmt_duration(secs)

    if quiet:
        stats = _run_analyze_quiet(
            folder, prompt_filter=prompt, sort=sort_mode, fmt_eta=fmt_eta,
        )
    else:
        stats = _run_analyze_visual(
            folder, prompt_filter=prompt, sort=sort_mode,
            n_todo=n_todo, fmt_eta=fmt_eta, window=window,
        )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("[green]analyzed[/green]", str(stats.analyzed))
    table.add_row("[dim]skipped[/dim]", str(stats.skipped))
    if stats.errors:
        table.add_row("[red]errors[/red]", str(stats.errors))
    table.add_row("files", str(stats.total_files))
    if stats.analyzed:
        table.add_row(
            "avg per pass",
            f"{stats.analyze_seconds / stats.analyzed:.1f}s",
        )
        for prom in sorted(stats.pass_counts):
            n = stats.pass_counts[prom]
            t = stats.pass_seconds[prom]
            label = prom if len(prom) <= 60 else prom[:57] + "…"
            table.add_row(f"  {label}", f"{t / n:.1f}s × {n}")
    console.print(table)
    if stats.errors:
        raise typer.Exit(1)


# Compact prompt for log/progress lines — full prompts are sentences.
def _prompt_label(prompt: str, max_len: int = 40) -> str:
    s = prompt.strip().replace("\n", " ")
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _log_event(ev: AnalyzeEvent, eta: str) -> None:
    """One plain-text line to stderr per event — `tee`/`tail` friendly."""
    parts = [
        f"[{ev.status}]",
        ev.video.name,
        f"prompt={_prompt_label(ev.prompt)!r}",
    ]
    if ev.status == "analyzed":
        parts.append(f"elapsed={ev.elapsed:.1f}s")
    if ev.detail:
        parts.append(f"detail={ev.detail[:80]!r}")
    parts.append(f"eta={eta}")
    print(" ".join(parts), file=sys.stderr, flush=True)


def _coerce_tags(output: Any) -> list[str]:
    """Pull a flat list of tag-like strings out of a pass output.

    Tag passes return lists. JSON passes return dicts whose values
    can be lists. Text passes return strings — we don't extract tags
    from those.
    """
    if isinstance(output, list):
        return [str(t).strip() for t in output if isinstance(t, str) and t.strip()]
    if isinstance(output, dict):
        tags: list[str] = []
        for v in output.values():
            if isinstance(v, list):
                for t in v:
                    if isinstance(t, str) and t.strip():
                        tags.append(t.strip())
        return tags
    return []


# Cycling palette for the tag stream — high-contrast, terminal-safe.
_TAG_PALETTE = (
    "cyan", "magenta", "yellow", "green", "blue",
    "bright_cyan", "bright_magenta", "bright_yellow", "bright_green",
)


def _run_analyze_visual(
    folder: Path,
    *,
    prompt_filter: list[str] | None,
    sort: str | None,
    n_todo: int,
    fmt_eta: Any,
    window: int = 100,
) -> AnalyzeStats:
    """Streaming tag-wall view.

    Layout (top → bottom):
      • header panel:  current file, pass, elapsed, ETA, N/M
      • burst row:     tags from the most-recent analyzed event
      • cloud panel:   tag counts as horizontal bars (top 20)

    *window*: rolling window of the last N tag-producing events.
    Keeps the cloud reactive. 0 = unlimited (lifetime counts).
    """
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold]{task.description}[/bold]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("eta [cyan]{task.fields[eta]}[/cyan]"),
        console=console,
        transient=False,
        expand=True,
    )
    task_id = progress.add_task("analyzing", total=n_todo, eta="—")

    # Rolling window of per-event tag lists. window=0 means unbounded.
    tag_events: deque[list[str]] = deque(maxlen=window if window > 0 else None)
    lifetime_total: int = 0   # total mentions ever, for the footer

    latest_file: str = "…"
    latest_label: str = ""
    latest_tags: list[str] = []
    latest_elapsed: float = 0.0
    burst_n: int = 0          # counter for palette cycling per event
    t_start = time.monotonic()

    def render_burst() -> Text:
        if not latest_tags:
            waited = time.monotonic() - t_start
            return Text(
                f"  warming up — loading model / decoding first video "
                f"({waited:.0f}s elapsed)",
                style="dim italic",
            )
        t = Text()
        t.append(f"  {latest_file}", style="bold white")
        t.append(f"  [{latest_label}]  ", style="dim")
        t.append(f"{latest_elapsed:.1f}s\n", style="dim")
        t.append("  ↳ ", style="dim")
        for i, tag in enumerate(latest_tags[:20]):
            color = _TAG_PALETTE[(burst_n + i) % len(_TAG_PALETTE)]
            t.append(tag, style=color)
            if i < min(len(latest_tags), 20) - 1:
                t.append("  ")
        if len(latest_tags) > 20:
            t.append(f"  …+{len(latest_tags) - 20}", style="dim")
        return t

    def render_cloud() -> Text:
        if not tag_events:
            return Text("  (no tags yet)", style="dim")
        counts: Counter[str] = Counter()
        for tags in tag_events:
            counts.update(tags)
        if not counts:
            return Text("  (no tags yet)", style="dim")
        top = counts.most_common(20)
        max_n = top[0][1]
        max_label = max(len(tag) for tag, _ in top)
        bar_width = 32
        t = Text()
        for tag, n in top:
            filled = max(1, round(bar_width * n / max_n))
            color = _TAG_PALETTE[hash(tag) % len(_TAG_PALETTE)]
            t.append(f"  {tag:<{max_label}}  ", style=color)
            t.append("█" * filled, style=color)
            t.append("·" * (bar_width - filled), style="dim")
            t.append(f"  {n}\n", style="bold")
        unique = len(counts)
        window_total = sum(counts.values())
        win_label = (
            f"window of {len(tag_events)}/{window}"
            if window > 0 else f"all {len(tag_events)} events"
        )
        t.append(
            f"\n  {unique} unique  ·  {window_total} in {win_label}"
            f"  ·  {lifetime_total} lifetime",
            style="dim italic",
        )
        return t

    def render() -> Any:
        from rich.console import Group
        return Group(
            progress,
            Panel(render_burst(), title="latest", border_style="dim",
                  padding=(0, 1)),
            Panel(render_cloud(), title="tag cloud", border_style="dim",
                  padding=(0, 1)),
        )

    with Live(render(), console=console, refresh_per_second=8,
              transient=False) as live:

        def on_event(ev: AnalyzeEvent, s: AnalyzeStats) -> None:
            nonlocal latest_file, latest_label, latest_tags
            nonlocal latest_elapsed, burst_n, lifetime_total
            eta = fmt_eta(s, ev)
            if ev.status != "skipped":
                progress.advance(task_id)
            progress.update(task_id, eta=eta)
            if ev.status == "analyzed":
                tags = _coerce_tags(ev.output)
                if tags:
                    latest_file = ev.video.name
                    latest_label = _prompt_label(ev.prompt, 30)
                    latest_tags = tags
                    latest_elapsed = ev.elapsed
                    burst_n += 1
                    tag_events.append(tags)
                    lifetime_total += len(tags)
            live.update(render())
            _log_event(ev, eta)

        return run_analyze(
            folder, prompt_filter=prompt_filter, sort=sort,
            on_event=on_event,
        )


def _run_analyze_quiet(
    folder: Path,
    *,
    prompt_filter: list[str] | None,
    sort: str | None,
    fmt_eta: Any,
) -> AnalyzeStats:
    def on_event(ev: AnalyzeEvent, s: AnalyzeStats) -> None:
        _log_event(ev, fmt_eta(s, ev))

    return run_analyze(
        folder, prompt_filter=prompt_filter, sort=sort,
        on_event=on_event,
    )


def _fmt_duration(seconds: float) -> str:
    """Compact `HhMmSs` formatting; never zero-pad to the seconds-only form."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------


@app.command("query")
def query_cmd(
    paths: Annotated[list[Path], typer.Argument(help="File(s) or folder(s)")],
    prompt: Annotated[
        list[str] | None,
        typer.Option(
            "--prompt", "-p",
            help="Show only entries with EXACTLY this prompt. Repeatable.",
        ),
    ] = None,
    exclude: Annotated[
        list[str] | None,
        typer.Option(
            "--exclude", "-x",
            help="Glob(s) to skip (fnmatch on full path).",
        ),
    ] = None,
    format_: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format"),
    ] = OutputFormat.table,
    full_paths: Annotated[
        bool,
        typer.Option("--full-paths", help="Show absolute paths in table"),
    ] = False,
    show_missing: Annotated[
        bool,
        typer.Option(
            "--show-missing/--hide-missing",
            help="Show files with no sidecar / no matching entries",
        ),
    ] = True,
) -> None:
    """Dump analysis results for the given file(s) and/or folder(s)."""
    backend = SidecarBackend()
    iterator = backend.get_analyses(
        paths, prompts=prompt, exclude=exclude,
    )

    if format_ is OutputFormat.jsonl:
        _emit_jsonl(iterator, show_missing=show_missing)
        return
    if format_ is OutputFormat.json:
        _emit_json(iterator, show_missing=show_missing)
        return
    _emit_table(iterator, show_missing=show_missing, full_paths=full_paths)


def _to_json_dict(fa: FileAnalyses) -> dict[str, Any]:
    return {
        "file": str(fa.file),
        "file_mtime": fa.file_mtime,
        "analyses": fa.analyses,
    }


def _emit_jsonl(
    iterator: Any,
    *,
    show_missing: bool,
) -> None:
    for fa in iterator:
        if not fa.analyses and not show_missing:
            continue
        console.print_json(data=_to_json_dict(fa))


def _emit_json(
    iterator: Any,
    *,
    show_missing: bool,
) -> None:
    items = [
        _to_json_dict(fa)
        for fa in iterator
        if fa.analyses or show_missing
    ]
    console.print_json(data=items)


def _emit_table(
    iterator: Any,
    *,
    show_missing: bool,
    full_paths: bool,
) -> None:
    any_emitted = False
    for fa in iterator:
        if not fa.analyses and not show_missing:
            continue
        any_emitted = True
        _print_file_block(fa, full_paths=full_paths)
    if not any_emitted:
        err_console.print("[yellow]No results.[/yellow]")


def _print_file_block(fa: FileAnalyses, *, full_paths: bool) -> None:
    label = str(fa.file) if full_paths else fa.file.name
    console.rule(f"[bold]{label}[/bold]", align="left")
    if not fa.analyses:
        console.print("[dim](no sidecar)[/dim]")
        return

    for name in sorted(fa.analyses):
        entry = fa.analyses[name]
        header = f"[cyan]{name}[/cyan]"
        if entry.get("parse_error"):
            header += " [red](parse error)[/red]"
        console.print(header)
        out = entry.get("output")
        _print_output(out)
        console.print()


def _print_output(out: Any) -> None:
    if isinstance(out, str):
        console.print(out)
    else:
        # rich.json.JSON renders nicely with syntax highlighting.
        console.print(JSON.from_data(out))


# ---------------------------------------------------------------------------
# cloud — ascii tag cloud across many sidecars
# ---------------------------------------------------------------------------


@app.command("cloud")
def cloud_cmd(
    prompt_substr: Annotated[
        str,
        typer.Argument(
            help=(
                "Substring of the prompt to match. Tags are pulled from "
                "the first analysis entry whose key contains this string. "
                "Case-insensitive."
            ),
        ),
    ],
    path: Annotated[
        Path,
        typer.Argument(help="Folder to scan for sidecars (recursive)."),
    ] = Path("."),
    limit: Annotated[
        int,
        typer.Option(
            "--limit", "-n",
            help="Max number of tags to render.",
        ),
    ] = 150,
    min_count: Annotated[
        int,
        typer.Option(
            "--min", "-m",
            help=(
                "Drop tags with fewer than this many occurrences. "
                "Default 1 keeps the rare ones in the tail view."
            ),
        ),
    ] = 1,
    tail_fraction: Annotated[
        float,
        typer.Option(
            "--tail",
            help=(
                "Fraction of rows to reserve for the LEAST frequent "
                "tags at the bottom (0 disables the tail section)."
            ),
        ),
    ] = 0.15,
    no_color: Annotated[
        bool,
        typer.Option(
            "--no-color",
            help="Plain ASCII output (default uses colour).",
        ),
    ] = False,
    lines: Annotated[
        int | None,
        typer.Option(
            "--lines",
            help=(
                "Total rows to render (overrides terminal-height auto-fit). "
                "Use this to scroll past the head/tail split, e.g. --lines 500."
            ),
        ),
    ] = None,
) -> None:
    """Render a deterministic tag-frequency bar chart across sidecars.

    Counts tags from every sidecar entry whose prompt key contains
    *prompt_substr* (case-insensitive). Fills the terminal vertically
    with a head + tail split: most-frequent tags at top, least-frequent
    at bottom, middle elided. Sqrt-scaled bars so the long tail stays
    readable. Designed for `watch -n 5 --color "uv run videodb cloud
    'Tag this' /path/to/footage"`.
    """
    import shutil

    counts, n_sidecars = _collect_tag_counts(path, prompt_substr)
    if not counts:
        err_console.print(
            f"[yellow]No sidecars under {path} matched "
            f"prompt substring {prompt_substr!r}[/yellow]"
        )
        raise typer.Exit(1)

    # Sort by (count desc, tag asc) so identical counts never reorder
    # across runs (deterministic for `watch`).
    ranked_all = sorted(
        ((tag, n) for tag, n in counts.items() if n >= min_count),
        key=lambda x: (-x[1], x[0]),
    )
    if not ranked_all:
        err_console.print(
            f"[yellow]No tags above min_count={min_count} "
            f"(total entries scanned: {sum(counts.values())})[/yellow]"
        )
        raise typer.Exit(1)

    term = shutil.get_terminal_size((100, 24))
    width = term.columns
    # Fill vertical space. Reserve: 1 title + 1 blank + (1 blank
    # above separator + 1 separator + 1 blank below when we have a
    # tail) = 5 rows. Otherwise 2 rows.
    total_rows = max(6, lines if lines is not None else term.lines - 1)

    # Allocate row budget. Tail rows are packed with multiple tags
    # per row, so we don't need many. Head rows are 1 bar per row.
    if tail_fraction <= 0 or total_rows < 10:
        tail_rows = 0
        head_rows = total_rows - 2
    else:
        tail_rows = max(1, int(round((total_rows - 4) * tail_fraction)))
        head_rows = total_rows - 4 - tail_rows

    # Head: at most `limit` bars, also capped to `head_rows`.
    head_n = min(limit, head_rows, len(ranked_all))
    head = ranked_all[:head_n]

    # Tail: pack tags by count desc into `tail_rows` lines. The
    # packer pulls from the rarest-first and stops when it can't
    # fit any more.
    if tail_rows > 0 and len(ranked_all) > head_n:
        tail_groups, tail_tag_count = _pack_tail(
            ranked_all[head_n:],
            rows=tail_rows,
            width=width,
        )
    else:
        tail_groups, tail_tag_count = [], 0

    omitted = len(ranked_all) - head_n - tail_tag_count

    _render_bars(
        head=head, tail_groups=tail_groups, omitted=omitted,
        max_count=ranked_all[0][1],
        width=width, color=not no_color,
        n_sidecars=n_sidecars, total_tags=len(counts),
    )


# Row format for tail groups:
#     "  N:  tag · tag · tag · …    (M more)"
# Indent + right-aligned "N:" + 2 spaces + selection of tags joined
# by " · ", optionally followed by an ellipsis hint when truncated.
_TAIL_SEP = " · "
_TAIL_INDENT = 2          # spaces before "N:"
_TAIL_COUNT_GAP = 2       # spaces between "N:" and first tag
_TAIL_OVERFLOW = " · …"   # appended when more tags exist than fit
_MORE_FMT = "  ({n} more)"  # suffix when many extras remain


def _pack_tail(
    tags_in_tail: list[tuple[str, int]],
    *,
    rows: int,
    width: int,
) -> tuple[list[tuple[int, list[str], int]], int]:
    """Pick a sample of rarest-first tags, one count bucket per row.

    Returns ((count, [shown_tags], n_hidden) per row, total tags shown).

    For each count bucket: sort tags deterministically, hash that
    list, use the hash to seed an RNG, shuffle, then take as many
    tags as fit on a single line. The hash-as-seed means the
    visible selection is *stable* until the tag set changes — `watch`
    won't reshuffle on every tick, but the moment a new tag enters
    or leaves a bucket the shuffle changes (signalling new content).
    """
    import hashlib
    import random
    from collections import defaultdict

    by_count: defaultdict[int, list[str]] = defaultdict(list)
    for tag, c in tags_in_tail:
        by_count[c].append(tag)
    for c in by_count:
        by_count[c].sort()  # canonical order for hashing

    ordered_counts = sorted(by_count)  # ascending: 1, 2, 3, ...

    # We pick the lowest-count buckets first (those are the
    # "rarest tags" tail). Take up to `rows` of them — but we'll
    # render them with the rarest at the BOTTOM so the chart flows
    # continuously: high counts at top → low counts at bottom.
    visible_counts = ordered_counts[:rows]
    if not visible_counts:
        return [], 0
    max_label_w = max(len(str(c)) for c in visible_counts) + 1  # +":"

    rows_used: list[tuple[int, list[str], int]] = []
    tags_shown = 0
    for c in visible_counts:
        all_tags = by_count[c]

        # Hash the sorted bucket → seed → shuffle.
        h = hashlib.blake2b(
            "\x00".join(all_tags).encode("utf-8"), digest_size=8,
        ).digest()
        seed = int.from_bytes(h, "big")
        shuffled = list(all_tags)
        random.Random(seed).shuffle(shuffled)

        # Pack as many tags as fit on one line.
        prefix_w = _TAIL_INDENT + max_label_w + _TAIL_COUNT_GAP
        avail = max(20, width - prefix_w)
        # Reserve room for the truncation marker so we don't over-fill.
        # We'll compute the exact suffix length below per-case.
        sep_w = len(_TAIL_SEP)

        picked: list[str] = []
        used = 0
        for tag in shuffled:
            tag_w = len(tag)
            need = tag_w + (sep_w if picked else 0)
            # Tentatively check whether we could still afford an
            # overflow marker if there'd be more after this one.
            remaining_after = len(shuffled) - len(picked) - 1
            overflow_w = (
                len(_TAIL_OVERFLOW)
                + (len(_MORE_FMT.format(n=remaining_after))
                   if remaining_after > 0 else 0)
            ) if remaining_after > 0 else 0
            if used + need + overflow_w > avail and picked:
                break
            picked.append(tag)
            used += need
        n_hidden = len(shuffled) - len(picked)
        rows_used.append((c, picked, n_hidden))
        tags_shown += len(picked)

    # Render highest count first → lowest count last, so the rare
    # tags sit at the very bottom of the chart.
    rows_used.reverse()
    return rows_used, tags_shown


def _collect_tag_counts(
    root: Path, prompt_substr: str,
) -> tuple["Counter[str]", int]:
    """Walk *root* for sidecars; return (tag counts, num matched sidecars).

    Per sidecar we take the FIRST analyses entry whose key contains
    *prompt_substr* (case-insensitive). Only list-typed outputs
    contribute — string outputs are ignored (they're descriptions,
    not tags).
    """
    needle = prompt_substr.lower()
    counts: Counter[str] = Counter()
    matched = 0
    for sc in root.rglob("*.videodb.json"):
        try:
            with open(sc, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
            err_console.print(f"[yellow]skipping unreadable sidecar {sc}: {e}[/yellow]")
            continue
        for key, entry in (data.get("analyses") or {}).items():
            if needle not in key.lower():
                continue
            out = entry.get("output")
            added = False
            if isinstance(out, list):
                for t in out:
                    if isinstance(t, str):
                        counts[t.strip()] += 1
                        added = True
            elif isinstance(out, dict):
                # Walk nested string-list values too (legacy JSON passes).
                for v in out.values():
                    if isinstance(v, list):
                        for t in v:
                            if isinstance(t, str):
                                counts[t.strip()] += 1
                                added = True
            if added:
                matched += 1
            break  # first matching entry per sidecar wins
    counts.pop("", None)
    return counts, matched


# Heavy horizontal line as the bar glyph. Half-cell partial at the
# tip for smooth fractional widths. Single-line height per row gives
# natural vertical breathing room — much easier to read at a glance
# than stacked full blocks.
_FULL_LINE = "━"     # U+2501 BOX DRAWINGS HEAVY HORIZONTAL
_HALF_LINE = "╸"    # U+2578 BOX DRAWINGS HEAVY LEFT  (half-width left)


def _make_bar(value: float, max_value: float, width: int) -> str:
    """Render a horizontal heavy-line bar of *width* cells.

    Uses a half-line partial at the tip so a 23-cell-wide bar can
    show 22.5 cells of fill rather than rounding to 23. Padded with
    spaces to *width* so subsequent columns stay aligned.
    """
    if max_value <= 0 or width <= 0:
        return ""
    frac = max(0.0, min(1.0, value / max_value))
    total_halves = round(frac * width * 2)
    full = total_halves // 2
    rem = total_halves % 2
    bar = _FULL_LINE * full
    if rem:
        bar += _HALF_LINE
    return bar.ljust(width)


# Rank-based colour gradient (rich tags). Walks from hot bright to
# cool dim across the visible range — nicer than a single colour.
_GRADIENT = (
    "bold bright_red",
    "bold red",
    "bold orange3",
    "bold yellow",
    "green",
    "cyan",
    "blue",
    "dim white",
)


def _style_for_rank(rank: int, total: int) -> str:
    """Pick a gradient style by fractional position in the ranking."""
    if total <= 1:
        return _GRADIENT[0]
    bucket = min(
        len(_GRADIENT) - 1,
        int((rank / max(1, total - 1)) * len(_GRADIENT)),
    )
    return _GRADIENT[bucket]


def _render_bars(
    *,
    head: list[tuple[str, int]],
    tail_groups: list[tuple[int, list[str], int]],
    omitted: int,
    max_count: int,
    width: int,
    color: bool,
    n_sidecars: int,
    total_tags: int,
) -> None:
    """Render the head bar chart + (optional) packed tail list.

    Head: one bar per row, sqrt-scaled (tag at 1/4 the top count
    gets ~50% of the bar, not 25%).
    Tail: each row is `  N:  tag · tag · tag`, packed densely so
    many rare tags fit in a few lines.
    """
    import math
    from rich.text import Text

    n_head = len(head)
    n_tail_tags = sum(len(tags) for _, tags, _ in tail_groups)
    max_count_str_len = len(str(max_count))
    # Tag column sized to longest head tag, capped so wide tags
    # don't squeeze the bar off the screen.
    if head:
        tag_col = min(28, max(8, max(len(t) for t, _ in head)))
    else:
        tag_col = 12
    fixed = tag_col + 2 + 2 + max_count_str_len
    bar_col = max(8, width - fixed)

    # Header
    showing = f"top {n_head}" + (
        f" + bottom {n_tail_tags}" if n_tail_tags else ""
    )
    title = (
        f"[bold]videodb tag cloud[/bold] — "
        f"{n_sidecars} clips, {total_tags} unique tags, "
        f"showing {showing}"
    )
    if color:
        console.print(title)
    else:
        console.print(Text.from_markup(title).plain)
    console.print()

    max_root = math.sqrt(max_count) if max_count > 0 else 1.0

    # Head: one bar per row, gradient-styled by rank.
    for i, (tag, c) in enumerate(head):
        bar = _make_bar(math.sqrt(c), max_root, bar_col)
        tag_str = tag[:tag_col].rjust(tag_col)
        count_str = f"{c:>{max_count_str_len}}"
        style = _style_for_rank(i, max(n_head, 1))
        if color:
            line = Text()
            line.append(f"{tag_str}  ", style=style)
            line.append(bar, style=style)
            line.append(f"  {count_str}", style="dim")
            console.print(line)
        else:
            console.print(
                f"{tag_str}  {bar}  {count_str}",
                highlight=False,
            )

    # Separator (only when there IS a tail and the middle has tags).
    if tail_groups and omitted > 0:
        _render_separator(width=width, omitted=omitted, color=color)
    elif tail_groups:
        console.print()  # just a blank line if no middle was elided

    # Tail: count-grouped sample lists, one row per count bucket.
    if tail_groups:
        max_label_w = max(len(str(c)) for c, _, _ in tail_groups)
        for c, tags, n_hidden in tail_groups:
            label = f"{c}:".rjust(max_label_w + 1)
            indent = " " * _TAIL_INDENT
            body = _TAIL_SEP.join(tags)
            suffix = ""
            if n_hidden > 0:
                suffix = _TAIL_OVERFLOW + _MORE_FMT.format(n=n_hidden)
            if color:
                line = Text()
                line.append(indent)
                line.append(label, style="bold dim")
                line.append(" " * _TAIL_COUNT_GAP)
                line.append(body, style="dim")
                if suffix:
                    line.append(suffix, style="dim italic")
                console.print(line)
            else:
                console.print(
                    f"{indent}{label}{' ' * _TAIL_COUNT_GAP}{body}{suffix}",
                    highlight=False,
                )


def _render_separator(
    *,
    width: int,
    omitted: int,
    color: bool,
) -> None:
    """Print a dotted unicode 'page break' with the omitted-tag count.

    Glyph is `┄` (box drawings light triple dash horizontal) — a
    three-dot dash per cell that visually connects without feeling
    aggressive. Centered label shows how many tags were skipped.
    """
    from rich.text import Text

    label = f"  ⋯ {omitted} tags omitted ⋯  "
    pad_total = max(0, width - len(label))
    left_pad = pad_total // 2
    right_pad = pad_total - left_pad
    left = "┄" * left_pad
    right = "┄" * right_pad

    console.print()  # blank line above
    if color:
        line = Text()
        line.append(left, style="dim")
        line.append(label, style="dim italic")
        line.append(right, style="dim")
        console.print(line)
    else:
        console.print(f"{left}{label}{right}", highlight=False)
    console.print()  # blank line below


# ---------------------------------------------------------------------------
# passes — list which passes would run for a given path
# ---------------------------------------------------------------------------


@app.command("passes")
def passes_cmd(
    for_path: Annotated[
        Path | None,
        typer.Option(
            "--for-path",
            help="Resolve settings as if for this path",
        ),
    ] = None,
) -> None:
    """List passes that `analyze` would run for *for_path* (or defaults).

    Useful for verifying that folder-level `.videodb_settings.toml`
    additions are being picked up.
    """
    settings = resolve_settings(for_path)
    passes = build_passes(settings)

    table = Table(title="passes")
    table.add_column("prompt", overflow="fold")
    table.add_column("parse", style="cyan")
    table.add_column("model", style="dim")
    table.add_column("nframes", justify="right")
    table.add_column("max_tokens", justify="right")
    table.add_column("settings (cache key)", style="dim", overflow="fold")

    for p in passes:
        table.add_row(
            p.prompt, p.parse, p.model, str(p.nframes),
            str(p.max_tokens), p.settings,
        )
    console.print(table)


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


@settings_app.command("list")
def settings_list_cmd() -> None:
    """Show the user settings file."""
    console.print(list_settings())


@settings_app.command("get")
def settings_get_cmd(
    key: Annotated[
        str | None,
        typer.Argument(help="Dotted key (e.g. analyze.nframes)"),
    ] = None,
    for_path: Annotated[
        Path | None,
        typer.Option("--for-path", help="Show merged settings for this path"),
    ] = None,
    show_sources: Annotated[
        bool,
        typer.Option("--show-sources", help="Show where each value came from"),
    ] = False,
) -> None:
    """Get a setting value. Without `--for-path`, reads user settings only."""
    try:
        out = get_settings(key=key, for_path=for_path, show_sources=show_sources)
        console.print(out)
    except KeyError:
        err_console.print(f"[red]Key not found: {key}[/red]")
        raise typer.Exit(1)


@settings_app.command("set")
def settings_set_cmd(
    key: Annotated[str, typer.Argument(help="Dotted key (e.g. analyze.nframes)")],
    value: Annotated[str, typer.Argument(help="Value to set (parsed heuristically)")],
    for_path: Annotated[
        Path | None,
        typer.Option("--for-path", help="Write to folder settings at this path"),
    ] = None,
) -> None:
    """Set a setting value. Without `--for-path`, writes to user settings."""
    try:
        msg = set_setting(key, value, for_path=for_path)
        console.print(msg)
    except NotADirectoryError:
        err_console.print(f"[red]Not a directory: {for_path}[/red]")
        raise typer.Exit(1)


@settings_app.command("path")
def settings_path_cmd() -> None:
    """Print the path to the user settings file."""
    from .settings import USER_SETTINGS_PATH
    console.print(str(USER_SETTINGS_PATH))


if __name__ == "__main__":  # pragma: no cover
    app()
