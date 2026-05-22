"""Pass system — every pass is just a prompt + parse mode.

Passes are loaded from settings. There are no built-in pass *classes*;
the three "built-in" prompts live in the settings DEFAULTS list so a
user can remove or reorder them like any other pass.

The Pass key in the sidecar is the **full prompt text**. Cache hit =
matching `settings` string (a `key=value&...` join of the prompt's
inference settings + parser version). See `passes/pass_def.py`.
"""

from __future__ import annotations

from typing import Any

from .pass_def import Pass, ParseMode, parse_modes

__all__ = ["Pass", "ParseMode", "build_passes", "parse_modes"]


def build_passes(settings: dict[str, Any]) -> list[Pass]:
    """Construct the pass list from a resolved settings dict.

    Top-level inference defaults live on `analyze`; per-pass overrides
    (model / nframes / max_tokens / parse) live on each entry in
    `analyze.passes`.
    """
    analyze = settings.get("analyze", {})
    default_model: str = analyze.get("model", "")
    default_nframes: int = int(analyze.get("nframes", 32))
    default_max_tokens: int = int(analyze.get("max_tokens", 256))

    raw_passes = analyze.get("passes") or []
    out: list[Pass] = []
    for i, entry in enumerate(raw_passes):
        if not isinstance(entry, dict) or "prompt" not in entry:
            raise ValueError(
                f"analyze.passes[{i}] must be a table with a `prompt` key, "
                f"got {entry!r}"
            )
        out.append(
            Pass(
                prompt=str(entry["prompt"]),
                model=str(entry.get("model", default_model)),
                nframes=int(entry.get("nframes", default_nframes)),
                max_tokens=int(entry.get("max_tokens", default_max_tokens)),
                parse=str(entry.get("parse", "raw")),
            )
        )
    return out
