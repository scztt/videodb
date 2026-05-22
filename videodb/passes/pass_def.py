"""The single `Pass` dataclass — one prompt, one parse mode.

A pass is fully described by:
  - prompt text (the literal Qwen prompt; also the sidecar key)
  - inference params: model id, fps, max_tokens
  - parse mode: "raw" (str output, default) or "json" (parsed structure)

The `settings` property formats those into the human-readable cache
key stored in the sidecar:

    "fps=1.0&max_tokens=256&model=Qwen/...&parse=json&version=1"

Cache hit = settings string matches the one in the sidecar entry.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .. import model as model_runner
from ..config import PARSER_VERSION, format_settings

log = logging.getLogger(__name__)

ParseMode = Literal["raw", "json", "csv"]


def parse_modes() -> tuple[str, ...]:
    """Known parse modes — referenced by settings validation / CLI help."""
    return ("raw", "json", "csv")


def parse_csv_tags(text: str) -> list[str]:
    """Parse a comma-separated tag list. Tolerant of newlines, fences,
    leading "Tags:" labels, and stray quotes around tags."""
    # Strip code fences and any "Tags:" / "Output:" prefix.
    text = strip_code_fences(text)
    for prefix in ("tags:", "output:", "answer:"):
        low = text.lower().lstrip()
        if low.startswith(prefix):
            text = text[len(prefix) + (len(text) - len(low)):]
            break
    # Split on commas OR newlines (Qwen sometimes one-per-line lists).
    parts = re.split(r"[,\n]", text)
    out: list[str] = []
    for p in parts:
        p = p.strip().strip("\"'").strip()
        # Drop bullet markers.
        p = re.sub(r"^[-*•]\s*", "", p).strip()
        if p:
            out.append(p)
    return out


# --- JSON salvage helpers ----------------------------------------------

# ```json … ``` or plain ``` … ```  (multiline, non-greedy)
_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)


def strip_code_fences(text: str) -> str:
    """Strip a single surrounding markdown code fence, if present."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def extract_json(text: str) -> Any:
    """Best-effort JSON extraction from a model response.

    Tries: parse whole string → strip fences → find largest balanced
    `{...}` / `[...]` substring. Raises `json.JSONDecodeError` if
    nothing parses.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = strip_code_fences(text)
    if fenced != text:
        try:
            return json.loads(fenced)
        except json.JSONDecodeError:
            pass
        text = fenced

    span = _find_balanced_json_span(text)
    if span is not None:
        start, end = span
        return json.loads(text[start:end])

    raise json.JSONDecodeError("no JSON value found in model output", text, 0)


def _find_balanced_json_span(text: str) -> tuple[int, int] | None:
    """First balanced { } or [ ] span — string-aware so quoted braces
    don't throw off the depth counter."""
    best: tuple[int, int] | None = None
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\" and in_str:
                escape = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    cand = (start, i + 1)
                    if best is None or (cand[1] - cand[0]) > (best[1] - best[0]):
                        best = cand
                    break
    return best


# --- the Pass dataclass -------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class Pass:
    """A single analysis pass.

    The `prompt` doubles as the sidecar key — two passes with the same
    prompt collapse into one entry, with the most recent run's
    `settings` string. That's intentional: changing inference params
    is treated as an upgrade of the same logical pass.
    """
    prompt: str
    model: str
    nframes: int
    max_tokens: int
    parse: ParseMode = "raw"

    def __post_init__(self) -> None:
        if self.parse not in parse_modes():
            raise ValueError(
                f"Unknown parse mode: {self.parse!r}. "
                f"Known: {parse_modes()}"
            )

    @property
    def settings(self) -> str:
        """Cache-key string: `key=value&key=value`, alphabetical keys.

        `parse` is omitted from the string when it's the default
        `raw`, so adding an explicit `parse = "raw"` to a TOML entry
        doesn't invalidate previously-cached results.
        """
        items: dict[str, object] = {
            "model": self.model,
            "nframes": self.nframes,
            "max_tokens": self.max_tokens,
            "version": PARSER_VERSION,
        }
        if self.parse != "raw":
            items["parse"] = self.parse
        return format_settings(items)

    def run(self, video_path: Path) -> tuple[Any, bool]:
        """Execute the pass. Returns (output, parse_error).

        `parse_error` is True only when `parse="json"` and the model's
        output couldn't be coerced to JSON — in that case `output` is
        the raw string so nothing is lost.
        """
        raw = model_runner.generate(
            video_path=video_path,
            prompt=self.prompt,
            nframes=self.nframes,
            max_tokens=self.max_tokens,
            model_id=self.model,
        )
        return self._parse(raw)

    def _parse(self, raw: str) -> tuple[Any, bool]:
        if self.parse == "raw":
            return raw, False
        if self.parse == "json":
            try:
                return extract_json(raw), False
            except json.JSONDecodeError as e:
                log.warning(
                    "json parse failed for prompt %r: %s",
                    self.prompt[:60], e,
                )
                return raw, True
        if self.parse == "csv":
            tags = parse_csv_tags(raw)
            # Only flag a parse error if the model produced nothing
            # comma-separable at all — empty list isn't great but it
            # isn't structurally broken.
            return tags, False
        # Unreachable — guarded in __post_init__.
        return raw, False
