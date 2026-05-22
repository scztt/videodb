"""Layered settings resolution.

Resolution order (lowest -> highest priority):
  1. Built-in DEFAULTS
  2. User settings  (~/.config/videodb/settings.toml)
  3. Folder settings (walk up from target path, merging every
                      `.videodb_settings.toml` found; deeper wins)

Modelled on audio-database's settings cascade. Pure functions except
the I/O helpers.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import toml

from .config import DEFAULT_MODEL_ID, DEFAULT_SIDECAR_MAPPING

USER_SETTINGS_PATH: Path = Path.home() / ".config" / "videodb" / "settings.toml"
FOLDER_SETTINGS_FILENAME: str = ".videodb_settings.toml"

DEFAULTS: dict[str, Any] = {
    "analyze": {
        "sidecar_mapping": DEFAULT_SIDECAR_MAPPING,
        "model": DEFAULT_MODEL_ID,
        # Number of frames to sample (evenly spaced) per clip. Same
        # count for short and long clips. Must be a multiple of 2.
        "nframes": 4,
        "max_tokens": 192,
        # Order videos are analyzed in. One of:
        #   path | size | size_desc | mtime | mtime_desc | random
        # `size` is the friendliest for big batches — quick wins
        # first, lets you ctrl-C early without wasting hours on a
        # giant clip. Has no effect on cache keys; results are the
        # same regardless of order.
        "sort": "path",
        # Passes are just (prompt, parse mode) tuples. Override the
        # whole list in user / folder settings to add custom queries.
        "passes": [
            # {
            #     "prompt": ("Describe what happens in this video in 1-2 " "sentences."),
            # },
            #
            # --- combined "flat tags" pass — single inference, low max_tokens.
            # Replaces the two structured-JSON passes below for fast
            # broad-overview tagging. Outputs a comma-separated list of
            # short tags spanning content + boolean attributes.
            {
                "prompt": (
                    "Tag this TV-broadcast clip for a searchable "
                    "database. Output ONLY a comma-separated list — "
                    "no prose, no JSON, no markdown.\n\n"
                    "Aim for 10-18 tags, one word or short phrase "
                    "each, lowercase except proper names. Cover any "
                    "that apply:\n"
                    "  • topic — what the clip is ABOUT (e.g. "
                    "medical, sports, fashion, politics, food, "
                    "technology, military, nature, architecture)\n"
                    "  • people — age and grouping (e.g. children, "
                    "teens, senior, crowd, boyband, news anchor) "
                    "plus celebrity names if recognizable\n"
                    "  • actions — concrete verbs people do (e.g. "
                    "sing, dance, eat, drink, kiss, laugh, cry, "
                    "cheer, wave, run, fight, talk, listen)\n"
                    "  • mood — energy/temperature words (e.g. dark, "
                    "warm, cold, bright, calm, hectic, scary, sexy, "
                    "serious, funny, sad, upbeat, downbeat)\n"
                    "  • music genre — if any musical performance is "
                    "visible, guess from cues (pop, rock, hiphop, "
                    "rnb, country, electronic, ballad)\n"
                    "  • notable objects (cars, guns, flags, fire, "
                    "money, etc.) and place if specific\n"
                    "  • camera — pick the closest: static, zoom, "
                    "push in, pull back, aerial, handheld, slowmo, "
                    "pan, tilt (omit if unclear)\n"
                    "  • flags ONLY if clearly true: 'has text', "
                    "'has logo', 'animated'\n\n"
                    "AVOID: 'video', 'footage', 'clip', 'scene', "
                    "'shot' — those apply to everything. Prefer "
                    "specific over generic ('electric guitar' > "
                    "'instrument', 'senior' > 'person').\n\n"
                    "Example for a live rock band clip:\n"
                    "rock, musician, crowd, sing, electric guitar, "
                    "stage lights, hectic, push in, has faces, "
                    "has logo"
                ),
                "parse": "csv",
                "max_tokens": 96,
            },
            # {
            #     "prompt": (
            #         "Describe this video and output ONLY a JSON object "
            #         "with keys: "
            #         '"location" (string, one word), '
            #         '"people" (array of strings, one word each — '
            #         'short abstract words like "man" / "child" / '
            #         '"baseball player" / "nurse"; use proper names '
            #         "ONLY for celebrities), "
            #         '"objects" (array of strings, one word each), '
            #         '"actions" (array of strings, one word each, '
            #         'present-tense verbs like "walk" / "speak"), '
            #         '"colors" (array of up to 3 strings, one word '
            #         "each), "
            #         '"visual_style" (string, one word), '
            #         '"mood" (string, one word). '
            #         "All lowercase except proper names. "
            #         "No prose outside the JSON, no markdown."
            #     ),
            #     "parse": "json",
            # },
            # {
            #     "prompt": (
            #         "Answer ONLY with a JSON object whose values are "
            #         "all booleans (true or false). Each key asks a "
            #         "yes/no question about this video — output true "
            #         "if yes, false if no. Keys: "
            #         '"text" (does the video contain prominent text, '
            #         "captions, signs, or readable "
            #         "text - not counting small titles, scenery, stock ticker, headlines?), "
            #         '"people" (are any people visible at any point?), '
            #         '"camera_moving" (does the camera move — pan, '
            #         "tilt, dolly, handheld shake, zoom — at any "
            #         "point? false only if every shot is locked-off "
            #         "static), "
            #         '"faces" (is at least one human face clearly '
            #         "visible — not just bodies, backs of heads, or "
            #         "silhouettes?), "
            #         '"animation" (is this video animated, CGI, '
            #         "motion graphics, or otherwise not live-action "
            #         "photography?), "
            #         '"logo" (is a recognizable brand, company, '
            #         "network, or product logo visible at any "
            #         "point?). "
            #         "Output ONLY the JSON object, no prose, no "
            #         "markdown. Every value must be true or false, "
            #         "not a string."
            #     ),
            #     "parse": "json",
            # },
        ],
    },
}


# --- pure helpers --------------------------------------------------------


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*. Override wins on conflicts.

    Returns a new dict — neither input is mutated. Lists are replaced,
    not merged (a `queries = [...]` in folder settings replaces the
    user-level list entirely).
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def get_nested(d: dict, dotted_key: str) -> Any:
    parts = dotted_key.split(".")
    cur: Any = d
    for part in parts:
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted_key)
        cur = cur[part]
    return cur


def set_nested(d: dict, dotted_key: str, value: Any) -> dict:
    result = copy.deepcopy(d)
    parts = dotted_key.split(".")
    cur: dict[str, Any] = result
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value
    return result


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    items: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items


# --- I/O -----------------------------------------------------------------


def load_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _write_toml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        toml.dump(data, f)


# --- folder walk ---------------------------------------------------------


def find_folder_settings(start_path: Path) -> tuple[dict, list[Path]]:
    """Walk from *start_path* up to /, merging every folder settings file.

    Returns (merged_dict, paths_found_in_root_first_order).
    """
    path = start_path.resolve()
    if path.is_file():
        path = path.parent

    found: list[tuple[dict[str, Any], Path]] = []
    while True:
        candidate = path / FOLDER_SETTINGS_FILENAME
        if candidate.is_file():
            found.append((load_toml(candidate), candidate))
        parent = path.parent
        if parent == path:
            break
        path = parent

    if not found:
        return {}, []

    found.reverse()  # root-most first
    merged: dict[str, Any] = {}
    paths: list[Path] = []
    for data, p in found:
        merged = deep_merge(merged, data)
        paths.append(p)

    return merged, paths


# --- resolution ----------------------------------------------------------


def resolve_settings(for_path: Path | None = None) -> dict[str, Any]:
    merged = deep_merge(DEFAULTS, load_toml(USER_SETTINGS_PATH))
    if for_path is not None:
        folder_data, _ = find_folder_settings(for_path)
        merged = deep_merge(merged, folder_data)
    return merged


def resolve_settings_with_sources(
    for_path: Path | None = None,
) -> tuple[dict, dict[str, str]]:
    """Like `resolve_settings` but also tracks where each key came from."""
    user_data = load_toml(USER_SETTINGS_PATH)
    if for_path is not None:
        folder_data, folder_paths = find_folder_settings(for_path)
    else:
        folder_data, folder_paths = {}, []

    merged = deep_merge(DEFAULTS, user_data)
    merged = deep_merge(merged, folder_data)

    flat_defaults = _flatten(DEFAULTS)
    flat_user = _flatten(user_data)

    folder_key_source: dict[str, Path] = {}
    for p in folder_paths:
        for k in _flatten(load_toml(p)):
            folder_key_source[k] = p

    sources: dict[str, str] = {}
    for key in _flatten(merged):
        if key in folder_key_source:
            sources[key] = f"folder:{folder_key_source[key]}"
        elif key in flat_user:
            sources[key] = "user"
        elif key in flat_defaults:
            sources[key] = "default"
        else:
            sources[key] = "unknown"
    return merged, sources


# --- sidecar mapping -----------------------------------------------------


def expand_sidecar_path(video_path: Path, mapping: str) -> Path:
    """Expand sidecar mapping template for a video file.

    Template vars: {fullpath}, {filename}, {parent_path}.
    """
    video_path = Path(video_path).resolve()
    result = mapping.format(
        fullpath=video_path,
        filename=video_path.name,
        parent_path=video_path.parent,
    )
    return Path(result)


def resolve_sidecar_mapping(for_path: Path) -> str:
    s = resolve_settings(for_path)
    return s.get("analyze", {}).get("sidecar_mapping", DEFAULT_SIDECAR_MAPPING)


# --- read/write helpers --------------------------------------------------


def read_user_settings() -> dict:
    return load_toml(USER_SETTINGS_PATH)


def write_user_settings(settings: dict) -> None:
    _write_toml(USER_SETTINGS_PATH, settings)


def read_folder_settings(folder: Path) -> dict:
    return load_toml(folder / FOLDER_SETTINGS_FILENAME)


def write_folder_settings(folder: Path, settings: dict) -> None:
    _write_toml(folder / FOLDER_SETTINGS_FILENAME, settings)


# --- value formatting / parsing (for `settings set` CLI) -----------------


def format_value(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, list):
        return "[" + ", ".join(str(v) for v in val) + "]"
    return str(val)


def parse_value(raw: str) -> Any:
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.lower() == "none":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [s.strip().strip('"').strip("'") for s in inner.split(",")]
    return raw


# --- high-level API (used by CLI) ----------------------------------------


def list_settings() -> str:
    data = read_user_settings()
    if not data:
        return f"No user settings file at {USER_SETTINGS_PATH}"
    return f"# {USER_SETTINGS_PATH}\n" + toml.dumps(data).rstrip()


def get_settings(
    key: str | None = None,
    for_path: Path | None = None,
    show_sources: bool = False,
) -> str:
    if for_path is not None:
        if show_sources:
            merged, sources = resolve_settings_with_sources(for_path)
            if key:
                val = get_nested(merged, key)
                src = sources.get(key, "unknown")
                return f"{key} = {format_value(val)}  # {src}"
            lines = []
            flat = _flatten(merged)
            for k in sorted(flat):
                src = sources.get(k, "unknown")
                lines.append(f"{k} = {format_value(flat[k])}  # {src}")
            return "\n".join(lines)
        merged = resolve_settings(for_path)
        if key:
            return format_value(get_nested(merged, key))
        flat = _flatten(merged)
        return "\n".join(f"{k} = {format_value(flat[k])}" for k in sorted(flat))

    data = read_user_settings()
    if key:
        return format_value(get_nested(data, key))
    flat = _flatten(data) if data else {}
    if not flat:
        return "No user settings."
    return "\n".join(f"{k} = {format_value(flat[k])}" for k in sorted(flat))


def set_setting(
    key: str,
    value: str,
    for_path: Path | None = None,
) -> str:
    parsed = parse_value(value)
    if for_path is not None:
        if not for_path.is_dir():
            raise NotADirectoryError(str(for_path))
        data = read_folder_settings(for_path)
        data = set_nested(data, key, parsed)
        write_folder_settings(for_path, data)
        target = for_path / FOLDER_SETTINGS_FILENAME
    else:
        data = read_user_settings()
        data = set_nested(data, key, parsed)
        write_user_settings(data)
        target = USER_SETTINGS_PATH
    return f"Set {key} = {format_value(parsed)} in {target}"
