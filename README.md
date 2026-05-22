# videodb

Per-file video analysis with sidecar caching. Each video file gets a
`{video}.videodb.json` sidecar holding the results of every analysis
*pass* — a description, a tag list, a structured JSON summary, and any
number of user-defined text queries — produced by Qwen2.5-VL.

Modelled on the [audio-database](../audio-database) tool: layered
settings (defaults → user → folder), incremental re-analysis driven by
a per-pass settings hash plus the file's mtime, and a thin CLI.

## Install

```sh
uv sync
```

First run downloads the Qwen2.5-VL model (~16 GB) into `HF_HOME`.
If you've already used `describe_video.py`, set `HF_HOME` to its cache
so the weights aren't re-downloaded:

```sh
export HF_HOME=/Volumes/chrys-rescue/_models/.hf-cache
```

The package picks that path up automatically when the volume is
mounted.

## Usage

```sh
# Analyze every video under a folder (recursive).
uv run videodb analyze /Volumes/PEACECORE/footage

# Just one pass, just one file.
uv run videodb analyze /path/to/clip.mov --pass tags

# Show all stored analyses for a file or folder.
uv run videodb query /path/to/clip.mov
uv run videodb query /Volumes/PEACECORE/footage --pass tags --exclude '*/proxies/*'

# Pipe out as JSON / JSONL.
uv run videodb query /Volumes/PEACECORE/footage --format jsonl

# Inspect which passes would run for a given path.
uv run videodb passes --for-path /Volumes/PEACECORE/footage

# Settings.
uv run videodb settings list
uv run videodb settings get analyze.fps
uv run videodb settings set analyze.fps 0.5
uv run videodb settings set analyze.fps 0.5 --for-path /Volumes/PEACECORE/footage
```

## Settings

User settings live at `~/.config/videodb/settings.toml`. Per-folder
overrides live in `.videodb_settings.toml` at any level above the
target — deepest wins.

```toml
[analyze]
sidecar_mapping = "{fullpath}.videodb.json"
passes = ["text", "json", "tags"]
model = "Qwen/Qwen2.5-VL-7B-Instruct"
fps = 1.0
max_tokens = 256

[[analyze.queries]]
prompt = "Is this footage indoors or outdoors?"

[[analyze.queries]]
prompt = "What is the dominant color palette?"
max_tokens = 64
```

Each query becomes a sidecar entry named `<slug16>__<hash8>`, where
the slug is the first 16 alnum chars of the prompt and the hash is the
content hash of `(name, version, model, fps, max_tokens, prompt)`.
Removing a query from settings does *not* remove its sidecar entry —
you can always find or reconstruct the prompt from the stored data.

## Re-analysis triggers

A pass re-runs when any of the following changes:

- the video file's mtime
- the pass's `version` (bump in code to invalidate)
- the prompt text
- the model id / fps / max_tokens

Otherwise the sidecar entry is reused.
