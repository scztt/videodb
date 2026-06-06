"""Lazy Qwen-VL loader. One instance per process.

Originally lifted from describe_video.py. Uses `AutoModelForImageTextToText`
so both Qwen2.5-VL and Qwen3-VL checkpoints load with the right
architecture (Qwen3-VL's Interleaved-MRoPE in particular needs the
qwen3_vl class — instantiating it as qwen2_5_vl silently drops the
positional encoding).

Loading the 7B model uses ~16GB RAM and takes ~10s. We keep it loaded
between calls (one process should analyze the whole batch).
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DEFAULT_MODEL_ID

if TYPE_CHECKING:
    import torch  # noqa: F401 — type-only

log = logging.getLogger(__name__)


# --- env --------------------------------------------------------------

# HF cache: default to the shared on-volume cache where describe_video.py
# already lives, so the ~16GB Qwen weights don't get re-downloaded.
# Override priority: existing HF_HOME env > on-volume cache (if mounted) >
# package-local `.hf-cache`. Users can set HF_HOME before launching to
# point elsewhere.
_HERE = Path(__file__).resolve().parent
_SHARED_HF_CACHE = Path("/Volumes/chrys-rescue/_models/.hf-cache")
if _SHARED_HF_CACHE.is_dir():
    _DEFAULT_HF_HOME = str(_SHARED_HF_CACHE)
else:
    _DEFAULT_HF_HOME = str(_HERE.parent / ".hf-cache")
os.environ.setdefault("HF_HOME", _DEFAULT_HF_HOME)
os.environ.setdefault("TRANSFORMERS_CACHE", _DEFAULT_HF_HOME)
os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "torchcodec")


# --- lazy globals -----------------------------------------------------

_model = None
_processor = None
_device: str | None = None
_loaded_model_id: str | None = None


def _pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pick_dtype(device: str):
    import torch
    if device == "mps":
        return torch.float16
    if device == "cuda":
        return torch.bfloat16
    return torch.float32


def _ensure_loaded(model_id: str) -> None:
    """Load the model once per process for a given *model_id*.

    Swapping models mid-run reloads — but you shouldn't be doing that.
    """
    global _model, _processor, _device, _loaded_model_id

    if _loaded_model_id == model_id and _model is not None:
        return

    from transformers import AutoModelForImageTextToText, AutoProcessor

    _device = _pick_device()
    dtype = _pick_dtype(_device)
    log.info("Loading %s on %s (%s)…", model_id, _device, dtype)
    t0 = time.monotonic()

    _processor = AutoProcessor.from_pretrained(model_id)
    _model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=_device,
    )
    _model.eval()
    _loaded_model_id = model_id

    log.info("Model ready in %.1fs", time.monotonic() - t0)


class VideoTooShortError(RuntimeError):
    """Raised when a clip has fewer frames than the model needs.

    qwen_vl_utils requires nframes to be a multiple of FRAME_FACTOR=2
    and at least 2. Clips with 0 or 1 native frames can't be sampled
    no matter what. We catch this up-front so the model never spins
    up for a doomed call (and so the error message is readable).
    """


# qwen_vl_utils requires nframes to be a multiple of this (and >= it).
_QWEN_FRAME_FACTOR = 2


def _probe_video(video_path: Path) -> tuple[int, float, float]:
    """Cheap duration probe via torchcodec.

    Returns ``(num_frames, duration_seconds, average_fps)``.
    Used for the pre-flight too-short check; full decode happens
    later inside qwen_vl_utils.
    """
    from torchcodec.decoders import VideoDecoder
    m = VideoDecoder(str(video_path)).metadata
    return (
        int(m.num_frames or 0),
        float(m.duration_seconds or 0.0),
        float(m.average_fps or 0.0),
    )


def generate(
    *,
    video_path: Path,
    prompt: str,
    nframes: int = 32,
    max_tokens: int = 256,
    model_id: str = DEFAULT_MODEL_ID,
) -> str:
    """Run Qwen2.5-VL on a video + text prompt. Returns generated text.

    Samples exactly *nframes* frames, evenly spaced. This is
    duration-independent — a 2s clip and a 5-minute clip both get
    the same 32-frame view of themselves. Short clips no longer
    suffer from sparse sampling; long clips no longer blow up
    memory.

    Raises `VideoTooShortError` if the clip has fewer native frames
    than *nframes* — we won't oversample beyond what exists.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    # Clamp to qwen_vl_utils's factor constraint.
    nframes = max(_QWEN_FRAME_FACTOR, (nframes // _QWEN_FRAME_FACTOR) * _QWEN_FRAME_FACTOR)

    # Pre-flight: skip clips qwen_vl_utils would fail on. Its
    # smart_nframes() requires `nframes <= total_frames`, so a clip
    # with fewer native frames than we want can't satisfy this.
    try:
        num_frames, duration_s, native_fps = _probe_video(video_path)
    except Exception as e:  # noqa: BLE001 — torchcodec varies on bad files
        raise RuntimeError(f"video probe failed: {e}") from e

    if num_frames < nframes:
        raise VideoTooShortError(
            f"too short ({duration_s:.3f}s, {num_frames} native "
            f"frames @ {native_fps:.1f}fps) — need ≥{nframes} frames"
        )

    _ensure_loaded(model_id)
    assert _model is not None and _processor is not None and _device is not None

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "nframes": nframes,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_device)

    with torch.inference_mode():
        out_ids = _model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
    out = _processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return out.strip()
