"""Encoding :class:`~sonarch.Audio` objects to files."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

import av
import numpy as np
import torch

from sonarch.core.audio import Audio
from sonarch.core.errors import AudioValidationError, ClippingError, EncodeError

__all__ = ["save"]

Encoding = Literal["pcm16", "pcm24", "pcm32", "float32"]

# container format, {encoding: (codec, sample_fmt)}
_WAV = {
    "pcm16": ("pcm_s16le", "s16"),
    "pcm24": ("pcm_s24le", "s32"),
    "pcm32": ("pcm_s32le", "s32"),
    "float32": ("pcm_f32le", "flt"),
}
_FLAC = {
    "pcm16": ("flac", "s16"),
    "pcm24": ("flac", "s32"),
}
_OPUS_RATES = (8_000, 12_000, 16_000, 24_000, 48_000)


def save(
    audio: Audio,
    path: str | os.PathLike[str],
    *,
    format: Literal["wav", "flac", "opus"] | None = None,
    encoding: Encoding | None = None,
    bitrate: str | int | None = None,
    clip: bool = False,
) -> None:
    """Write audio to a file.

    Args:
        audio: The audio to write.
        path: Destination path. The format is inferred from the extension
            (``.wav``, ``.flac``, ``.opus``) unless ``format`` is given.
        format: ``"wav"``, ``"flac"`` or ``"opus"``.
        encoding: Sample encoding for WAV (``"pcm16"`` default, ``"pcm24"``,
            ``"pcm32"``, ``"float32"``) or FLAC (``"pcm16"`` default,
            ``"pcm24"``). Not used for Opus.
        bitrate: Opus bitrate, e.g. ``"64kbps"`` or ``64000``. Default 64 kbps.
        clip: Integer encodings and Opus can't store values outside
            ``[-1, 1]``. By default sonarch raises :class:`ClippingError`
            instead of silently clipping; pass ``clip=True`` to allow it.

    Raises:
        ClippingError: Audio exceeds ``[-1, 1]`` and ``clip`` is False.
        SampleRateMismatchError: Opus was requested at an unsupported rate.
        EncodeError: FFmpeg failed to write the file.
    """
    if not isinstance(audio, Audio):
        raise TypeError(f"save() expects an Audio object, got {type(audio).__name__}")

    path = Path(path)
    fmt = format or _format_from_suffix(path)
    codec_name, sample_fmt, options = _codec_for(fmt, encoding, bitrate, audio.sample_rate)

    data = audio.tensor.detach().to(device="cpu", dtype=torch.float32)
    if not torch.isfinite(data).all():
        raise AudioValidationError("audio contains NaN or infinite values; refusing to write")
    peak = float(data.abs().max()) if data.numel() else 0.0
    if peak > 1.0 and not (fmt == "wav" and encoding == "float32"):
        if not clip:
            raise ClippingError(
                f"audio peak is {peak:.3f}, outside [-1, 1]; {fmt} "
                f"({encoding or 'default encoding'}) would clip it. Normalize first, "
                "save with encoding='float32' (WAV), or pass clip=True to clip on purpose."
            )
        data = data.clamp(-1.0, 1.0)

    layout = {1: "mono", 2: "stereo"}.get(audio.channels, f"{audio.channels}c")
    array = np.ascontiguousarray(data.numpy())

    try:
        with av.open(str(path), mode="w", format=_CONTAINERS[fmt]) as container:
            stream = container.add_stream(
                codec_name, rate=audio.sample_rate, options=options, layout=layout
            )
            stream.format = sample_fmt
            container.start_encoding()  # write the header even if there are no samples

            if array.shape[1] > 0:
                frame = av.AudioFrame.from_ndarray(array, format="fltp", layout=layout)
                frame.sample_rate = audio.sample_rate
                frame.pts = 0
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
    except av.error.FFmpegError as exc:
        raise EncodeError(f"failed to write {str(path)!r}: {exc}") from exc


_CONTAINERS = {"wav": "wav", "flac": "flac", "opus": "ogg"}


def _format_from_suffix(path: Path) -> Literal["wav", "flac", "opus"]:
    suffix = path.suffix.lower()
    if suffix in (".wav", ".wave"):
        return "wav"
    if suffix == ".flac":
        return "flac"
    if suffix in (".opus", ".ogg"):
        return "opus"
    raise ValueError(
        f"can't infer the format from {path.name!r}; use a .wav, .flac or .opus "
        "extension or pass format=..."
    )


def _codec_for(
    fmt: str,
    encoding: Encoding | None,
    bitrate: str | int | None,
    sample_rate: int,
) -> tuple[str, str, dict[str, str]]:
    if fmt == "opus":
        if encoding is not None:
            raise ValueError("encoding= is not used for Opus; set bitrate= instead")
        if sample_rate not in _OPUS_RATES:
            from sonarch.core.errors import SampleRateMismatchError

            raise SampleRateMismatchError(
                48_000,
                sample_rate,
                hint=f"Opus supports {', '.join(map(str, _OPUS_RATES))} Hz; resample first",
            )
        return "libopus", "flt", {"b": str(_parse_bitrate(bitrate))}

    if bitrate is not None:
        raise ValueError(f"bitrate= only applies to Opus, not {fmt}")
    table = {"wav": _WAV, "flac": _FLAC}.get(fmt)
    if table is None:
        raise ValueError(f"unsupported format {fmt!r}; choose 'wav', 'flac' or 'opus'")
    key = encoding or "pcm16"
    if key not in table:
        raise ValueError(f"{fmt} supports encodings {sorted(table)}, got {key!r}")
    codec_name, sample_fmt = table[key]
    return codec_name, sample_fmt, {}


def _parse_bitrate(bitrate: str | int | None) -> int:
    if bitrate is None:
        return 64_000
    if isinstance(bitrate, int):
        value = bitrate
    else:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(k|kbps|kb/s|bps)?\s*", bitrate.lower())
        if not match:
            raise ValueError(f"can't parse bitrate {bitrate!r}; use e.g. '64kbps' or 64000")
        number, unit = float(match.group(1)), match.group(2)
        value = int(number * 1000) if unit in ("k", "kbps", "kb/s") else int(number)
    if not 6_000 <= value <= 510_000:
        raise ValueError(f"Opus bitrate must be between 6 and 510 kbps, got {value} bps")
    return value
