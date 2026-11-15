"""Time-domain editing: silence trimming, padding, cropping, fixed length, chunking.

All operations are tensor slicing/padding, so they run on any device. Times are
in seconds and converted to samples with ``round(seconds * sample_rate)``.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from sonarch.core.audio import Audio

__all__ = ["chunk", "crop", "fix_length", "nonsilent_range", "pad", "trim_silence"]


def _samples(audio: Audio, seconds: float, name: str) -> int:
    is_number = isinstance(seconds, (int, float)) and not isinstance(seconds, bool)
    if not is_number or not math.isfinite(seconds):
        raise ValueError(f"{name} must be a finite number of seconds, got {seconds!r}")
    if seconds < 0:
        raise ValueError(f"{name} must be >= 0, got {seconds}")
    return round(seconds * audio.sample_rate)


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def nonsilent_range(
    audio: Audio, top_db: float = 60.0, frame_length: int = 2048, hop_length: int = 512
) -> tuple[int, int]:
    """Sample range ``(start, end)`` that :func:`trim_silence` would keep.

    Same algorithm and defaults as ``librosa.effects.trim``: frame RMS levels
    are compared with the loudest frame (across all channels); frames within
    ``top_db`` dB of it on any channel count as non-silent. Because levels are
    relative, uniformly silent audio (e.g. all zeros) is kept whole, as in librosa.
    """
    if not math.isfinite(top_db) or top_db <= 0:
        raise ValueError(f"top_db must be a positive number of dB, got {top_db!r}")
    frame_length = _positive_int(frame_length, "frame_length")
    hop_length = _positive_int(hop_length, "hop_length")

    x = audio.tensor
    half = frame_length // 2
    frames = torch.nn.functional.pad(x, (half, half)).unfold(-1, frame_length, hop_length)
    power = frames.pow(2).mean(dim=-1)  # (channels, n_frames) mean square = rms**2

    amin_power = 1e-10  # librosa's amin (1e-5) squared
    ref = power.max().clamp(min=amin_power)
    db = 10.0 * torch.log10(power.clamp(min=amin_power)) - 10.0 * torch.log10(ref)
    loud = (db.amax(dim=0) > -top_db).nonzero().flatten()
    if loud.numel() == 0:
        return 0, 0
    start = int(loud[0]) * hop_length
    end = min(audio.num_samples, (int(loud[-1]) + 1) * hop_length)
    return start, end


def trim_silence(
    audio: Audio, top_db: float = 60.0, frame_length: int = 2048, hop_length: int = 512
) -> Audio:
    """Remove leading and trailing silence, like ``librosa.effects.trim``.

    Args:
        audio: Input audio.
        top_db: Frames more than this many dB below the loudest frame are silence.
        frame_length: Analysis frame length in samples.
        hop_length: Hop between frames in samples (the trim resolution).
    """
    start, end = nonsilent_range(audio, top_db, frame_length, hop_length)
    return audio.replace(tensor=audio.tensor[:, start:end])


def pad(audio: Audio, start: float = 0.0, end: float = 0.0) -> Audio:
    """Add ``start`` seconds of silence before and ``end`` seconds after."""
    before = _samples(audio, start, "start")
    after = _samples(audio, end, "end")
    if before == after == 0:
        return audio
    return audio.replace(tensor=torch.nn.functional.pad(audio.tensor, (before, after)))


def crop(audio: Audio, start: float = 0.0, end: float | None = None) -> Audio:
    """Keep the audio between ``start`` and ``end`` seconds.

    ``end=None`` keeps everything after ``start``. An ``end`` past the end of
    the audio is clamped, like slicing.

    Raises:
        ValueError: ``start`` is past the end of the audio or after ``end``.
    """
    lo = _samples(audio, start, "start")
    hi = audio.num_samples if end is None else _samples(audio, end, "end")
    if end is not None and hi < lo:
        raise ValueError(f"end ({end} s) must not be before start ({start} s)")
    if lo > audio.num_samples:
        raise ValueError(f"start ({start} s) is past the end of the audio ({audio.duration:.3f} s)")
    return audio.replace(tensor=audio.tensor[:, lo : min(hi, audio.num_samples)])


def fix_length(
    audio: Audio,
    seconds: float,
    mode: Literal["pad_or_crop", "pad", "crop"] = "pad_or_crop",
) -> Audio:
    """Make audio exactly ``seconds`` long by zero-padding the end and/or cropping it.

    Args:
        seconds: Target duration.
        mode: ``"pad_or_crop"`` (default) does whichever is needed; ``"pad"``
            only pads (raises if the audio is longer); ``"crop"`` only crops
            (raises if shorter).
    """
    if mode not in ("pad_or_crop", "pad", "crop"):
        raise ValueError(f"mode must be 'pad_or_crop', 'pad' or 'crop', got {mode!r}")
    target = _samples(audio, seconds, "seconds")
    n = audio.num_samples
    if n == target:
        return audio
    if n < target:
        if mode == "crop":
            raise ValueError(
                f"audio is {audio.duration:.3f} s, shorter than {seconds} s (mode='crop')"
            )
        return audio.replace(tensor=torch.nn.functional.pad(audio.tensor, (0, target - n)))
    if mode == "pad":
        raise ValueError(f"audio is {audio.duration:.3f} s, longer than {seconds} s (mode='pad')")
    return audio.replace(tensor=audio.tensor[:, :target])


def chunk(
    audio: Audio,
    seconds: float,
    hop: float | None = None,
    last: Literal["pad", "keep", "drop"] = "pad",
) -> list[Audio]:
    """Split into windows of ``seconds``, starting every ``hop`` seconds.

    Args:
        seconds: Window length.
        hop: Step between window starts; defaults to ``seconds`` (no overlap).
        last: What to do with a final window that runs past the end:
            ``"pad"`` (default) zero-pads it to full length so every chunk has
            the same shape, ``"keep"`` keeps it shorter, ``"drop"`` discards it.
    """
    if last not in ("pad", "keep", "drop"):
        raise ValueError(f"last must be 'pad', 'keep' or 'drop', got {last!r}")
    size = _samples(audio, seconds, "seconds")
    step = size if hop is None else _samples(audio, hop, "hop")
    if size == 0 or step == 0:
        raise ValueError("chunk length and hop must be at least one sample")

    n = audio.num_samples
    chunks = []
    for begin in range(0, max(n, 1), step):
        piece = audio.tensor[:, begin : begin + size]
        if piece.shape[1] < size:
            if last == "drop" or (piece.shape[1] == 0 and n > 0):
                break
            if last == "pad":
                piece = torch.nn.functional.pad(piece, (0, size - piece.shape[1]))
        chunks.append(audio.replace(tensor=piece))
        if begin + size >= n:
            break
    return chunks
