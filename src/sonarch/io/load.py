"""Decoding audio files into :class:`~sonarch.Audio` objects."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import av
import numpy as np
import torch

from sonarch.core.audio import Audio
from sonarch.core.errors import DecodeError
from sonarch.io._source import Source, describe, first_audio_stream, open_input

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["load"]


def load(
    source: Source,
    *,
    offset: float = 0.0,
    duration: float | None = None,
) -> Audio:
    """Decode an audio file into an :class:`~sonarch.Audio` object.

    Audio keeps its original sample rate and channels; nothing is converted
    behind your back. Use ``.resample()`` / ``.to_mono()`` explicitly.

    Args:
        source: Local path, URL (``http(s)://``...), or binary file object.
            Any format FFmpeg can decode works, including the audio track of
            video files.
        offset: Start reading this many seconds into the file.
        duration: Read at most this many seconds. ``None`` reads to the end.

    Returns:
        Float32 ``Audio`` with shape ``(channels, samples)``.

    Raises:
        FileNotFoundError: A local path does not exist.
        DecodeError: The input is not decodable audio.
        ValueError: ``offset`` / ``duration`` are invalid, or ``offset`` is
            past the end of the file.

    Example:
        >>> audio = sa.load("speech.mp3", offset=1.5, duration=3.0)
    """
    if not math.isfinite(offset) or offset < 0:
        raise ValueError(f"offset must be a finite number >= 0, got {offset}")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError(f"duration must be a finite number > 0 or None, got {duration}")

    with open_input(source) as container:
        stream = first_audio_stream(container, source)
        sample_rate = int(stream.codec_context.sample_rate)
        if sample_rate <= 0:
            raise DecodeError(f"{describe(source)!r} reports an invalid sample rate")

        start = round(offset * sample_rate)
        stop = None if duration is None else start + round(duration * sample_rate)
        # Lossy decoders (MP3, AAC, Opus...) produce wrong samples for a short
        # while after a seek, so seek earlier and discard the warm-up audio.
        preroll = 0 if _is_lossless(stream.codec_context.name) else sample_rate // 2
        seek_to = max(start - preroll, 0)

        try:
            first, data = _decode(container, stream, sample_rate, seek_to, stop)
        except av.error.FFmpegError as exc:
            raise DecodeError(f"failed to decode {describe(source)!r}: {exc}") from exc

    lo = max(start - first, 0)
    hi = data.shape[1] if stop is None else max(stop - first, 0)
    if start > 0 and lo >= data.shape[1]:
        total = (first + data.shape[1]) / sample_rate
        raise ValueError(
            f"offset {offset}s is past the end of {describe(source)!r} ({total:.3f}s long)"
        )
    return Audio(torch.from_numpy(np.ascontiguousarray(data[:, lo:hi])), sample_rate)


_LOSSLESS_CODECS = frozenset({"flac", "alac", "wavpack", "tta", "ape", "mlp", "truehd"})


def _is_lossless(codec_name: str) -> bool:
    return codec_name.startswith("pcm_") or codec_name in _LOSSLESS_CODECS


def _decode(
    container: av.container.InputContainer,
    stream: av.audio.stream.AudioStream,
    sample_rate: int,
    start: int,
    stop: int | None,
) -> tuple[int, NDArray[Any]]:
    """Decode samples covering ``[start, stop)``.

    Returns the index of the first decoded sample and a float32 array of shape
    ``(channels, n)`` that begins at that index.
    """
    seeked = start > 0 and _seek(container, stream, start / sample_rate)
    resampler = av.AudioResampler(format="fltp")  # keep rate & layout, convert to float32
    stream_start = stream.start_time or 0

    chunks: list[NDArray[Any]] = []
    first: int | None = None
    position = 0
    for frame in container.decode(stream):
        if first is None:
            if seeked and frame.pts is None:
                # Can't tell where the seek landed; restart from the beginning.
                return _decode_from_start(container, stream, sample_rate, start, stop)
            first = (
                round(float((frame.pts - stream_start) * frame.time_base) * sample_rate)
                if seeked and frame.pts is not None
                else 0
            )
            first = max(first, 0)
            position = first
        for out in resampler.resample(frame):
            chunks.append(out.to_ndarray())
        position += frame.samples
        if stop is not None and position >= stop:
            break
    for out in resampler.resample(None):
        chunks.append(out.to_ndarray())

    channels = stream.codec_context.channels or 1
    if not chunks:
        return (first or 0), np.zeros((channels, 0), dtype=np.float32)
    return (first or 0), np.concatenate(chunks, axis=1)


def _decode_from_start(
    container: av.container.InputContainer,
    stream: av.audio.stream.AudioStream,
    sample_rate: int,
    start: int,
    stop: int | None,
) -> tuple[int, NDArray[Any]]:
    container.seek(0, stream=stream)
    return _decode(container, stream, sample_rate, 0, stop)


def _seek(
    container: av.container.InputContainer,
    stream: av.audio.stream.AudioStream,
    seconds: float,
) -> bool:
    """Seek to (at or before) ``seconds``. Returns False if the input can't seek."""
    if stream.time_base is None:
        return False
    target = int(seconds / stream.time_base) + (stream.start_time or 0)
    try:
        container.seek(target, stream=stream, backward=True, any_frame=False)
    except av.error.FFmpegError:
        return False  # nothing decoded yet, so we're still at the start
    return True
