"""Decoding audio files into :class:`~sonarch.Audio` objects."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import av
import numpy as np
import torch

from sonarch.core.audio import Audio
from sonarch.core.errors import DecodeError
from sonarch.io._source import Source, _is_url, describe, first_audio_stream, open_input

try:
    from sonarch import _core
except ImportError:  # pragma: no cover - built without the Rust extension
    _core = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["load", "load_many"]

Backend = Literal["auto", "native", "ffmpeg"]

# Formats the Rust engine decodes exactly like FFmpeg (gapless, bit-exact for
# PCM/FLAC). Everything else (AAC/M4A, Opus, video...) goes through FFmpeg.
# .ogg may hold Opus; the engine reports that as unsupported and we fall back.
_NATIVE_EXTENSIONS = frozenset(
    {".wav", ".wave", ".flac", ".mp3", ".aif", ".aiff", ".aifc", ".ogg", ".oga"}
)


def load(
    source: Source,
    *,
    offset: float = 0.0,
    duration: float | None = None,
    backend: Backend = "auto",
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
        backend: ``"auto"`` (default) uses the fast Rust engine for local
            WAV/AIFF/FLAC/MP3/Vorbis files and FFmpeg for everything else.
            ``"native"`` requires the Rust engine; ``"ffmpeg"`` always uses
            FFmpeg. Both engines return the same samples.

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
    _validate(offset, duration, backend)

    if backend != "ffmpeg":
        path = _native_candidate(source, offset)
        if path is None and backend == "native":
            raise ValueError(
                "backend='native' needs a local WAV/AIFF/FLAC/MP3/OGG file, offset=0 and the "
                "compiled engine; use backend='auto' to fall back to FFmpeg automatically"
            )
        if path is not None:
            try:
                array, rate = _core.decode(path, duration)
                return _to_audio(array, rate)
            except (_core.UnsupportedError, _core.NativeDecodeError) as exc:
                if backend == "native":
                    raise DecodeError(f"native engine failed on {path!r}: {exc}") from exc
                # FFmpeg handles more edge cases; let it try.

    return _load_ffmpeg(source, offset, duration)


def load_many(
    sources: Sequence[str | os.PathLike[str]],
    *,
    duration: float | None = None,
    num_threads: int | None = None,
) -> list[Audio]:
    """Decode many files in parallel, using every CPU core.

    Decoding runs in Rust with Python's GIL released, so this scales with
    cores instead of being stuck on one. Files the engine doesn't handle
    (AAC, Opus...) are decoded with FFmpeg instead, transparently.

    Args:
        sources: Local file paths.
        duration: Read at most this many seconds of each file.
        num_threads: Worker threads. ``None`` uses one per CPU core.

    Returns:
        One ``Audio`` per path, in the same order.

    Raises:
        FileNotFoundError: Any path does not exist (checked before decoding).
        DecodeError: A file could not be decoded by either engine.

    Example:
        >>> clips = sa.load_many(sorted(Path("data").glob("*.flac")))
    """
    _validate(0.0, duration, "auto")
    if num_threads is not None and num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")

    paths = [os.fspath(s) for s in sources]
    for path in paths:
        if not _is_url(path) and not Path(path).exists():
            raise FileNotFoundError(f"no such file: {path!r}")

    results: list[Audio | None] = [None] * len(paths)
    native = [i for i, p in enumerate(paths) if _native_candidate(p, 0.0) is not None]
    if native:
        decoded = _core.decode_many([paths[i] for i in native], duration, num_threads)
        for i, item in zip(native, decoded, strict=True):
            if item[0] is not None:
                results[i] = _to_audio(item[0], item[1])

    for i, audio in enumerate(results):
        if audio is None:
            results[i] = _load_ffmpeg(paths[i], 0.0, duration)
    return [audio for audio in results if audio is not None]


def _validate(offset: float, duration: float | None, backend: str) -> None:
    if not math.isfinite(offset) or offset < 0:
        raise ValueError(f"offset must be a finite number >= 0, got {offset}")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError(f"duration must be a finite number > 0 or None, got {duration}")
    if backend not in ("auto", "native", "ffmpeg"):
        raise ValueError(f"backend must be 'auto', 'native' or 'ffmpeg', got {backend!r}")


def _native_candidate(source: Source, offset: float) -> str | None:
    """Return the path if the Rust engine should decode ``source``, else None."""
    if _core is None or offset != 0 or not isinstance(source, (str, os.PathLike)):
        return None
    path = os.fspath(source)
    if _is_url(path) or Path(path).suffix.lower() not in _NATIVE_EXTENSIONS:
        return None
    if not Path(path).is_file():
        raise FileNotFoundError(f"no such file: {path!r}")
    return path


def _to_audio(array: NDArray[Any], sample_rate: int) -> Audio:
    return Audio(torch.from_numpy(array), sample_rate)


def _load_ffmpeg(source: Source, offset: float, duration: float | None) -> Audio:
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
