"""Fast metadata probing without decoding."""

from __future__ import annotations

from dataclasses import dataclass

import av

from sonarch.io._source import Source, first_audio_stream, open_input

__all__ = ["AudioInfo", "info"]


@dataclass(frozen=True)
class AudioInfo:
    """Metadata about an audio file, read from its headers.

    Attributes:
        sample_rate: Sample rate in Hz.
        channels: Number of channels.
        duration: Duration in seconds, or ``None`` if the container does not
            say. For compressed formats (MP3, AAC...) this is an estimate from
            the headers and can differ slightly from the decoded length.
        num_samples: ``duration * sample_rate`` rounded, or ``None``.
        codec: Codec name, e.g. ``"pcm_s16le"``, ``"flac"``, ``"mp3"``.
        format: Container format, e.g. ``"wav"``, ``"flac"``, ``"mp3"``.
        bit_rate: Bit rate in bits/s, or ``None`` if unknown.
    """

    sample_rate: int
    channels: int
    duration: float | None
    num_samples: int | None
    codec: str
    format: str
    bit_rate: int | None


def info(source: Source) -> AudioInfo:
    """Read metadata from an audio file without decoding it.

    Args:
        source: Local path, URL, or binary file object.

    Example:
        >>> sa.info("clip.flac")
        AudioInfo(sample_rate=44100, channels=2, duration=12.4, ...)
    """
    with open_input(source) as container:
        stream = first_audio_stream(container, source)
        codec = stream.codec_context
        sample_rate = int(codec.sample_rate)

        duration: float | None = None
        if stream.duration is not None and stream.time_base is not None:
            duration = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration = container.duration / av.time_base

        return AudioInfo(
            sample_rate=sample_rate,
            channels=int(codec.channels),
            duration=duration,
            num_samples=None if duration is None else round(duration * sample_rate),
            codec=codec.name,
            format=container.format.name.split(",")[0],
            bit_rate=(codec.bit_rate or container.bit_rate) or None,
        )
