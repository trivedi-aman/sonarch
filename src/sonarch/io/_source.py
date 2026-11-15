"""Opening audio sources with PyAV, with sonarch's error handling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO, Any, Union, cast

import av
import av.container

from sonarch.core.errors import DecodeError

Source = Union[str, "os.PathLike[str]", IO[bytes]]


def describe(source: Source) -> str:
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)
    return getattr(source, "name", None) or type(source).__name__


def _is_url(path: str) -> bool:
    return "://" in path


def open_input(source: Source) -> av.container.InputContainer:
    """Open a path, URL or binary file object for reading.

    Raises:
        FileNotFoundError: A local path does not exist.
        DecodeError: FFmpeg cannot open or understand the input.
    """
    target: Any
    if isinstance(source, (str, os.PathLike)):
        target = os.fspath(source)
        if not _is_url(target) and not Path(target).exists():
            raise FileNotFoundError(f"no such file: {target!r}")
    else:
        target = source

    try:
        return cast(av.container.InputContainer, av.open(target, mode="r"))
    except av.error.FFmpegError as exc:
        raise DecodeError(f"could not open {describe(source)!r} as audio: {exc}") from exc


def first_audio_stream(
    container: av.container.InputContainer, source: Source
) -> av.audio.stream.AudioStream:
    if not container.streams.audio:
        raise DecodeError(f"{describe(source)!r} has no audio stream")
    return container.streams.audio[0]
