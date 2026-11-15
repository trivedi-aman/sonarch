"""Exception hierarchy for sonarch.

Every error sonarch raises on purpose derives from :class:`SonarchError`, so
callers can catch everything from this library with a single ``except``.
Validation errors also derive from :class:`ValueError` so generic handlers
keep working.
"""

from __future__ import annotations

__all__ = [
    "AudioValidationError",
    "ChannelMismatchError",
    "ClippingError",
    "DecodeError",
    "EncodeError",
    "LengthMismatchError",
    "SampleRateMismatchError",
    "SonarchError",
]


class SonarchError(Exception):
    """Base class for all errors raised by sonarch."""


class AudioValidationError(SonarchError, ValueError):
    """Audio data or metadata is invalid (bad shape, dtype, sample rate...)."""


class SampleRateMismatchError(AudioValidationError):
    """Two pieces of audio, or audio and a consumer, disagree on sample rate."""

    def __init__(self, expected: int, got: int, hint: str | None = None) -> None:
        self.expected = expected
        self.got = got
        message = f"sample rate mismatch: expected {expected} Hz, got {got} Hz"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)


class ChannelMismatchError(AudioValidationError):
    """Two pieces of audio disagree on channel count."""

    def __init__(self, expected: int, got: int, hint: str | None = None) -> None:
        self.expected = expected
        self.got = got
        message = f"channel mismatch: expected {expected} channel(s), got {got}"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)


class LengthMismatchError(AudioValidationError):
    """Two pieces of audio disagree on length (in samples)."""

    def __init__(self, expected: int, got: int, hint: str | None = None) -> None:
        self.expected = expected
        self.got = got
        message = f"length mismatch: expected {expected} samples, got {got}"
        if hint:
            message = f"{message}. {hint}"
        super().__init__(message)


class ClippingError(AudioValidationError):
    """Audio exceeds [-1, 1] and would be clipped by an integer encoding."""


class DecodeError(SonarchError):
    """A file or stream could not be decoded as audio."""


class EncodeError(SonarchError):
    """Audio could not be encoded or written."""
