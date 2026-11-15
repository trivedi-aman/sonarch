"""Core types: the Audio object and the error hierarchy."""

from sonarch.core.audio import Audio, mix
from sonarch.core.errors import (
    AudioValidationError,
    ChannelMismatchError,
    ClippingError,
    DecodeError,
    EncodeError,
    LengthMismatchError,
    SampleRateMismatchError,
    SonarchError,
)

__all__ = [
    "Audio",
    "AudioValidationError",
    "ChannelMismatchError",
    "ClippingError",
    "DecodeError",
    "EncodeError",
    "LengthMismatchError",
    "SampleRateMismatchError",
    "SonarchError",
    "mix",
]
