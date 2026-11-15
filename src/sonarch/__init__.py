"""sonarch: the one import you need for audio work.

>>> import sonarch as sa
>>> audio = sa.load("clip.flac")
>>> audio.sample_rate, audio.channels, audio.duration
"""

from importlib.metadata import PackageNotFoundError, version

from sonarch import ops
from sonarch.core import (
    Audio,
    AudioValidationError,
    ChannelMismatchError,
    ClippingError,
    DecodeError,
    EncodeError,
    LengthMismatchError,
    SampleRateMismatchError,
    SonarchError,
    mix,
)
from sonarch.io import AudioInfo, info, load, load_many, save
from sonarch.ops import MFCC, STFT, LogMel, MelSpectrogram

try:
    __version__ = version("sonarch")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+unknown"

__all__ = [
    "MFCC",
    "STFT",
    "Audio",
    "AudioInfo",
    "AudioValidationError",
    "ChannelMismatchError",
    "ClippingError",
    "DecodeError",
    "EncodeError",
    "LengthMismatchError",
    "LogMel",
    "MelSpectrogram",
    "SampleRateMismatchError",
    "SonarchError",
    "__version__",
    "info",
    "load",
    "load_many",
    "mix",
    "ops",
    "save",
]
