"""The :class:`Audio` object: a waveform that carries its own metadata."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch

from sonarch.core.errors import (
    AudioValidationError,
    ChannelMismatchError,
    LengthMismatchError,
    SampleRateMismatchError,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["Audio", "mix"]

_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)


def _as_tensor(data: torch.Tensor | NDArray[Any]) -> torch.Tensor:
    """Convert input to a float tensor of shape (channels, samples), or raise."""
    if isinstance(data, np.ndarray):
        if not np.issubdtype(data.dtype, np.floating):
            raise AudioValidationError(
                f"audio data must be floating point in [-1, 1], got dtype {data.dtype}. "
                "Convert integer PCM first, e.g. `data.astype(np.float32) / 32768` for int16."
            )
        tensor = torch.from_numpy(np.ascontiguousarray(data))
    elif isinstance(data, torch.Tensor):
        tensor = data
        if tensor.dtype not in _FLOAT_DTYPES:
            raise AudioValidationError(
                f"audio data must be floating point in [-1, 1], got dtype {tensor.dtype}. "
                "Convert integer PCM first, e.g. `x.float() / 32768` for int16."
            )
    else:
        raise AudioValidationError(
            f"audio data must be a torch.Tensor or numpy.ndarray, got {type(data).__name__}"
        )

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 2:
        raise AudioValidationError(
            f"audio data must have shape (samples,) or (channels, samples), "
            f"got {tuple(tensor.shape)}"
        )

    channels, samples = tensor.shape
    if channels == 0:
        raise AudioValidationError("audio must have at least one channel")
    if samples > 0 and channels > samples:
        raise AudioValidationError(
            f"got shape {tuple(tensor.shape)}, which looks like (samples, channels). "
            "sonarch uses (channels, samples): pass `data.T` instead."
        )
    return tensor


def _check_sample_rate(sample_rate: object) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise AudioValidationError(
            f"sample_rate must be an integer number of Hz, got {sample_rate!r}"
        )
    rate = int(sample_rate)
    if rate <= 0:
        raise AudioValidationError(f"sample_rate must be positive, got {rate}")
    return rate


class Audio:
    """A waveform plus the metadata needed to use it correctly.

    The waveform is always a floating-point tensor of shape
    ``(channels, samples)`` with values nominally in ``[-1, 1]``. Operations
    that combine audio check sample rate, channels and length, and raise
    instead of silently producing wrong results.

    Args:
        data: Waveform as a ``torch.Tensor`` or ``numpy.ndarray``, shape
            ``(samples,)`` or ``(channels, samples)``. NumPy input is wrapped
            without copying when possible.
        sample_rate: Sample rate in Hz.

    Example:
        >>> audio = Audio(torch.zeros(2, 16_000), sample_rate=16_000)
        >>> audio.duration
        1.0
    """

    __slots__ = ("_sample_rate", "_tensor")

    def __init__(self, data: torch.Tensor | NDArray[Any], sample_rate: int) -> None:
        self._tensor = _as_tensor(data)
        self._sample_rate = _check_sample_rate(sample_rate)

    # ------------------------------------------------------------------ metadata

    @property
    def tensor(self) -> torch.Tensor:
        """The waveform, shape ``(channels, samples)``."""
        return self._tensor

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """Number of channels."""
        return int(self._tensor.shape[0])

    @property
    def num_samples(self) -> int:
        """Number of samples per channel."""
        return int(self._tensor.shape[1])

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.num_samples / self._sample_rate

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the waveform, ``(channels, samples)``."""
        return (self.channels, self.num_samples)

    @property
    def device(self) -> torch.device:
        """Device the waveform lives on."""
        return self._tensor.device

    @property
    def dtype(self) -> torch.dtype:
        """Floating-point dtype of the waveform."""
        return self._tensor.dtype

    # --------------------------------------------------------------- conversion

    def replace(
        self,
        *,
        tensor: torch.Tensor | NDArray[Any] | None = None,
        sample_rate: int | None = None,
    ) -> Audio:
        """Return a new ``Audio`` with some fields replaced.

        Only use ``sample_rate=`` when relabelling audio whose rate was wrong;
        to change the rate of real audio, resample it.
        """
        return Audio(
            self._tensor if tensor is None else tensor,
            self._sample_rate if sample_rate is None else sample_rate,
        )

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Audio:
        """Move to a device and/or cast to another floating-point dtype."""
        if dtype is not None and dtype not in _FLOAT_DTYPES:
            raise AudioValidationError(f"dtype must be a floating-point dtype, got {dtype}")
        return self.replace(tensor=self._tensor.to(device=device, dtype=dtype))

    def cpu(self) -> Audio:
        """Shortcut for ``audio.to("cpu")``."""
        return self.to("cpu")

    def numpy(self) -> NDArray[Any]:
        """Return the waveform as a NumPy array of shape ``(channels, samples)``.

        bfloat16 audio is returned as float32, since NumPy has no bfloat16.
        """
        tensor = self._tensor.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return tensor.numpy()

    def resample(
        self, sample_rate: int, *, quality: Literal["fast", "high", "best"] = "high"
    ) -> Audio:
        """Resample to ``sample_rate`` Hz. See :func:`sonarch.ops.resample`.

        Runs in the Rust engine for CPU float32 audio and in PyTorch on GPU.
        Returns ``self`` if the rate already matches.
        """
        from sonarch.ops.resample import resample

        return resample(self, sample_rate, quality=quality)

    def to_mono(self) -> Audio:
        """Average all channels into one. Mono audio is returned unchanged."""
        if self.channels == 1:
            return self
        return self.replace(tensor=self._tensor.mean(dim=0, keepdim=True))

    def to_stereo(self) -> Audio:
        """Duplicate mono audio into two channels. Stereo audio is returned unchanged."""
        if self.channels == 2:
            return self
        if self.channels != 1:
            raise ChannelMismatchError(
                1, self.channels, hint="to_stereo() only converts mono audio; downmix first"
            )
        return self.replace(tensor=self._tensor.expand(2, -1).clone())

    def save(self, path: str | os.PathLike[str], **kwargs: Any) -> None:
        """Write to a file. See :func:`sonarch.save` for options."""
        from sonarch.io import save

        save(self, path, **kwargs)

    # ------------------------------------------------------------------ dunders

    def __len__(self) -> int:
        return self.num_samples

    def __repr__(self) -> str:
        device = "" if self.device.type == "cpu" else f", device={self.device}"
        return (
            f"Audio(channels={self.channels}, sample_rate={self._sample_rate}, "
            f"duration={self.duration:.3f}s, dtype={str(self.dtype).removeprefix('torch.')}"
            f"{device})"
        )

    def __add__(self, other: object) -> Audio:
        if not isinstance(other, Audio):
            return NotImplemented
        _check_compatible(self, other, check_length=True)
        return self.replace(tensor=self._tensor + other._tensor)

    __hash__ = None  # type: ignore[assignment]  # mutable tensor inside: not hashable


def _check_compatible(a: Audio, b: Audio, *, check_length: bool) -> None:
    if a.sample_rate != b.sample_rate:
        raise SampleRateMismatchError(
            a.sample_rate, b.sample_rate, hint="Resample one clip to match the other"
        )
    if a.channels != b.channels:
        raise ChannelMismatchError(
            a.channels, b.channels, hint="Use to_mono() or to_stereo() to match channels"
        )
    if check_length and a.num_samples != b.num_samples:
        raise LengthMismatchError(
            a.num_samples,
            b.num_samples,
            hint="Use sa.mix(..., align='pad') or align='crop' to combine clips of "
            "different lengths",
        )


def mix(
    *clips: Audio,
    align: Literal["error", "pad", "crop"] = "error",
    weights: list[float] | None = None,
) -> Audio:
    """Sum several clips into one.

    All clips must share sample rate, channel count and device.

    Args:
        *clips: Two or more ``Audio`` objects.
        align: What to do when lengths differ. ``"error"`` (default) raises,
            ``"pad"`` zero-pads shorter clips to the longest, ``"crop"`` cuts
            every clip to the shortest.
        weights: Optional gain for each clip, e.g. ``[1.0, 0.3]``.

    Note:
        The result is a plain sum, so it can exceed ``[-1, 1]``. Normalize
        afterwards if needed.
    """
    if len(clips) < 2:
        raise ValueError(f"mix() needs at least two clips, got {len(clips)}")
    if align not in ("error", "pad", "crop"):
        raise ValueError(f"align must be 'error', 'pad' or 'crop', got {align!r}")
    if weights is not None and len(weights) != len(clips):
        raise ValueError(f"got {len(weights)} weights for {len(clips)} clips")

    first = clips[0]
    for clip in clips[1:]:
        _check_compatible(first, clip, check_length=align == "error")

    lengths = [clip.num_samples for clip in clips]
    target = max(lengths) if align == "pad" else min(lengths)
    gains = weights if weights is not None else [1.0] * len(clips)

    out = torch.zeros((first.channels, target), dtype=first.dtype, device=first.device)
    for clip, gain in zip(clips, gains, strict=True):
        n = min(clip.num_samples, target)
        out[:, :n] += gain * clip.tensor[:, :n]
    return first.replace(tensor=out)
