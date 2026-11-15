"""Band-limited resampling.

CPU float32 audio is resampled by the Rust engine; GPU tensors (and other
dtypes) use an equivalent PyTorch implementation. Both use the same
windowed-sinc filter design as ``torchaudio.functional.resample``, so results
agree with torchaudio and with each other to float32 precision.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import numpy as np
import torch

from sonarch.core.audio import Audio
from sonarch.core.errors import AudioValidationError

try:
    from sonarch import _core
except ImportError:  # pragma: no cover - built without the Rust extension
    _core = None  # type: ignore[assignment]

__all__ = ["QUALITIES", "Quality", "resample", "resample_tensor"]

Quality = Literal["fast", "high", "best"]

# (lowpass_filter_width, rolloff, window, kaiser beta). Must match rust/src/dsp.rs.
QUALITIES: dict[str, tuple[int, float, str, float | None]] = {
    "fast": (6, 0.99, "hann", None),  # torchaudio's default
    "high": (16, 0.945, "kaiser", 8.5551783568115),
    "best": (64, 0.9475937167399596, "kaiser", 14.769656459379492),  # kaiser_best
}


def resample(audio: Audio, sample_rate: int, *, quality: Quality = "high") -> Audio:
    """Resample ``audio`` to ``sample_rate`` Hz.

    Output length is ``ceil(num_samples * sample_rate / audio.sample_rate)``,
    the same as torchaudio.

    Args:
        audio: Input audio, on any device.
        sample_rate: Target rate in Hz.
        quality: ``"fast"`` (torchaudio's default filter), ``"high"``
            (default; steeper anti-aliasing), or ``"best"`` (librosa's former
            ``kaiser_best``; slowest).
    """
    target = _check_rate(sample_rate)
    if quality not in QUALITIES:
        raise ValueError(f"quality must be 'fast', 'high' or 'best', got {quality!r}")
    if target == audio.sample_rate:
        return audio

    tensor = audio.tensor
    if _core is not None and tensor.device.type == "cpu" and tensor.dtype == torch.float32:
        array = np.ascontiguousarray(tensor.detach().numpy())
        out = _core.resample(array, audio.sample_rate, target, quality)
        return Audio(torch.from_numpy(out), target)
    return Audio(resample_tensor(tensor, audio.sample_rate, target, quality=quality), target)


def resample_tensor(
    waveform: torch.Tensor, orig_rate: int, new_rate: int, *, quality: Quality = "high"
) -> torch.Tensor:
    """PyTorch implementation for tensors of shape ``(..., samples)`` on any device."""
    if orig_rate == new_rate:
        return waveform
    gcd = math.gcd(orig_rate, new_rate)
    orig, new = orig_rate // gcd, new_rate // gcd

    compute_dtype = (
        waveform.dtype if waveform.dtype in (torch.float32, torch.float64) else torch.float32
    )
    kernel, width = _kernel(orig, new, quality, waveform.device, compute_dtype)

    shape = waveform.shape
    x = waveform.reshape(-1, 1, shape[-1]).to(compute_dtype)
    n = shape[-1]
    x = torch.nn.functional.pad(x, (width, width + orig))
    y = torch.nn.functional.conv1d(x, kernel, stride=orig)  # (batch, new, blocks)
    y = y.transpose(1, 2).reshape(x.shape[0], -1)
    out_len = math.ceil(new * n / orig)
    return y[:, :out_len].reshape(*shape[:-1], out_len).to(waveform.dtype)


@lru_cache(maxsize=32)
def _kernel(
    orig: int, new: int, quality: str, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, int]:
    lpw, rolloff, window, beta = QUALITIES[quality]
    base_freq = min(orig, new) * rolloff
    width = math.ceil(lpw * orig / base_freq)
    idx = torch.arange(-width, width + orig, dtype=torch.float64) / orig
    shift = -torch.arange(new, dtype=torch.float64)[:, None] / new
    t = ((shift + idx) * base_freq).clamp(-lpw, lpw)
    if window == "hann":
        win = torch.cos(t * math.pi / lpw / 2) ** 2
    else:
        assert beta is not None
        beta_t = torch.tensor(beta, dtype=torch.float64)
        win = torch.special.i0(
            beta_t * torch.sqrt((1 - (t / lpw) ** 2).clamp(min=0))
        ) / torch.special.i0(beta_t)
    x = t * math.pi
    sinc = torch.where(x == 0, torch.ones_like(x), torch.sin(x) / x)
    kernel = (sinc * win * (base_freq / orig)).unsqueeze(1)  # (new, 1, taps)
    return kernel.to(device=device, dtype=dtype), width


def _check_rate(sample_rate: object) -> int:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, (int, np.integer)):
        raise AudioValidationError(
            f"sample_rate must be an integer number of Hz, got {sample_rate!r}"
        )
    if int(sample_rate) <= 0:
        raise AudioValidationError(f"sample_rate must be positive, got {sample_rate}")
    return int(sample_rate)
