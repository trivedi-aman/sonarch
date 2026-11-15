"""Level operations: peak / RMS / loudness measurement and normalization, DC removal.

Gain changes are plain tensor maths, so they run on any device. Loudness
(LUFS, ITU-R BS.1770-4) is measured by the Rust engine and matches pyloudnorm.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from sonarch.core.audio import Audio
from sonarch.core.errors import AudioValidationError

try:
    from sonarch import _core
except ImportError:  # pragma: no cover - built without the Rust extension
    _core = None  # type: ignore[assignment]

__all__ = [
    "loudness",
    "loudness_normalize",
    "peak",
    "peak_normalize",
    "remove_dc",
    "rms",
    "rms_normalize",
]


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def _check_db(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number of dB, got {value!r}")
    return float(value)


def peak(audio: Audio) -> float:
    """Largest absolute sample value across all channels (linear, 1.0 = 0 dBFS)."""
    if audio.num_samples == 0:
        return 0.0
    return float(audio.tensor.abs().max())


def rms(audio: Audio) -> float:
    """Root-mean-square level across all channels (linear)."""
    if audio.num_samples == 0:
        return 0.0
    return float(audio.tensor.double().pow(2).mean().sqrt())


def loudness(audio: Audio) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770-4 / EBU R128), like pyloudnorm.

    Channel order for weighting: up to 5 channels are [L, R, C, Ls, Rs];
    6 channels are 5.1 [L, R, C, LFE, Ls, Rs] with the LFE ignored.

    Returns ``-inf`` for silence.

    Raises:
        AudioValidationError: Audio is shorter than 0.4 s or has more than 6 channels.
    """
    if _core is None:  # pragma: no cover
        raise RuntimeError("loudness() needs sonarch's compiled engine")
    array = np.ascontiguousarray(audio.tensor.detach().to("cpu", torch.float32).numpy())
    try:
        return float(_core.loudness(array, audio.sample_rate))
    except ValueError as exc:
        raise AudioValidationError(str(exc)) from None


def peak_normalize(audio: Audio, target_db: float = 0.0) -> Audio:
    """Scale so the peak sample is at ``target_db`` dBFS (0 dBFS = 1.0).

    Silent audio is returned unchanged.
    """
    target = _check_db("target_db", target_db)
    current = peak(audio)
    if current == 0.0:
        return audio
    return audio.replace(tensor=audio.tensor * (_db_to_gain(target) / current))


def rms_normalize(audio: Audio, target_db: float = -20.0) -> Audio:
    """Scale so the RMS level (over all channels) is ``target_db`` dBFS.

    Silent audio is returned unchanged. The result can exceed [-1, 1].
    """
    target = _check_db("target_db", target_db)
    current = rms(audio)
    if current == 0.0:
        return audio
    return audio.replace(tensor=audio.tensor * (_db_to_gain(target) / current))


def loudness_normalize(audio: Audio, target_lufs: float = -23.0) -> Audio:
    """Scale to an integrated loudness of ``target_lufs`` (EBU R128 default: -23).

    Silent audio (``-inf`` LUFS) is returned unchanged. The result can exceed
    [-1, 1]; saving it to an integer format then raises ``ClippingError``
    rather than clipping silently.
    """
    target = _check_db("target_lufs", target_lufs)
    current = loudness(audio)
    if not math.isfinite(current):
        return audio
    return audio.replace(tensor=audio.tensor * _db_to_gain(target - current))


def remove_dc(audio: Audio) -> Audio:
    """Subtract each channel's mean, removing any constant (DC) offset."""
    if audio.num_samples == 0:
        return audio
    return audio.replace(tensor=audio.tensor - audio.tensor.mean(dim=1, keepdim=True))
