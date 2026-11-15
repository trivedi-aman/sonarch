"""Spectral features: STFT, mel spectrogram, log-mel, MFCC.

Defaults and maths follow librosa 1.0 (`librosa.stft`, `librosa.filters.mel`,
`librosa.feature.melspectrogram`, `librosa.power_to_db`,
`librosa.feature.mfcc`), and results match it to float32 precision. Everything
is PyTorch, so features are computed on whatever device the audio is on, and
batched over channels.

Functions return tensors shaped ``(channels, freq_or_coeff, frames)``. The
classes (``STFT``, ``MelSpectrogram``, ``LogMel``, ``MFCC``) hold settings and
are called on audio: ``sa.MelSpectrogram(n_mels=80)(audio)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
import torch

from sonarch.core.audio import Audio

__all__ = [
    "MFCC",
    "STFT",
    "LogMel",
    "MelSpectrogram",
    "log_mel",
    "mel_filterbank",
    "melspectrogram",
    "mfcc",
    "power_to_db",
    "stft",
]

Norm = Literal["slaney"] | None


# --------------------------------------------------------------------------- STFT


def stft(
    audio: Audio,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    center: bool = True,
) -> torch.Tensor:
    """Complex STFT, shape ``(channels, 1 + n_fft // 2, frames)``. Matches ``librosa.stft``.

    Uses a periodic Hann window and, with ``center=True``, zero-pads
    ``n_fft // 2`` samples on both sides (librosa's default ``pad_mode``).
    """
    n_fft = _positive_int(n_fft, "n_fft")
    hop = n_fft // 4 if hop_length is None else _positive_int(hop_length, "hop_length")
    win = n_fft if win_length is None else _positive_int(win_length, "win_length")
    if win > n_fft:
        raise ValueError(f"win_length ({win}) must be <= n_fft ({n_fft})")

    x = audio.tensor
    if not x.is_floating_point() or x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    if center:
        x = torch.nn.functional.pad(x, (n_fft // 2, n_fft // 2))
    if x.shape[-1] < n_fft:
        raise ValueError(
            f"audio is too short for n_fft={n_fft}: {audio.num_samples} samples"
            + ("" if center else " (try center=True)")
        )
    window = torch.hann_window(win, periodic=True, dtype=x.dtype, device=x.device)
    return torch.stft(
        x, n_fft, hop_length=hop, win_length=win, window=window, center=False, return_complex=True
    )


# ---------------------------------------------------------------------- mel scale


def _hz_to_mel(freqs: np.ndarray, htk: bool) -> np.ndarray:
    if htk:
        return 2595.0 * np.log10(1.0 + freqs / 700.0)
    f_sp = 200.0 / 3
    mels = freqs / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = freqs >= min_log_hz
    mels[log_region] = min_log_mel + np.log(freqs[log_region] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray, htk: bool) -> np.ndarray:
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    freqs[log_region] = min_log_hz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freqs


@lru_cache(maxsize=64)
def _mel_filterbank_np(
    sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float, htk: bool, norm: Norm
) -> np.ndarray:
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mel_points = np.linspace(
        _hz_to_mel(np.array([fmin], dtype=np.float64), htk)[0],
        _hz_to_mel(np.array([fmax], dtype=np.float64), htk)[0],
        n_mels + 2,
    )
    mel_f = _mel_to_hz(mel_points, htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_freqs)
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    if norm == "slaney":
        weights *= (2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels]))[:, None]
    result: np.ndarray = np.asarray(weights, dtype=np.float32)
    result.setflags(write=False)
    return result


def mel_filterbank(
    sample_rate: int,
    n_fft: int = 2048,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: Norm = "slaney",
) -> torch.Tensor:
    """Mel filterbank, shape ``(n_mels, 1 + n_fft // 2)``. Matches ``librosa.filters.mel``."""
    fmax = sample_rate / 2.0 if fmax is None else float(fmax)
    if not 0 <= fmin < fmax:
        raise ValueError(f"need 0 <= fmin < fmax, got fmin={fmin}, fmax={fmax}")
    if norm not in ("slaney", None):
        raise ValueError(f"norm must be 'slaney' or None, got {norm!r}")
    weights = _mel_filterbank_np(
        int(sample_rate),
        _positive_int(n_fft, "n_fft"),
        _positive_int(n_mels, "n_mels"),
        float(fmin),
        fmax,
        bool(htk),
        norm,
    )
    return torch.from_numpy(weights.copy())


@lru_cache(maxsize=64)
def _filterbank_on(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float | None,
    htk: bool,
    norm: Norm,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, htk, norm).to(device, dtype)


# ----------------------------------------------------------------- spectrograms


def melspectrogram(
    audio: Audio,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    center: bool = True,
    power: float = 2.0,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: Norm = "slaney",
) -> torch.Tensor:
    """Mel spectrogram, shape ``(channels, n_mels, frames)``. Matches
    ``librosa.feature.melspectrogram`` (``power=2.0`` is a power spectrogram,
    ``1.0`` a magnitude spectrogram)."""
    if not power > 0:
        raise ValueError(f"power must be > 0, got {power}")
    spec = stft(audio, n_fft, hop_length, win_length, center).abs()
    if power != 1.0:
        spec = spec.pow(power)
    bank = _filterbank_on(
        audio.sample_rate, n_fft, n_mels, fmin, fmax, htk, norm, spec.device, spec.dtype
    )
    return torch.matmul(bank, spec)


def power_to_db(
    spec: torch.Tensor, ref: float = 1.0, amin: float = 1e-10, top_db: float | None = 80.0
) -> torch.Tensor:
    """``10 * log10(spec / ref)``, floored at ``amin`` and, if ``top_db`` is set,
    at ``top_db`` below each channel's peak. Matches ``librosa.power_to_db``."""
    if amin <= 0:
        raise ValueError(f"amin must be > 0, got {amin}")
    db = 10.0 * torch.log10(spec.clamp(min=amin)) - 10.0 * math.log10(max(amin, abs(ref)))
    if top_db is not None:
        if top_db < 0:
            raise ValueError(f"top_db must be >= 0, got {top_db}")
        dims = (-2, -1) if db.ndim >= 2 else (-1,)
        db = torch.maximum(db, db.amax(dim=dims, keepdim=True) - top_db)
    return db


def log_mel(audio: Audio, top_db: float | None = 80.0, **kwargs: object) -> torch.Tensor:
    """``power_to_db(melspectrogram(audio, **kwargs))`` in dB."""
    return power_to_db(melspectrogram(audio, **kwargs), top_db=top_db)  # type: ignore[arg-type]


@lru_cache(maxsize=32)
def _dct_matrix(n_in: int, n_out: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Orthonormal DCT-II as a matrix (rows = output coefficients), like scipy norm='ortho'."""
    n = torch.arange(n_in, dtype=torch.float64)
    k = torch.arange(n_out, dtype=torch.float64)[:, None]
    basis = torch.cos(math.pi / n_in * (n + 0.5) * k) * math.sqrt(2.0 / n_in)
    basis[0] /= math.sqrt(2.0)
    return basis.to(device, dtype)


def mfcc(
    audio: Audio,
    n_mfcc: int = 20,
    lifter: float = 0.0,
    top_db: float | None = 80.0,
    **kwargs: object,
) -> torch.Tensor:
    """MFCCs, shape ``(channels, n_mfcc, frames)``. Matches ``librosa.feature.mfcc``
    (DCT-II, ``norm='ortho'``). Extra keyword arguments go to :func:`melspectrogram`."""
    db = log_mel(audio, top_db=top_db, **kwargs)
    n_mels = db.shape[-2]
    if not 0 < _positive_int(n_mfcc, "n_mfcc") <= n_mels:
        raise ValueError(f"n_mfcc must be between 1 and n_mels ({n_mels}), got {n_mfcc}")
    coeffs = torch.matmul(_dct_matrix(n_mels, n_mfcc, db.device, db.dtype), db)
    if lifter > 0:
        k = torch.arange(1, n_mfcc + 1, dtype=db.dtype, device=db.device)
        coeffs = coeffs * (1 + (lifter / 2) * torch.sin(math.pi * k / lifter))[:, None]
    elif lifter < 0:
        raise ValueError(f"lifter must be >= 0, got {lifter}")
    return coeffs


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


# ------------------------------------------------------------------- op classes


@dataclass(frozen=True)
class STFT:
    """``STFT(n_fft=512)(audio)`` -> complex tensor ``(channels, freq, frames)``."""

    n_fft: int = 2048
    hop_length: int | None = None
    win_length: int | None = None
    center: bool = True

    def __call__(self, audio: Audio) -> torch.Tensor:
        return stft(audio, self.n_fft, self.hop_length, self.win_length, self.center)


@dataclass(frozen=True)
class MelSpectrogram:
    """``MelSpectrogram(n_mels=80)(audio)`` -> tensor ``(channels, n_mels, frames)``."""

    n_fft: int = 2048
    hop_length: int = 512
    win_length: int | None = None
    center: bool = True
    power: float = 2.0
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float | None = None
    htk: bool = False
    norm: Norm = "slaney"

    def __call__(self, audio: Audio) -> torch.Tensor:
        return melspectrogram(audio, **self.__dict__)


@dataclass(frozen=True)
class LogMel(MelSpectrogram):
    """``LogMel(n_mels=80)(audio)`` -> log-mel spectrogram in dB."""

    top_db: float | None = 80.0

    def __call__(self, audio: Audio) -> torch.Tensor:
        settings = dict(self.__dict__)
        top_db = settings.pop("top_db")
        return power_to_db(melspectrogram(audio, **settings), top_db=top_db)


@dataclass(frozen=True)
class MFCC(MelSpectrogram):
    """``MFCC(n_mfcc=13)(audio)`` -> tensor ``(channels, n_mfcc, frames)``."""

    n_mfcc: int = 20
    lifter: float = 0.0
    top_db: float | None = 80.0

    def __call__(self, audio: Audio) -> torch.Tensor:
        settings = dict(self.__dict__)
        extras = {key: settings.pop(key) for key in ("n_mfcc", "lifter", "top_db")}
        return mfcc(audio, **extras, **settings)
