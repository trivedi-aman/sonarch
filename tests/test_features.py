from __future__ import annotations

import warnings

import librosa
import numpy as np
import pytest
import torch

import sonarch as sa
from sonarch.ops import features as F
from tests.conftest import tones

warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture
def signal() -> np.ndarray:
    x = tones(1.5, 22_050, channels=2)
    x[1] *= 0.01  # different levels per channel exercise per-channel top_db
    return x


def _close(ours: torch.Tensor, ref: np.ndarray, rel: float = 1e-5) -> None:
    out = ours.numpy()
    assert out.shape == ref.shape
    assert np.abs(out - ref).max() <= rel * max(np.abs(ref).max(), 1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"n_fft": 512, "hop_length": 128, "win_length": 400}, {"n_fft": 1024, "center": False}],
)
def test_stft(signal: np.ndarray, kwargs: dict[str, object]) -> None:
    _close(F.stft(sa.Audio(signal, 22_050), **kwargs), librosa.stft(signal, **kwargs))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sr": 22_050, "n_fft": 2048},
        {"sr": 16_000, "n_fft": 400, "n_mels": 80},
        {"sr": 16_000, "n_fft": 512, "n_mels": 40, "fmin": 20.0, "fmax": 7_600.0, "htk": True},
        {"sr": 44_100, "n_fft": 2048, "n_mels": 64, "norm": None},
    ],
)
def test_mel_filterbank(kwargs: dict[str, object]) -> None:
    ours = dict(kwargs)
    rate = ours.pop("sr")
    _close(F.mel_filterbank(rate, **ours), librosa.filters.mel(**kwargs))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs", [{}, {"n_mels": 80, "n_fft": 1024, "hop_length": 256}, {"power": 1.0, "n_mels": 40}]
)
def test_melspectrogram(signal: np.ndarray, kwargs: dict[str, object]) -> None:
    ref = librosa.feature.melspectrogram(y=signal, sr=22_050, **kwargs)  # type: ignore[arg-type]
    _close(F.melspectrogram(sa.Audio(signal, 22_050), **kwargs), ref)  # type: ignore[arg-type]
    _close(sa.MelSpectrogram(**kwargs)(sa.Audio(signal, 22_050)), ref)  # type: ignore[arg-type]


@pytest.mark.parametrize("top_db", [80.0, 40.0, None])
def test_log_mel(signal: np.ndarray, top_db: float | None) -> None:
    ref = librosa.power_to_db(librosa.feature.melspectrogram(y=signal, sr=22_050), top_db=top_db)
    ours = sa.LogMel(top_db=top_db)(sa.Audio(signal, 22_050)).numpy()
    # Without a floor, very quiet bins are float32 rounding noise; compare
    # everything within 80 dB (librosa's default floor) of each channel's peak.
    relevant = ref > ref.max(axis=(-2, -1), keepdims=True) - 80
    assert np.abs(ours - ref)[relevant].max() < 5e-3  # dB


@pytest.mark.parametrize(
    "kwargs", [{}, {"n_mfcc": 13}, {"n_mfcc": 13, "lifter": 22}, {"n_mfcc": 40, "n_mels": 40}]
)
def test_mfcc(signal: np.ndarray, kwargs: dict[str, object]) -> None:
    ref = librosa.feature.mfcc(y=signal, sr=22_050, **kwargs)  # type: ignore[arg-type]
    ours = sa.MFCC(**kwargs)(sa.Audio(signal, 22_050)).numpy()  # type: ignore[arg-type]
    assert ours.shape == ref.shape
    assert np.abs(ours - ref).max() < 1e-2


def test_mono_shapes() -> None:
    audio = sa.Audio(tones(1.0, 16_000, channels=1), 16_000)
    assert sa.MelSpectrogram(n_mels=80, n_fft=400, hop_length=160)(audio).shape == (1, 80, 101)
    assert sa.MFCC(n_mfcc=13)(audio).shape[:2] == (1, 13)


def test_float64_and_float16_inputs() -> None:
    x = torch.from_numpy(tones(0.5, 16_000))
    ref = sa.MelSpectrogram()(sa.Audio(x, 16_000))
    assert sa.MelSpectrogram()(sa.Audio(x.double(), 16_000)).dtype == torch.float64
    half = sa.MelSpectrogram()(sa.Audio(x.half(), 16_000))
    assert half.dtype == torch.float32
    torch.testing.assert_close(half, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_matches_cpu() -> None:
    audio = sa.Audio(tones(1.0, 16_000), 16_000)
    cpu = sa.LogMel(n_mels=80)(audio)
    gpu = sa.LogMel(n_mels=80)(audio.to("mps"))
    assert gpu.device.type == "mps"
    torch.testing.assert_close(gpu.cpu(), cpu, atol=1e-3, rtol=0)


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda a: F.stft(a, n_fft=0), "n_fft"),
        (lambda a: F.stft(a, n_fft=256, win_length=512), "win_length"),
        (lambda a: F.stft(sa.Audio(torch.zeros(10), 16_000), n_fft=512, center=False), "too short"),
        (lambda a: F.mel_filterbank(16_000, fmin=9_000.0), "fmin"),
        (lambda a: F.mel_filterbank(16_000, norm="area"), "norm"),  # type: ignore[arg-type]
        (lambda a: F.melspectrogram(a, power=0.0), "power"),
        (lambda a: F.mfcc(a, n_mfcc=200), "n_mfcc"),
        (lambda a: F.power_to_db(torch.ones(2, 2), amin=0.0), "amin"),
    ],
)
def test_errors(call: object, match: str) -> None:
    audio = sa.Audio(tones(0.5, 16_000), 16_000)
    with pytest.raises(ValueError, match=match):
        call(audio)  # type: ignore[operator]


def test_op_classes_are_frozen_settings() -> None:
    op = sa.MFCC(n_mfcc=13)
    with pytest.raises(AttributeError):
        op.n_mfcc = 20  # type: ignore[misc]
    assert op == sa.MFCC(n_mfcc=13)
