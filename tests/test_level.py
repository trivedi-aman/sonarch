from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import sonarch as sa
from tests.conftest import tones


@pytest.mark.parametrize("rate", [16_000, 44_100, 48_000])
@pytest.mark.parametrize("channels", [1, 2, 5])
def test_loudness_matches_pyloudnorm(rate: int, channels: int) -> None:
    pyln = pytest.importorskip("pyloudnorm")
    x = tones(2.5, rate, channels=channels)
    x[:, : x.shape[1] // 4] *= 1e-3  # quiet intro exercises the gating
    expected = pyln.Meter(rate).integrated_loudness(x.T.astype(np.float64))
    assert sa.Audio(x, rate).loudness == pytest.approx(expected, abs=1e-9)


def test_loudness_reference_tone() -> None:
    # BS.1770: a 0 dBFS 997 Hz sine in one channel reads about -3.01 LKFS.
    t = np.arange(48_000 * 5) / 48_000
    sine = np.sin(2 * np.pi * 997 * t).astype(np.float32)
    level = sa.Audio(np.stack([sine, np.zeros_like(sine)]), 48_000).loudness
    assert level == pytest.approx(-3.01, abs=0.05)


def test_loudness_of_51_ignores_lfe() -> None:
    x = tones(1.0, 48_000, channels=6)
    with_lfe = sa.Audio(x, 48_000).loudness
    x[3] = 0.0
    assert sa.Audio(x, 48_000).loudness == pytest.approx(with_lfe)


def test_loudness_silence_is_minus_inf() -> None:
    assert sa.Audio(torch.zeros(2, 48_000), 48_000).loudness == -math.inf


@pytest.mark.parametrize(
    ("shape", "match"), [((2, 1_000), "at least 0.4"), ((8, 48_000), "1-6 channels")]
)
def test_loudness_rejects(shape: tuple[int, int], match: str) -> None:
    with pytest.raises(sa.AudioValidationError, match=match):
        _ = sa.Audio(torch.zeros(shape), 48_000).loudness


@pytest.mark.parametrize("target", [-23.0, -16.0, -30.0])
def test_loudness_normalize_hits_target(target: float) -> None:
    audio = sa.Audio(tones(3.0, 48_000) * 0.1, 48_000).loudness_normalize(target)
    assert audio.loudness == pytest.approx(target, abs=1e-6)


def test_loudness_normalize_silence_unchanged() -> None:
    audio = sa.Audio(torch.zeros(1, 48_000), 48_000)
    assert audio.loudness_normalize() is audio


def test_loudness_on_other_devices_and_dtypes() -> None:
    x = torch.from_numpy(tones(1.0, 48_000))
    reference = sa.Audio(x, 48_000).loudness
    assert sa.Audio(x.double(), 48_000).loudness == pytest.approx(reference, abs=1e-5)
    if torch.backends.mps.is_available():
        assert sa.Audio(x, 48_000).to("mps").loudness == pytest.approx(reference, abs=1e-9)


def test_peak_and_peak_normalize() -> None:
    audio = sa.Audio(torch.tensor([[0.1, -0.5], [0.2, 0.25]]), 8_000)
    assert audio.peak == 0.5
    torch.testing.assert_close(audio.peak_normalize().tensor.abs().max(), torch.tensor(1.0))
    assert audio.peak_normalize(-6.0).peak == pytest.approx(10 ** (-6 / 20))


def test_rms_and_rms_normalize() -> None:
    audio = sa.Audio(torch.full((2, 100), 0.5), 8_000)
    assert audio.rms == pytest.approx(0.5)
    assert audio.rms_normalize(-20.0).rms == pytest.approx(0.1)


@pytest.mark.parametrize("method", ["peak_normalize", "rms_normalize"])
def test_normalize_silence_unchanged(method: str) -> None:
    audio = sa.Audio(torch.zeros(1, 100), 8_000)
    assert getattr(audio, method)() is audio


def test_empty_audio_levels() -> None:
    audio = sa.Audio(torch.zeros(1, 0), 8_000)
    assert audio.peak == 0.0
    assert audio.rms == 0.0
    assert audio.remove_dc() is audio


def test_remove_dc() -> None:
    x = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -1.0, -1.0]])
    out = sa.Audio(x, 8_000).remove_dc()
    torch.testing.assert_close(out.tensor, torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "loud", True])
def test_bad_targets(bad: object) -> None:
    audio = sa.Audio(tones(1.0, 48_000), 48_000)
    for method in ("peak_normalize", "rms_normalize", "loudness_normalize"):
        with pytest.raises(ValueError):
            getattr(audio, method)(bad)
