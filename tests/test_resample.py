from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import sonarch as sa
from sonarch.ops.resample import QUALITIES, resample_tensor
from tests.conftest import SR, encode_with_pyav, tones

RATIOS = [(48_000, 16_000), (44_100, 16_000), (16_000, 44_100), (22_050, 48_000), (8_000, 8_001)]


def _torchaudio_kwargs(quality: str) -> dict[str, object]:
    lpw, rolloff, window, beta = QUALITIES[quality]
    kwargs: dict[str, object] = {
        "lowpass_filter_width": lpw,
        "rolloff": rolloff,
        "resampling_method": f"sinc_interp_{window}",
    }
    if beta is not None:
        kwargs["beta"] = beta
    return kwargs


@pytest.mark.parametrize(("orig", "new"), RATIOS)
@pytest.mark.parametrize("quality", ["fast", "high", "best"])
def test_matches_torchaudio(orig: int, new: int, quality: str) -> None:
    functional = pytest.importorskip("torchaudio.functional")
    if not hasattr(functional, "resample"):
        pytest.skip("this torchaudio version has no functional.resample")
    x = torch.from_numpy(tones(0.5, orig))
    reference = functional.resample(x.double(), orig, new, **_torchaudio_kwargs(quality)).float()
    out = sa.Audio(x, orig).resample(new, quality=quality)  # type: ignore[arg-type]
    assert out.sample_rate == new
    assert out.shape == tuple(reference.shape)
    torch.testing.assert_close(out.tensor, reference, atol=1e-6, rtol=0)


@pytest.mark.parametrize(("orig", "new"), RATIOS)
@pytest.mark.parametrize("quality", ["fast", "high"])
def test_rust_matches_torch_path(orig: int, new: int, quality: str) -> None:
    x = torch.from_numpy(tones(0.5, orig, channels=3))
    rust = sa.Audio(x, orig).resample(new, quality=quality).tensor  # type: ignore[arg-type]
    torch_path = resample_tensor(x, orig, new, quality=quality)  # type: ignore[arg-type]
    torch.testing.assert_close(rust, torch_path, atol=1e-6, rtol=0)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 7, 100, 44_101])
def test_output_length_is_ceil(n: int) -> None:
    out = sa.Audio(torch.zeros(1, n), 44_100).resample(16_000)
    assert out.num_samples == math.ceil(n * 16_000 / 44_100)


def test_preserves_in_band_tone() -> None:
    t = np.arange(48_000) / 48_000
    x = np.sin(2 * np.pi * 1_000 * t).astype(np.float32)
    out = sa.Audio(x, 48_000).resample(16_000).numpy()[0]
    expected = np.sin(2 * np.pi * 1_000 * np.arange(out.size) / 16_000)
    middle = slice(200, -200)  # ignore filter edge effects
    np.testing.assert_allclose(out[middle], expected[middle], atol=2e-3)


@pytest.mark.parametrize(("quality", "min_db"), [("fast", 30), ("high", 70), ("best", 90)])
def test_rejects_aliasing(quality: str, min_db: float) -> None:
    """A 12 kHz tone can't exist at 16 kHz; it must be filtered, not folded to 4 kHz."""
    t = np.arange(48_000 * 2) / 48_000
    x = np.sin(2 * np.pi * 12_000 * t).astype(np.float32)
    out = sa.Audio(x, 48_000).resample(16_000, quality=quality).numpy()[0]  # type: ignore[arg-type]
    rms = float(np.sqrt(np.mean(out[1000:-1000] ** 2)))
    attenuation_db = 20 * np.log10((1 / np.sqrt(2)) / max(rms, 1e-12))
    assert attenuation_db > min_db


def test_same_rate_is_noop() -> None:
    audio = sa.Audio(torch.zeros(10), 16_000)
    assert audio.resample(16_000) is audio


@pytest.mark.parametrize("dtype", [torch.float64, torch.float16])
def test_other_dtypes_use_torch_path(dtype: torch.dtype) -> None:
    x = torch.from_numpy(tones(0.2, 48_000)).to(dtype)
    out = sa.Audio(x, 48_000).resample(16_000)
    assert out.dtype == dtype
    reference = sa.Audio(x.float(), 48_000).resample(16_000).tensor
    torch.testing.assert_close(
        out.tensor.float(), reference, atol=2e-3 if dtype == torch.float16 else 1e-6, rtol=0
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_matches_cpu() -> None:
    x = torch.from_numpy(tones(0.5, 44_100))
    cpu = sa.Audio(x, 44_100).resample(16_000).tensor
    mps = sa.Audio(x, 44_100).to("mps").resample(16_000)
    assert mps.device.type == "mps"
    torch.testing.assert_close(mps.tensor.cpu(), cpu, atol=1e-5, rtol=0)


@pytest.mark.parametrize("bad", [0, -16_000, 16_000.0, True])
def test_bad_rate(bad: object) -> None:
    with pytest.raises(sa.AudioValidationError):
        sa.Audio(torch.zeros(10), 8_000).resample(bad)  # type: ignore[arg-type]


def test_bad_quality() -> None:
    with pytest.raises(ValueError, match="quality"):
        sa.Audio(torch.zeros(10), 8_000).resample(16_000, quality="ultra")  # type: ignore[arg-type]


class TestLoadConversion:
    @pytest.fixture
    def files(self, tmp_path: Path) -> list[Path]:
        flac = tmp_path / "a.flac"
        sa.Audio(tones(1.0, 48_000), 48_000).save(flac)
        mp3 = tmp_path / "b.mp3"
        encode_with_pyav(mp3, tones(1.0, SR), SR, "libmp3lame")
        m4a = tmp_path / "c.m4a"
        encode_with_pyav(m4a, tones(1.0, SR), SR, "aac", "ipod")
        return [flac, mp3, m4a]

    def test_load_converts(self, files: list[Path]) -> None:
        for path in files:
            audio = sa.load(path, sample_rate=16_000, mono=True)
            expected = sa.load(path).to_mono().resample(16_000)
            assert audio.sample_rate == 16_000
            assert audio.channels == 1
            torch.testing.assert_close(audio.tensor, expected.tensor)

    @pytest.mark.parametrize("mono", [False, True])
    def test_load_many_matches_load(self, files: list[Path], mono: bool) -> None:
        clips = sa.load_many(files, sample_rate=16_000, mono=mono, num_threads=2)
        for path, clip in zip(files, clips, strict=True):
            single = sa.load(path, sample_rate=16_000, mono=mono)
            assert clip.sample_rate == 16_000
            assert clip.shape == single.shape
            torch.testing.assert_close(clip.tensor, single.tensor, atol=1e-6, rtol=0)

    def test_load_many_duration_then_resample(self, files: list[Path]) -> None:
        for clip in sa.load_many(files, duration=0.5, sample_rate=16_000):
            assert clip.num_samples == 8_000

    @pytest.mark.parametrize(
        "kwargs", [{"sample_rate": 0}, {"sample_rate": 16_000.0}, {"quality": "ultra"}]
    )
    def test_bad_arguments(self, files: list[Path], kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            sa.load(files[0], **kwargs)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            sa.load_many(files, **kwargs)  # type: ignore[arg-type]
