"""The Rust engine must return the same audio as the FFmpeg path, and fall back cleanly."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import sonarch as sa
from tests.conftest import SR, encode_with_pyav, tones

pytest.importorskip("sonarch._core", reason="Rust engine not built")


def _both(path: Path, **kwargs: object) -> tuple[sa.Audio, sa.Audio]:
    native = sa.load(path, backend="native", **kwargs)  # type: ignore[arg-type]
    ffmpeg = sa.load(path, backend="ffmpeg", **kwargs)  # type: ignore[arg-type]
    return native, ffmpeg


@pytest.mark.parametrize(
    ("name", "channels", "subtype"),
    [
        ("pcm16.wav", 2, "PCM_16"),
        ("pcm24.wav", 2, "PCM_24"),
        ("pcm32.wav", 1, "PCM_32"),
        ("float.wav", 2, "FLOAT"),
        ("pcm8.wav", 1, "PCM_U8"),
        ("six.wav", 6, "PCM_16"),
        ("pcm16.flac", 2, "PCM_16"),
        ("pcm24.flac", 1, "PCM_24"),
        ("six.flac", 6, "PCM_16"),
        ("pcm16.aiff", 2, "PCM_16"),
    ],
)
def test_lossless_is_bit_exact(tmp_path: Path, name: str, channels: int, subtype: str) -> None:
    path = tmp_path / name
    sf.write(path, tones(1.0, channels=channels).T, SR, subtype=subtype)
    native, ffmpeg = _both(path)
    assert native.sample_rate == ffmpeg.sample_rate == SR
    assert native.shape == ffmpeg.shape
    np.testing.assert_array_equal(native.numpy(), ffmpeg.numpy())


@pytest.mark.parametrize("rate", [22_050, 44_100, 48_000])
def test_mp3_matches_ffmpeg(tmp_path: Path, rate: int) -> None:
    path = tmp_path / "a.mp3"
    encode_with_pyav(path, tones(2.0, rate), rate, "libmp3lame")
    native, ffmpeg = _both(path)
    assert native.shape == ffmpeg.shape
    np.testing.assert_allclose(native.numpy(), ffmpeg.numpy(), atol=1e-5)


@pytest.mark.parametrize("duration", [0.1, 0.5, 1.3, 10.0, 1 / 3])
def test_duration_matches_ffmpeg(tmp_path: Path, duration: float) -> None:
    path = tmp_path / "a.flac"
    sa.Audio(tones(1.5), SR).save(path)
    native, ffmpeg = _both(path, duration=duration)
    assert native.shape == ffmpeg.shape
    np.testing.assert_array_equal(native.numpy(), ffmpeg.numpy())


def test_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.wav"
    sa.Audio(torch.zeros(2, 0), SR).save(path)
    native, ffmpeg = _both(path)
    assert native.shape == ffmpeg.shape == (2, 0)


def test_returns_writable_float32(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    sa.Audio(tones(0.2), SR).save(path)
    audio = sa.load(path, backend="native")
    assert audio.dtype == torch.float32
    audio.tensor.mul_(0.5)  # must not raise: memory is owned and writable


class TestFallback:
    def test_opus_in_ogg_falls_back(self, tmp_path: Path) -> None:
        path = tmp_path / "a.ogg"
        encode_with_pyav(path, tones(1.0, 48_000), 48_000, "libopus", "ogg")
        assert sa.load(path).sample_rate == 48_000
        with pytest.raises(sa.DecodeError, match="native engine"):
            sa.load(path, backend="native")

    def test_m4a_uses_ffmpeg(self, tmp_path: Path) -> None:
        path = tmp_path / "a.m4a"
        encode_with_pyav(path, tones(1.0), SR, "aac", "ipod")
        assert sa.load(path).channels == 2
        with pytest.raises(ValueError, match="backend='native'"):
            sa.load(path, backend="native")

    def test_offset_uses_ffmpeg(self, tmp_path: Path) -> None:
        path = tmp_path / "a.wav"
        sa.Audio(tones(1.0), SR).save(path)
        with pytest.raises(ValueError, match="offset=0"):
            sa.load(path, offset=0.5, backend="native")

    def test_file_object_uses_ffmpeg(self, tmp_path: Path) -> None:
        path = tmp_path / "a.wav"
        sa.Audio(tones(0.2), SR).save(path)
        with open(path, "rb") as f:
            assert sa.load(f).num_samples == round(0.2 * SR)

    def test_corrupt_file_raises_decode_error(self, tmp_path: Path) -> None:
        path = tmp_path / "junk.mp3"
        path.write_bytes(b"\x00not audio at all" * 100)
        with pytest.raises(sa.DecodeError):
            sa.load(path)

    def test_bad_backend(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="backend"):
            sa.load(tmp_path / "a.wav", backend="gpu")  # type: ignore[arg-type]

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sa.load(tmp_path / "nope.flac")


class TestLoadMany:
    @pytest.fixture
    def mixed(self, tmp_path: Path) -> list[Path]:
        paths = []
        for i in range(12):
            path = tmp_path / f"clip{i}.flac"
            sa.Audio(tones(0.2 + 0.05 * i, channels=1 + i % 2), SR).save(path)
            paths.append(path)
        opus = tmp_path / "x.ogg"
        encode_with_pyav(opus, tones(0.5, 48_000), 48_000, "libopus", "ogg")
        m4a = tmp_path / "y.m4a"
        encode_with_pyav(m4a, tones(0.5), SR, "aac", "ipod")
        mp3 = tmp_path / "z.mp3"
        encode_with_pyav(mp3, tones(0.5), SR, "libmp3lame")
        return [*paths[:5], opus, *paths[5:], m4a, mp3]

    def test_matches_load_in_order(self, mixed: list[Path]) -> None:
        clips = sa.load_many(mixed)
        assert len(clips) == len(mixed)
        for path, clip in zip(mixed, clips, strict=True):
            single = sa.load(path)
            assert clip.sample_rate == single.sample_rate
            np.testing.assert_array_equal(clip.numpy(), single.numpy())

    @pytest.mark.parametrize("threads", [1, 3])
    def test_num_threads(self, mixed: list[Path], threads: int) -> None:
        assert len(sa.load_many(mixed, num_threads=threads)) == len(mixed)

    def test_duration(self, mixed: list[Path]) -> None:
        for clip in sa.load_many(mixed, duration=0.1):
            assert clip.num_samples == round(0.1 * clip.sample_rate)

    def test_empty(self) -> None:
        assert sa.load_many([]) == []

    def test_missing_file_checked_first(self, mixed: list[Path], tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="nope"):
            sa.load_many([*mixed, tmp_path / "nope.wav"])

    def test_bad_threads(self, mixed: list[Path]) -> None:
        with pytest.raises(ValueError, match="num_threads"):
            sa.load_many(mixed, num_threads=0)
