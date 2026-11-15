"""Shared fixtures: test signals and reference files written by independent tools."""

from __future__ import annotations

import wave
from collections.abc import Callable
from pathlib import Path

import av
import numpy as np
import pytest
import soundfile as sf

SR = 44_100


def tones(seconds: float = 2.0, sample_rate: int = SR, channels: int = 2) -> np.ndarray:
    """Deterministic tonal test signal, shape (channels, samples), float32 in [-0.6, 0.6].

    Tonal rather than noise on purpose: lossy codecs (AAC in particular) may
    resynthesize noise randomly, which makes decodes non-reproducible.
    """
    t = np.arange(round(seconds * sample_rate)) / sample_rate
    freqs = [440.0, 554.4, 659.3, 880.0, 1234.5, 3000.0, 220.0, 330.0]
    rows = [
        0.4 * np.sin(2 * np.pi * freqs[c % len(freqs)] * t)
        + 0.2 * np.sin(2 * np.pi * 2.5 * freqs[c % len(freqs)] * t)
        for c in range(channels)
    ]
    return np.stack(rows).astype(np.float32)


def write_wav_stdlib(path: Path, data: np.ndarray, sample_rate: int) -> np.ndarray:
    """Write 16-bit PCM WAV with the stdlib `wave` module.

    Returns the exact float32 values a correct decoder must produce.
    """
    pcm = np.clip(np.round(data * 32767), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(data.shape[0])
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.T.tobytes())
    return pcm.astype(np.float32) / 32768


def encode_with_pyav(
    path: Path, data: np.ndarray, sample_rate: int, codec: str, fmt: str | None = None
) -> None:
    layout = {1: "mono", 2: "stereo"}.get(data.shape[0], f"{data.shape[0]}c")
    with av.open(str(path), "w", format=fmt) as container:
        stream = container.add_stream(codec, rate=sample_rate, layout=layout)
        frame = av.AudioFrame.from_ndarray(data, format="fltp", layout=layout)
        frame.sample_rate = sample_rate
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


@pytest.fixture
def signal() -> np.ndarray:
    return tones()


@pytest.fixture
def wav_file(tmp_path: Path, signal: np.ndarray) -> tuple[Path, np.ndarray]:
    path = tmp_path / "ref.wav"
    expected = write_wav_stdlib(path, signal, SR)
    return path, expected


@pytest.fixture
def flac_file(tmp_path: Path, signal: np.ndarray) -> tuple[Path, np.ndarray]:
    path = tmp_path / "ref.flac"
    sf.write(path, signal.T, SR, subtype="PCM_16")
    expected, _ = sf.read(path, dtype="float32", always_2d=True)
    return path, expected.T


LOSSY = {
    "mp3": ("libmp3lame", SR, None),
    "m4a": ("aac", SR, "ipod"),
    "opus": ("libopus", 48_000, "ogg"),
}


@pytest.fixture(params=sorted(LOSSY))
def lossy_file(request: pytest.FixtureRequest, tmp_path: Path) -> tuple[Path, int]:
    ext = request.param
    codec, rate, fmt = LOSSY[ext]
    path = tmp_path / f"ref.{ext}"
    encode_with_pyav(path, tones(3.0, rate), rate, codec, fmt)
    return path, rate


MakeFile = Callable[[str, bytes], Path]


@pytest.fixture
def make_file(tmp_path: Path) -> MakeFile:
    def _make(name: str, content: bytes) -> Path:
        path = tmp_path / name
        path.write_bytes(content)
        return path

    return _make
