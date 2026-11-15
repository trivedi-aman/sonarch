from __future__ import annotations

import librosa
import numpy as np
import pytest
import torch

import sonarch as sa
from sonarch.ops.time import nonsilent_range
from tests.conftest import tones


def _with_silence(channels: int, lead: int, tail: int, rate: int = 22_050) -> np.ndarray:
    body = tones(1.0, rate, channels=channels)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal((channels, lead + tail)).astype(np.float32) * 1e-5
    return np.concatenate([noise[:, :lead], body, noise[:, lead:]], axis=1)


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize(("lead", "tail"), [(0, 0), (5_000, 3_000), (12_345, 777)])
@pytest.mark.parametrize("top_db", [20.0, 60.0])
def test_trim_matches_librosa(channels: int, lead: int, tail: int, top_db: float) -> None:
    x = _with_silence(channels, lead, tail)
    _, (start, end) = librosa.effects.trim(x, top_db=top_db)
    assert nonsilent_range(sa.Audio(x, 22_050), top_db=top_db) == (start, end)
    trimmed = sa.Audio(x, 22_050).trim_silence(top_db=top_db)
    np.testing.assert_array_equal(trimmed.numpy(), x[:, start:end])


def test_trim_custom_frames_matches_librosa() -> None:
    x = _with_silence(1, 4_000, 4_000)
    _, (start, end) = librosa.effects.trim(x, top_db=40, frame_length=1024, hop_length=256)
    assert nonsilent_range(sa.Audio(x, 22_050), 40, 1024, 256) == (start, end)


def test_trim_all_zeros_kept_like_librosa() -> None:
    # Levels are relative to the loudest frame, so uniform silence is kept whole.
    x = np.zeros((1, 10_000), dtype=np.float32)
    _, (start, end) = librosa.effects.trim(x)
    assert nonsilent_range(sa.Audio(x, 16_000)) == (start, end) == (0, 10_000)


@pytest.mark.parametrize("kwargs", [{"top_db": 0}, {"frame_length": 0}, {"hop_length": -1}])
def test_trim_bad_args(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        sa.Audio(torch.zeros(1, 100), 16_000).trim_silence(**kwargs)  # type: ignore[arg-type]


def test_pad() -> None:
    audio = sa.Audio(torch.ones(2, 10), 10).pad(start=0.2, end=0.3)
    assert audio.shape == (2, 15)
    assert audio.tensor[:, :2].abs().sum() == 0
    assert audio.tensor[:, -3:].abs().sum() == 0
    same = sa.Audio(torch.ones(1, 10), 10)
    assert same.pad() is same


def test_crop() -> None:
    audio = sa.Audio(torch.arange(100.0), 10)
    torch.testing.assert_close(audio.crop(2.0, 3.5).tensor, torch.arange(20.0, 35.0)[None])
    assert audio.crop(9.0).num_samples == 10
    assert audio.crop(5.0, 50.0).num_samples == 50  # end clamped
    assert audio.crop(10.0).num_samples == 0


@pytest.mark.parametrize(
    ("start", "end", "match"),
    [(3.0, 2.0, "before"), (11.0, None, "past the end"), (-1.0, None, ">= 0")],
)
def test_crop_errors(start: float, end: float | None, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        sa.Audio(torch.zeros(100), 10).crop(start, end)


def test_fix_length() -> None:
    audio = sa.Audio(torch.ones(1, 25), 10)
    assert audio.fix_length(3.0).num_samples == 30
    assert audio.fix_length(2.0).num_samples == 20
    assert audio.fix_length(2.5) is audio
    with pytest.raises(ValueError, match="mode='pad'"):
        audio.fix_length(2.0, mode="pad")
    with pytest.raises(ValueError, match="mode='crop'"):
        audio.fix_length(3.0, mode="crop")
    with pytest.raises(ValueError, match="mode"):
        audio.fix_length(3.0, mode="stretch")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("n", "kwargs", "lengths"),
    [
        (100, {}, [30, 30, 30, 30]),
        (100, {"last": "keep"}, [30, 30, 30, 10]),
        (100, {"last": "drop"}, [30, 30, 30]),
        (90, {}, [30, 30, 30]),
        (100, {"hop": 2.0, "last": "drop"}, [30, 30, 30, 30]),
        (100, {"hop": 2.0}, [30, 30, 30, 30, 30]),
        (10, {}, [30]),
        (0, {}, [30]),
    ],
)
def test_chunk(n: int, kwargs: dict[str, object], lengths: list[int]) -> None:
    data = torch.arange(float(n))[None] if n else torch.zeros(1, 0)
    pieces = sa.Audio(data, 10).chunk(3.0, **kwargs)  # type: ignore[arg-type]
    assert [p.num_samples for p in pieces] == lengths


def test_chunk_content_and_overlap() -> None:
    pieces = sa.Audio(torch.arange(10.0), 1).chunk(4.0, hop=2.0, last="keep")
    assert [p.tensor[0].tolist() for p in pieces] == [
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0, 9.0],
    ]


def test_chunk_bad_args() -> None:
    audio = sa.Audio(torch.zeros(100), 10)
    with pytest.raises(ValueError):
        audio.chunk(0.0)
    with pytest.raises(ValueError):
        audio.chunk(1.0, last="wrap")  # type: ignore[arg-type]
