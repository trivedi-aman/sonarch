from __future__ import annotations

import numpy as np
import pytest
import torch

import sonarch as sa


class TestConstruction:
    def test_1d_becomes_mono(self) -> None:
        audio = sa.Audio(torch.zeros(100), 16_000)
        assert audio.shape == (1, 100)
        assert audio.channels == 1

    def test_numpy_is_wrapped_without_copy(self) -> None:
        data = np.zeros((2, 100), dtype=np.float32)
        audio = sa.Audio(data, 16_000)
        data[0, 0] = 0.5
        assert audio.tensor[0, 0].item() == 0.5

    def test_keeps_float_dtype(self) -> None:
        assert sa.Audio(np.zeros(10, dtype=np.float64), 8_000).dtype == torch.float64

    @pytest.mark.parametrize(
        "data", [np.zeros(10, dtype=np.int16), torch.zeros(10, dtype=torch.int32)]
    )
    def test_rejects_integer_pcm(self, data: object) -> None:
        with pytest.raises(sa.AudioValidationError, match="floating point"):
            sa.Audio(data, 16_000)  # type: ignore[arg-type]

    def test_rejects_3d(self) -> None:
        with pytest.raises(sa.AudioValidationError, match="shape"):
            sa.Audio(torch.zeros(1, 2, 100), 16_000)

    def test_rejects_samples_first_layout(self) -> None:
        with pytest.raises(sa.AudioValidationError, match=r"data\.T"):
            sa.Audio(torch.zeros(1000, 2), 16_000)

    def test_allows_empty(self) -> None:
        audio = sa.Audio(torch.zeros(2, 0), 16_000)
        assert audio.duration == 0.0

    def test_rejects_zero_channels(self) -> None:
        with pytest.raises(sa.AudioValidationError, match="at least one channel"):
            sa.Audio(torch.zeros(0, 10), 16_000)

    @pytest.mark.parametrize("rate", [0, -1, 16_000.0, True, "16000", None])
    def test_rejects_bad_sample_rate(self, rate: object) -> None:
        with pytest.raises(sa.AudioValidationError, match="sample_rate"):
            sa.Audio(torch.zeros(10), rate)  # type: ignore[arg-type]

    def test_accepts_numpy_integer_rate(self) -> None:
        assert sa.Audio(torch.zeros(10), np.int64(8_000)).sample_rate == 8_000  # type: ignore[arg-type]

    def test_rejects_other_types(self) -> None:
        with pytest.raises(sa.AudioValidationError):
            sa.Audio([0.0, 0.1], 16_000)  # type: ignore[arg-type]


class TestMetadata:
    def test_properties(self) -> None:
        audio = sa.Audio(torch.zeros(2, 24_000), 16_000)
        assert audio.sample_rate == 16_000
        assert audio.num_samples == len(audio) == 24_000
        assert audio.duration == 1.5
        assert audio.device.type == "cpu"

    def test_repr(self) -> None:
        text = repr(sa.Audio(torch.zeros(2, 16_000), 16_000))
        assert text == "Audio(channels=2, sample_rate=16000, duration=1.000s, dtype=float32)"

    def test_not_hashable(self) -> None:
        with pytest.raises(TypeError):
            hash(sa.Audio(torch.zeros(10), 8_000))


class TestConversion:
    def test_to_mono_averages(self) -> None:
        audio = sa.Audio(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), 8_000).to_mono()
        assert audio.shape == (1, 2)
        torch.testing.assert_close(audio.tensor, torch.tensor([[0.5, 0.5]]))

    def test_to_mono_on_mono_is_identity(self) -> None:
        audio = sa.Audio(torch.zeros(10), 8_000)
        assert audio.to_mono() is audio

    def test_to_stereo(self) -> None:
        audio = sa.Audio(torch.arange(4.0), 8_000).to_stereo()
        assert audio.shape == (2, 4)
        audio.tensor[0, 0] = 9.0  # channels must not share memory
        assert audio.tensor[1, 0] == 0.0

    def test_to_stereo_rejects_multichannel(self) -> None:
        with pytest.raises(sa.ChannelMismatchError):
            sa.Audio(torch.zeros(6, 100), 8_000).to_stereo()

    def test_to_dtype(self) -> None:
        assert sa.Audio(torch.zeros(10), 8_000).to(dtype=torch.float16).dtype == torch.float16

    def test_to_rejects_integer_dtype(self) -> None:
        with pytest.raises(sa.AudioValidationError):
            sa.Audio(torch.zeros(10), 8_000).to(dtype=torch.int16)

    def test_numpy(self) -> None:
        array = sa.Audio(torch.ones(2, 5), 8_000).numpy()
        assert isinstance(array, np.ndarray)
        assert array.shape == (2, 5)

    def test_numpy_from_bfloat16(self) -> None:
        array = sa.Audio(torch.ones(5, dtype=torch.bfloat16), 8_000).numpy()
        assert array.dtype == np.float32

    def test_replace_validates(self) -> None:
        audio = sa.Audio(torch.zeros(10), 8_000)
        assert audio.replace(sample_rate=16_000).sample_rate == 16_000
        with pytest.raises(sa.AudioValidationError):
            audio.replace(sample_rate=0)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
    def test_to_mps(self) -> None:
        audio = sa.Audio(torch.zeros(10), 8_000).to("mps")
        assert audio.device.type == "mps"
        assert audio.numpy().shape == (1, 10)


class TestCombining:
    def test_add(self) -> None:
        a = sa.Audio(torch.ones(2, 4), 8_000)
        torch.testing.assert_close((a + a).tensor, torch.full((2, 4), 2.0))

    def test_add_rate_mismatch(self) -> None:
        with pytest.raises(sa.SampleRateMismatchError, match="16000 Hz"):
            sa.Audio(torch.zeros(10), 8_000) + sa.Audio(torch.zeros(10), 16_000)

    def test_add_channel_mismatch(self) -> None:
        with pytest.raises(sa.ChannelMismatchError):
            sa.Audio(torch.zeros(1, 10), 8_000) + sa.Audio(torch.zeros(2, 10), 8_000)

    def test_add_length_mismatch_suggests_mix(self) -> None:
        with pytest.raises(sa.LengthMismatchError, match=r"sa\.mix"):
            sa.Audio(torch.zeros(10), 8_000) + sa.Audio(torch.zeros(12), 8_000)

    def test_add_non_audio(self) -> None:
        with pytest.raises(TypeError):
            sa.Audio(torch.zeros(10), 8_000) + 1  # type: ignore[operator]

    def test_mix_pad(self) -> None:
        a = sa.Audio(torch.ones(3), 8_000)
        b = sa.Audio(torch.ones(5), 8_000)
        out = sa.mix(a, b, align="pad")
        torch.testing.assert_close(out.tensor, torch.tensor([[2.0, 2.0, 2.0, 1.0, 1.0]]))

    def test_mix_crop_with_weights(self) -> None:
        a = sa.Audio(torch.ones(3), 8_000)
        b = sa.Audio(torch.ones(5), 8_000)
        out = sa.mix(a, b, align="crop", weights=[1.0, 0.5])
        torch.testing.assert_close(out.tensor, torch.full((1, 3), 1.5))

    def test_mix_default_requires_equal_length(self) -> None:
        with pytest.raises(sa.LengthMismatchError):
            sa.mix(sa.Audio(torch.ones(3), 8_000), sa.Audio(torch.ones(5), 8_000))

    def test_mix_still_checks_rate_when_padding(self) -> None:
        with pytest.raises(sa.SampleRateMismatchError):
            sa.mix(sa.Audio(torch.ones(3), 8_000), sa.Audio(torch.ones(5), 16_000), align="pad")

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"align": "stretch"}, "align"), ({"weights": [1.0]}, "weights")],
    )
    def test_mix_bad_args(self, kwargs: dict[str, object], match: str) -> None:
        a = sa.Audio(torch.ones(3), 8_000)
        with pytest.raises(ValueError, match=match):
            sa.mix(a, a, **kwargs)  # type: ignore[arg-type]

    def test_mix_needs_two(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            sa.mix(sa.Audio(torch.ones(3), 8_000))
