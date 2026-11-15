from __future__ import annotations

import io
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch

import sonarch as sa
from tests.conftest import SR, MakeFile, encode_with_pyav, tones, write_wav_stdlib


class TestLoad:
    def test_wav_matches_stdlib_reference(self, wav_file: tuple[Path, np.ndarray]) -> None:
        path, expected = wav_file
        audio = sa.load(path)
        assert audio.sample_rate == SR
        assert audio.dtype == torch.float32
        np.testing.assert_array_equal(audio.numpy(), expected)

    def test_flac_matches_soundfile(self, flac_file: tuple[Path, np.ndarray]) -> None:
        path, expected = flac_file
        np.testing.assert_array_equal(sa.load(path).numpy(), expected)

    def test_matches_librosa(self, flac_file: tuple[Path, np.ndarray]) -> None:
        path, _ = flac_file
        reference, rate = librosa.load(path, sr=None, mono=False)
        audio = sa.load(path)
        assert audio.sample_rate == rate
        np.testing.assert_allclose(audio.numpy(), reference, atol=1e-7)

    def test_str_path_and_file_object(self, wav_file: tuple[Path, np.ndarray]) -> None:
        path, expected = wav_file
        np.testing.assert_array_equal(sa.load(str(path)).numpy(), expected)
        with open(path, "rb") as f:
            np.testing.assert_array_equal(sa.load(f).numpy(), expected)
        np.testing.assert_array_equal(sa.load(io.BytesIO(path.read_bytes())).numpy(), expected)

    def test_mono_file(self, tmp_path: Path) -> None:
        path = tmp_path / "mono.wav"
        expected = write_wav_stdlib(path, tones(channels=1), SR)
        audio = sa.load(path)
        assert audio.channels == 1
        np.testing.assert_array_equal(audio.numpy(), expected)

    def test_multichannel_file(self, tmp_path: Path) -> None:
        path = tmp_path / "six.wav"
        expected = write_wav_stdlib(path, tones(channels=6), SR)
        np.testing.assert_array_equal(sa.load(path).numpy(), expected)

    @pytest.mark.parametrize(("offset", "duration"), [(0.25, 0.5), (0.0, 0.1), (1.9, None)])
    def test_offset_duration_is_sample_exact(
        self, wav_file: tuple[Path, np.ndarray], offset: float, duration: float | None
    ) -> None:
        path, expected = wav_file
        start = round(offset * SR)
        stop = None if duration is None else start + round(duration * SR)
        audio = sa.load(path, offset=offset, duration=duration)
        np.testing.assert_array_equal(audio.numpy(), expected[:, start:stop])

    def test_duration_past_end_is_truncated(self, wav_file: tuple[Path, np.ndarray]) -> None:
        path, expected = wav_file
        audio = sa.load(path, offset=1.5, duration=10.0)
        np.testing.assert_array_equal(audio.numpy(), expected[:, round(1.5 * SR) :])

    def test_lossy_seek_matches_full_decode(self, lossy_file: tuple[Path, int]) -> None:
        path, rate = lossy_file
        full = sa.load(path).numpy()
        for offset in (0.3, 1.0, 2.1):
            part = sa.load(path, offset=offset, duration=0.4).numpy()
            start = round(offset * rate)
            assert part.shape[1] == round(0.4 * rate)
            np.testing.assert_allclose(part, full[:, start : start + part.shape[1]], atol=1e-4)

    def test_offset_past_end_raises(self, wav_file: tuple[Path, np.ndarray]) -> None:
        path, _ = wav_file
        with pytest.raises(ValueError, match="past the end"):
            sa.load(path, offset=5.0)

    @pytest.mark.parametrize(
        "kwargs",
        [{"offset": -1.0}, {"offset": float("nan")}, {"duration": 0.0}, {"duration": -2.0}],
    )
    def test_bad_offset_duration(
        self, wav_file: tuple[Path, np.ndarray], kwargs: dict[str, float]
    ) -> None:
        path, _ = wav_file
        with pytest.raises(ValueError):
            sa.load(path, **kwargs)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sa.load(tmp_path / "nope.wav")

    def test_garbage_file(self, make_file: MakeFile) -> None:
        with pytest.raises(sa.DecodeError):
            sa.load(make_file("junk.wav", b"definitely not audio" * 10))

    def test_audio_track_of_video(self, tmp_path: Path) -> None:
        import av

        path = tmp_path / "clip.mp4"
        data = tones(1.0)
        with av.open(str(path), "w") as container:
            video = container.add_stream("mpeg4", rate=10)
            video.width, video.height = 32, 32
            audio = container.add_stream("aac", rate=SR, layout="stereo")
            for i in range(10):
                frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
                frame.pts = i
                for packet in video.encode(frame):
                    container.mux(packet)
            aframe = av.AudioFrame.from_ndarray(data, format="fltp", layout="stereo")
            aframe.sample_rate = SR
            for packet in audio.encode(aframe):
                container.mux(packet)
            for stream in (video, audio):
                for packet in stream.encode(None):
                    container.mux(packet)
        loaded = sa.load(path)
        assert loaded.sample_rate == SR
        assert loaded.channels == 2
        assert loaded.duration == pytest.approx(1.0, abs=0.1)

    def test_video_without_audio(self, tmp_path: Path) -> None:
        import av

        path = tmp_path / "silent.mp4"
        with av.open(str(path), "w") as container:
            video = container.add_stream("mpeg4", rate=10)
            video.width, video.height = 32, 32
            frame = av.VideoFrame.from_ndarray(np.zeros((32, 32, 3), np.uint8), format="rgb24")
            for packet in [*video.encode(frame), *video.encode(None)]:
                container.mux(packet)
        with pytest.raises(sa.DecodeError, match="no audio stream"):
            sa.load(path)


class TestInfo:
    def test_wav(self, wav_file: tuple[Path, np.ndarray]) -> None:
        path, expected = wav_file
        meta = sa.info(path)
        assert meta.sample_rate == SR
        assert meta.channels == 2
        assert meta.num_samples == expected.shape[1]
        assert meta.duration == pytest.approx(expected.shape[1] / SR)
        assert meta.codec == "pcm_s16le"
        assert meta.format == "wav"

    def test_lossy(self, lossy_file: tuple[Path, int]) -> None:
        path, rate = lossy_file
        meta = sa.info(path)
        assert meta.sample_rate == rate
        assert meta.channels == 2
        assert meta.duration == pytest.approx(3.0, abs=0.05)
        assert meta.bit_rate is not None

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            sa.info(tmp_path / "nope.flac")


class TestSave:
    @pytest.mark.parametrize(
        ("name", "encoding", "subtype", "atol"),
        [
            ("a.wav", None, "PCM_16", 1 / 32768),
            ("a.wav", "pcm24", "PCM_24", 1 / 2**23),
            ("a.wav", "pcm32", "PCM_32", 1e-7),
            ("a.wav", "float32", "FLOAT", 0.0),
            ("a.flac", None, "PCM_16", 1 / 32768),
            ("a.flac", "pcm24", "PCM_24", 1 / 2**23),
        ],
    )
    def test_lossless_round_trip(
        self, tmp_path: Path, name: str, encoding: str | None, subtype: str, atol: float
    ) -> None:
        audio = sa.Audio(tones(1.0), SR)
        path = tmp_path / name
        audio.save(path, encoding=encoding)
        assert sf.info(path).subtype == subtype
        loaded = sa.load(path)
        assert loaded.shape == audio.shape
        np.testing.assert_allclose(loaded.numpy(), audio.numpy(), atol=atol)

    def test_opus(self, tmp_path: Path) -> None:
        audio = sa.Audio(tones(1.0, 48_000), 48_000)
        path = tmp_path / "a.opus"
        sa.save(audio, path, bitrate="96kbps")
        loaded = sa.load(path)
        assert loaded.sample_rate == 48_000
        assert loaded.shape == audio.shape
        assert sa.info(path).codec == "opus"

    def test_opus_rejects_unsupported_rate(self, tmp_path: Path) -> None:
        with pytest.raises(sa.SampleRateMismatchError, match="Opus supports"):
            sa.Audio(tones(0.1), SR).save(tmp_path / "a.opus")

    def test_multichannel(self, tmp_path: Path) -> None:
        audio = sa.Audio(tones(0.5, channels=6), SR)
        audio.save(tmp_path / "six.flac")
        assert sa.load(tmp_path / "six.flac").channels == 6

    def test_mono_from_gpu_dtype(self, tmp_path: Path) -> None:
        audio = sa.Audio(torch.from_numpy(tones(0.5, channels=1)).half(), SR)
        audio.save(tmp_path / "half.wav")
        assert sa.load(tmp_path / "half.wav").channels == 1

    def test_empty(self, tmp_path: Path) -> None:
        sa.Audio(torch.zeros(1, 0), SR).save(tmp_path / "empty.wav")
        assert sa.load(tmp_path / "empty.wav").num_samples == 0

    def test_clipping_raises_by_default(self, tmp_path: Path) -> None:
        loud = sa.Audio(torch.full((1, 100), 1.5), SR)
        with pytest.raises(sa.ClippingError, match=r"1\.500"):
            loud.save(tmp_path / "loud.wav")
        assert not (tmp_path / "loud.wav").exists()

    def test_clip_true_clips(self, tmp_path: Path) -> None:
        loud = sa.Audio(torch.full((1, 100), 1.5), SR)
        loud.save(tmp_path / "loud.wav", clip=True)
        assert sa.load(tmp_path / "loud.wav").numpy().max() == pytest.approx(1.0, abs=1e-4)

    def test_float_wav_keeps_out_of_range(self, tmp_path: Path) -> None:
        loud = sa.Audio(torch.full((1, 100), 1.5), SR)
        loud.save(tmp_path / "loud.wav", encoding="float32")
        assert sa.load(tmp_path / "loud.wav").numpy().max() == 1.5

    def test_refuses_nan(self, tmp_path: Path) -> None:
        bad = sa.Audio(torch.tensor([0.0, float("nan")]), SR)
        with pytest.raises(sa.AudioValidationError, match="NaN"):
            bad.save(tmp_path / "bad.wav")

    @pytest.mark.parametrize(
        ("name", "kwargs", "match"),
        [
            ("a.mp4", {}, "can't infer"),
            ("a.wav", {"encoding": "pcm8"}, "supports encodings"),
            ("a.flac", {"encoding": "float32"}, "supports encodings"),
            ("a.wav", {"bitrate": "64k"}, "only applies to Opus"),
            ("a.opus", {"encoding": "pcm16"}, "not used for Opus"),
            ("a.opus", {"bitrate": "lots"}, "can't parse"),
            ("a.opus", {"bitrate": 1_000_000}, "between 6 and 510"),
        ],
    )
    def test_bad_arguments(
        self, tmp_path: Path, name: str, kwargs: dict[str, object], match: str
    ) -> None:
        audio = sa.Audio(tones(0.1, 48_000), 48_000)
        with pytest.raises(ValueError, match=match):
            audio.save(tmp_path / name, **kwargs)

    def test_explicit_format(self, tmp_path: Path) -> None:
        sa.Audio(tones(0.1), SR).save(tmp_path / "noext", format="flac")
        assert sa.info(tmp_path / "noext").codec == "flac"

    def test_save_rejects_non_audio(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="Audio object"):
            sa.save(np.zeros(10), tmp_path / "a.wav")  # type: ignore[arg-type]


def test_lossy_fixture_encoder_helper_roundtrip(tmp_path: Path) -> None:
    """Guard the test helper itself: an MP3 it writes decodes to the right length."""
    path = tmp_path / "x.mp3"
    encode_with_pyav(path, tones(1.0), SR, "libmp3lame")
    assert sa.load(path).num_samples == SR
