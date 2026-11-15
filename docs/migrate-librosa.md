# Switching from librosa

sonarch covers librosa's loading, resampling, trimming and spectral features,
with matching defaults so you get the same numbers. The differences are in how
data is carried around: sonarch passes an `Audio` object that knows its own
sample rate, and runs on CPU or GPU.

```python
import sonarch as sa
```

## Key differences

| | librosa | sonarch |
|---|---|---|
| Data | `np.ndarray` + a separate `sr` you pass around | `Audio` object: `.tensor`, `.sample_rate`, `.channels` |
| Default on load | resamples to 22050 Hz and downmixes to mono | keeps the file's rate and channels; ask for conversions explicitly |
| Shape | mono `(samples,)`, multichannel `(channels, samples)` | always `(channels, samples)`; mono is `(1, samples)` |
| Feature shape | mono `(n_mels, frames)` | always `(channels, n_mels, frames)` |
| Device | CPU (NumPy) | CPU or GPU (PyTorch); `.numpy()` when you need an array |
| Mismatched rates | silently wrong results | `SampleRateMismatchError` |

## Loading

| librosa | sonarch |
|---|---|
| `y, sr = librosa.load(path)` | `audio = sa.load(path, sample_rate=22_050, mono=True)` (librosa's defaults) |
| `y, sr = librosa.load(path, sr=None, mono=False)` | `audio = sa.load(path)` |
| `y, sr = librosa.load(path, sr=16000)` | `audio = sa.load(path, sample_rate=16_000, mono=True)` |
| `librosa.load(path, offset=1.0, duration=2.0)` | `sa.load(path, offset=1.0, duration=2.0)` |
| `[librosa.load(p, sr=16000) for p in paths]` | `sa.load_many(paths, sample_rate=16_000, mono=True)` (parallel) |
| `librosa.get_duration(path=path)` | `sa.info(path).duration` |
| `librosa.get_samplerate(path)` | `sa.info(path).sample_rate` |
| `soundfile.write(path, y, sr)` | `audio.save(path)` |

Samples from `sa.load` match librosa/soundfile exactly for WAV and FLAC.
Resampled output differs slightly: librosa uses soxr by default, sonarch uses a
windowed-sinc filter (the same design as torchaudio). Both are high quality;
pick `quality="fast" | "high" | "best"` in sonarch.

## Conversions and effects

| librosa | sonarch |
|---|---|
| `librosa.resample(y, orig_sr=sr, target_sr=16000)` | `audio.resample(16_000)` |
| `librosa.to_mono(y)` | `audio.to_mono()` |
| `librosa.effects.trim(y, top_db=30)` | `audio.trim_silence(top_db=30)` (identical result) |
| `librosa.effects.trim(y)[1]` (the interval) | `sa.ops.nonsilent_range(audio)` |
| `librosa.util.fix_length(y, size=n)` | `audio.fix_length(seconds)` |
| `librosa.util.normalize(y)` | `audio.peak_normalize()` |
| `librosa.util.frame(y, ...)` | `audio.chunk(seconds, hop=...)` |

## Spectral features

All of these match librosa 1.0 to float32 precision with the same arguments.

| librosa | sonarch |
|---|---|
| `librosa.stft(y)` | `sa.STFT()(audio)` or `sa.ops.stft(audio)` |
| `np.abs(librosa.stft(y, n_fft=512))` | `sa.STFT(n_fft=512)(audio).abs()` |
| `librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)` | `sa.MelSpectrogram(n_mels=80)(audio)` |
| `librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))` | `sa.LogMel()(audio)` |
| `librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)` | `sa.MFCC(n_mfcc=13)(audio)` |
| `librosa.filters.mel(sr=sr, n_fft=2048)` | `sa.ops.mel_filterbank(sr, n_fft=2048)` |
| `librosa.power_to_db(S)` | `sa.ops.power_to_db(S)` |

Note that you never pass `sr`: the `Audio` object already knows it.

## Getting NumPy back

```python
mel = sa.MelSpectrogram(n_mels=80)(audio)   # torch.Tensor (channels, 80, frames)
mel_np = mel.cpu().numpy()[0]               # librosa-style (80, frames) for mono
```

## Not covered yet

Chroma, CQT, onset/beat tracking and pitch estimation are not in sonarch yet.
Keep using librosa for those: `librosa.feature.chroma_stft(y=audio.numpy()[0], sr=audio.sample_rate)`.
