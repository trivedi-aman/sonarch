# Switching from torchaudio

torchaudio's I/O (`torchaudio.load`, `torchaudio.save`, `torchaudio.info`) is
being retired in favour of TorchCodec. sonarch gives you loading, saving and
the common transforms in one package, keeps everything as PyTorch tensors, and
matches torchaudio's resampling exactly.

```python
import sonarch as sa
```

## Key differences

| | torchaudio | sonarch |
|---|---|---|
| Data | `(tensor, sample_rate)` tuple | `Audio` object: `.tensor`, `.sample_rate`, `.channels` |
| Shape | `(channels, samples)` | `(channels, samples)`, the same |
| Rate bookkeeping | you pass `orig_freq` / `new_freq` by hand | the `Audio` object tracks it; mismatches raise errors |
| Backends | FFmpeg / sox / soundfile, depending on install | Rust engine for WAV/FLAC/MP3/Vorbis, bundled FFmpeg for everything else; nothing to install |

## I/O

| torchaudio | sonarch |
|---|---|
| `waveform, sr = torchaudio.load(path)` | `audio = sa.load(path)`; `audio.tensor`, `audio.sample_rate` |
| `torchaudio.load(path, frame_offset=n, num_frames=m)` | `sa.load(path, offset=n / sr, duration=m / sr)` |
| `torchaudio.save(path, waveform, sr)` | `sa.Audio(waveform, sr).save(path)` |
| `torchaudio.save(path, w, sr, bits_per_sample=24)` | `audio.save(path, encoding="pcm24")` |
| `torchaudio.info(path)` | `sa.info(path)` |
| a `DataLoader` loop over `torchaudio.load` | `sa.load_many(paths, ...)`, or `sa.load` inside workers (releases the GIL) |

One behavioural difference: if audio exceeds [-1, 1], `torchaudio.save` to an
integer format clips silently, while sonarch raises `ClippingError`. Pass
`clip=True` to clip on purpose, or save as `encoding="float32"`.

## Transforms

| torchaudio | sonarch |
|---|---|
| `F.resample(w, 44100, 16000)` | `audio.resample(16_000, quality="fast")` (identical to torchaudio's default) |
| `F.resample(w, 44100, 16000, lowpass_filter_width=64, rolloff=0.9475937, resampling_method="sinc_interp_kaiser", beta=14.7697)` | `audio.resample(16_000, quality="best")` |
| `w.mean(dim=0, keepdim=True)` | `audio.to_mono()` |
| `T.Spectrogram(n_fft=512)(w)` | `sa.STFT(n_fft=512, hop_length=256)(audio).abs() ** 2` (see note) |
| `T.MelSpectrogram(sample_rate=sr, n_mels=80)(w)` | `sa.MelSpectrogram(n_mels=80)(audio)` (see note) |
| `T.AmplitudeToDB()(mel)` | `sa.ops.power_to_db(mel)` |
| `T.MFCC(sample_rate=sr, n_mfcc=13)(w)` | `sa.MFCC(n_mfcc=13)(audio)` (see note) |
| `T.Vol(gain_db, "db")(w)` | `audio.replace(tensor=audio.tensor * 10 ** (gain_db / 20))` |
| `F.loudness(w, sr)` | `audio.loudness` |

**Notes on matching torchaudio exactly:**

- sonarch's defaults follow librosa (`n_fft=2048`, `hop_length=n_fft // 4`
  for `STFT` and 512 for mel, slaney mel scale with slaney normalisation),
  not torchaudio's (`n_fft=400`, `hop_length=n_fft // 2`, HTK scale, no
  normalisation). Pass torchaudio's values explicitly.
- sonarch pads centered frames with zeros, like librosa. torchaudio's
  `Spectrogram` / `MelSpectrogram` default to `pad_mode="reflect"`, which
  changes the first and last frames. With `pad_mode="constant"` on the
  torchaudio side, results agree to float32 precision:

```python
T.MelSpectrogram(sample_rate=16_000, n_mels=128, pad_mode="constant")(w)
# equals
sa.MelSpectrogram(n_fft=400, hop_length=200, n_mels=128, htk=True, norm=None)(audio)
```

`audio.resample` runs in Rust on CPU (faster than torchaudio for common
ratios; see the README's Speed section) and in PyTorch on GPU tensors, so you
can keep audio on the GPU.

## Not covered yet

Kaldi-compatible `fbank`, `F.lfilter`/biquads, pitch/VAD and the augmentation
transforms aren't in sonarch yet. You can use torchaudio on `audio.tensor`
alongside sonarch in the meantime.
