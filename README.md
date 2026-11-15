# sonarch

**The one import you need for audio work: loading, preprocessing, features, codecs and datasets, ready for ML.**

```python
import sonarch as sa

audio = sa.load("anything.mp3")              # any format, any source; no system FFmpeg needed
audio = audio.resample(16_000).to_mono()
mel   = sa.MelSpectrogram(n_mels=80)(audio)  # CPU or GPU, matches librosa within tolerance
```

> ⚠️ **Status: early development.** This README describes the design target. Every feature section is tagged:
> ✅ available · 🚧 in progress · 📋 planned. See the [Roadmap](#roadmap).

---

## Why sonarch?

Working with audio today means juggling several libraries:

- `librosa` for features (CPU-only, slow at scale)
- `torchaudio` for loading and transforms (loading is being deprecated in favour of `torchcodec`)
- `audiomentations` / `torch-audiomentations` for augmentation
- `ffmpeg` / `sox` for format conversion
- A separate repo for every neural codec (EnCodec, DAC, Mimi, SNAC…)

The glue code between them is where bugs live: **silent sample-rate mismatches**, **preprocessing that differs between training and inference**, and **data pipelines stuck on the CPU**.

sonarch gives you **one consistent API** for all of it. Where a great tool already exists, sonarch wraps it behind that API instead of reinventing it, so you get one `Audio` object, one style and one install.

| Problem | sonarch's answer |
|---|---|
| "Which library do I even use to load audio now?" | `sa.load()`: every common format, local or remote, with bundled decoders |
| Sample-rate / channel mismatch bugs | `Audio` objects carry their own metadata and check it at every step |
| Train/inference preprocessing drift | Pipelines save to YAML, so you ship the exact same steps with your model |
| Slow CPU preprocessing | Core ops run on CPU, CUDA or MPS, on single clips or batches |
| Fragmented neural codecs | One `Codec` interface over EnCodec, DAC, Mimi, SNAC |
| Huge raw datasets | Pack audio into compressed (FLAC / Opus) shards and stream them |
| "How much did compression hurt?" | Quality metrics: PESQ, STOI, SI-SDR, mel distance |

---

## Design principles

1. **Easy by default, powerful on demand.** One-liners and presets for common cases; every setting is exposed for advanced users.
2. **Metadata travels with data.** Sample rate, channels and loudness are never "assumed".
3. **Reproducible.** A saved pipeline reloads with exactly the same configuration. Outputs match across CPU, CUDA and MPS within a tested numerical tolerance.
4. **Device-agnostic.** The same code runs on CPU, CUDA or MPS. NumPy arrays and PyTorch tensors are both accepted.
5. **Small core, optional extras.** The core needs only PyTorch, NumPy and a bundled decoder. Codecs, augmentation and metrics are opt-in extras.
6. **Fail loudly.** Invalid shapes, sample rates or configs raise clear errors, never silent corruption.
7. **Wrap first, replace later.** sonarch builds on proven libraries where they exist and only reimplements where it can be clearly faster or more correct.

---

## Installation

```bash
pip install sonarch                 # core: load/save, resample, normalize, trim, features
pip install sonarch[augment]        # augmentation ops
pip install sonarch[codecs]         # neural codecs (EnCodec, DAC, Mimi, SNAC)
pip install sonarch[metrics]        # PESQ, STOI, SI-SDR
pip install sonarch[cloud]          # S3 / GCS / HTTP reading via fsspec
pip install sonarch[all]            # everything
```

Requires Python 3.10+ and PyTorch 2.x. Decoding uses bundled FFmpeg libraries (via PyAV), so **no system FFmpeg install is needed** on Linux, macOS or Windows.

---

## Quickstart

### Load, inspect, convert 🚧

```python
import sonarch as sa

audio = sa.load("clip.flac")
audio.sample_rate     # 44100
audio.channels        # 2
audio.duration        # 12.4 (seconds)
audio.loudness        # -18.2 (LUFS, computed on first access)

audio = audio.resample(16_000).to_mono().to("cuda")
audio.tensor          # torch.Tensor, shape (channels, samples)
audio.numpy()         # np.ndarray

sa.info("clip.flac")  # fast metadata probe, no decoding
audio.save("clip_16k.wav")
```

Mismatches fail loudly:

```python
a, b = sa.load("a.wav"), sa.load("b_48k.wav")
a + b                 # ❌ SampleRateMismatchError: 16000 Hz vs 48000 Hz
sa.mix(a, b.resample(16_000), align="pad")   # ✅ explicit, pads the shorter clip
```

### Build a pipeline 🚧

```python
pipe = sa.Pipeline([
    sa.Resample(16_000),
    sa.ToMono(),
    sa.LoudnessNorm(target_lufs=-23),
    sa.TrimSilence(top_db=30),
    sa.FixLength(seconds=10, mode="pad_or_crop"),
    sa.MelSpectrogram(n_mels=80, n_fft=400, hop_length=160),
])

mel = pipe("clip.wav")                 # path, Audio, tensor or array

pipe.save("preprocess.yaml")           # ship this with your model
pipe = sa.Pipeline.load("preprocess.yaml")
```

### Batches on GPU 📋

```python
pipe = pipe.to("cuda")
batch = pipe(["a.wav", "b.flac", "c.mp3"])
batch.tensor          # (batch, n_mels, frames), padded to the longest item
batch.lengths         # true length of each item, in frames
batch.mask            # bool mask, True where values are real
```

If the pipeline ends in `FixLength`, every item has the same length and no padding is added.

### Presets 📋

```python
pipe = sa.preset("whisper")       # 16 kHz mono, 80-bin log-mel, 30 s (Whisper up to large-v2)
pipe = sa.preset("whisper-v3")    # same, but 128 bins (Whisper large-v3)
features = pipe("speech.mp3")
```

---

## Coming from another library?

| You used to write | In sonarch |
|---|---|
| `librosa.load(path, sr=16000, mono=True)` | `sa.load(path).resample(16_000).to_mono()` |
| `librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80)` | `sa.MelSpectrogram(n_mels=80)(audio)` |
| `librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)` | `sa.MFCC(n_mfcc=13)(audio)` |
| `librosa.effects.trim(y, top_db=30)` | `audio.trim_silence(top_db=30)` |
| `torchaudio.load(path)` | `sa.load(path)` |
| `torchaudio.functional.resample(x, 44100, 16000)` | `audio.resample(16_000)` |
| `torchaudio.compliance.kaldi.fbank(x, ...)` | `sa.Fbank(...)(audio)` |
| `WhisperFeatureExtractor(...)(x)` | `sa.preset("whisper")(audio)` |

Numerical ops are tested against librosa / torchaudio so outputs match within tolerance. Full migration guides: 📋 `docs/migrate-librosa.md`, `docs/migrate-torchaudio.md`.

---

## Features

### 1. Loading & I/O 🚧
- Decode WAV, FLAC, MP3, OGG/Opus, M4A/AAC, and audio tracks from video files
- Save to WAV, FLAC and Opus
- Fast metadata probing without decoding
- Streaming / chunked reads for long files 📋
- Read from URLs, S3 and GCS via `fsspec` (`sonarch[cloud]`) 📋

### 2. Preprocessing ops
| Category | Ops | Status |
|---|---|---|
| Format | `Resample`, `ToMono`, `ToStereo`, `ChangeDtype` | 🚧 |
| Level | `PeakNorm`, `RMSNorm`, `LoudnessNorm` (EBU R128 / LUFS), `DCRemove` | 🚧 |
| Time | `TrimSilence`, `FixLength`, `Pad`, `Crop`, `Chunk` | 🚧 |
| Time | `VAD` | 📋 |
| Filters | `HighPass`, `LowPass`, `BandPass`, `PreEmphasis` | 📋 |
| Features | `STFT`, `MelSpectrogram`, `LogMel`, `MFCC` | 🚧 |
| Features | `Fbank` (Kaldi-compatible), `CQT`, `Chroma` | 📋 |

### 3. Neural codecs and tokenization (`sonarch[codecs]`) 📋
One interface for discrete audio tokens, for audio language models and compression.

```python
codec = sa.Codec("dac", bitrate="6kbps", device="cuda")
codec.sample_rate                   # 44100: the rate the model expects

tokens = codec.encode(audio)        # (codebooks, frames) int tensor
recon  = codec.decode(tokens)       # Audio object

codec.encode(audio_16k)             # ❌ SampleRateMismatchError: codec expects 44100 Hz
codec.encode(audio_16k, resample=True)   # ✅ explicit

sa.Codec.available()                # ["encodec", "dac", "mimi", "snac", ...]
```

> **Model licenses:** sonarch is MIT, but pretrained codec weights come from their original authors and may carry different licenses. `sa.Codec.info("dac")` shows the license for each model; check it before commercial use.

### 4. Datasets: compressed shards and streaming 📋
Turn large folders of WAV files into compact, streamable shards. Built on WebDataset-style tar shards, so they also work with existing tooling.

```python
sa.dataset.pack(
    source="raw_wavs/",
    dest="shards/",
    format="flac",           # lossless; or "opus" with bitrate="64kbps"
    shard_size="1GB",
    preprocess=pipe,         # optionally store preprocessed audio
)

ds = sa.dataset.stream("shards/", pipeline=pipe)   # PyTorch IterableDataset
```

Also reads Hugging Face `datasets` audio columns directly.

### 5. Augmentation (`sonarch[augment]`) 📋
`AddNoise`, `AddReverb` (RIR), `GainJitter`, `PitchShift`, `TimeStretch`, `SpecAugment`, `Mixup`, `CodecAugment` (simulate MP3/Opus artifacts). All are probabilistic and seedable. Most run batched on GPU. `CodecAugment` and high-quality `PitchShift` / `TimeStretch` run on CPU.

```python
train_pipe = pipe.with_augment([
    sa.AddNoise(snr_db=(5, 20), p=0.5),         # waveform op: placed before MelSpectrogram
    sa.SpecAugment(time_masks=2, freq_masks=2), # spectrogram op: placed after it
], seed=42)
```

Each op declares whether it works on waveforms or spectrograms, and `with_augment` puts it in the right place automatically. `train_pipe.describe()` prints the final order.

### 6. Quality evaluation (`sonarch[metrics]`) 📋
Know what a compression setting costs *before* you train on it.

```python
report = sa.evaluate(original, recon, metrics=["pesq", "stoi", "si_sdr", "mel_distance"])
print(report)

sa.compare_codecs(
    "eval_clips/",
    codecs=[("opus", "24kbps"), ("dac", "6kbps"), ("encodec", "3kbps")],
)   # → table of bitrate vs. quality
```

PESQ only supports 8 kHz and 16 kHz audio; sonarch resamples for it and says so in the report.

### 7. Feature caching 📋
Compute features once and store them compactly.

```python
sa.cache.build(dataset="shards/", pipeline=pipe, dest="mel_cache/", dtype="float16")
```

---

## Writing a custom op

Every op subclasses `sa.Op` and declares what kind of data it takes and returns. sonarch uses this to validate pipelines and place augmentations.

```python
class Invert(sa.Op):
    input = "wave"           # "wave" or "spec"
    output = "wave"

    def __call__(self, audio: sa.Audio) -> sa.Audio:
        return audio.replace(tensor=-audio.tensor)
```

Ops are automatically saved to YAML if their constructor arguments are plain types (numbers, strings, lists, dicts).

---

## Roadmap

| Milestone | Scope | Status |
|---|---|---|
| **v0.1: Core** | `sa.load` / `save` / `info`, `Audio` object, resample, mono/stereo, normalization, trim, pad/crop, STFT / mel / MFCC matched to librosa, migration guides | 🚧 In progress |
| **v0.2: ML-ready** | `Pipeline` + YAML, batched GPU execution with padding and masks, `whisper` / `wav2vec2` presets matched to Hugging Face feature extractors, benchmarks vs. librosa / torchaudio | 📋 Planned |
| **v0.3: Codecs** | Unified `Codec` API: EnCodec, DAC, Mimi, SNAC | 📋 Planned |
| **v0.4: Data** | `dataset.pack` / `stream`, FLAC/Opus shards, HF `datasets` support, cloud reading | 📋 Planned |
| **v0.5: Augment & metrics** | Augmentation suite, `evaluate`, `compare_codecs` | 📋 Planned |
| **v1.0** | Remaining features (filters, VAD, Fbank, CQT, caching), full docs, stable API | 📋 Planned |

---

## Project structure

```
sonarch/
├── src/sonarch/
│   ├── __init__.py
│   ├── core/
│   │   ├── audio.py          # Audio object + metadata validation
│   │   ├── pipeline.py       # Pipeline, YAML save/load
│   │   ├── op.py             # Base class for all ops
│   │   └── errors.py
│   ├── io/                   # load, save, info, decoding backends
│   ├── ops/
│   │   ├── format.py         # resample, channels, dtype
│   │   ├── level.py          # normalization, loudness
│   │   ├── time.py           # trim, pad/crop, chunk
│   │   ├── filters.py
│   │   └── features.py       # STFT, mel, MFCC
│   ├── codecs/               # Codec interface + backends
│   ├── dataset/              # pack, stream, shards
│   ├── augment/
│   ├── metrics/
│   ├── cache/
│   └── presets/              # YAML preset definitions
├── tests/
├── benchmarks/               # speed vs. librosa / torchaudio
├── docs/
├── examples/
├── pyproject.toml
└── README.md
```

---

## Development

```bash
git clone https://github.com/trivedi-aman/sonarch.git
cd sonarch
pip install -e ".[dev,all]"
pytest
```

Conventions:
- Type hints everywhere; `mypy --strict` on `sonarch/core`
- Formatting and linting with `ruff`
- Every op needs: a unit test, a CPU/GPU parity test, and a YAML save/load round-trip test
- Numerical ops are tested against a reference (librosa / torchaudio) within tolerance

---

## Contributing

Contributions are welcome. Open an issue to discuss new ops, codecs or presets before sending a large PR.

## License

MIT for the sonarch code. Pretrained model weights used by `sonarch[codecs]` keep their original licenses.
