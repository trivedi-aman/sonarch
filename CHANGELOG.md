# Changelog

## 0.1.0 (2026-11-15)

First alpha release: the v0.1 "Core" milestone.

### Loading and saving
- `sa.load` for any format FFmpeg can decode, from paths, URLs or file objects,
  with `offset=` / `duration=` (sample-exact seeking) and `sample_rate=` / `mono=`.
- Rust decoding engine for WAV, AIFF, FLAC, MP3 and Ogg Vorbis, with automatic
  FFmpeg fallback. Output is bit-exact with FFmpeg for lossless formats.
- `sa.load_many`: decode, downmix and resample many files in parallel on all cores.
- `sa.save` to WAV (16/24/32-bit PCM, float), FLAC (16/24-bit) and Opus; raises
  `ClippingError` instead of silently clipping.
- `sa.info` for metadata without decoding.

### The `Audio` object
- Waveform plus sample rate, always `(channels, samples)`, on any device.
- Mismatched sample rates, channels or lengths raise clear errors; `sa.mix`
  combines clips explicitly.

### Processing
- `resample` with `fast` / `high` / `best` quality: Rust on CPU, PyTorch on GPU,
  matching torchaudio to float32 precision.
- Levels: `loudness` (ITU-R BS.1770-4, matches pyloudnorm), `peak`, `rms`,
  `peak_normalize`, `rms_normalize`, `loudness_normalize`, `remove_dc`.
- Time: `trim_silence` (matches librosa exactly), `pad`, `crop`, `fix_length`, `chunk`.
- Features: `STFT`, `MelSpectrogram`, `LogMel`, `MFCC`, matching librosa 1.0.

### Docs
- Migration guides from librosa and torchaudio.
- Benchmarks in `benchmarks/`.

## 0.1.0.dev1
- Rust resampling; `sample_rate=` / `mono=` on `load` and `load_many`.

## 0.1.0.dev0
- Initial preview: `Audio`, `load`, `save`, `info`.
