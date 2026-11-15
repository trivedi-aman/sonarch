"""Loading speed: sonarch vs soundfile vs librosa (and torchaudio if installed).

Usage:
    python benchmarks/bench_load.py                 # default: 60 s stereo, 7 runs
    python benchmarks/bench_load.py --seconds 10 --runs 20
    python benchmarks/bench_load.py --files 200     # many-files throughput

Prints the median time per file in ms (lower is better) and a speed-up vs
sonarch, so results are easy to compare across commits and machines.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import tempfile
import time
import warnings
from collections.abc import Callable
from pathlib import Path

import av
import numpy as np

import sonarch as sa

warnings.filterwarnings("ignore")

FORMATS = {  # extension -> (PyAV codec, container format)
    "wav": ("pcm_s16le", "wav"),
    "flac": ("flac", "flac"),
    "mp3": ("libmp3lame", "mp3"),
    "m4a": ("aac", "ipod"),
    "ogg": ("libopus", "ogg"),
}


def make_signal(seconds: float, sample_rate: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(round(seconds * sample_rate)) / sample_rate
    tone = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 1234 * t)
    noise = 0.05 * rng.standard_normal((2, t.size))
    return (tone + noise).astype(np.float32)


def write(path: Path, data: np.ndarray, sample_rate: int, codec: str, fmt: str) -> None:
    with av.open(str(path), "w", format=fmt) as container:
        stream = container.add_stream(codec, rate=sample_rate, layout="stereo")
        frame = av.AudioFrame.from_ndarray(data, format="fltp", layout="stereo")
        frame.sample_rate = sample_rate
        for packet in [*stream.encode(frame), *stream.encode(None)]:
            container.mux(packet)


def loaders() -> dict[str, Callable[[str], object]]:
    found: dict[str, Callable[[str], object]] = {"sonarch": lambda p: sa.load(p)}
    try:
        import soundfile as sf

        found["soundfile"] = lambda p: sf.read(p, dtype="float32", always_2d=True)
    except ImportError:
        pass
    try:
        import librosa

        found["librosa"] = lambda p: librosa.load(p, sr=None, mono=False)
    except ImportError:
        pass
    try:
        import torchaudio

        found["torchaudio"] = lambda p: torchaudio.load(p)
    except ImportError:
        pass
    return found


def time_it(fn: Callable[[], object], runs: int) -> float | None:
    try:
        fn()  # warm-up, and detect unsupported formats
    except Exception:
        return None
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=60.0, help="clip length")
    parser.add_argument("--runs", type=int, default=7, help="timed runs per cell")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--files", type=int, default=0, help="also time loading N short files")
    args = parser.parse_args()

    libs = loaders()
    print(
        f"sonarch {sa.__version__} · Python {platform.python_version()} · "
        f"{platform.system()} {platform.machine()}"
    )
    print(
        f"{args.seconds:g} s stereo @ {args.sample_rate} Hz · "
        f"median of {args.runs} runs · ms/file\n"
    )

    header = f"{'format':8}" + "".join(f"{name:>12}" for name in libs)
    print(header)
    print("-" * len(header))

    with tempfile.TemporaryDirectory() as tmp:
        signal = make_signal(args.seconds, args.sample_rate)
        for ext, (codec, fmt) in FORMATS.items():
            path = Path(tmp) / f"bench.{ext}"
            write(path, signal, args.sample_rate, codec, fmt)
            row = f"{ext:8}"
            base = None
            for name, load in libs.items():
                ms = time_it(lambda load=load, p=str(path): load(p), args.runs)
                if name == "sonarch":
                    base = ms
                if ms is None:
                    row += f"{'n/a':>12}"
                elif name == "sonarch" or base is None:
                    row += f"{ms:12.1f}"
                else:
                    row += f"{ms:7.1f} ({base / ms:.1f}x)".rjust(12)
            print(row)

        if args.files:
            from concurrent.futures import ThreadPoolExecutor

            workers = os.cpu_count() or 1
            print(f"\n{args.files} files x 5 s, total seconds (best of 3), {workers} cores:")
            short = make_signal(5.0, args.sample_rate)
            for ext in ("flac", "mp3"):
                codec, fmt = FORMATS[ext]
                paths = []
                for i in range(args.files):
                    path = Path(tmp) / f"many_{i}.{ext}"
                    write(path, short, args.sample_rate, codec, fmt)
                    paths.append(str(path))

                def best(fn: Callable[[], object]) -> float:
                    fn()
                    runs = []
                    for _ in range(3):
                        start = time.perf_counter()
                        fn()
                        runs.append(time.perf_counter() - start)
                    return min(runs)

                print(f"  {ext}:")
                rows = {"sonarch.load_many (Rust)": lambda paths=paths: sa.load_many(paths)}
                for name, load in libs.items():
                    rows[f"{name}, one by one"] = lambda load=load, paths=paths: [
                        load(p) for p in paths
                    ]
                    rows[f"{name}, {workers} Python threads"] = lambda load=load, paths=paths: list(
                        ThreadPoolExecutor(workers).map(load, paths)
                    )
                for name, fn in rows.items():
                    print(f"    {name:32}{best(fn):8.3f} s")

    print("\n(x) = how many times faster than sonarch; < 1.0x means slower.")


if __name__ == "__main__":
    main()
