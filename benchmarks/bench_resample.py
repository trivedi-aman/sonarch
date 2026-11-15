"""Resampling speed: sonarch vs torchaudio vs librosa (soxr).

Usage:
    python benchmarks/bench_resample.py
    python benchmarks/bench_resample.py --files 300   # also: load + mono + 16 kHz for N files

Prints the best-of-5 time in ms (lower is better).
"""

from __future__ import annotations

import argparse
import os
import platform
import tempfile
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

import sonarch as sa
from sonarch.ops.resample import QUALITIES

warnings.filterwarnings("ignore")


def best(fn: Callable[[], object], runs: int = 5) -> float:
    fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def signal(seconds: float, rate: int) -> np.ndarray:
    t = np.arange(round(seconds * rate)) / rate
    rng = np.random.default_rng(0)
    mono = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(t.size)
    return np.stack([mono, mono[::-1]]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--files", type=int, default=0)
    args = parser.parse_args()

    try:
        import torchaudio.functional as taf
    except ImportError:
        taf = None
    try:
        import librosa
    except ImportError:
        librosa = None

    print(
        f"sonarch {sa.__version__} · {platform.system()} {platform.machine()} · "
        f"{args.seconds:g} s stereo · best of 5 · ms\n"
    )
    print(f"{'ratio':16}{'quality':9}{'sonarch':>9}{'torchaudio':>12}{'librosa':>10}")
    for orig, new in [(44_100, 16_000), (48_000, 16_000), (44_100, 48_000)]:
        x = signal(args.seconds, orig)
        audio = sa.Audio(x, orig)
        xt = torch.from_numpy(x)
        for quality, (lpw, rolloff, window, beta) in QUALITIES.items():
            ours = best(lambda: audio.resample(new, quality=quality)) * 1000
            ta = ""
            if taf is not None:
                kw = {
                    "lowpass_filter_width": lpw,
                    "rolloff": rolloff,
                    "resampling_method": f"sinc_interp_{window}",
                }
                if beta is not None:
                    kw["beta"] = beta
                ta = f"{best(lambda: taf.resample(xt, orig, new, **kw)) * 1000:.1f}"
            lr = ""
            if librosa is not None and quality == "high":
                lr = f"{best(lambda: librosa.resample(x, orig_sr=orig, target_sr=new)) * 1000:.1f}"
            print(f"{orig}->{new:<8}{quality:9}{ours:9.1f}{ta:>12}{lr:>10}")
    print("\nlibrosa uses soxr_hq; compared on the 'high' row only.")

    if args.files and librosa is not None:
        workers = os.cpu_count() or 1
        print(
            f"\nload + mono + 16 kHz, {args.files} FLAC files x 5 s @ 44.1 kHz, {workers} cores, s:"
        )
        with tempfile.TemporaryDirectory() as tmp:
            short = sa.Audio(signal(5.0, 44_100), 44_100)
            paths = []
            for i in range(args.files):
                path = Path(tmp) / f"f{i}.flac"
                short.save(path)
                paths.append(str(path))
            rows = {
                "sonarch.load_many": lambda: sa.load_many(paths, sample_rate=16_000, mono=True),
                "librosa.load, one by one": lambda: [librosa.load(p, sr=16_000) for p in paths],
                f"librosa.load, {workers} threads": lambda: list(
                    ThreadPoolExecutor(workers).map(lambda p: librosa.load(p, sr=16_000), paths)
                ),
            }
            for name, fn in rows.items():
                print(f"  {name:32}{best(fn, 3):8.3f}")


if __name__ == "__main__":
    main()
