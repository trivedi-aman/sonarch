"""Type stubs for sonarch's native Rust engine."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

class UnsupportedError(Exception): ...
class NativeDecodeError(Exception): ...

def decode(path: str, max_seconds: float | None = None) -> tuple[NDArray[np.float32], int]: ...
def resample(
    array: NDArray[np.float32], orig_rate: int, new_rate: int, quality: str = "high"
) -> NDArray[np.float32]: ...
def decode_many(
    paths: list[str],
    max_seconds: float | None = None,
    num_threads: int | None = None,
    sample_rate: int | None = None,
    mono: bool = False,
    quality: str = "high",
) -> list[tuple[NDArray[np.float32], int] | tuple[None, Literal["unsupported", "failed"], str]]: ...
