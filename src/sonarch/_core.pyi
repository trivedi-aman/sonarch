"""Type stubs for sonarch's native Rust engine."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

class UnsupportedError(Exception): ...
class NativeDecodeError(Exception): ...

def decode(path: str, max_seconds: float | None = None) -> tuple[NDArray[np.float32], int]: ...
def decode_many(
    paths: list[str],
    max_seconds: float | None = None,
    num_threads: int | None = None,
) -> list[tuple[NDArray[np.float32], int] | tuple[None, Literal["unsupported", "failed"], str]]: ...
