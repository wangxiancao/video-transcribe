"""Synthesis cache: key = voice + segment text + speed (issue #2).

Stores the raw synthesized audio (pre-fitting) as PCM16 wav so a rerun with
the same voice, text and speed skips the GPU entirely. Entries are written
atomically; unreadable entries are treated as misses, never as failures.
Durations are not stored: the stage measures whatever it loads, so plan
timestamps always come from the actual audio.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class CachedAudio:
    samples: "np.ndarray"  # (time,) float32
    sample_rate: int


class SynthesisCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(voice_name: str, text: str, speed: float) -> str:
        payload = f"{voice_name}\x00{text}\x00{speed:.6f}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / f"{key}.wav"

    def get(self, key: str) -> CachedAudio | None:
        path = self._path(key)
        try:
            samples, rate = sf.read(str(path), dtype="float32", always_2d=False)
        except (OSError, RuntimeError):
            return None
        samples = np.asarray(samples, dtype="float32").reshape(-1)
        rate = int(rate)
        if rate <= 0 or samples.size == 0:
            return None
        return CachedAudio(samples=samples, sample_rate=rate)

    def put(self, key: str, samples: "np.ndarray", sample_rate: int) -> None:
        path = self._path(key)
        # Keep a .wav suffix: soundfile infers the container from the name.
        temp = path.with_name(f".{path.name}.tmp.wav")
        try:
            sf.write(str(temp), samples, sample_rate, subtype="PCM_16")
            os.replace(temp, path)
        except (OSError, RuntimeError) as exc:
            if temp.exists():
                temp.unlink()
            raise RuntimeError(f"tts: cannot write synthesis cache entry: {exc}") from exc
