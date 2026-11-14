"""Slot fitting for dubbed speech (ADR-0003).

Each synthesized segment must fit the speech slot its segment timeline
provides. The only fitting mechanism is ffmpeg `atempo` (ADR-0004), capped
at MAX_SPEED; anything still longer than its slot overflows into the
adjacent silence and is accounted in the dub plan. The stage never
truncates or shifts speech: the timeline must stay aligned with the
subtitles, and overflow is the gate harness's metric, not a silent fixup.
"""

import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from pipeline.contracts import StageError

# Fixed contract, not config: the acceptance gate measures against this cap.
MAX_SPEED = 1.25
# F5-TTS + the vocos vocoder output at this rate; the dub plan and track are
# documented and validated at it.
DUB_SAMPLE_RATE = 24000
# The engine's native synthesis-speed range (ADR-0005): the F5 infer speed
# argument accepts 0.5..2.0 and the stage rejects anything else before an
# engine is ever built.
MIN_SYNTH_SPEED = 0.5
MAX_SYNTH_SPEED = 2.0
# Overshoot within 0.2% of the slot is measurement noise; stretching it
# would produce a factor the validator cannot distinguish from 1.0.
FIT_EPSILON = 0.002


@dataclass(frozen=True)
class FitDecision:
    speed_factor: float
    method: str  # "none" (fits as synthesized) or "atempo"


def plan_fit(slot_duration: float, synth_duration: float) -> FitDecision:
    """Decide how one measured synthesis fits its slot.

    Speech that already fits is left untouched; anything longer is
    time-stretched by the smallest sufficient factor, never beyond
    MAX_SPEED. The residual beyond the cap is overflow, not extra stretch.
    Overshoot within FIT_EPSILON counts as fitting: sub-permille factors
    would round onto 1.0 and still claim a stretch.
    """
    factor = synth_duration / slot_duration
    if factor <= 1.0 + FIT_EPSILON:
        return FitDecision(1.0, "none")
    return FitDecision(min(factor, MAX_SPEED), "atempo")


def atempo_wav(source: Path, factor: float, target: Path, *, ffmpeg="ffmpeg") -> Path:
    """Time-stretch one PCM wav by `factor`; return the target path.

    The factor is written with six decimals so the filter argument is
    deterministic across runs; the caller measures the output file instead
    of trusting the factor (timestamps must come from measured audio).
    """
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            f"atempo={factor:.6f}",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not target.is_file() or target.stat().st_size == 0:
        raise StageError(
            f"tts: cannot time-stretch speech by {factor:.3f}x: "
            f"{(result.stderr or '').strip()[-500:]}"
        )
    return target


def wav_duration(path: Path) -> float:
    """Duration of a PCM wav in seconds, measured off the file header."""
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            if rate <= 0:
                raise wave.Error(f"invalid sample rate: {rate}")
            return wav.getnframes() / float(rate)
    except (OSError, wave.Error) as exc:
        raise StageError(f"tts: cannot measure duration of {path}: {exc}") from exc
