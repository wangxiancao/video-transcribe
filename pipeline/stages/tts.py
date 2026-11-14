"""tts stage: per-segment synthesis into a dub plan and a full dub track.

Consumes translation.json (tts.input selects the spoken field: the target
by default, the source with input=source). Every segment is synthesized
separately from a fixed reference voice (ADR-0004) and fitted into its
speech slot with ffmpeg atempo, capped at 1.25x (ADR-0003); anything still
longer overflows into the adjacent silence and is accounted in the plan.

The stage writes tts/dub_plan.json — every timestamp measured off the
actual audio, never estimated — and tts/tts.wav, a silence bed of the
source duration with each segment placed at its slot start (overlapping
overflow tails mix additively; nothing is clipped). Video muxing belongs
to the dub stage; this stage never touches video.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from pipeline.contracts import ConfigError, StageError
from pipeline.tts.cache import SynthesisCache
from pipeline.tts.factory import create_tts_engine
from pipeline.tts.fitting import atempo_wav, plan_fit, wav_duration
from pipeline.tts.voices import VoiceLibrary

# ADR-0004: fitting happens after synthesis, via atempo; synthesis itself
# always runs at native speed. The speed stays in the cache key so a future
# synthesis-speed knob cannot collide with entries made without it.
SYNTH_SPEED = 1.0
DEFAULT_VOICE = "zh-female-01"
# Which translation.json field the stage speaks; validators share this
# single source so the stage and its contract cannot drift apart.
INPUT_MODES = ("translated", "source")


def spoken_field(mode):
    """The translation.json field a dub input mode speaks."""
    return "source" if mode == "source" else "target"


def _input_mode(config):
    mode = config.get("tts.input", "translated")
    if mode not in INPUT_MODES:
        raise ConfigError(
            f"tts.input must be one of {', '.join(INPUT_MODES)}, got {mode!r}"
        )
    return mode


def _voice(config):
    name = config.get("tts.voice", DEFAULT_VOICE)
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("tts.voice must be a non-empty string")
    voices_dir = config.get("tts.voices_dir")
    if voices_dir:
        library = VoiceLibrary(Path(str(voices_dir)).expanduser())
    else:
        library = VoiceLibrary()
    try:
        return library.get(name.strip())
    except ValueError as exc:
        raise StageError(f"tts: {exc}") from exc


def _build_engine(config):
    try:
        return create_tts_engine(config)
    except FileNotFoundError as exc:
        # F5-TTS assets are pre-downloaded only; a missing file is a setup
        # error, not something the stage can heal.
        raise StageError(f"tts: {exc}") from exc


def _speak_text(segment, mode):
    return str(segment[spoken_field(mode)]).strip()


def _write_track(path, samples, sample_rate):
    # Keep a .wav suffix: soundfile infers the container from the name.
    temp = path.with_name(f".{path.name}.tmp.wav")
    try:
        sf.write(str(temp), samples, sample_rate, subtype="PCM_16")
        os.replace(temp, path)
    except (OSError, RuntimeError) as exc:
        if temp.exists():
            temp.unlink()
        raise StageError(f"tts: cannot write {path}: {exc}") from exc


def _fit_samples(samples, sample_rate, decision):
    """Apply the atempo decision; return samples measured off the result."""
    if decision.method == "none":
        return samples
    with tempfile.TemporaryDirectory(prefix="tts-fit-") as workdir:
        raw = Path(workdir) / "raw.wav"
        fitted = Path(workdir) / "fitted.wav"
        sf.write(str(raw), samples, sample_rate, subtype="PCM_16")
        atempo_wav(raw, decision.speed_factor, fitted)
        stretched, _rate = sf.read(str(fitted), dtype="float32", always_2d=False)
    return np.asarray(stretched, dtype="float32").reshape(-1)


def run(ctx) -> None:
    mode = _input_mode(ctx.config)
    voice = _voice(ctx.config)
    segments = ctx.read("translation")["segments"]
    if not segments:
        raise StageError("tts: translation segments must not be empty")
    source_duration = wav_duration(ctx.path("audio"))

    cache = None
    if ctx.config.get("tts.cache", True):
        cache = SynthesisCache(ctx.workdir / ".tts-cache")

    engine = None
    sample_rate = None
    entries = []
    placements = []
    for segment in segments:
        text = _speak_text(segment, mode)
        slot_start = float(segment["start"])
        slot_end = float(segment["end"])
        key = SynthesisCache.key(voice.name, text, SYNTH_SPEED)
        audio = cache.get(key) if cache else None
        if audio is None:
            if engine is None:
                engine = _build_engine(ctx.config)
            wav = engine.synth(text, voice, speed=SYNTH_SPEED)
            samples = np.asarray(wav.samples, dtype="float32").reshape(-1)
            rate = int(wav.sample_rate)
            if cache:
                cache.put(key, samples, rate)
        else:
            samples, rate = audio.samples, audio.sample_rate
        if sample_rate is None:
            sample_rate = rate
        elif rate != sample_rate:
            raise StageError(
                f"tts: segment {segment['id']} synthesized at {rate} Hz, "
                f"expected {sample_rate} Hz"
            )

        synth_duration = samples.shape[-1] / sample_rate
        decision = plan_fit(slot_end - slot_start, synth_duration)
        samples = _fit_samples(samples, sample_rate, decision)
        duration = samples.shape[-1] / sample_rate  # measured, post-fitting
        placed_end = slot_start + duration
        overflow = max(0.0, round(placed_end - slot_end, 6))
        entries.append(
            {
                "id": segment["id"],
                "slot_start": round(slot_start, 6),
                "slot_end": round(slot_end, 6),
                "text": text,
                "synthesized_duration_s": round(synth_duration, 6),
                "speed_factor": round(decision.speed_factor, 6),
                "method": decision.method,
                "placed_start": round(slot_start, 6),
                "placed_end": round(placed_end, 6),
                "overflow_s": overflow,
            }
        )
        placements.append((int(round(slot_start * sample_rate)), samples))
        if overflow > 0:
            print(
                f"tts: segment {segment['id']} overflows its slot by "
                f"{overflow:.3f}s (fitted {duration:.3f}s into "
                f"{slot_end - slot_start:.3f}s at {decision.speed_factor:.2f}x)",
                file=sys.stderr,
            )

    # Silence bed of the source duration; a final overflow past the source
    # end extends the bed rather than truncating speech.
    track_frames = max(
        int(round(source_duration * sample_rate)),
        max(start + samples.shape[-1] for start, samples in placements),
    )
    track = np.zeros(track_frames, dtype="float32")
    for start, samples in placements:
        track[start : start + samples.shape[-1]] += samples  # overlaps mix, none clipped
    _write_track(ctx.path("tts_audio"), track, sample_rate)

    total_overflow = sum(entry["overflow_s"] for entry in entries)
    if total_overflow > 0:
        print(
            f"tts: total slot overflow {total_overflow:.3f}s "
            f"({total_overflow / source_duration * 100:.2f}% of "
            f"{source_duration:.1f}s source audio)",
            file=sys.stderr,
        )
    ctx.write(
        "dub_plan",
        {
            "schema_version": 1,
            "input": mode,
            "voice": voice.name,
            "sample_rate": sample_rate,
            "source_duration_s": round(source_duration, 6),
            "track_duration_s": round(track_frames / sample_rate, 6),
            "segments": entries,
            "overflow": {
                "count": sum(1 for entry in entries if entry["overflow_s"] > 0),
                "total_s": round(total_overflow, 6),
                "ratio": round(total_overflow / source_duration, 6),
            },
        },
    )
