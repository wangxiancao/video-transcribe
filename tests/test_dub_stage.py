"""Dub stage tests: whole-track replacement mux (#3).

The public seam is the stage runner (stage id + workdir -> artifacts and exit
code) and the artifact validators, per docs/agents/verification.md. The dub
track is built by the tts stage with a deterministic fake engine so the mux
is exercised on real fixture media without F5-TTS assets.
"""

import hashlib
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from pipeline.runner import run
from pipeline.tts.engine import SynthResult

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSLATION = REPO_ROOT / "tests/fixtures/translation/sample_6s.json"
SEGMENTS = REPO_ROOT / "tests/fixtures/segments/sample_6s.json"
VIDEO = REPO_ROOT / "tests/fixtures/video/sample_5s.mp4"


def _config(tmp_path, **tts_overrides):
    tts = {"voice": "test-voice", "voices_dir": str(tmp_path / "voices")}
    tts.update(tts_overrides)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"tts": tts}), encoding="utf-8")
    return path


def _voices(tmp_path):
    """A minimal valid named-voice library for the tts stage to select from."""
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    sf.write(
        voices_dir / "test-voice.wav",
        (0.2 * np.sin(2 * math.pi * 220 * np.arange(2400) / 24000)).astype("float32"),
        24000,
        subtype="PCM_16",
    )
    (voices_dir / "voices.json").write_text(
        json.dumps([
            {
                "name": "test-voice",
                "ref_audio": "test-voice.wav",
                "ref_text": "测试参考文本。",
                "language": "zh",
            },
        ]),
        encoding="utf-8",
    )
    return voices_dir


def _workdir(tmp_path, audio_duration=5.0):
    """Workdir with the video fixture and a matching-length audio bed.

    The tts stage sizes its silence bed off media/audio.wav, so the bed must
    match the video fixture duration for the dub drift gate to pass.
    """
    workdir = tmp_path / "work"
    media = workdir / "media"
    media.mkdir(parents=True)
    shutil.copyfile(TRANSLATION, workdir / "translation.json")
    shutil.copyfile(SEGMENTS, workdir / "segments.json")
    shutil.copyfile(VIDEO, media / "source.mp4")
    sf.write(
        media / "audio.wav",
        0.2 * np.sin(2 * math.pi * 440 * np.arange(int(16000 * audio_duration)) / 16000),
        16000,
        subtype="PCM_16",
    )
    return workdir


class FakeEngine:
    """Deterministic stand-in for F5TTSEngine; one second of 440 Hz speech."""

    def synth(self, text, voice, *, speed=1.0, seed=0):
        samples = (0.3 * np.sin(2 * math.pi * 440 * np.arange(24000) / 24000)).astype("float32")
        return SynthResult(samples=samples, sample_rate=24000, duration_s=1.0)


def _probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def _wav_duration(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def _run_tts(workdir, config_path, monkeypatch):
    monkeypatch.setattr(
        "pipeline.stages.tts.create_tts_engine", lambda config: FakeEngine()
    )
    assert run("tts", workdir, config_path) == 0


def test_registry_places_dub_after_tts_and_keeps_default_subtitle_product():
    import video_transcribe

    from pipeline.registry import ARTIFACTS, STAGES, producer_of, validate_registry

    validate_registry()
    ids = [stage.id for stage in STAGES]
    assert ids.index("dub") == ids.index("tts") + 1
    assert ids.index("burn") == ids.index("dub") + 1
    dub = STAGES[ids.index("dub")]
    assert dub.requires == ("source_video", "tts_audio")
    assert dub.produces == ("output_dub",)
    assert ARTIFACTS["output_dub"].relpath == "output_dub.mp4"
    assert ARTIFACTS["output_video"].relpath == "output.mp4"
    assert producer_of("output_dub") == "dub"
    # The subtitle product's default entry never dubs either.
    assert video_transcribe.ALL_STAGES == "ingest,transcribe,segment,translate,subtitle,burn"
    assert "dub" not in video_transcribe.ALL_STAGES


def test_dub_stage_requires_tts_track_and_source_video(tmp_path, monkeypatch):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    monkeypatch.setattr(
        "pipeline.stages.tts.create_tts_engine", lambda config: FakeEngine()
    )

    # No tts/tts.wav yet: missing input, not a stage failure.
    assert run("dub", workdir, _config(tmp_path)) == 2
    assert not (workdir / "output_dub.mp4").exists()

    _run_tts(workdir, _config(tmp_path), monkeypatch)
    (workdir / "media/source.mp4").unlink()
    assert run("dub", workdir, _config(tmp_path)) == 2
    assert not (workdir / "output_dub.mp4").exists()


def test_dub_stage_copies_video_and_replaces_audio(tmp_path, monkeypatch):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    _run_tts(workdir, _config(tmp_path), monkeypatch)

    assert run("dub", workdir, _config(tmp_path)) == 0

    output = workdir / "output_dub.mp4"
    assert output.is_file() and output.stat().st_size > 0
    output_probe = _probe(output)
    source_probe = _probe(workdir / "media/source.mp4")
    streams = {stream["codec_type"]: stream["codec_name"] for stream in output_probe["streams"]}
    # Whole-track replacement: the source video stream goes through untouched,
    # the audio is the synthesized track encoded as AAC (ADR-0003).
    assert streams == {"video": source_probe["streams"][0]["codec_name"], "audio": "aac"}
    assert abs(
        float(output_probe["format"]["duration"]) - float(source_probe["format"]["duration"])
    ) <= 0.5
    assert run("dub", workdir, _config(tmp_path), validate_only=True) == 0


def test_dub_stage_fails_when_track_overshoots_the_drift_gate(tmp_path, monkeypatch, capsys):
    # A silence bed 3s longer than the video (tts tail overflow) must fail the
    # 0.5s gate, not trim the speech (ADR-0003: nothing is clipped).
    _voices(tmp_path)
    workdir = _workdir(tmp_path, audio_duration=8.0)
    _run_tts(workdir, _config(tmp_path), monkeypatch)
    assert _wav_duration(workdir / "tts/tts.wav") == pytest.approx(8.0, abs=0.05)

    assert run("dub", workdir, _config(tmp_path)) == 1

    error = capsys.readouterr().err
    assert "stage 'dub'" in error
    assert "differs from source" in error
    assert not (workdir / "output_dub.mp4").exists()


def test_dub_stage_leaves_burn_output_untouched(tmp_path, monkeypatch):
    """output.mp4 and output_dub.mp4 are independent artifacts."""
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    subs = workdir / "subs"
    subs.mkdir()
    shutil.copyfile(REPO_ROOT / "tests/fixtures/srt/sample_5s_source.srt", subs / "source.srt")
    shutil.copyfile(REPO_ROOT / "tests/fixtures/srt/sample_5s_target.srt", subs / "target.srt")
    config = _config(tmp_path)

    assert run("burn", workdir, config) == 0
    burned = (workdir / "output.mp4").read_bytes()
    _run_tts(workdir, config, monkeypatch)
    assert run("dub", workdir, config) == 0

    assert (workdir / "output.mp4").read_bytes() == burned
    assert (workdir / "output_dub.mp4").is_file()
    # Both artifacts survive independently in the same workdir.
    for name in ("output.mp4", "output_dub.mp4"):
        probe = _probe(workdir / name)
        assert {stream["codec_type"] for stream in probe["streams"]} == {"video", "audio"}


def test_dub_stage_rejects_corrupt_source_without_output(tmp_path, monkeypatch, capsys):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    _run_tts(workdir, _config(tmp_path), monkeypatch)
    (workdir / "media/source.mp4").write_bytes(b"not a video")

    assert run("dub", workdir, _config(tmp_path)) == 1

    error = capsys.readouterr().err
    assert "stage 'dub'" in error
    assert "ffprobe" in error.lower() or "cannot read" in error.lower()
    assert not (workdir / "output_dub.mp4").exists()
