"""TTS stage tests: per-segment synthesis, slot fitting, artifact contract (#2).

The public seam is the stage runner (stage id + workdir -> artifacts and exit
code) and the artifact validators, per docs/agents/verification.md. Real F5
synthesis is marked slow; the fast suite drives a deterministic fake engine
through the same stage seam.
"""

import json
import math
import shutil
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
AUDIO = REPO_ROOT / "tests/fixtures/audio/zh_30s.wav"
CKPT = REPO_ROOT / "_model_cache" / "F5-TTS-Emilia-ZH-EN" / "model_1250000.safetensors"
VOCAB = REPO_ROOT / "_model_cache" / "F5-TTS-Emilia-ZH-EN" / "vocab.txt"
VOCODER = REPO_ROOT / "_model_cache" / "vocos-mel-24khz"
VOICES = REPO_ROOT / "_model_cache" / "tts_voices"
ASSETS_OK = CKPT.is_file() and VOCAB.is_file() and VOCODER.is_dir() and VOICES.is_dir()


def _config(tmp_path, **tts_overrides):
    tts = {"voice": "test-voice", "voices_dir": str(tmp_path / "voices")}
    tts.update(tts_overrides)
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps({"tts": tts}), encoding="utf-8")
    return path


def _voices(tmp_path):
    """A minimal valid named-voice library for the stage to select from."""
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()
    sf.write(
        voices_dir / "test-voice.wav",
        (0.2 * np.sin(2 * math.pi * 220 * np.arange(2400) / 24000)).astype("float32"),
        24000,
        subtype="PCM_16",
    )
    sf.write(
        voices_dir / "alt-voice.wav",
        (0.2 * np.sin(2 * math.pi * 180 * np.arange(2400) / 24000)).astype("float32"),
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
            {
                "name": "alt-voice",
                "ref_audio": "alt-voice.wav",
                "ref_text": "另一个参考文本。",
                "language": "zh",
            },
        ]),
        encoding="utf-8",
    )
    return voices_dir


def _workdir(tmp_path):
    workdir = tmp_path / "work"
    (workdir / "media").mkdir(parents=True)
    shutil.copyfile(TRANSLATION, workdir / "translation.json")
    shutil.copyfile(SEGMENTS, workdir / "segments.json")
    shutil.copyfile(AUDIO, workdir / "media/audio.wav")
    return workdir


class FakeEngine:
    """Deterministic stand-in for F5TTSEngine; records every synth call."""

    def __init__(self, duration_by_text=None, default_duration=1.0):
        self.duration_by_text = duration_by_text or {}
        self.default_duration = default_duration
        self.calls = []

    def synth(self, text, voice, *, speed=1.0, seed=0):
        self.calls.append({"text": text, "voice": voice.name, "speed": speed})
        duration = self.duration_by_text.get(text, self.default_duration)
        samples = (0.3 * np.sin(2 * math.pi * 440 * np.arange(int(24000 * duration)) / 24000)).astype("float32")
        return SynthResult(samples=samples, sample_rate=24000, duration_s=float(samples.shape[-1] / 24000))


def _slot_energies(track_path, plan):
    samples, sr = sf.read(str(track_path), dtype="float32", always_2d=False)
    rate = int(sr)
    energies = []
    for segment in plan["segments"]:
        lo = int(segment["placed_start"] * rate)
        hi = int(segment["placed_end"] * rate)
        window = samples[lo:hi]
        energies.append(float(np.sqrt(np.mean(window * window))))
    return energies


def test_tts_stage_produces_plan_and_silence_bed_track(tmp_path, monkeypatch):
    voices_dir = _voices(tmp_path)
    workdir = _workdir(tmp_path)
    engine = FakeEngine(default_duration=1.0)
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: engine)

    assert run("tts", workdir, _config(tmp_path)) == 0

    plan = json.loads((workdir / "tts/dub_plan.json").read_text(encoding="utf-8"))
    assert plan["schema_version"] == 1
    assert plan["input"] == "translated"
    assert plan["voice"] == "test-voice"
    assert plan["sample_rate"] == 24000
    assert plan["source_duration_s"] == pytest.approx(30.0, abs=1e-3)
    assert [segment["id"] for segment in plan["segments"]] == [1, 2]
    # Timestamps come from the measured audio: 1.0s of speech at each slot start.
    first, second = plan["segments"]
    assert first["slot_start"] == pytest.approx(0.5, abs=1e-3)
    assert first["placed_start"] == pytest.approx(0.5, abs=1e-3)
    assert first["placed_end"] == pytest.approx(1.5, abs=0.05)
    assert first["overflow_s"] == pytest.approx(0.0, abs=1e-6)
    assert second["placed_start"] == pytest.approx(4.0, abs=1e-3)

    # Full-track replacement semantics: speech at placements, silence elsewhere.
    energies = _slot_energies(workdir / "tts/tts.wav", plan)
    assert energies[0] > 0.01
    assert energies[1] > 0.01
    samples, sr = sf.read(str(workdir / "tts/tts.wav"), dtype="float32")
    tail = samples[int(6.0 * sr):int(8.0 * sr)]
    assert float(np.sqrt(np.mean(tail * tail))) < 1e-4

    assert engine.calls[0]["voice"] == "test-voice"
    assert engine.calls[0]["text"].startswith("这是一段")
    assert run("tts", workdir, _config(tmp_path), validate_only=True) == 0


SEGMENT_ONE_TEXT = (
    "这是一段故意写得很长的中文译文，用来强制字幕按显示长度拆分成多条字幕，"
    "并验证源文和译文的时间戳完全一致。"
)
SEGMENT_TWO_TEXT = "短句。"


def test_tts_stage_stretches_overlong_speech_into_its_slot(tmp_path, monkeypatch):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    # Slot 1 is 3.5s; 4.0s of speech needs 1.1429x -- inside the cap.
    # Slot 2 is 2.0s; speech synthesized to exactly 2.0s fits untouched.
    engine = FakeEngine(
        duration_by_text={SEGMENT_ONE_TEXT: 4.0, SEGMENT_TWO_TEXT: 2.0}
    )
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: engine)

    assert run("tts", workdir, _config(tmp_path)) == 0

    plan = json.loads((workdir / "tts/dub_plan.json").read_text(encoding="utf-8"))
    first, second = plan["segments"]
    assert first["method"] == "atempo"
    assert first["synthesized_duration_s"] == pytest.approx(4.0, abs=1e-3)
    assert first["speed_factor"] == pytest.approx(4.0 / 3.5, abs=1e-5)
    assert first["speed_factor"] <= 1.25
    assert first["placed_end"] == pytest.approx(4.0, abs=0.05)
    assert first["overflow_s"] == pytest.approx(0.0, abs=1e-3)
    assert second["method"] == "none"
    # The stretched speech really fills the slot: energy right up to slot end.
    samples, sr = sf.read(str(workdir / "tts/tts.wav"), dtype="float32")
    edge = samples[int(3.8 * sr):int(4.0 * sr)]
    assert float(np.sqrt(np.mean(edge * edge))) > 0.01
    assert run("tts", workdir, _config(tmp_path), validate_only=True) == 0


def test_tts_stage_caps_speed_and_accounts_overflow(tmp_path, monkeypatch):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    # Slot 1 is 3.5s: 6.0s of speech overflows even at the 1.25x cap.
    engine = FakeEngine(
        duration_by_text={SEGMENT_ONE_TEXT: 6.0, SEGMENT_TWO_TEXT: 1.0}
    )
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: engine)

    assert run("tts", workdir, _config(tmp_path)) == 0

    plan = json.loads((workdir / "tts/dub_plan.json").read_text(encoding="utf-8"))
    first, second = plan["segments"]
    assert first["method"] == "atempo"
    assert first["speed_factor"] == pytest.approx(1.25)  # capped, not 1.714
    assert first["placed_end"] == pytest.approx(0.5 + 6.0 / 1.25, abs=0.05)
    assert first["overflow_s"] == pytest.approx(6.0 / 1.25 - 3.5, abs=0.05)
    assert second["method"] == "none"
    assert plan["overflow"]["count"] == 1
    assert plan["overflow"]["total_s"] == pytest.approx(first["overflow_s"], abs=1e-3)
    assert plan["overflow"]["ratio"] == pytest.approx(
        first["overflow_s"] / plan["source_duration_s"], abs=1e-5
    )
    assert run("tts", workdir, _config(tmp_path), validate_only=True) == 0


def test_tts_stage_cache_hit_skips_synthesis(tmp_path, monkeypatch):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    engine = FakeEngine(default_duration=1.0)
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: engine)
    config = _config(tmp_path)

    assert run("tts", workdir, config) == 0
    assert len(engine.calls) == 2
    cache_dir = workdir / ".tts-cache"
    assert len(list(cache_dir.glob("*.wav"))) == 2

    # Warm cache: no new synthesis, the engine is never rebuilt.
    warm_calls = len(engine.calls)
    assert run("tts", workdir, config) == 0
    assert len(engine.calls) == warm_calls
    assert run("tts", workdir, config, validate_only=True) == 0

    # The key is voice + text + speed: another voice is a miss.
    assert run("tts", workdir, _config(tmp_path, voice="alt-voice")) == 0
    assert [call["voice"] for call in engine.calls[warm_calls:]] == [
        "alt-voice",
        "alt-voice",
    ]

    # Cache disabled: synthesis happens again, no cache entries are added.
    before = len(list(cache_dir.glob("*.wav")))
    assert run("tts", workdir, _config(tmp_path, cache=False)) == 0
    assert len(engine.calls) == warm_calls + 4
    assert len(list(cache_dir.glob("*.wav"))) == before


def test_tts_stage_input_source_speaks_source_text(tmp_path, monkeypatch):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    engine = FakeEngine(default_duration=1.0)
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: engine)

    assert run("tts", workdir, _config(tmp_path, input="source")) == 0

    plan = json.loads((workdir / "tts/dub_plan.json").read_text(encoding="utf-8"))
    assert plan["input"] == "source"
    assert engine.calls[0]["text"].startswith("This is a deliberately long")
    assert plan["segments"][0]["text"].startswith("This is a deliberately long")
    assert run("tts", workdir, _config(tmp_path, input="source"), validate_only=True) == 0


def test_tts_stage_synthesis_speed_flows_to_engine_plan_and_cache(tmp_path, monkeypatch):
    """tts.speed (ADR-0005) drives the engine call, the plan contract and the
    cache key: entries made at one speed never serve another."""
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    engine = FakeEngine(default_duration=1.0)
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: engine)

    assert run("tts", workdir, _config(tmp_path, speed=1.6)) == 0

    assert {call["speed"] for call in engine.calls} == {1.6}
    plan = json.loads((workdir / "tts/dub_plan.json").read_text(encoding="utf-8"))
    assert plan["synth_speed"] == pytest.approx(1.6)
    cache_dir = workdir / ".tts-cache"
    entries_at_1_6 = len(list(cache_dir.glob("*.wav")))

    # The same workdir at the default speed is a full cache miss: the key
    # includes the speed, so the engine synthesizes both segments again.
    assert run("tts", workdir, _config(tmp_path)) == 0
    assert {call["speed"] for call in engine.calls[len(engine.calls) - 2:]} == {1.0}
    assert len(list(cache_dir.glob("*.wav"))) == entries_at_1_6 + 2
    assert run("tts", workdir, _config(tmp_path, speed=1.6), validate_only=True) == 0


def test_tts_stage_rejects_out_of_range_speed(tmp_path, monkeypatch):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: FakeEngine())

    # Config errors exit 2 before any engine is built or artifact written.
    assert run("tts", workdir, _config(tmp_path, speed=2.5)) == 2
    assert run("tts", workdir, _config(tmp_path, speed="fast")) == 2
    assert not (workdir / "tts/dub_plan.json").exists()


def test_tts_stage_requires_source_audio(tmp_path, monkeypatch):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    (workdir / "media/audio.wav").unlink()
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: FakeEngine())

    assert run("tts", workdir, _config(tmp_path)) == 2
    assert not (workdir / "tts/dub_plan.json").exists()


def test_tts_stage_leaves_sub_permille_overshoot_unstretched(tmp_path, monkeypatch):
    """A factor below FIT_EPSILON must stay 'none', or the stage produces a
    plan its own validator rejects (a stretched speed rounds onto 1.0)."""
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    engine = FakeEngine(
        duration_by_text={SEGMENT_ONE_TEXT: 3.5007, SEGMENT_TWO_TEXT: 1.0}
    )
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: engine)

    assert run("tts", workdir, _config(tmp_path)) == 0

    plan = json.loads((workdir / "tts/dub_plan.json").read_text(encoding="utf-8"))
    first = plan["segments"][0]
    assert first["method"] == "none"
    assert first["speed_factor"] == pytest.approx(1.0)
    assert first["overflow_s"] == pytest.approx(0.0007, abs=1e-3)
    assert run("tts", workdir, _config(tmp_path), validate_only=True) == 0


def test_tts_plan_and_audio_validators_reject_tampering(tmp_path, monkeypatch):
    _voices(tmp_path)
    workdir = _workdir(tmp_path)
    # Segment 1 needs an atempo fit so the cap/method mutations are real.
    engine = FakeEngine(
        duration_by_text={SEGMENT_ONE_TEXT: 4.0, SEGMENT_TWO_TEXT: 1.0}
    )
    monkeypatch.setattr("pipeline.stages.tts.create_tts_engine", lambda config: engine)
    config = _config(tmp_path)
    assert run("tts", workdir, config) == 0

    plan_path = workdir / "tts/dub_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    def _validate():
        return run("tts", workdir, config, validate_only=True)

    def _restore():
        plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    def _mutate(mutator):
        mutated = json.loads(json.dumps(plan))
        mutator(mutated)
        plan_path.write_text(json.dumps(mutated, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            return _validate()
        finally:
            _restore()

    # Speed beyond the fixed 1.25x cap breaks the contract...
    assert _mutate(lambda p: p["segments"][0].__setitem__("speed_factor", 1.3)) == 1
    assert _mutate(lambda p: p["segments"][0].__setitem__("method", "none")) == 1
    assert _mutate(lambda p: p["segments"][0].__setitem__("text", "被篡改的文本。")) == 1
    assert _mutate(lambda p: p["segments"][0].__setitem__("slot_end", 99.0)) == 1
    assert _mutate(lambda p: p.__setitem__("schema_version", 2)) == 1
    # The synthesis speed is part of the contract and the cache key.
    assert _mutate(lambda p: p.pop("synth_speed")) == 1
    assert _mutate(lambda p: p.__setitem__("synth_speed", 3.0)) == 1
    assert _mutate(lambda p: p["segments"].pop()) == 1
    assert _mutate(lambda p: p["overflow"].__setitem__("total_s", 99.0)) == 1
    # The track contract is 24 kHz mono, enforced against the plan too.
    assert _mutate(lambda p: p.__setitem__("sample_rate", 48000)) == 1

    # ...and so does audio that no longer matches the plan (50 ms gate).
    samples, sr = sf.read(str(workdir / "tts/tts.wav"), dtype="float32")
    sf.write(str(workdir / "tts/tts.wav"), samples[: int(29.0 * sr)], sr, subtype="PCM_16")
    assert _validate() == 1

    sf.write(str(workdir / "tts/tts.wav"), samples, sr, subtype="PCM_16")
    assert _validate() == 0


def test_registry_places_tts_after_subtitle_and_keeps_default_six_stages():
    import video_transcribe

    from pipeline.registry import ARTIFACTS, STAGES, validate_registry

    validate_registry()
    ids = [stage.id for stage in STAGES]
    assert ids.index("tts") == ids.index("subtitle") + 1
    assert set(STAGES[ids.index("tts")].produces) == {"dub_plan", "tts_audio"}
    assert ARTIFACTS["dub_plan"].relpath == "tts/dub_plan.json"
    assert ARTIFACTS["tts_audio"].relpath == "tts/tts.wav"
    # The subtitle product's default entry never synthesizes speech.
    assert video_transcribe.ALL_STAGES == "ingest,transcribe,segment,translate,subtitle,burn"
    assert "tts" not in video_transcribe.ALL_STAGES


@pytest.mark.slow
@pytest.mark.skipif(
    not (ASSETS_OK and VOICES.is_dir()),
    reason="F5-TTS weights, vocoder, or voice assets missing",
)
def test_real_tts_stage_synthesis_and_cache(tmp_path, monkeypatch):
    """Real F5-TTS through the stage seam, on the real voice library."""
    workdir = _workdir(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        json.dumps({"tts": {"voice": "zh-female-01", "voices_dir": str(VOICES)}}),
        encoding="utf-8",
    )

    import pipeline.stages.tts as tts_stage
    from pipeline.tts.factory import create_tts_engine as real_create

    builds = []

    def counting_create(config):
        builds.append(1)
        return real_create(config)

    monkeypatch.setattr(tts_stage, "create_tts_engine", counting_create)

    assert run("tts", workdir, config_path) == 0
    assert len(builds) == 1  # one engine for both segments

    plan = json.loads((workdir / "tts/dub_plan.json").read_text(encoding="utf-8"))
    track, sr = sf.read(str(workdir / "tts/tts.wav"), dtype="float32")
    assert plan["sample_rate"] == sr
    assert plan["source_duration_s"] == pytest.approx(30.0, abs=1e-3)
    for segment in plan["segments"]:
        lo = int(segment["placed_start"] * sr)
        hi = int(segment["placed_end"] * sr)
        # Plan timestamps come from measured audio: the placed window holds
        # speech, and its length is the plan duration within 50 ms.
        assert (hi - lo) / sr == pytest.approx(
            segment["placed_end"] - segment["placed_start"], abs=0.05
        )
        window = track[lo:hi]
        assert float(np.sqrt(np.mean(window * window))) > 0.01
        assert segment["speed_factor"] <= 1.25

    # Cache hit: rerunning synthesizes nothing and still validates.
    builds.clear()
    assert run("tts", workdir, config_path) == 0
    assert builds == []
    assert run("tts", workdir, config_path, validate_only=True) == 0
