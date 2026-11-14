"""Tests for the F5-TTS seam and the named voice library (ADR-0004).

The real-synthesis tests need the pre-downloaded weights, the voice assets,
and a GPU; they are skipped (not failed) when any of those is absent.
"""

import json
import math
from pathlib import Path

import pytest

from pipeline.tts.engine import (
    F5TTSEngine,
    install_audio_loader_patch,
)
from pipeline.tts.voices import VoiceLibrary

REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "_model_cache" / "F5-TTS-Emilia-ZH-EN" / "model_1250000.safetensors"
VOCAB = REPO / "_model_cache" / "F5-TTS-Emilia-ZH-EN" / "vocab.txt"
VOCODER = REPO / "_model_cache" / "vocos-mel-24khz"
VOICES = REPO / "_model_cache" / "tts_voices"
ASSETS_OK = CKPT.is_file() and VOCAB.is_file() and VOCODER.is_dir() and VOICES.is_dir()


def _write_wav(path, seconds=1.0, sr=24000):
    import numpy as np
    import soundfile as sf

    t = np.arange(int(sr * seconds)) / sr
    sf.write(path, (0.2 * np.sin(2 * math.pi * 220 * t)).astype("float32"), sr)


def test_audio_loader_patch_routes_through_soundfile(tmp_path):
    import soundfile as sf
    import torch
    import torchaudio

    wav = tmp_path / "a.wav"
    _write_wav(wav)
    install_audio_loader_patch()
    tensor, sr = torchaudio.load(str(wav))
    data, native_sr = sf.read(str(wav), dtype="float32", always_2d=True)
    assert sr == native_sr
    assert tensor.shape == (data.shape[1], data.shape[0])  # (channels, time)
    assert torch.allclose(tensor, torch.from_numpy(data.T))


def test_voice_library_roundtrip(tmp_path):
    _write_wav(tmp_path / "v.wav")
    (tmp_path / "voices.json").write_text(json.dumps([
        {"name": "zh-female-01", "ref_audio": "v.wav", "ref_text": "你好。", "language": "zh"},
        {"name": "en-male-01", "ref_audio": "v.wav", "ref_text": "Hello.", "language": "en"},
    ]))
    lib = VoiceLibrary(tmp_path)
    voices = lib.load()
    assert set(voices) == {"zh-female-01", "en-male-01"}
    assert lib.get("en-male-01").ref_text == "Hello."
    assert lib.get("zh-female-01").ref_audio == tmp_path / "v.wav"


def test_voice_library_reports_all_problems(tmp_path):
    _write_wav(tmp_path / "v.wav")
    (tmp_path / "voices.json").write_text(json.dumps([
        {"name": "a", "ref_audio": "missing.wav", "ref_text": "", "language": "zh"},
        {"name": "a", "ref_audio": "v.wav", "ref_text": "x", "language": ""},
        {"name": "", "ref_audio": "v.wav", "ref_text": "x", "language": "en"},
    ]))
    with pytest.raises(ValueError) as exc:
        VoiceLibrary(tmp_path).load()
    text = str(exc.value)
    assert "missing.wav" in text and "empty ref_text" in text
    assert "duplicate voice name" in text and "missing language" in text
    assert "manifest entry missing name" in text


def test_voice_library_unknown_name_lists_available(tmp_path):
    _write_wav(tmp_path / "v.wav")
    (tmp_path / "voices.json").write_text(json.dumps([
        {"name": "zh-female-01", "ref_audio": "v.wav", "ref_text": "你好。", "language": "zh"},
    ]))
    with pytest.raises(ValueError) as exc:
        VoiceLibrary(tmp_path).get("nope")
    assert "zh-female-01" in str(exc.value)


@pytest.mark.slow
@pytest.mark.skipif(
    not (ASSETS_OK and VOICES.is_dir()),
    reason="F5-TTS weights, vocos vocoder, or voice assets missing",
)
class TestRealSynthesis:
    @pytest.fixture(scope="class")
    def engine(self):
        return F5TTSEngine()

    @pytest.fixture(scope="class")
    def voices(self):
        return VoiceLibrary(VOICES).load()

    def test_synth_zh_and_en_measured_duration(self, engine, voices):
        for name, text in [
            ("zh-female-01", "这是语音合成验收的第一段中文测试文本。"),
            ("en-male-01", "This is the first English acceptance sentence."),
        ]:
            result = engine.synth(text, voices[name], seed=7)
            assert result.sample_rate == 24000
            assert 0.3 <= result.duration_s <= 30.0
            assert result.samples.shape[-1] == round(result.duration_s * result.sample_rate)
            assert all(v == v for v in result.samples.tolist()[:100])  # no NaN

    def test_seed_determinism(self, engine, voices):
        import numpy as np

        text = "固定随机种子应当带来可复现的合成结果。"
        a = engine.synth(text, voices["zh-female-01"], seed=42)
        b = engine.synth(text, voices["zh-female-01"], seed=42)
        assert a.duration_s == b.duration_s
        assert a.samples.shape == b.samples.shape
        assert np.allclose(a.samples, b.samples, rtol=1e-4, atol=1e-4)

    def test_unsupported_attn_backend_rejected(self):
        # Constructor validation fires before any weight load, so this
        # stays cheap even under --run-slow.
        with pytest.raises(ValueError):
            F5TTSEngine(attn_backend="warp-drive")
