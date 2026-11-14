"""F5-TTS inference seam (ADR-0004).

Local-only runtime: the checkpoint, vocab, and vocoder are all resolved from
paths downloaded ahead of time (ModelScope weights plus the vocos vocoder);
nothing downloads at inference time. Synthesis is per-segment from a fixed
reference clip, which is what keeps the voice stable across a 1h run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pipeline.tts.fitting import MAX_SYNTH_SPEED, MIN_SYNTH_SPEED

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT_FILE = REPO_ROOT / "_model_cache" / "F5-TTS-Emilia-ZH-EN" / "model_1250000.safetensors"
DEFAULT_VOCAB_FILE = REPO_ROOT / "_model_cache" / "F5-TTS-Emilia-ZH-EN" / "vocab.txt"
DEFAULT_VOCODER_PATH = REPO_ROOT / "_model_cache" / "vocos-mel-24khz"

SUPPORTED_ATTN_BACKENDS = ("sdpa",)


def _soundfile_load(filepath, *_args, **_kwargs):
    """`torchaudio.load` replacement backed by soundfile.

    torchaudio 2.11 routes load() through torchcodec, whose bundled library
    links against libnvrtc.so.13 and fails to import on this driver stack,
    so every torchaudio.load call site — f5_tts.infer.utils_infer included —
    raises OSError as-is. The replacement restores the documented contract
    (a (channels, time) float32 tensor plus the sample rate) without
    touching the dependency; the function it replaces was already unusable
    in this environment.
    """
    import soundfile as sf

    data, sr = sf.read(str(filepath), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), int(sr)


def install_audio_loader_patch() -> None:
    """Idempotently route torchaudio.load through soundfile (see above)."""
    import torchaudio

    if getattr(torchaudio, "_vt_sf_load_patched", False):
        return
    torchaudio.load = _soundfile_load
    torchaudio._vt_sf_load_patched = True


@dataclass(frozen=True)
class SynthResult:
    samples: "np.ndarray"  # (time,) float32, CPU — ffmpeg/soundfile-ready
    sample_rate: int
    duration_s: float


class F5TTSEngine:
    """Thin wrapper over f5_tts.api.F5TTS with local-only asset resolution."""

    def __init__(
        self,
        ckpt_file: Path | str = DEFAULT_CKPT_FILE,
        vocab_file: Path | str = DEFAULT_VOCAB_FILE,
        vocoder_local_path: Path | str = DEFAULT_VOCODER_PATH,
        device: str | None = None,
        attn_backend: str = "sdpa",
    ):
        missing = [str(p) for p in (ckpt_file, vocab_file) if not Path(p).is_file()]
        if missing:
            raise FileNotFoundError(
                "F5-TTS assets not pre-downloaded (runtime must not download): "
                + ", ".join(missing)
            )
        if not Path(vocoder_local_path).is_dir():
            raise FileNotFoundError(
                f"vocos vocoder not pre-downloaded: {vocoder_local_path}"
            )
        if attn_backend not in SUPPORTED_ATTN_BACKENDS:
            raise ValueError(
                f"unsupported attn_backend {attn_backend!r}; "
                f"available: {', '.join(SUPPORTED_ATTN_BACKENDS)}"
            )
        self.attn_backend = attn_backend
        install_audio_loader_patch()
        from f5_tts.api import F5TTS

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=str(ckpt_file),
            vocab_file=str(vocab_file),
            vocoder_local_path=str(vocoder_local_path),
            device=self.device,
        )

    def synth(
        self,
        text: str,
        voice,
        *,
        speed: float = 1.0,
        seed: int = 0,
        nfe_step: int = 32,
        cross_fade_duration: float = 0.15,
    ) -> SynthResult:
        """Synthesize one segment; duration is measured off the actual audio."""
        if not text.strip():
            raise ValueError("synthesis text is empty")
        if not (MIN_SYNTH_SPEED <= speed <= MAX_SYNTH_SPEED):
            raise ValueError(
                f"speed out of supported range "
                f"[{MIN_SYNTH_SPEED}, {MAX_SYNTH_SPEED}]: {speed}"
            )
        wav, sr, _spec = self.model.infer(
            ref_file=str(voice.ref_audio),
            ref_text=voice.ref_text,
            gen_text=text,
            speed=float(speed),
            seed=int(seed),
            nfe_step=nfe_step,
            cross_fade_duration=cross_fade_duration,
            show_info=lambda *_: None,
        )
        # f5_tts.api.infer hands back the final wave as a numpy array.
        import numpy as np

        samples = np.asarray(wav, dtype="float32").reshape(-1)
        return SynthResult(
            samples=samples,
            sample_rate=int(sr),
            duration_s=float(samples.shape[-1] / int(sr)),
        )
