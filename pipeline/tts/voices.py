"""Named voice library (ADR-0004).

A voice is a fixed reference clip plus its transcript: selecting a voice
means picking a name from the manifest, not cloning on the fly. The assets
live in a gitignored directory so no NC-licensed audio enters the public
repository; the manifest (`voices.json`) sits next to the clips it points
to, so the whole library is relocatable as one directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VOICES_DIR = Path(__file__).resolve().parents[2] / "_model_cache" / "tts_voices"


@dataclass(frozen=True)
class Voice:
    name: str
    ref_audio: Path
    ref_text: str
    language: str


class VoiceLibrary:
    def __init__(self, voices_dir: Path | str = DEFAULT_VOICES_DIR):
        self.voices_dir = Path(voices_dir)
        self.manifest_path = self.voices_dir / "voices.json"

    def load(self) -> dict[str, Voice]:
        """Load and validate the manifest; report every problem at once."""
        if not self.manifest_path.is_file():
            raise ValueError(f"voice manifest not found: {self.manifest_path}")
        entries = json.loads(self.manifest_path.read_text())
        voices: dict[str, Voice] = {}
        problems: list[str] = []
        for entry in entries:
            name = entry.get("name")
            if not name:
                problems.append("manifest entry missing name")
                continue
            if name in voices:
                problems.append(f"{name}: duplicate voice name")
            ref_audio = self.voices_dir / str(entry.get("ref_audio", ""))
            if not ref_audio.is_file() or ref_audio.stat().st_size == 0:
                problems.append(f"{name}: reference audio missing or empty: {ref_audio}")
            if not str(entry.get("ref_text") or "").strip():
                problems.append(f"{name}: empty ref_text")
            if not str(entry.get("language") or "").strip():
                problems.append(f"{name}: missing language")
            voices[name] = Voice(
                name=name,
                ref_audio=ref_audio,
                ref_text=str(entry.get("ref_text") or ""),
                language=str(entry.get("language") or ""),
            )
        if problems:
            raise ValueError(
                "invalid voice library:\n  " + "\n  ".join(problems)
            )
        return voices

    def get(self, name: str) -> Voice:
        voices = self.load()
        if name not in voices:
            raise ValueError(
                f"unknown voice {name!r}; available: {', '.join(sorted(voices))}"
            )
        return voices[name]
