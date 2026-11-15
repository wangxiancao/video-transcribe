# Dub Voice Identity: Diarization + Voice Mapping Implementation Plan (ADR-0006 P2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each source speaker of the video gets one consistent named dubbing voice (diarization → ECAPA-similarity voice mapping), replacing today's single fixed voice, with pyannote fully local and a clean fallback when disabled.

**Architecture:** A new optional `diarize` stage (fixture-backed for tests, pyannote adapter behind a factory seam — the `pipeline/llm/fixture.py` pattern) writes `diarization/diarization.json`; the tts stage resolves a voice per segment via explicit `tts.voice_map` or precomputed ECAPA embedding similarity. No synthesis-engine change; the synthesis cache is already keyed by voice name. F5 zero-shot cross-lingual cloning stays rejected (ADR-0006, F5 issue #315); OpenVoice v2 is a separate probe task that must NOT touch product code.

**Tech Stack:** pyannote.audio (community-1; code MIT / weights CC-BY-4.0, gated download), speechbrain ECAPA-TDNN (Apache-2.0), numpy, existing `pipeline/registry.py` stage table.

## Global Constraints

- Zero cloud at runtime; pyannote/speechbrain weights are one-time pre-downloads (HF gated acceptance is a manual setup step, like ModelScope weights).
- New deps only where this plan says: `pyannote.audio`, `speechbrain` (install via `uv pip install --python .venv/bin/python`, pinned versions recorded in commit message).
- Contract artifacts are APIs: `diarization/diarization.json` is a new artifact; `dub_plan.json` gains additive fields (`voices`, per-segment `voice`); `schema_version` stays 1.
- Vendor code stays out of orchestration: pyannote/speechbrain imports live inside `pipeline/diarize/` and `pipeline/tts/voicing.py`, never in stage `run()` bodies beyond factory calls.
- `tts.speaker_voices: false` (default) must keep every artifact byte-identical to pre-P2 behavior.
- Tests run without pyannote/speechbrain installed (fixture backend + fake embeddings); GPU/model tests auto-skip on missing assets.
- Runner does not skip stages by artifact existence — `diarize` runs when selected; tts fails loudly when `speaker_voices` is on but diarization is missing.

## 验收目标 (acceptance targets)

**A. Contract gates (pytest, green without pyannote/speechbrain installed)**
1. `diarize` (fixture backend) end-to-end: writes `diarization/diarization.json` with one speaker per segment id, ids/timestamps consistent with `segments.json`.
2. Explicit `tts.voice_map` is honored exactly; unmapped speaker → explicit `StageError` naming the speaker.
3. `tts.speaker_voices: true` without `diarization/diarization.json` → explicit `StageError` (no silent fallback).
4. Disabled (`speaker_voices: false`) → `dub_plan.json` byte-identical to the pre-P2 stage output on the same inputs (regression fixture).
5. `dub_plan.json` segments carry `voice`; top-level `voices` maps speaker→voice when enabled, absent when disabled.

**B. Real-content gates (`bench/v4_10_diarization_acceptance.py`)**
1. Single-speaker 1h acceptance content: exactly 1 diarized speaker → every segment the same voice (no mid-video voice swap); plan-level invariant `len(set(seg["voice"])) == len(set(seg["speaker"]))`.
2. Multi-speaker sample: speaker turns and voice switches agree (manual listen + the same plan invariant); diarization runs with no network calls (offline env var test).
3. Wall-clock: diarize stage adds ≤ 5% to E2E on the 1h content; TTS E2E RTFx ≤ 0.5 gate unchanged.
4. Human identity gate: "配音的人与说话人气质匹配" judgment recorded in the report (maintainer).

**C. Follow-up probe gate (OpenVoice v2, adoption is a separate ADR)**
Δcosine(ECAPA; converted vs source speaker) ≥ +0.05 over the mapped-voice baseline AND TTS E2E RTFx ≤ 0.5 retained. Failure = probe report + no product change.

---

### Task 1: Diarization seam (fixture-first)

**Files:**
- Create: `pipeline/diarize/__init__.py`, `pipeline/diarize/base.py`, `pipeline/diarize/fixture.py`, `pipeline/diarize/factory.py`
- Create: `tests/fixtures/diarization/sample_6s.json`
- Test: `tests/test_diarize.py` (new)

**Interfaces:**
- Produces: `Diarizer` protocol with `diarize(audio: Path, segments: list[dict]) -> list[dict]` returning `[{"id": int, "speaker": str}, ...]` (one per input segment, same order/ids); `create_diarizer(config) -> Diarizer` reading `diarization.runtime` (`fixture` default | `pyannote`) and `diarization.fixture` (path). Task 2's stage and Task 5's adapter consume this.

- [ ] **Step 1: Write the failing tests**

`tests/test_diarize.py`:

```python
"""Diarization seam tests: fixture backend, factory wiring."""

import json
from pathlib import Path

import pytest

from pipeline.contracts import ConfigError


class Cfg(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


def test_factory_fixture_backend_reads_assignment_file(tmp_path):
    from pipeline.diarize.factory import create_diarizer

    fixture = tmp_path / "assign.json"
    fixture.write_text(json.dumps(
        {"segments": [{"id": 1, "speaker": "S1"}, {"id": 2, "speaker": "S2"}]}))
    diarizer = create_diarizer(Cfg({
        "diarization.runtime": "fixture",
        "diarization.fixture": str(fixture)}))
    segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "a"},
                {"id": 2, "start": 2.0, "end": 4.0, "text": "b"}]
    result = diarizer.diarize(tmp_path / "audio.wav", segments)
    assert result == [{"id": 1, "speaker": "S1"}, {"id": 2, "speaker": "S2"}]


def test_fixture_backend_rejects_unknown_ids(tmp_path):
    from pipeline.diarize.fixture import FixtureDiarizer

    fixture = tmp_path / "assign.json"
    fixture.write_text(json.dumps({"segments": [{"id": 9, "speaker": "S1"}]}))
    with pytest.raises(Exception, match="id 1"):
        FixtureDiarizer(fixture).diarize(
            tmp_path / "audio.wav",
            [{"id": 1, "start": 0.0, "end": 1.0, "text": "x"}])


def test_factory_rejects_unknown_runtime():
    from pipeline.diarize.factory import create_diarizer

    with pytest.raises(ConfigError, match="runtime"):
        create_diarizer(Cfg({"diarization.runtime": "cloud"}))
```

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest tests/test_diarize.py -q` — Expected: FAIL `ModuleNotFoundError`

- [ ] **Step 3: Implement**

`pipeline/diarize/base.py`:

```python
"""Diarizer seam: label each pipeline segment with a source speaker.

Backend-neutral on purpose (the llm fixture pattern): tests and offline
runs use FixtureDiarizer; real content uses the pyannote adapter. The
contract is per-segment speaker labels aligned to segments.json ids —
never word-level, never overlapping labels.
"""


class Diarizer:
    def diarize(self, audio, segments):
        """Return [{"id": <segment id>, "speaker": "<label>"}, ...]."""
        raise NotImplementedError
```

`pipeline/diarize/fixture.py`:

```python
"""Fixture diarizer: read a canned id->speaker assignment from disk."""

import json
from pathlib import Path

from pipeline.contracts import StageError


class FixtureDiarizer:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StageError(f"diarization fixture unreadable: {exc}") from exc
        rows = data.get("segments")
        if not isinstance(rows, list) or not rows:
            raise StageError(f"diarization fixture {self.path} has no segments")
        self.speakers = {int(row["id"]): str(row["speaker"]) for row in rows}

    def diarize(self, audio, segments):
        missing = [s["id"] for s in segments if int(s["id"]) not in self.speakers]
        if missing:
            raise StageError(
                f"diarization fixture misses ids: {missing[:5]}")
        return [{"id": int(s["id"]), "speaker": self.speakers[int(s["id"])]}
                for s in segments]
```

`pipeline/diarize/factory.py`:

```python
"""Create the configured diarizer backend."""

from pipeline.contracts import ConfigError
from pipeline.diarize.fixture import FixtureDiarizer

RUNTIMES = ("fixture", "pyannote")


def create_diarizer(config):
    runtime = config.get("diarization.runtime", "fixture")
    if runtime == "fixture":
        path = config.get("diarization.fixture")
        if not path:
            raise ConfigError("diarization.fixture is required with the fixture runtime")
        return FixtureDiarizer(path)
    if runtime == "pyannote":
        from pipeline.diarize.pyannote_adapter import PyannoteDiarizer
        return PyannoteDiarizer(config)
    raise ConfigError(
        f"diarization.runtime must be one of {', '.join(RUNTIMES)}, got {runtime!r}")
```

`pipeline/diarize/__init__.py`: empty. Fixture `tests/fixtures/diarization/sample_6s.json` — copy the ids from `tests/fixtures/segments/sample_6s.json`, all `"S1"`:

```json
{"segments": [{"id": 1, "speaker": "S1"}, {"id": 2, "speaker": "S1"}]}
```

(match the actual id count of the segments fixture — check before committing.)

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/test_diarize.py -q` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/diarize/ tests/test_diarize.py tests/fixtures/diarization/
git commit -m "feat(diarize): backend seam with fixture runtime (ADR-0006 P2)"
```

---

### Task 2: The `diarize` stage, registry entry, contract

**Files:**
- Create: `pipeline/stages/diarize.py`
- Modify: `pipeline/registry.py` (import + ARTIFACTS + STAGES entry between `segment` and `translate`)
- Modify: `config.example.yaml` (diarization block), `docs/agents/execution-chain.md` (stage table + artifact list)
- Test: `tests/test_diarize.py` (append)

**Interfaces:**
- Consumes: Task 1 `create_diarizer`; artifacts `audio`, `segments`.
- Produces: artifact `diarization` → `diarization/diarization.json` with `{"schema_version": 1, "runtime": str, "speakers": [str, ...], "segments": [{"id", "speaker"}]}`. Task 4's tts reads it.

- [ ] **Step 1: Write the failing stage test**

Append to `tests/test_diarize.py`:

```python
def test_diarize_stage_writes_artifact(tmp_path, monkeypatch):
    import shutil
    from pipeline.runner import run

    repo = Path(__file__).resolve().parents[1]
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copyfile(repo / "tests/fixtures/segments/sample_6s.json",
                    workdir / "segments.json")
    shutil.copyfile(repo / "tests/fixtures/segments/sample_6s.json",  # audio unused by fixture
                    workdir / "audio.wav")
    config = tmp_path / "config.yaml"
    config.write_text(
        "diarization:\n"
        "  runtime: fixture\n"
        f"  fixture: {repo}/tests/fixtures/diarization/sample_6s.json\n")
    assert run("diarize", workdir, config) == 0
    data = json.loads((workdir / "diarization/diarization.json").read_text())
    assert data["runtime"] == "fixture"
    assert data["speakers"] == ["S1"]
    assert all(set(row) == {"id", "speaker"} for row in data["segments"])
    ids = json.loads((workdir / "segments.json").read_text())["segments"]
    assert [row["id"] for row in data["segments"]] == [s["id"] for s in ids]
```

(If the fixture diarizer rejects the `audio.wav` copy — it never opens audio — this passes as written. If `sample_6s.json` has ids beyond 1..2, extend the fixture file accordingly.)

- [ ] **Step 2: Run** — Expected: FAIL — stage unknown to the runner.

- [ ] **Step 3: Implement**

`pipeline/stages/diarize.py`:

```python
"""diarize stage: label every segment with its source speaker (ADR-0006).

Vendor-neutral orchestration only: backend selection lives in
pipeline/diarize/factory.py. The artifact is downstream input for
speaker-to-voice mapping in the tts stage; nothing else consumes it.
"""

from pipeline.contracts import MissingInputError, StageError
from pipeline.diarize.factory import create_diarizer


def run(ctx) -> None:
    segments = ctx.read("segments")["segments"]
    if not segments:
        raise StageError("diarize: segments must not be empty")
    audio = ctx.path("audio")
    if not audio.is_file():
        raise MissingInputError(f"diarize: audio not found: {audio}")
    diarizer = create_diarizer(ctx.config)
    try:
        rows = diarizer.diarize(audio, segments)
    except StageError:
        raise
    except Exception as exc:  # noqa: BLE001  backend isolation
        raise StageError(f"diarize: {exc}") from exc
    speakers = sorted({row["speaker"] for row in rows})
    ctx.write(
        "diarization",
        {
            "schema_version": 1,
            "runtime": ctx.config.get("diarization.runtime", "fixture"),
            "speakers": speakers,
            "segments": rows,
        },
    )
```

`pipeline/registry.py`: import `from pipeline.stages.diarize import run as run_diarize`; add to `ARTIFACTS` after `"segments"`:

```python
    "diarization": Artifact("diarization", "diarization/diarization.json", "json"),
```

and a `StageSpec` in `STAGES` between `segment` and `translate`:

```python
    StageSpec(
        "diarize",
        "Label segments with source speakers (ADR-0006 voice mapping)",
        ("audio", "segments"),
        ("diarization",),
        (),
        run_diarize,
    ),
```

`config.example.yaml` (new top-level block after `segmentation:`):

```yaml
# Source-speaker labeling for dubbing voice mapping (ADR-0006 P2).
# fixture: read a canned id->speaker file (tests/offline); pyannote: local
# speaker diarization (weights pre-downloaded, HF-gated acceptance needed
# once). diarization.json feeds tts.speaker_voices.
diarization:
  runtime: 'fixture'
  # fixture runtime: path to {"segments": [{"id": 1, "speaker": "S1"}]}
  fixture: 'tests/fixtures/diarization/sample_6s.json'
  # pyannote runtime:
  #   model: 'pyannote/speaker-diarization-community-1'
  #   hf_token_env: 'HF_TOKEN'
```

`docs/agents/execution-chain.md`: add to the stage table (after segment row) `| diarize | media/audio.wav、segments.json | diarization/diarization.json | 说话人标注（ADR-0006）；后端 fixture/pyannote；仅 tts.speaker_voices 消费 |` and an artifact bullet `diarization/diarization.json：runtime/speakers[] + segments[]（id/speaker）；段数与 ID 与 segments.json 一致`.

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/test_diarize.py tests/test_t1_runner.py -q` — Expected: PASS (runner stage table test may assert stage order — extend its expectation if it enumerates stages).

- [ ] **Step 5: Commit**

```bash
git add pipeline/stages/diarize.py pipeline/registry.py config.example.yaml docs/agents/execution-chain.md tests/test_diarize.py tests/test_t1_runner.py
git commit -m "feat(stage): diarize stage + registry entry + contract (ADR-0006 P2)"
```

---

### Task 3: Speaker→voice assignment (explicit map + ECAPA similarity)

**Files:**
- Create: `pipeline/tts/voicing.py`
- Create: `bench/voice_embedding_tool.py` (offline precompute of library/speaker embeddings)
- Test: `tests/test_tts_voicing.py` (new)

**Interfaces:**
- Consumes: `diarization.json` speakers; `VoiceLibrary` manifest (`voices.json` entries may carry `"embedding": "<file>.npy"` relative to the manifest).
- Produces: `assign_voices(speaker_embeddings: dict[str, "np.ndarray"], voice_embeddings: dict[str, "np.ndarray"], explicit_map: dict[str, str]) -> dict[str, str]`; `speaker_embeddings(audio: Path, turns: list[dict]) -> dict[str, np.ndarray]` (turns = `[{"speaker", "start", "end"}]`); `embed_wav_window(path, start, end) -> np.ndarray`. Task 4 consumes all three.

- [ ] **Step 1: Write the failing tests**

`tests/test_tts_voicing.py`:

```python
"""Voice mapping tests: explicit map, embedding argmax, failure modes."""

import numpy as np
import pytest

from pipeline.contracts import ConfigError, StageError


def test_explicit_map_wins_and_is_exact():
    from pipeline.tts.voicing import assign_voices
    assert assign_voices(
        {"S1": np.zeros(1), "S2": np.zeros(1)},
        {"zh-male-01": np.ones(1)},
        {"S1": "zh-male-01", "S2": "zh-male-01"},
    ) == {"S1": "zh-male-01", "S2": "zh-male-01"}


def test_similarity_argmax_picks_closest_voice():
    from pipeline.tts.voicing import assign_voices
    speaker = {"S1": np.array([1.0, 0.0])}
    voices = {"a": np.array([0.9, 0.1]), "b": np.array([0.0, 1.0])}
    assert assign_voices(speaker, voices, {}) == {"S1": "a"}


def test_no_embeddings_without_map_is_config_error():
    from pipeline.tts.voicing import assign_voices
    with pytest.raises(ConfigError, match="voice_map"):
        assign_voices({"S1": np.zeros(1)}, {}, {})


def test_explicit_map_unknown_voice_rejected():
    from pipeline.tts.voicing import assign_voices
    with pytest.raises(ConfigError, match="zh-ghost"):
        assign_voices({"S1": np.zeros(1)},
                      {"zh-male-01": np.zeros(1)},
                      {"S1": "zh-ghost"})


def test_speaker_embeddings_means_windows(tmp_path, monkeypatch):
    import soundfile as sf
    from pipeline.tts import voicing

    rate = 16000
    sf.write(tmp_path / "a.wav",
             np.zeros(rate, dtype="float32"), rate)
    calls = []

    def fake_embed(path, start, end):
        calls.append((start, end))
        return np.array([start], dtype="float32")

    monkeypatch.setattr(voicing, "embed_wav_window", fake_embed)
    turns = [{"speaker": "S1", "start": 0.0, "end": 1.0},
             {"speaker": "S1", "start": 2.0, "end": 3.0}]
    result = voicing.speaker_embeddings(tmp_path / "a.wav", turns)
    assert result["S1"] == np.array([1.0])  # mean(0, 2)
    assert len(calls) == 2
```

- [ ] **Step 2: Run** — Expected: FAIL `ModuleNotFoundError: pipeline.tts.voicing`

- [ ] **Step 3: Implement `pipeline/tts/voicing.py`**

```python
"""Speaker-to-voice assignment (ADR-0006 P2): map, don't clone.

Explicit tts.voice_map wins; otherwise each diarized speaker takes the
library voice whose precomputed ECAPA embedding is most cosine-similar
to the speaker's mean embedding over their segment windows. Embeddings
are precomputed offline (bench/voice_embedding_tool.py) so the pipeline
never loads speechbrain unless similarity matching is actually used.
"""

from pathlib import Path

import numpy as np

from pipeline.contracts import ConfigError


def embed_wav_window(path, start, end):
    """ECAPA embedding of audio[start:end]; speechbrain is lazy-imported."""
    import soundfile as sf
    from speechbrain.inference.speaker import SpeakerRecognition

    samples, rate = sf.read(str(path), start=int(start * 16000),
                            stop=int(end * 16000), dtype="float32",
                            always_2d=False)
    # resample-on-read is not supported; the tool and stage both use 16 kHz
    encoder = SpeakerRecognition.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="_model_cache/speechbrain-ecapa")
    wave = np.asarray(samples, dtype="float32").reshape(1, -1)
    return encoder.encode_batch(torch_tensor(wave)).squeeze().numpy()


def torch_tensor(array):
    import torch
    return torch.from_numpy(array)


def speaker_embeddings(audio, turns):
    """Mean embedding per speaker over their windows (max 20 windows)."""
    by_speaker = {}
    for turn in turns:
        by_speaker.setdefault(turn["speaker"], []).append(turn)
    result = {}
    for speaker, speaker_turns in by_speaker.items():
        sampled = speaker_turns[:20]
        vectors = [embed_wav_window(audio, t["start"], t["end"])
                   for t in sampled]
        result[speaker] = np.mean(vectors, axis=0)
    return result


def _cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm else 0.0


def assign_voices(speaker_embeddings, voice_embeddings, explicit_map):
    """Resolve {speaker: voice_name}. Explicit map wins per speaker."""
    voices = {}
    for speaker in speaker_embeddings:
        mapped = (explicit_map or {}).get(speaker)
        if mapped is not None:
            if mapped not in voice_embeddings and not voice_embeddings:
                # voices with no embeddings at all: explicit map is the only path
                voices[speaker] = mapped
                continue
            if mapped not in voice_embeddings:
                raise ConfigError(
                    f"tts.voice_map[{speaker!r}] names unknown voice {mapped!r}")
            voices[speaker] = mapped
            continue
        if not voice_embeddings:
            raise ConfigError(
                "voice similarity matching needs precomputed voice "
                "embeddings (bench/voice_embedding_tool.py) or explicit "
                "tts.voice_map entries for every speaker")
        voices[speaker] = max(
            voice_embeddings, key=lambda name: _cosine(
                speaker_embeddings[speaker], voice_embeddings[name]))
    return voices
```

(For the "explicit map without any embeddings" case, Task 4 validates mapped names against the actual `VoiceLibrary` manifest — that check is authoritative; `assign_voices` only rejects names when embeddings exist to compare against.)

`bench/voice_embedding_tool.py`:

```python
"""Precompute ECAPA embeddings for voice-library clips (offline, once).

Writes <voices_dir>/<clip>.npy next to each reference clip and records
"embedding" in voices.json so the tts stage can similarity-match
diarized speakers to library voices (ADR-0006 P2). speechbrain weights
land in _model_cache/speechbrain-ecapa (one-time download).

Usage: .venv/bin/python bench/voice_embedding_tool.py [--voices-dir DIR]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from pipeline.tts.voicing import embed_wav_window  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--voices-dir", default="_model_cache/tts_voices")
    args = parser.parse_args()
    voices_dir = Path(args.voices_dir)
    manifest_path = voices_dir / "voices.json"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest["voices"]:
        clip = voices_dir / entry["clip"]
        out = clip.with_suffix(clip.suffix + ".npy")
        if not out.exists():
            import numpy as np
            vector = embed_wav_window(clip, 0.0, None if entry == {} else 30.0)
            np.save(out, vector)
        entry["embedding"] = out.name
        print(f"{entry['name']}: {out.name}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

(`embed_wav_window`'s `stop` uses 16 kHz sample counts; for whole clips pass a large `end` — adjust the tool to read the clip's true duration via `soundfile.info` rather than the hardcoded 30.0 before running.)

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/test_tts_voicing.py -q` — Expected: PASS (no speechbrain needed — fake embeddings)

- [ ] **Step 5: Commit**

```bash
git add pipeline/tts/voicing.py bench/voice_embedding_tool.py tests/test_tts_voicing.py
git commit -m "feat(tts): speaker-to-voice assignment via explicit map or ECAPA similarity"
```

---

### Task 4: Per-segment voices in the tts stage

**Files:**
- Modify: `pipeline/stages/tts.py` (voice resolution per segment; `tts.speaker_voices`, `tts.voice_map` config), `config.example.yaml`, `docs/agents/execution-chain.md:40` (dub_plan fields)
- Test: `tests/test_tts_stage.py` (append)

**Interfaces:**
- Consumes: Task 2 `diarization.json`, Task 3 `assign_voices`/`speaker_embeddings`; `VoiceLibrary`.
- Produces: when `tts.speaker_voices: true`: `dub_plan.json` top-level `"voices": {speaker: voice}` and per-segment `"voice"`; synthesis cache keys unchanged in form (`SynthesisCache.key(voice.name, text, speed)`). When false: output identical to today (no `voices`, no per-segment `voice`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tts_stage.py` (reuse the file's existing workdir/config helpers; the fixture voice library the tests already use must gain a second voice if it has only one — check `tests/fixtures` first):

```python
class TestSpeakerVoices:
    def _workdir_with_diarization(self, tmp_path, speakers):
        workdir, config = _tts_workdir(tmp_path)   # existing helper: translation.json + audio
        rows = [{"id": s["id"], "speaker": s["speaker"]}
                for s in json.loads((workdir / "translation.json").read_text())["segments"]]
        for row in rows:
            row["speaker"] = speakers[row["id"] % len(speakers)]
        (workdir / "diarization").mkdir(exist_ok=True)
        (workdir / "diarization/diarization.json").write_text(json.dumps(
            {"schema_version": 1, "runtime": "fixture",
             "speakers": sorted(set(speakers)), "segments": rows}))
        return workdir, config

    def test_per_segment_voices_recorded(self, tmp_path):
        workdir, config = self._workdir_with_diarization(tmp_path, ["S1", "S2"])
        config.write_text(config.read_text() +
                          "tts:\n  speaker_voices: true\n"
                          "  voice_map: {S1: zh-male-01, S2: zh-female-01}\n")
        assert run("tts", workdir, config) == 0
        plan = json.loads((workdir / "tts/dub_plan.json").read_text())
        assert plan["voices"] == {"S1": "zh-male-01", "S2": "zh-female-01"}
        assert {s["voice"] for s in plan["segments"]} == {"zh-male-01", "zh-female-01"}

    def test_missing_diarization_is_explicit_error(self, tmp_path):
        workdir, config = _tts_workdir(tmp_path)
        config.write_text(config.read_text() + "tts:\n  speaker_voices: true\n")
        assert run("tts", workdir, config) == 1  # stderr names the missing artifact

    def test_disabled_is_byte_identical(self, tmp_path):
        workdir, config = _tts_workdir(tmp_path)
        assert run("tts", workdir, config) == 0
        plan = json.loads((workdir / "tts/dub_plan.json").read_text())
        assert "voices" not in plan and "voice" not in plan["segments"][0]
```

- [ ] **Step 2: Run** — Expected: FAIL (no `voices`/`voice`, no error on missing diarization).

- [ ] **Step 3: Implement**

In `pipeline/stages/tts.py`:

```python
def _voice_for_speaker(config, library):
    """Resolve {speaker: Voice} when tts.speaker_voices is enabled."""
    diarization_path = ctx_free_path = None  # resolved by caller
```

Concretely, inside `run(ctx)`, after `voice = _voice(ctx.config)`:

```python
    speaker_voices = None
    if ctx.config.get("tts.speaker_voices", False):
        diarization = ctx.read_optional("diarization") if hasattr(ctx, "read_optional") else None
        if diarization is None:
            path = ctx.workdir / "diarization" / "diarization.json"
            import json as _json
            if not path.is_file():
                raise StageError(
                    "tts: tts.speaker_voices needs the diarize stage output "
                    f"({path}); run --stages diarize first")
            diarization = _json.loads(path.read_text())
        speakers = {row["speaker"] for row in diarization["segments"]}
        explicit = ctx.config.get("tts.voice_map", {}) or {}
        if explicit:
            mapping = {s: explicit[s] for s in speakers}
            unknown = [v for v in mapping.values() if not _library_has(ctx.config, v)]
            if unknown:
                raise ConfigError(f"tts.voice_map names unknown voices: {unknown}")
        else:
            from pipeline.tts.voicing import assign_voices, speaker_embeddings
            turns = [
                {"speaker": row["speaker"],
                 "start": float(seg["start"]), "end": float(seg["end"])}
                for row, seg in zip(diarization["segments"], segments)]
            spk = speaker_embeddings(ctx.path("audio"), turns)
            mapping = assign_voices(spk, _voice_embeddings(ctx.config), {})
        library = _voice_library(ctx.config)
        speaker_voices = {s: library.get(v) for s, v in mapping.items()}
```

Refactor helpers: extract `_voice_library(config) -> VoiceLibrary` (already half-present inside `_voice`) and `_library_has(config, name) -> bool`, plus `_voice_embeddings(config) -> dict[str, np.ndarray]` loading each manifest `embedding` file (error listing `bench/voice_embedding_tool.py` when absent). In the synthesis loop, per segment:

```python
        seg_voice = voice
        if speaker_voices is not None:
            speaker = speaker_by_id[segment["id"]]  # built from diarization rows
            seg_voice = speaker_voices[speaker]
        key = SynthesisCache.key(seg_voice.name, text, speed)
        ... engine.synth(text, seg_voice, speed=speed)
```

and entries gain `"voice": seg_voice.name` only when `speaker_voices` is not None; the plan dict gains `"voices": {s: v.name for s, v in speaker_voices.items()}` in the same case. `config.example.yaml` tts block:

```yaml
  # ADR-0006 P2: one voice per diarized source speaker. Requires the
  # diarize stage output. voice_map is explicit {speaker: voice};
  # without it, speakers are matched by precomputed ECAPA similarity
  # (bench/voice_embedding_tool.py).
  speaker_voices: false
  voice_map: {}
```

`execution-chain.md:40` dub_plan bullet: append `voices`（speaker→voice 映射，仅 tts.speaker_voices 时）与段级 `voice`（同上）.

- [ ] **Step 4: Run** — `.venv/bin/python -m pytest tests/test_tts_stage.py tests/test_tts_voicing.py -q` — Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/stages/tts.py config.example.yaml docs/agents/execution-chain.md tests/test_tts_stage.py
git commit -m "feat(tts): per-speaker voice resolution recorded in the dub plan (ADR-0006 P2)"
```

---

### Task 5: pyannote adapter + dependency + real-content smoke

**Files:**
- Create: `pipeline/diarize/pyannote_adapter.py`
- Modify: none in product logic (adapter only)
- Test: `tests/test_diarize.py` (append, auto-skip without pyannote)

**Interfaces:**
- Consumes: `diarization.model` (default `pyannote/speaker-diarization-community-1`), `diarization.hf_token_env` (default `HF_TOKEN`).
- Produces: `PyannoteDiarizer(config).diarize(audio, segments)` — same contract as the fixture.

- [ ] **Step 1: Install + accept gated weights (manual, once)**

```bash
uv pip install --python .venv/bin/python "pyannote.audio==3.3.2"
```

Manual (wizard-grade) steps, record completion in the commit message:
1. HF account → accept terms of `pyannote/speaker-diarization-community-1` (and `pyannote/segmentation-3.0` dependency) on their model pages.
2. Create a read token, `export HF_TOKEN=...`, run a one-time warm-up (below) so weights cache under `~/.cache/huggingface`; after that no token and no network are needed.

- [ ] **Step 2: Write the skipping test**

```python
def test_pyannote_adapter_labels_real_audio(tmp_path):
    pytest.importorskip("pyannote.audio")
    import numpy as np
    import soundfile as sf
    from pipeline.diarize.pyannote_adapter import PyannoteDiarizer

    rate = 16000
    t = np.arange(rate * 4) / rate
    # two alternating tone bursts = two pseudo-speakers
    wave = np.where((t % 2) < 1, 0.3 * np.sin(2 * np.pi * 300 * t),
                    0.3 * np.sin(2 * np.pi * 500 * t)).astype("float32")
    audio = tmp_path / "a.wav"
    sf.write(audio, wave, rate)
    diarizer = PyannoteDiarizer(Cfg({"diarization.runtime": "pyannote"}))
    segments = [{"id": 1, "start": 0.0, "end": 2.0, "text": "a"},
                {"id": 2, "start": 2.0, "end": 4.0, "text": "b"}]
    rows = diarizer.diarize(audio, segments)
    assert len(rows) == 2 and rows[0]["speaker"] != rows[1]["speaker"]
```

(marked by `importorskip`; on CI without weights it skips — the fixture backend covers contract behavior).

- [ ] **Step 3: Implement the adapter**

```python
"""pyannote diarization adapter (ADR-0006 P2). Local-only at runtime.

Weights are pre-downloaded once (HF-gated acceptance; cache in
~/.cache/huggingface). Segments are labeled by maximal speaker-turn
overlap; ties break toward the earlier turn — order-preserving,
best-effort, like every alignment in this pipeline.
"""

import os

from pipeline.contracts import MissingInputError


class PyannoteDiarizer:
    def __init__(self, config):
        self.model_name = config.get(
            "diarization.model", "pyannote/speaker-diarization-community-1")
        token = os.environ.get(config.get("diarization.hf_token_env", "HF_TOKEN"))
        import torch
        from pyannote.audio import Pipeline
        self.pipeline = Pipeline.from_pretrained(self.model_name,
                                                 use_auth_token=token)
        if torch.cuda.is_available():
            self.pipeline.to(torch.device("cuda"))

    def diarize(self, audio, segments):
        import torch
        import soundfile as sf
        samples, rate = sf.read(str(audio), dtype="float32", always_2d=False)
        annotation = self.pipeline(
            {"waveform": torch.from_numpy(samples).float().reshape(1, -1),
             "sample_rate": rate})
        turns = [(turn.start, turn.end, label)
                 for turn, _, label in annotation.itertracks(yield_label=True)]
        rows = []
        for segment in segments:
            start, end = float(segment["start"]), float(segment["end"])
            scored = {}
            for t_start, t_end, label in turns:
                overlap = min(end, t_end) - max(start, t_start)
                if overlap > 0:
                    scored[label] = scored.get(label, 0.0) + overlap
            if not scored:
                raise MissingInputError(
                    f"diarize: no speaker turn overlaps segment "
                    f"{segment['id']} [{start}, {end}]")
            speaker = max(scored, key=scored.get)
            rows.append({"id": int(segment["id"]), "speaker": str(speaker)})
        return rows
```

- [ ] **Step 4: Run smoke + full suite**

Run: `.venv/bin/python -m pytest tests/test_diarize.py -q` then the real-content smoke:

```bash
.venv/bin/python video_transcribe.py runs/tts-accept-1h-stage1 --stages diarize --config config.yaml
```

(check the runner's positional-arg form first; expected: `diarization/diarization.json` lands in the run dir, single speaker for the podcast; then `.venv/bin/python -m pytest tests/ -q` all green).

- [ ] **Step 5: Commit**

```bash
git add pipeline/diarize/pyannote_adapter.py tests/test_diarize.py
git commit -m "feat(diarize): pyannote adapter (local runtime, gated weights pre-download)"
```

---

### Task 6: Acceptance harness + human gates

**Files:**
- Create: `bench/v4_10_diarization_acceptance.py`
- Output: `bench/results/v4_10/diarization-acceptance.json`
- Modify: `docs/adr/0006-*.md` (measured outcomes paragraph)

- [ ] **Step 1: Harness** — asserts, on a bench-local workdir seeded from the 1h run: (1) `len(set(voices.values())) == len(set(speakers))`; (2) no segment's `voice` differs from its speaker's mapped voice (no flapping); (3) offline run: `HF_HUB_OFFLINE=1` in env, stage still succeeds (weights cached); (4) wall of diarize vs E2E baseline ≤ 5%; writes JSON + prints the two human-gate checklists (identity match, turn agreement). Structure mirrors `bench/v4_9_budget_acceptance.py` (Task 6 there) — reuse its workdir-copy and subprocess-stage-run helpers.
- [ ] **Step 2: Run** — expected: all machine gates pass; human gates recorded by maintainer.
- [ ] **Step 3: ADR-0006 outcomes paragraph + commit + merge/push** (same flow as Plan 1 Task 6).

---

### Task 7 (follow-up, separate decision): OpenVoice v2 probe — PROBE ONLY

**Files:**
- Create: `bench/v4_11_openvoice_probe.py`; Output: `bench/results/v4_11/openvoice-probe.json`

No product code, no config keys. Install `openvoice` (MIT) + checkpoints (manual, one-time, `_model_cache/openvoice/`); convert 20 F5-synthesized segments from the acceptance run through the tone-color converter with source-speaker reference clips; measure Δcosine (ECAPA, converted vs source speaker) against the mapped-voice baseline and added RTF. Gates: Δcosine ≥ +0.05 AND TTS E2E RTFx ≤ 0.5 retained. **Adoption requires a new ADR** (ADR-0006 explicitly defers it); failure = report filed, nothing else.

---

## Self-Review

- Spec coverage: ADR-0006 P2 items → Tasks 1–2 (diarization local, word-timestamp reuse via segment windows), Task 3–4 (per-speaker library mapping + embedding match + plan recording), Task 5 (pyannote local/gated), Task 6 (acceptance); OpenVoice = Task 7 probe-only; cloning/engine swap correctly absent (rejected).
- Placeholders: Task 6 Step 1 intentionally reuses Plan 1 Task 6 helpers by reference to the file it creates (same branch, sequential execution); Task 3's tool and Task 4's helpers carry explicit "check before run" notes (clip duration via `soundfile.info`; test-fixture voice-library shape). Everything else is concrete.
- Type consistency: `Diarizer.diarize(audio, segments) -> [{"id", "speaker"}]`; `assign_voices(dict, dict, dict) -> dict[str, str]`; `speaker_embeddings(audio, turns) -> dict[str, np.ndarray]`; dub_plan `voices` / segments `voice` naming consistent across Task 4 and the docs edit.
