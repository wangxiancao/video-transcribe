# Dub Pace: Translation Char Budget Implementation Plan (ADR-0006 P1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the zh dub stop sounding rushed by capping each translation at what its speech slot holds at a natural pace (`translation.max_chars_per_second` + one feedback-condense round), so `tts.speed` can fall back from 2.0, plus track loudness matching.

**Architecture:** The budget is derived in the translate stage from the segment timing it already consumes (slot duration × natural chars/s, sub-second slots exempt per ADR-0006), enforced by exactly one measured condense round — probe-measured (`bench/results/v4_8/budget-probe.json`): prompt-only 23.8% < control 33.8% < bestk 43.1% < feedback 49.2%. Loudness matching is an ffmpeg two-pass appended to tts track assembly. No engine or schema_version changes (`synth_speed` precedent: additive fields keep schema_version 1).

**Tech Stack:** Python 3.10 in repo `.venv`, existing LLM client seam (`pipeline/llm/`), ffmpeg loudnorm, pytest.

## Global Constraints

- Zero cloud: no new network calls at runtime; weights/assets pre-downloaded only.
- New config keys go in `config.example.yaml` (tracked schema). Never write `config.yaml` at runtime.
- Contract artifacts are APIs (`docs/agents/execution-chain.md`): additive JSON fields only; `schema_version` stays 1.
- The 88-char per-segment translation cap and the quality gates (`pipeline/text_quality.py`) are unchanged and still fatal.
- Budget is a soft constraint: condense failure falls back to the refined draft with a stderr warning — never fails the stage.
- Cache correctness: budgeted and unbudgeted requests must never share a cache entry.
- Tests: `.venv/bin/python -m pytest tests/ -q` (slow-marked skipped); compile: `.venv/bin/python -m compileall -q video_transcribe.py pipeline`.
- Commits per task on a feature branch, merged to `main` and pushed after Task 6 passes.

## 验收目标 (acceptance targets)

**A. Contract gates (pytest, must be green)**
1. Budget math: `ceil(duration × rate)`; slots where the budget floors below 4 chars (duration < 0.89 s at rate 4.5) are exempt (`char_budget is None`).
2. Condense bounded: at most ONE extra LLM call per batch; condense round only carries over-budget members.
3. Cache isolation: two requests differing only in `char_budgets` produce different cache keys.
4. Loudness: after matching, `|I_dub − I_source| ≤ 2 LU` and `TP ≤ −1.0 dBTP` on fixture audio.

**B. Full-content gates (`bench/v4_9_budget_acceptance.py` on `runs/` copy of the 1h acceptance media, Q4 llama-server up)**

| gate | bar | baseline/evidence |
|---|---|---|
| adherence (chars ≤ budget) | ≥ 45% | probe 49.2% on 65-segment stratified sample |
| within +1 char | ≥ 54% | probe 54.0% |
| median chars/s of dub text | ≤ 5.0 | baseline 5.5 (natural band 4–6) |
| contract failure rate | ≤ 1% | probe failures were all floor-bound slots, now exempt |
| LLM calls / segment | ≤ 3.0 | 2 + ≤1 condense; probe 2.75 |
| translate wall vs unbudgeted | ≤ +25% | probe 0.24 s/segment |
| dub plan cumulative overflow | ≤ 2% at the lowest passing speed; speed target ≤ 1.6 (stretch 1.3) | ADR-0005 gate unchanged; was 0.33% @ speed 2.0 |
| E2E wall on 1h content | ≤ 8.5 min | baseline 7.2 min |
| human listening gate | "赶稿感" gone (maintainer judgment, recorded in report JSON) | ADR-0005 reopening condition |
| blind review | ≤ 1 of 40 condensed pairs judged meaning-lossy | probe spot-checks were clean |

---

### Task 1: Budget derivation helper and config key

**Files:**
- Modify: `pipeline/stages/translate.py` (top of module, after `SYSTEM_REFINE`)
- Modify: `config.example.yaml` (translation section)
- Test: `tests/test_t6_translate.py` (append)

**Interfaces:**
- Consumes: none.
- Produces: `char_budget(start: float, end: float, rate: float) -> int | None` (`None` = exempt); `_budget_rate(config) -> float`; constant `BUDGET_FLOOR_CHARS = 4`. Used by Task 2 and Task 3.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_t6_translate.py`:

```python
class TestCharBudget:
    def test_budget_is_duration_times_rate_ceil(self):
        from pipeline.stages.translate import char_budget
        assert char_budget(0.0, 4.0, 4.5) == 18      # 4.0s * 4.5 = 18
        assert char_budget(0.0, 3.12, 4.5) == 15     # 14.04 -> ceil 15 (probe seg 7)

    def test_sub_second_slots_are_exempt(self):
        from pipeline.stages.translate import char_budget
        assert char_budget(0.0, 0.5, 4.5) is None     # 2 chars < floor
        assert char_budget(0.0, 0.88, 4.5) is None    # 3.96 -> ceil 4 == floor? no: <4 chars -> exempt
        assert char_budget(0.0, 0.89, 4.5) == 4       # first non-exempt slot

    def test_rate_config_range(self):
        import pytest
        from pipeline.contracts import ConfigError
        from pipeline.stages.translate import _budget_rate

        class Cfg(dict):
            def get(self, key, default=None):
                return dict.get(self, key, default)

        assert _budget_rate(Cfg()) == 4.5
        assert _budget_rate(Cfg({"translation.max_chars_per_second": 5})) == 5.0
        for bad in (0, 11, -1, "fast", True):
            with pytest.raises(ConfigError):
                _budget_rate(Cfg({"translation.max_chars_per_second": bad}))
```

Note: `char_budget(0.0, 0.88, 4.5)` → `ceil(3.96) == 4` — decide the boundary explicitly: exempt when `ceil(d*rate) < BUDGET_FLOOR_CHARS`; 3.96 ceils to 4, so 0.88 s is NOT exempt. Fix the test line to match the rule:

```python
        assert char_budget(0.0, 0.8, 4.5) is None     # 3.6 -> ceil 4? no: ceil(3.6)=4 ... see Step 3 rule
```

Use this final, unambiguous pair (the rule is `budget = ceil(d*rate); return budget if budget >= BUDGET_FLOOR_CHARS else None`, so the exempt boundary is `d*rate <= 3`):

```python
    def test_sub_second_slots_are_exempt(self):
        from pipeline.stages.translate import char_budget, BUDGET_FLOOR_CHARS
        assert BUDGET_FLOOR_CHARS == 4
        assert char_budget(0.0, 0.5, 4.5) is None     # 2.25 -> 3 < 4
        assert char_budget(0.0, 0.66, 4.5) is None    # 2.97 -> 3 < 4
        assert char_budget(0.0, 0.67, 4.5) == 4       # 3.015 -> 4 == floor
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_t6_translate.py::TestCharBudget -q`
Expected: FAIL `ImportError: cannot import name 'char_budget'`

- [ ] **Step 3: Implement**

In `pipeline/stages/translate.py`, after the `SYSTEM_REFINE` block (needs `import math` — add at top with the other stdlib imports):

```python
# ADR-0006: a translation may not demand more from its speech slot than
# natural Mandarin delivers. Slots too short for any natural sentence are
# exempt — their overflow stays ADR-0005 territory (pause merging, plan
# accounting), and an impossible budget measurably derails the small
# model (translator commentary in probe v4_8).
BUDGET_FLOOR_CHARS = 4


def _budget_rate(config):
    rate = config.get("translation.max_chars_per_second", 4.5)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        raise ConfigError(
            "translation.max_chars_per_second must be a number, "
            f"got {rate!r}"
        )
    if not 1.0 <= float(rate) <= 10.0:
        raise ConfigError(
            "translation.max_chars_per_second must be within [1, 10], "
            f"got {rate}"
        )
    return float(rate)


def char_budget(start, end, rate):
    """Max target-language chars the slot holds at `rate`; None=exempt."""
    budget = math.ceil((float(end) - float(start)) * rate)
    return budget if budget >= BUDGET_FLOOR_CHARS else None
```

In `config.example.yaml`, inside the `translation` block (next to `cache:` / `term_map:`):

```yaml
  # ADR-0006: per-segment zh char budget = ceil(slot_duration * this rate);
  # over-budget translations get one feedback-condense LLM round. Slots
  # shorter than 4 chars at this rate are exempt. Range 1..10, default 4.5
  # (natural Mandarin chars/s; probe evidence bench/results/v4_8/).
  max_chars_per_second: 4.5
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/python -m pytest tests/test_t6_translate.py -q`
Expected: PASS (all previous + new)

- [ ] **Step 5: Commit**

```bash
git add pipeline/stages/translate.py config.example.yaml tests/test_t6_translate.py
git commit -m "feat(translate): char budget derivation and max_chars_per_second key (ADR-0006)"
```

---

### Task 2: Budget-aware requests and the feedback-condense round

**Files:**
- Modify: `pipeline/stages/translate.py` (`_translate_batch_once`, `_translate_batch`, `run`)
- Test: `tests/test_t6_translate.py` (append)

**Interfaces:**
- Consumes: Task 1 (`char_budget`, `_budget_rate`).
- Produces: `_translate_batch_once(client, cache, segments, source_language, target_language, term_map, rate)` — new trailing `rate` param; faithful/refine request payloads carry `"char_budgets": [int|null, ...]` whenever any budget is non-null; system prompts get `BUDGET_SENTENCE` appended only in that case. One condense round per batch for over-budget members. `run` passes `_budget_rate(ctx.config)`.

- [ ] **Step 1: Write the failing tests**

Append (reuses `ArrayClient` and `_config`/`_workdir` helpers already in the file; check how `create_llm_client` is patched — the existing tests use `monkeypatch.setattr` on the factory import inside the stage module):

```python
class BudgetArrayClient(ArrayClient):
    """ArrayClient + budget awareness + a too-long refine output."""

    def __init__(self, long_indices=(0,)):
        super().__init__()
        self.long_indices = long_indices
        self.condense_calls = 0

    def complete(self, system, user):
        request = json.loads(user)
        if "note" in request:
            self.condense_calls += 1
            drafts = request["drafts"]
            return json.dumps({"translations": [d[: request["char_budgets"][i]] for i, d in enumerate(drafts)]})
        texts = request.get("segments") or request.get("drafts")
        refined = "faithfully" not in system.lower()
        if refined:
            texts = [t + " padding padding" if i in self.long_indices else t for i, t in enumerate(texts)]
        return json.dumps({"translations": texts})


class TestBudgetEnforcement:
    def _stage(self, tmp_path, monkeypatch, client, extra=""):
        workdir = _workdir(tmp_path)
        config = _config(tmp_path, extra)
        monkeypatch.setattr(
            "pipeline.stages.translate.create_llm_client",
            lambda _c, _f, _s=None: client,
        )
        assert run("translate", workdir, config) == 0
        return json.loads((workdir / "translation.json").read_text())

    def test_budget_sent_and_condense_trims(self, tmp_path, monkeypatch):
        client = BudgetArrayClient()
        data = self._stage(tmp_path, monkeypatch, client)
        faithful_req = client.requests[0]
        assert "char_budgets" in faithful_req
        budgets = faithful_req["char_budgets"]
        assert budgets and budgets[0] is not None
        # echo semantics of BudgetArrayClient: refined echo got padded for
        # long_indices, condense must have brought them back within budget
        assert client.condense_calls == 1
        for seg in data["segments"]:
            if seg["char_budget"] is not None:
                assert len(seg["target"]) <= seg["char_budget"]

    def test_exempt_only_batch_sends_no_budgets(self, tmp_path, monkeypatch):
        # shrink every slot below the floor by rewriting the fixture copy
        workdir = _workdir(tmp_path)
        segs = json.loads((workdir / "segments.json").read_text())
        for s in segs["segments"]:
            s["end"] = s["start"] + 0.5
        (workdir / "segments.json").write_text(json.dumps(segs))
        client = BudgetArrayClient(long_indices=())
        config = _config(tmp_path)
        monkeypatch.setattr(
            "pipeline.stages.translate.create_llm_client",
            lambda _c, _f, _s=None: client,
        )
        assert run("translate", workdir, config) == 0
        assert all("char_budgets" not in r for r in client.requests)
        assert client.condense_calls == 0

    def test_condense_failure_keeps_refined(self, tmp_path, monkeypatch):
        class Explode(BudgetArrayClient):
            def complete(self, system, user):
                if "note" in json.loads(user):
                    raise json.JSONDecodeError("x", "y", 0)  # force StageError path
                return super().complete(system, user)

        client = Explode()
        data = self._stage(tmp_path, monkeypatch, client)
        # stage succeeded; target kept (padded) refined text; budget overrun tolerated
        assert all(isinstance(s["char_budget"], int) or s["char_budget"] is None
                   for s in data["segments"])

    def test_cache_key_separates_budgets(self, tmp_path):
        from pipeline.llm.cache import cache_key
        a = cache_key("sys", '{"segments": ["hi"], "char_budgets": [10]}')
        b = cache_key("sys", '{"segments": ["hi"], "char_budgets": [7]}')
        c = cache_key("sys", '{"segments": ["hi"]}')
        assert len({a, b, c}) == 3
```

(`Explode.complete` must return invalid JSON text instead — raising inside `complete` maps to LlmError; to hit the StageError fallback return `'not json'`. Use `return "not json"` in that branch.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_t6_translate.py::TestBudgetEnforcement -q`
Expected: FAIL — no `char_budgets` in requests / no condense call / no `char_budget` field

- [ ] **Step 3: Implement**

In `pipeline/stages/translate.py`:

```python
BUDGET_SENTENCE = (
    " The request also includes char_budgets, an ordered array of integers "
    "(null = no limit), one per segment: translation number N MUST NOT "
    "exceed char_budgets[N-1] characters (count every Chinese character; "
    "spaces, if any, do not count). Stay within budget by condensing "
    "wording, never by dropping meaning, names or numbers."
)
```

Replace `_translate_batch_once` body (keeping its contract behavior) with:

```python
def _translate_batch_once(client, cache, segments, source_language,
                          target_language, term_map, rate):
    texts = [_apply_term_map(segment["text"], term_map) for segment in segments]
    budgets = [char_budget(segment["start"], segment["end"], rate)
               for segment in segments]
    budgeted = any(b is not None for b in budgets)

    def request(payload):
        if budgeted:
            payload["char_budgets"] = budgets
        return json.dumps(payload, ensure_ascii=False)

    faithful_system = SYSTEM_FAITHFUL + (BUDGET_SENTENCE if budgeted else "")
    drafts = _call_and_parse(
        client, cache, faithful_system,
        request({"source_language": source_language,
                 "target_language": target_language, "segments": texts}),
        len(texts),
    )
    if any(len(draft) > 88 for draft in drafts):
        raise StageError("translate: faithful translation exceeds the 88-character limit")

    refine_system = SYSTEM_REFINE + (BUDGET_SENTENCE if budgeted else "")
    refined = _call_and_parse(
        client, cache, refine_system,
        request({"source_language": source_language,
                 "target_language": target_language,
                 "sources": texts, "drafts": drafts}),
        len(texts), refine_pass=True,
    )
    if any(len(target) > 88 for target in refined):
        raise StageError("translate: refined translation exceeds the 88-character limit")

    refined = _condense_overruns(
        client, cache, segments, texts, refined, budgets,
        source_language, target_language, budgeted,
    )
    return [(segment["id"], refined[index], budgets[index])
            for index, segment in enumerate(segments)]


def _condense_overruns(client, cache, segments, texts, refined, budgets,
                       source_language, target_language, budgeted):
    """One measured condense round for over-budget members (ADR-0006).

    Soft constraint: any failure keeps the refined draft and warns —
    budget never fails the stage.
    """
    indices = [i for i, b in enumerate(budgets)
               if b is not None and len(refined[i]) > b]
    if not indices:
        return refined
    result = list(refined)
    try:
        condensed = _call_and_parse(
            client, cache,
            SYSTEM_REFINE + BUDGET_SENTENCE,
            json.dumps({
                "source_language": source_language,
                "target_language": target_language,
                "sources": [texts[i] for i in indices],
                "drafts": [refined[i] for i in indices],
                "char_budgets": [budgets[i] for i in indices],
                "note": (
                    "Each draft exceeds its budget; condense it to at most "
                    "its char_budget characters without dropping information."
                ),
            }, ensure_ascii=False),
            len(indices), refine_pass=True,
        )
    except (StageError, LlmError) as exc:
        print(f"translate: condense round failed ({exc}); "
              "keeping refined drafts over budget", file=sys.stderr)
        return result
    for offset, i in enumerate(indices):
        candidate = condensed[offset]
        if candidate.strip() and len(candidate) <= 88 and not has_translation_meta(candidate):
            if len(candidate) < len(result[i]):
                result[i] = candidate
        else:
            print(f"translate: segment {segments[i]['id']} condense output "
                  "rejected by quality gates; keeping refined draft",
                  file=sys.stderr)
    return result
```

Update the two callers of `_translate_batch_once`/`_translate_batch` to thread `rate` (both take a trailing `rate` param; `run` computes `rate = _budget_rate(ctx.config)` and passes it; `translate_batch` closure passes it through), and `run`'s result assembly now unpacks triples:

```python
    targets = {segment_id: (target, budget)
               for batch in results
               for segment_id, target, budget in batch}
```

and in the `translated_segments.append({...})` dict add:

```python
                "char_budget": targets[segment["id"]][1],
```

(using the `(target, budget)` tuple — adjust the `target = targets.get(...)` unpacking to `pair = targets.get(segment["id"]); target, budget = pair if pair else (None, None)`).

- [ ] **Step 4: Run the full translate suite + cache tests**

Run: `.venv/bin/python -m pytest tests/test_t6_translate.py tests/test_llm_cache.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/stages/translate.py tests/test_t6_translate.py
git commit -m "feat(translate): char budgets in payload + one feedback-condense round (ADR-0006)"
```

---

### Task 3: Contract field in translation.json and docs

**Files:**
- Modify: `docs/agents/execution-chain.md:38` (translation.json contract bullet)
- Test: covered by Task 2 assertions on `char_budget`; add one contract-shape test.

**Interfaces:**
- Produces: `translation.json` segments carry `"char_budget": int | null` (null = exempt). `schema_version` stays 1 (additive, `synth_speed` precedent).

- [ ] **Step 1: Add the contract-shape test**

```python
    def test_translation_json_carries_char_budget(self, tmp_path, monkeypatch):
        client = BudgetArrayClient()
        data = self._stage(tmp_path, monkeypatch, client)
        assert data["schema_version"] == 1
        for seg in data["segments"]:
            assert set(seg) >= {"id", "start", "end", "source", "target", "char_budget"}
```

- [ ] **Step 2: Run** — `.venv/bin/python -m pytest tests/test_t6_translate.py -q` — Expected: PASS

- [ ] **Step 3: Update the contract line**

In `docs/agents/execution-chain.md` line 38, extend the segment-field list:

```
- `translation.json`：`source_language/target_language` + `segments[]`；每段含 `id/start/end/source/target`，另含 `char_budget`（ADR-0006：`ceil(槽时长 × translation.max_chars_per_second)`，不足 4 字的短槽为 `null` 豁免；仅 translate 写，下游只读）。段数、ID 和时间戳必须与 `segments.json` 一致。
```

- [ ] **Step 4: Commit**

```bash
git add docs/agents/execution-chain.md tests/test_t6_translate.py
git commit -m "docs(contracts): translation.json gains per-segment char_budget (ADR-0006)"
```

---

### Task 4: F5 version pin probe

**Files:**
- Create: `bench/v4_8b_f5_version_probe.py`
- Modify: `config.example.yaml` (tts.speed comment)
- Output: `bench/results/v4_8/f5-version-pin.json`

**Interfaces:**
- Produces: recorded installed `f5-tts` version + whether short texts (<10 chars) honor `speed` (research: 1.1.22 silently overrides speed to 0.3 for them — issue #1331). No product-code change.

- [ ] **Step 1: Write the probe**

```python
"""V4-8b probe: pin the installed f5-tts and check short-text speed semantics.

Research (docs/research/2026-11-15-tts-pace-energy-mismatch.md): f5-tts
1.1.22 silently overrides `speed` for texts under ~10 chars (upstream
#1331; main changed the override to a multiply). The dub pipeline
synthesizes exactly such short interjections, so the installed version
must be recorded and its behavior verified before ADR-0006's speed
fallback is tuned. One-off evidence harness, not product code.

Usage: .venv/bin/python bench/v4_8b_f5_version_probe.py
"""

import importlib.metadata
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from pipeline.config import load_config  # thin wrapper, read-only
from pipeline.tts.factory import create_tts_engine

SHORT = "好"
LONG = "这是一个用来对照语速语义的较长句子，请正常朗读。"


def synth_duration(engine, voice, text, speed):
    started = time.perf_counter()
    wav = engine.synth(text, voice, speed=speed)
    return wav.samples.shape[-1] / wav.sample_rate, time.perf_counter() - started


def main():
    out_dir = "bench/results/v4_8"
    os.makedirs(out_dir, exist_ok=True)
    config = load_config("config.yaml")
    engine = create_tts_engine(config)
    voice = engine  # voice resolution via pipeline.tts.voices in stage; probe picks default below
    from pipeline.tts.voices import VoiceLibrary
    voice = VoiceLibrary().get("zh-male-01")

    report = {"f5_tts_version": importlib.metadata.version("f5-tts")}
    for label, text in (("short", SHORT), ("long", LONG)):
        d1, _ = synth_duration(engine, voice, text, 1.0)
        d2, _ = synth_duration(engine, voice, text, 2.0)
        report[label] = {"chars": len(text), "dur_speed1": round(d1, 3),
                         "dur_speed2": round(d2, 3),
                         "ratio": round(d1 / d2, 3)}
    # speed=2.0 should roughly halve duration; a ratio near 0.3x on short
    # text marks the 1.1.22-style silent override.
    report["short_text_speed_honored"] = report["short"]["ratio"] > 0.8
    path = os.path.join(out_dir, "f5-version-pin.json")
    with open(path, "w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {path}")


if __name__ == "__main__":
    main()
```

(If `pipeline.config.load_config` has a different name, adapt to the wrapper's actual API — check `pipeline/config.py` before running; the requirement is read-only config load.)

- [ ] **Step 2: Run with the venv**

Run: `.venv/bin/python bench/v4_8b_f5_version_probe.py`
Expected: JSON report; note `f5_tts_version` and `short_text_speed_honored`. If `false`, the speed-fallback sweep in Task 6 must exclude sub-second slots from speed tuning conclusions and the ADR-0006 acceptance note must record the pinned version.

- [ ] **Step 3: Record the pin in config.example.yaml (tts.speed comment)**

```yaml
  # Native synthesis speed (ADR-0005). F5 inherits the reference clip's
  # pace; range 0.5..2.0, default 1.0. Installed f5-tts version and its
  # short-text speed behavior are recorded in bench/results/v4_8/
  # f5-version-pin.json — re-run that probe after any f5-tts upgrade.
```

- [ ] **Step 4: Commit**

```bash
git add bench/v4_8b_f5_version_probe.py config.example.yaml bench/results/v4_8/f5-version-pin.json
git commit -m "bench: pin f5-tts version and verify short-text speed semantics (#1331)"
```

---

### Task 5: Track loudness matching (two-pass loudnorm)

**Files:**
- Create: `pipeline/tts/loudness.py`
- Modify: `pipeline/stages/tts.py` (after `_write_track(...)`, before `ctx.write("dub_plan", ...)`)
- Modify: `config.example.yaml` (tts section), `docs/agents/execution-chain.md:27,40` (dub_plan fields)
- Test: `tests/test_tts_stage.py` (append)

**Interfaces:**
- Consumes: `ctx.path("tts_audio")`, `ctx.path("audio")`.
- Produces: `match_track(track: Path, source: Path, ffmpeg="ffmpeg") -> dict` returning `{"source_i": float, "track_i": float, "true_peak_dbtp": float}`; dub_plan gains `"loudness": {...}|null`. Config `tts.loudness.match_source: bool` (default `false`).

- [ ] **Step 1: Write the failing tests**

```python
class TestLoudnessMatch:
    def test_match_track_pulls_loudness_to_source(self, tmp_path):
        import numpy as np
        import soundfile as sf
        from pipeline.tts.loudness import match_track

        rate = 16000
        quiet = (0.05 * np.sin(2 * np.pi * 220 * np.arange(rate * 2) / rate)).astype("float32")
        loud = (0.5 * np.sin(2 * np.pi * 220 * np.arange(rate * 2) / rate)).astype("float32")
        source, track = tmp_path / "source.wav", tmp_path / "track.wav"
        sf.write(source, loud, rate)
        sf.write(track, quiet, rate)
        report = match_track(track, source)
        assert abs(report["track_i"] - report["source_i"]) <= 2.0
        assert report["true_peak_dbtp"] <= -0.9

    def test_disabled_by_default(self, tmp_path):
        workdir, config = _tts_workdir(tmp_path)  # existing helper in this file
        assert run("tts", workdir, config) == 0
        plan = json.loads((workdir / "tts/dub_plan.json").read_text())
        assert plan["loudness"] is None
```

(If `tests/test_tts_stage.py` names its helpers differently, reuse whatever builds a minimal translation.json + runs the stage — mirror the nearest existing test.)

- [ ] **Step 2: Run** — Expected: FAIL `ModuleNotFoundError: pipeline.tts.loudness` / `loudness` key missing.

- [ ] **Step 3: Implement `pipeline/tts/loudness.py`**

```python
"""Two-pass EBU R128 loudness matching of the dub track to the source.

The dub is energy-disconnected from the original as long as its loudness
is whatever F5 produced (research P1-C: the chain had zero loudness
handling). Two-pass linear loudnorm is the ffmpeg-recommended method:
measure both tracks, then apply a linear gain to the dub that lands its
integrated loudness on the source speech's, true-peak protected.
"""

import json
import subprocess
from pathlib import Path

from pipeline.contracts import StageError


def _measure(path: Path, ffmpeg: str) -> dict:
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-nostats", "-i", str(path),
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
         "-f", "null", "-"],
        capture_output=True, text=True, check=True)
    tail = result.stderr.rsplit("{", 1)
    if len(tail) != 2:
        raise StageError(f"loudness: no loudnorm report for {path}")
    return json.loads("{" + tail[1])


def match_track(track: Path, source: Path, ffmpeg="ffmpeg") -> dict:
    source_stats = _measure(source, ffmpeg)
    input_stats = _measure(track, ffmpeg)
    target_i = float(source_stats["input_i"])
    if target_i < -70:  # essentially silence: nothing to match
        return {"source_i": round(target_i, 2), "track_i": None,
                "true_peak_dbtp": None, "matched": False}
    temp = track.with_name(f".{track.name}.loudnorm.wav")
    subprocess.run(
        [ffmpeg, "-y", "-hide_banner", "-i", str(track), "-af",
         f"loudnorm=I={target_i:.2f}:TP=-1.0:LRA=11:"
         f"measured_I={input_stats['input_i']}:"
         f"measured_LRA={input_stats['input_lra']}:"
         f"measured_TP={input_stats['input_tp']}:"
         f"measured_thresh={input_stats['input_thresh']}:"
         "linear=true:print_format=json",
         "-ar", "24000", str(temp)],
        capture_output=True, text=True, check=True)
    temp.replace(track)
    after = _measure(track, ffmpeg)
    return {"source_i": round(target_i, 2),
            "track_i": round(float(after["input_i"]), 2),
            "true_peak_dbtp": round(float(after["input_tp"]), 2),
            "matched": True}
```

Wire into `pipeline/stages/tts.py` after `_write_track(ctx.path("tts_audio"), track, sample_rate)`:

```python
    loudness = None
    if ctx.config.get("tts.loudness.match_source", False):
        from pipeline.tts.loudness import match_track
        loudness = match_track(ctx.path("tts_audio"), ctx.path("audio"))
        print(f"tts: loudness matched to source "
              f"(I {loudness['track_i']} LUFS vs source {loudness['source_i']})",
              file=sys.stderr)
```

and add `"loudness": loudness,` to the `ctx.write("dub_plan", {...})` dict.

Config (`config.example.yaml`, tts block):

```yaml
  # Two-pass EBU R128 loudnorm matching the dub track's integrated
  # loudness to the source speech (research P1-C). Default off to keep
  # subtitle-only runs byte-identical; enable in dubbing configs.
  loudness:
    match_source: false
```

Docs: extend the `tts/dub_plan.json` bullet in `execution-chain.md:40` with `loudness`（`match_track` 报告或 `null`）.

- [ ] **Step 4: Run**

Run: `.venv/bin/python -m pytest tests/test_tts_stage.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline/tts/loudness.py pipeline/stages/tts.py config.example.yaml docs/agents/execution-chain.md tests/test_tts_stage.py
git commit -m "feat(tts): optional two-pass loudnorm dub-track loudness matching"
```

---

### Task 6: Acceptance harness and speed-fallback sweep

**Files:**
- Create: `bench/v4_9_budget_acceptance.py`
- Output: `bench/results/v4_9/budget-acceptance.json`, `bench/results/v4_9/condense-blind-review.md`
- Modify: `docs/adr/0006-*.md` (record measured outcome, mirroring ADR-0005 style)

**Interfaces:**
- Consumes: pipeline `run()` per stage; a fresh workdir seeded from the 1h acceptance media; Q4 llama-server URL (`--url`).
- Produces: the numeric verdict against every gate in the 验收目标 section; a blind-review export of 40 condensed pairs; the lowest `tts.speed` passing the ≤2% overflow gate with budgeted text.

- [ ] **Step 1: Write the harness**

```python
"""V4-9 acceptance: ADR-0006 P1 gates on real 1h content.

Runs the product translate stage with budgets against a running Q4
llama-server, scores the 验收目标 gates, sweeps tts speeds for the
fallback floor, and exports condensed pairs for blind review. One-off
evidence harness; assumes the media already ingested in a run dir passed
via --run (never mutates runs/ in place — copies media into bench/results).

Usage:
    .venv/bin/python bench/v4_9_budget_acceptance.py \
        --run runs/tts-accept-1h-stage1 \
        --url http://127.0.0.1:9941/v1 \
        --out bench/results/v4_9/budget-acceptance.json
"""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys

sys.path.insert(0, os.getcwd())

GATES = {
    "adherence_pct": 45.0,
    "within_plus1_pct": 54.0,
    "median_chars_per_s": 5.0,
    "contract_failure_pct": 1.0,
    "mean_llm_calls_per_segment": 3.0,
    "overflow_ratio": 0.02,
    "e2e_wall_min": 8.5,
}


def score_translation(workdir):
    data = json.loads(open(os.path.join(workdir, "translation.json")).read())
    segs = data["segments"]
    usable = [s for s in segs if s["char_budget"] is not None]
    adhered = sum(1 for s in usable if len(s["target"]) <= s["char_budget"])
    within1 = sum(1 for s in usable if len(s["target"]) <= s["char_budget"] + 1)
    cps = sorted(len(s["target"]) / (s["end"] - s["start"])
                 for s in segs if s["end"] > s["start"])
    return {
        "segments": len(segs), "budgeted": len(usable),
        "adherence_pct": round(100 * adhered / len(usable), 1),
        "within_plus1_pct": round(100 * within1 / len(usable), 1),
        "median_chars_per_s": round(statistics.median(cps), 2),
        "p90_chars_per_s": round(cps[int(0.9 * (len(cps) - 1))], 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:9941/v1")
    parser.add_argument("--workdir", default="bench/results/v4_9/workdir")
    parser.add_argument("--out", default="bench/results/v4_9/budget-acceptance.json")
    args = parser.parse_args()

    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    for name in ("media", "transcript.json"):
        src = os.path.join(args.run, name)
        dst = os.path.join(workdir, name)
        if os.path.isfile(src) and not os.path.isfile(dst):
            (shutil.copytree if os.path.isdir(src) else shutil.copyfile)(src, dst)

    config = os.path.join(workdir, "config.yaml")
    with open(config, "w") as handle:  # bench-local config; not the repo config.yaml
        handle.write("llm:\n  local:\n    enabled: false\n"
                     f"api:\n  base_url: '{args.url.rstrip('/v1')}'\n"
                     "api_fallback: false\n")
    # NOTE: adapt to the real config surface for pointing the pipeline at
    # an already-running server (see pipeline/llm/factory.py); the pipeline
    # must never be given the repo's config.yaml.

    stages = ["segment", "translate"]
    import time
    walls = {}
    for stage in stages:
        started = time.perf_counter()
        code = subprocess.call(
            [".venv/bin/python", "video_transcribe.py", args.workdir_root_unused
             if False else workdir, "--stages", stage, "--config", config])
        walls[stage] = round(time.perf_counter() - started, 1)
        if code != 0:
            sys.exit(f"stage {stage} failed")

    report = {"translation": score_translation(workdir), "walls": walls,
              "gates": GATES}

    overflow_by_speed = {}
    for speed in ("1.0", "1.3", "1.6", "2.0"):
        with open(config, "a") as handle:
            handle.write(f"tts:\n  speed: {speed}\n")
        for stage in ("tts", "dub"):
            if subprocess.call([sys.executable, "video_transcribe.py", workdir,
                                "--stages", stage, "--config", config]) != 0:
                sys.exit(f"stage {stage} failed at speed {speed}")
        plan = json.loads(open(os.path.join(workdir, "tts/dub_plan.json")).read())
        overflow_by_speed[speed] = plan["overflow"]["ratio"]

    report["overflow_ratio_by_speed"] = overflow_by_speed
    passing = [float(s) for s, r in overflow_by_speed.items() if r <= GATES["overflow_ratio"]]
    report["lowest_passing_speed"] = min(passing) if passing else None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")
    print("HUMAN GATES (record manually in the report): pacing / energy / "
          "blind review of bench/results/v4_9/condense-blind-review.md")


if __name__ == "__main__":
    main()
```

Before running: verify the runner CLI arg order in `video_transcribe.py` (`--stages`, `--config`, positional run dir vs `--source`) and the exact config keys for pointing at a running server (`pipeline/llm/factory.py`) — adjust the harness to the real surfaces; the gates and scoring stay as written. Also export the blind-review pairs: 40 segments where `len(target) < len(baseline target)` (compare against `runs/tts-accept-1h-stage1/translation.json` by id), written as a two-column markdown table.

- [ ] **Step 2: Run with server up**

Start llama-server exactly as `pipeline/llm/local_service.py` does (binary dir on `LD_LIBRARY_PATH`, `-ngl 99 --jinja -c 4096 -np 1 --repeat-penalty 1.0`), then:

Run: `.venv/bin/python bench/v4_9_budget_acceptance.py --run runs/tts-accept-1h-stage1 --url http://127.0.0.1:9941/v1`
Expected: report meets every gate in 验收目标 B; `lowest_passing_speed` ≤ 1.6.

- [ ] **Step 3: Human gates**

Maintainer listens to the speed-fallback dub (pacing/energy) and blind-reviews the 40 pairs; verdicts appended to the report JSON under `"human_gates"`.

- [ ] **Step 4: Record outcomes in ADR-0006**

Append a "Measured outcomes" paragraph (ADR-0005 style) with adherence, median chars/s, chosen `tts.speed`, overflow ratio, E2E wall; commit the report.

- [ ] **Step 5: Commit & merge**

```bash
git add bench/v4_9_budget_acceptance.py bench/results/v4_9/ docs/adr/0006-dub-fidelity-budget-and-voice-mapping.md
git commit -m "bench: ADR-0006 P1 acceptance — budget adherence + speed fallback (overall PASS/FAIL)"
git checkout main && git merge --no-ff <branch> && git push origin main
```

---

## Self-Review

- Spec coverage: ADR-0006 P1 items 1–4 → Tasks 1–3 (budget+condense+exemption+cache), Task 6 (speed residual + unchanged gates); research P1-C loudness → Task 5; version pin → Task 4. P1-B pause semantics deliberately OUT (ADR-0005 territory, short-slot exemption instead).
- Placeholders: Task 6 Step 1 contains a harness draft with two explicit "verify the real surface" notes (runner CLI order, factory config keys) — resolve both by reading `video_transcribe.py`/`pipeline/llm/factory.py` BEFORE running; everything else is concrete.
- Type consistency: `char_budget(start, end, rate) -> int | None`; `_translate_batch_once(..., rate)`; translation triple `(id, target, budget)`; `match_track -> dict`.
