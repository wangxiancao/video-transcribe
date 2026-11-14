# TTS native synthesis speed becomes the `tts.speed` config knob

Status: accepted (2026-11-15)

## Context

ADR-0004 pinned F5-TTS synthesis at native speed (`SYNTH_SPEED = 1.0`): fitting happens after synthesis via ffmpeg `atempo`, capped at 1.25x (ADR-0003), and residual length is overflow accounted in the dub plan. The bench acceptance gate quantifies cumulative overflow at ≤ 2% of source audio (`docs/agents/execution-chain.md`).

The stage-1 acceptance run (issue #5: real 1h English tech podcast, `target_language: zh`, default voice `zh-female-01`) measured **65.44% cumulative overflow** (2575 s over 3935.6 s of source audio; 1092 of 1102 segments overflow) and the dub stage correctly refused the resulting 3.38 s duration drift. Diagnosis from the measured dub plan and same-text voice probes:

- `zh-female-01`'s reference clip is itself slow-paced (15 chars / 6.76 s ≈ 2.2 chars/s), and F5-TTS inherits the reference's pace: pure-zh synthesis measured 2.4–2.6 chars/s across all text lengths (natural Mandarin ≈ 4–6 chars/s). The overflow is rate-driven, not term-driven (segments containing Latin tokens overflow barely more than pure-zh ones).
- A same-text probe with `zh-male-01` (reference pace 4.7 chars/s) measures ≈ 5.2–5.5 chars/s. Applying the measured per-segment rate factor to the whole run predicts ≈ 10.5% overflow at speed 1.0 — still 5x over the gate.
- No in-design configuration passes: raising the atempo cap increases overflow by definition (overflow measures how far speech extends past its slot, and a higher cap lets it extend further); `tts.input=source` with the fast `en-female-01` still predicts ≈ 8.3% on this content; the residual is driven by fast conversational slots (median slot raw-ratio 1.22 vs the 1.25 cap) plus a per-segment synthesis floor (~0.6–2.2 s fixed overhead).

The F5 infer API already accepts a native `speed` argument (0.5–2.0), the engine exposes it, and ADR-0004 deliberately keyed the synthesis cache by speed "so a future synthesis-speed knob cannot collide" — the knob was anticipated, only the config surface was missing.

## Decision

Expose native synthesis speed as `tts.speed` (float, default 1.0, clamped to the engine range 0.5–2.0; values outside it are a config error before any engine is built). The stage passes it to every synthesis call and into the cache key, and records it as `synth_speed` in `tts/dub_plan.json` so a plan fully determines its audio. The validator enforces the range on the plan.

For real content the acceptance configuration is `voice: zh-male-01` + `speed: 1.6` (predicted ≈ 1.2% overflow); the default remains `zh-female-01` + `speed: 1.0` for existing behavior. The atempo fitting cap stays 1.25 (ADR-0003 unchanged), and the ≤ 2% overflow gate is unchanged — the knob makes the gate reachable, it does not relax it.

## Alternatives rejected

- **Revise the ≤ 2% overflow gate** to a measured reality (e.g. ≤ 10%): the gate is design consensus in `execution-chain.md`; lowering it to fit a slow voice inverts the contract and hides the product defect (a dub whose speech trails 65% of its slots is unusable).
- **Accept `input=source` English dubbing for the acceptance**: it changes what is validated (an English dub, not the `target_language: zh` product path) and still fails the gate without a speed increase.
- **Raise the atempo cap** (ADR-0003): monotonic increase of overflow; also degrades prosody beyond what native speed compression achieves.
- **Re-record the voice library** with faster-paced references: right long-term fix for voice quality, but new NC-licensed audio assets are a maintainer decision and cannot unblock the acceptance today; the knob composes with better references later.

## Reopening conditions

Synthesized speech at the acceptance configuration sounds unacceptably rushed in the human listening gate; a faster-paced zh reference clip enters the library (then `speed` can return toward 1.0); or the F5 infer speed argument changes semantics in an upgrade.
