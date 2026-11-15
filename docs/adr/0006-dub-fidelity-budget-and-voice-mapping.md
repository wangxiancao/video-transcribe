# Dub fidelity: budget the translation, map voices by speaker — don't speed up synthesis and don't clone across languages

Status: accepted (2026-11-15)

## Context

The 1h acceptance dub (ADR-0005 configuration: `zh-male-01` + `speed: 2.0` + `pause_s: 0.8`) triggered ADR-0005's own reopening condition: in the human listening gate the Mandarin sounds rushed next to the calm English original, and the dubbing voice is visibly a different "person" from the source speaker. Research against primary sources is recorded in `docs/research/2026-11-15-tts-pace-energy-mismatch.md` and `docs/research/2026-11-15-tts-voice-identity-mismatch.md`; the decisive measurements:

- **The text, not the engine, sets the pace.** On the 933-segment acceptance corpus (`runs/tts-accept-1h-stage1/translation.json`), the zh translation alone demands a **median 5.5 chars/s** of its speech slot to fit naturally (natural Mandarin ≈ 4–6 chars/s); **73.4% of slots exceed 4.5 chars/s**, p95 = 9.12. The worst non-artifact slot needs 18.8 chars/s. `speed: 2.0` + atempo 1.25 is the tts stage paying the translate stage's debt — which is why the dub sounds like a rushed read.
- **Prompt-only budgeting fails on the Q4 runtime.** Probe `bench/v4_8_budget_probe.py` (65 stratified segments, product prompts + product parser, results in `bench/results/v4_8/budget-probe.json`): a budget sentence in both system prompts achieved **23.8% adherence vs 33.8% for the unmodified historical output** — the instruction is noise to HY-MT2-1.8B Q4, and it destabilized two responses into translator commentary.
- **Measured enforcement works.** Best-of-4 sampling: 43.1% adherence at 8 LLM calls/segment. One explicit feedback-condense round when the result exceeds budget: **49.2% adherence (+1 char: 54.0%), median 5.21 chars/s (back inside the natural band), mean overrun 2.62 chars, 2.75 calls/segment, 0.24 s/segment wall** — negligible against the 7.2 min E2E. Spot checks show meaning survives condensation; full blind review stays with the implementation ticket.
- **Budget floors break the model.** Both probe contract failures were the same two segments whose slot duration forces a ≤4–8 char budget; the model answers the impossible budget with translator commentary. Sub-second slots are structurally unfittable at the text level — ADR-0005 already showed their overflow is dominated by the per-segment synthesis floor, and `pause_s` merging is the tool for them.
- **Voice identity: the intuitive shortcut is officially broken.** F5-TTS zero-shot cross-lingual cloning (English reference → Chinese output) has accent/timbre leakage confirmed by the maintainers (F5 issue #315), so "use the source speaker's voice as the reference clip" is not available on the current engine. ADR-0004's engine rejections otherwise stand (RTFx gates, licenses); its GPT-SoVITS OOM rejection is weakened (issue #1519 closed as batch-size-related, and this pipeline synthesizes batch=1) but that alone is not a swap trigger.

## Decision

**Pace (P1): the translation carries a char budget; the tts stage stops over-paying.**

1. New config key `translation.max_chars_per_second` (default 4.5). The translate stage derives each segment's budget as `ceil(slot_duration × max_chars_per_second)` from the segment timing it already consumes, records it per segment in `translation.json`, and enforces it with **one feedback-condense round** (measured overrun → "condense to ≤N chars without dropping information"). Prompt-only enforcement is rejected by measurement; best-of-k is the fallback if feedback regresses.
2. Slots whose budget floors below a minimal natural sentence (~4 chars, i.e. sub-second slots) are exempted from the budget and flagged; their overflow remains ADR-0005 territory (`pause_s` merging, accepted overflow accounting).
3. `tts.speed` becomes a fitting residual, not the primary pace control: once budgets hold, the dubbing configuration returns toward natural synthesis speed instead of pinning `speed: 2.0`. The ≤ 2% overflow gate and the atempo 1.25 cap (ADR-0003) are unchanged.
4. The budget enters the translation cache key, so budgeted and unbudgeted runs cannot collide in cache.

**Voice identity (P2): speaker-aware voice mapping, not cloning.**

1. Diarization (pyannote community-1: code MIT, weights CC-BY-4.0, gated download, fully local at runtime) labels segments with source speakers using the existing word-level timestamps; each diarized speaker is mapped to one voice from the named library (selected by ECAPA embedding similarity), recorded per segment in the dub plan. One source speaker keeps one voice for the whole video.
2. Zero-shot cross-lingual cloning on F5-TTS is rejected for the product path (official accent leakage); an engine swap is rejected (ADR-0004 stands). OpenVoice v2 tone-color conversion (MIT) is the designated follow-up probe for true-to-source timbre — adopted only if a probe on F5 24kHz zh output passes similarity and RTFx gates.

## Consequences

- translate grows a cross-stage dependency on segment *timing* (it already consumes the segment list); the budget is a contract field in `translation.json` like any other (`docs/agents/execution-chain.md`).
- Budgeted zh text is bounded by slot length (~≤37 chars at the 8.16 s max segment), comfortably inside the subtitle 75-char contract; subtitle behavior is unchanged.
- Expected translate cost: ~+0.75 LLM calls/segment (measured 0.24 s/segment); RTFx gates unaffected.
- Voice mapping adds pyannote as the first diarization dependency; its weights are gated (HF acceptance) which the one-time-download policy already accommodates.
- Blind quality review of the 47 probe-condensed outputs is pending; if condensation is judged lossy in review, the fallback ladder is best-of-k, then a larger translation model.

## Reopening conditions

Feedback-condense adherence regresses below the prompt-only baseline on real content; condensation is judged meaning-lossy in blind review; pyannote diarization quality is unacceptable on multi-speaker real content (then speaker mapping falls back to single-voice); the OpenVoice v2 probe fails its gates and true-to-source timbre becomes a requirement (then the engine question reopens with GPT-SoVITS's cleared OOM record).
