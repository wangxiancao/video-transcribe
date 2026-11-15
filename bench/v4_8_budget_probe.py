"""V4-8 probe: can HY-MT2 (Q4) obey a per-segment Chinese char budget?

Evidence harness for the translation char-budget direction selected from
docs/research/2026-11-15-tts-pace-energy-mismatch.md (方案 A): the zh
translation text alone already demands a median 5.5 chars/s of its speech
slot on the 1h acceptance content (63% of slots >5 chars/s), which is why
the tts stage pays with speed 2.0 + atempo and sounds rushed. Before the
translate stage grows a budget knob, this probe measures whether the
small model can actually follow the instruction.

Method: reuse the product SYSTEM_FAITHFUL / SYSTEM_REFINE prompts and the
product JSON parser (private import is deliberate — fidelity to the real
contract, repairs included), append a char_budgets array to both request
payloads and one budget sentence to both system prompts, and compare
against the historical pipeline output for the same segments
(translation.json of the same run). One segment per request, mirroring
llm.local.batch_segments=1 for the Q4 runtime. One-off evidence harness,
not product code.

Usage:
    .venv/bin/python bench/v4_8_budget_probe.py \
        --run runs/tts-accept-1h-stage1 \
        --url http://127.0.0.1:9941/v1 \
        --out bench/results/v4_8/budget-probe.json
"""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.getcwd())

from pipeline.stages.translate import (  # noqa: E402  (fidelity to product contract)
    SYSTEM_FAITHFUL,
    SYSTEM_REFINE,
    _parse_translation_strings,
)
from pipeline.contracts import StageError  # noqa: E402

# The budget sentence appended to both system prompts. Speaks in characters
# because that is what the budget counts and small models confuse 字/词.
BUDGET_SENTENCE = (
    " The request also includes char_budgets, an ordered array of integers, "
    "one per segment: translation number N MUST NOT exceed char_budgets[N-1] "
    "characters (count every Chinese character; spaces, if any, do not "
    "count). Stay within budget by condensing wording, never by dropping "
    "meaning, names or numbers."
)


def post_chat(base_url, system, user, max_tokens=512, timeout=300,
              temperature=0.0, top_p=1.0):
    payload = json.dumps(
        {"model": "probe",
         "messages": [{"role": "system", "content": system},
                      {"role": "user", "content": user}],
         "temperature": temperature, "top_p": top_p,
         "max_tokens": max_tokens}).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", data=payload,
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.loads(response.read())
    wall = time.perf_counter() - started
    completion = (result.get("usage") or {}).get("completion_tokens")
    return {
        "content": result["choices"][0]["message"]["content"],
        "wall_s": round(wall, 3),
        "completion_tokens": completion,
        "tok_per_s": round(completion / wall, 1) if completion and wall else None,
    }


def call_and_parse(base_url, system, user, refine_pass=False, temperature=0.0):
    """One LLM round with a single fresh-call retry on contract failure,
    mirroring the stage's _call_and_parse semantics."""
    response = post_chat(base_url, system, user, temperature=temperature)
    timing = {k: response[k] for k in ("wall_s", "completion_tokens", "tok_per_s")}
    try:
        strings = _parse_translation_strings(
            response["content"], 1, refine_pass=refine_pass)
        return strings[0], timing
    except StageError:
        response = post_chat(base_url, system, user, temperature=temperature)
        timing = {k: response[k] for k in ("wall_s", "completion_tokens", "tok_per_s")}
        strings = _parse_translation_strings(
            response["content"], 1, refine_pass=refine_pass)
        return strings[0], timing


def load_corpus(run_dir):
    """Join segments.json slots with translation.json targets by id."""
    segments = json.load(
        open(os.path.join(run_dir, "segments.json")))["segments"]
    translations = json.load(
        open(os.path.join(run_dir, "translation.json")))["segments"]
    slots = {s["id"]: s for s in segments}
    rows, degenerate = [], 0
    for item in translations:
        slot = slots.get(item["id"])
        if slot is None:
            continue
        duration = slot["end"] - slot["start"]
        if duration <= 0.05:
            # ~1 ms segmenter artifacts (3/933 here) cannot hold speech at
            # any rate; they are a segmentation bug, not a budget question.
            degenerate += 1
            continue
        rows.append({
            "id": item["id"],
            "source": item["source"],
            "target_baseline": item["target"],
            "duration_s": round(duration, 3),
            "baseline_chars": len(item["target"]),
            "baseline_chars_per_s": round(len(item["target"]) / duration, 2),
        })
    return rows, degenerate


def pick_sample(rows, sample_size, seed):
    """Stratify by how badly the baseline misses natural pace, always
    keeping some slots that already fit (regression guard) and some
    sub-second slots (the interjection disaster zone)."""
    bands = [
        ("fits", lambda r: r["baseline_chars_per_s"] <= 4.5),
        ("mild", lambda r: 4.5 < r["baseline_chars_per_s"] <= 6.0),
        ("tight", lambda r: 6.0 < r["baseline_chars_per_s"] <= 8.0),
        ("over", lambda r: r["baseline_chars_per_s"] > 8.0),
    ]
    rng = random.Random(seed)
    picked = []
    per_band = max(2, sample_size // len(bands))
    for _, predicate in bands:
        pool = [r for r in rows if predicate(r)]
        picked.extend(rng.sample(pool, min(per_band, len(pool))))
    short = [r for r in rows if r["duration_s"] < 1.0
             and r["baseline_chars_per_s"] > 4.5]
    for r in rng.sample(short, min(6, len(short))):
        if r["id"] not in {p["id"] for p in picked}:
            picked.append(r)
    picked.sort(key=lambda r: r["id"])
    return picked


def faithful_call(base_url, source, budget, temperature=0.0):
    request = {"source_language": "English", "target_language": "简体中文",
               "segments": [source], "char_budgets": [budget]}
    return call_and_parse(
        base_url, SYSTEM_FAITHFUL + BUDGET_SENTENCE,
        json.dumps(request, ensure_ascii=False), temperature=temperature)


def refine_call(base_url, source, draft, budget, note=None, temperature=0.0):
    request = {"source_language": "English", "target_language": "简体中文",
               "sources": [source], "drafts": [draft],
               "char_budgets": [budget]}
    if note:
        request["note"] = note
    return call_and_parse(
        base_url, SYSTEM_REFINE + BUDGET_SENTENCE,
        json.dumps(request, ensure_ascii=False), refine_pass=True,
        temperature=temperature)


def summarize(name, records):
    """Adherence stats over per-segment records carrying `chars`/`budget`."""
    usable = [r for r in records if not r.get("contract_fail")]
    adhered = [r for r in usable if r["chars"] <= r["budget"]]
    within1 = [r for r in usable if r["chars"] <= r["budget"] + 1]
    overruns = [max(0, r["chars"] - r["budget"]) for r in usable]
    cps = [r["chars"] / r["duration_s"] for r in usable]
    cps.sort()
    summary = {
        "variant": name,
        "n": len(records),
        "contract_failures": len(records) - len(usable),
        "adherence": f"{len(adhered)}/{len(usable)}"
                     f" ({len(adhered) / len(usable):.1%})" if usable else "n/a",
        "within_plus1": f"{len(within1)}/{len(usable)}"
                        f" ({len(within1) / len(usable):.1%})" if usable else "n/a",
        "mean_overrun_chars": round(statistics.mean(overruns), 2) if usable else None,
        "max_overrun_chars": max(overruns) if usable else None,
        "median_chars_per_s": round(statistics.median(cps), 2) if cps else None,
        "p95_chars_per_s": round(cps[int(0.95 * (len(cps) - 1))], 2) if cps else None,
    }
    return summary


def run_prompt(base_url, row, budget):
    """Strategy `prompt`: budget sentence in both passes, greedy."""
    draft, faithful_timing = faithful_call(base_url, row["source"], budget)
    refined, refine_timing = refine_call(base_url, row["source"], draft, budget)
    return {"chars": len(refined), "text": refined, "draft_chars": len(draft),
            "draft": draft, "calls": 2,
            "wall_s": round(faithful_timing["wall_s"] + refine_timing["wall_s"], 3)}


def run_bestk(base_url, row, budget, k, temperature):
    """Strategy `bestk`: k sampled faithful+refine candidates, pick the
    first that fits the budget, else the shortest. A garbled candidate is
    skipped, not fatal; only all-garbled fails the segment."""
    candidates, walls, garbled = [], 0.0, 0
    for _ in range(k):
        try:
            draft, t1 = faithful_call(base_url, row["source"], budget, temperature)
            refined, t2 = refine_call(base_url, row["source"], draft, budget,
                                      temperature=temperature)
        except StageError:
            garbled += 1
            continue
        candidates.append(refined)
        walls += t1["wall_s"] + t2["wall_s"]
    if not candidates:
        raise StageError("bestk: all candidates failed the contract")
    fitting = [c for c in candidates if len(c) <= budget]
    chosen = fitting[0] if fitting else min(candidates, key=len)
    return {"chars": len(chosen), "text": chosen,
            "candidates_chars": [len(c) for c in candidates],
            "picked_fitting": bool(fitting), "garbled_candidates": garbled,
            "calls": 2 * k, "wall_s": round(walls, 3)}


def run_feedback(base_url, row, budget):
    """Strategy `feedback`: greedy prompt strategy, then one explicit
    condense round when the result is over budget."""
    result = run_prompt(base_url, row, budget)
    if result["chars"] > budget:
        draft = result["text"]
        note = (f"Draft is {len(draft)} characters, over budget {budget} by "
                f"{len(draft) - budget}; condense it to at most {budget} "
                "characters without dropping information.")
        refined, timing = refine_call(base_url, row["source"], draft, budget,
                                      note=note)
        result = {"chars": len(refined), "text": refined,
                  "draft_chars": len(draft), "condensed_from_chars": len(draft),
                  "calls": result["calls"] + 1,
                  "wall_s": round(result["wall_s"] + timing["wall_s"], 3)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="runs/tts-accept-1h-stage1")
    parser.add_argument("--url", default="http://127.0.0.1:9941/v1")
    parser.add_argument("--rate", type=float, default=4.5,
                        help="natural zh pace the budget targets, chars/s")
    parser.add_argument("--floor-chars", type=int, default=4,
                        help="minimum budget; below this a slot cannot hold "
                             "any natural zh sentence")
    parser.add_argument("--sample", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--strategies", default="prompt,bestk,feedback")
    parser.add_argument("--bestk", type=int, default=4)
    parser.add_argument("--sample-temp", type=float, default=0.8)
    parser.add_argument("--out", default="bench/results/v4_8/budget-probe.json")
    args = parser.parse_args()

    rows, degenerate = load_corpus(args.run)
    all_cps = sorted(r["baseline_chars_per_s"] for r in rows)
    corpus = {
        "run": args.run,
        "segments_usable": len(rows),
        "segments_degenerate_zero_duration": degenerate,
        "baseline_median_chars_per_s": round(statistics.median(all_cps), 2),
        "baseline_p90_chars_per_s": round(all_cps[int(0.9 * (len(all_cps) - 1))], 2),
        "baseline_p95_chars_per_s": round(all_cps[int(0.95 * (len(all_cps) - 1))], 2),
        "baseline_over_rate_pct": round(
            100 * sum(1 for c in all_cps if c > args.rate) / len(all_cps), 1),
        "rate_chars_per_s": args.rate,
    }

    sample = pick_sample(rows, args.sample, args.seed)
    control_records = []
    for row in sample:
        budget = max(args.floor_chars,
                     math.ceil(row["duration_s"] * args.rate))
        control_records.append({
            "id": row["id"], "duration_s": row["duration_s"],
            "budget": budget, "chars": row["baseline_chars"],
            "chars_per_s": row["baseline_chars_per_s"],
            "floor_bound": row["duration_s"] * args.rate < args.floor_chars,
            "text": row["target_baseline"],
        })

    runners = {
        "prompt": lambda r, b: run_prompt(args.url, r, b),
        "bestk": lambda r, b: run_bestk(args.url, r, b, args.bestk,
                                        args.sample_temp),
        "feedback": lambda r, b: run_feedback(args.url, r, b),
    }
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    records = {name: [] for name in strategies}
    for row in sample:
        budget = max(args.floor_chars,
                     math.ceil(row["duration_s"] * args.rate))
        outcomes = {}
        for name in strategies:
            record = {"id": row["id"], "duration_s": row["duration_s"],
                      "budget": budget,
                      "floor_bound": row["duration_s"] * args.rate
                      < args.floor_chars}
            try:
                outcome = runners[name](row, budget)
                outcome["chars_per_s"] = round(
                    outcome["chars"] / row["duration_s"], 2)
                record.update(outcome)
            except StageError as exc:
                record.update({"contract_fail": str(exc)[:200]})
            records[name].append(record)
            outcomes[name] = record.get("chars", "FAIL")
        print(f"seg {row['id']:>3}: budget {budget:>2} "
              f"baseline {row['baseline_chars']:>2} "
              + " ".join(f"{name}={outcomes[name]}" for name in strategies),
              file=sys.stderr)

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "corpus": corpus,
        "sample": {"size": len(sample), "seed": args.seed,
                   "bestk": args.bestk, "sample_temp": args.sample_temp},
        "control_from_translation_json": summarize("control", control_records),
        "control_records": control_records,
    }
    for name in strategies:
        summary = summarize(name, records[name])
        walls = [r["wall_s"] for r in records[name] if "wall_s" in r]
        calls = [r["calls"] for r in records[name] if "calls" in r]
        if walls:
            summary["mean_wall_s_per_segment"] = round(statistics.mean(walls), 2)
            summary["mean_llm_calls_per_segment"] = round(
                statistics.mean(calls), 2)
        report[f"strategy_{name}"] = summary
        report[f"records_{name}"] = records[name]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    summary_keys = (["corpus", "control_from_translation_json"]
                    + [f"strategy_{name}" for name in strategies])
    print(json.dumps({k: report[k] for k in summary_keys},
                     ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
