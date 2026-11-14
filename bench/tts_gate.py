"""TTS/dub acceptance gate: quantify the design-consensus gates on one run.

Post-hoc probe over a completed run workdir (issue #4). Produces one JSON
report where every gate carries a measured value, its fixed threshold and a
PASS/FAIL verdict — thresholds are design consensus, never relaxed here:

- overflow.ratio        <= 2% of source audio (execution-chain.md)
- timeline spot check   <= 50 ms sound beyond a declared window boundary
                          (plan timestamps must match the written audio)
- dub duration drift    <= 0.5 s vs source video (MAX_DURATION_DRIFT)
- TTS E2E RTF           <= 0.5 wall/media, cold incl. model load (ADR-0004;
                          RTF here is the ADR convention wall/media, so the
                          gate is an upper bound on processing time)
- VRAM sustained growth <= 256 MB after the first-half load peak (documented
                          harness default: a leak trend, not a peak gate)

With --timed-tts the probe re-synthesizes the run's translation in a scratch
workdir (cache off) as a timed subprocess while sampling nvidia-smi; that
measurement is independent of the run's own tts artifacts. Without it, only
the artifact gates are judged and the timed gates report null.

Usage:
    .venv/bin/python bench/tts_gate.py --workdir <run> --config <config.yaml> \
        [--timed-tts] [--samples 9] [--gpu-index 0] [--out PATH]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.getcwd())

# Design-consensus thresholds; printed into the report so a value can never
# drift away from its gate without an explicit harness change.
OVERFLOW_MAX_RATIO = 0.02
TIMELINE_TOLERANCE_S = 0.05
DUB_DRIFT_MAX_S = 0.5
RTF_MAX = 0.5
VRAM_GROWTH_MAX_MB = 256.0
# Fraction of a placed window that must sit above the onset threshold for
# the spot check to count as speech present.
SPEECH_PRESENCE_MIN = 0.2


def gate(value, threshold, verdict=None, **extra):
    return {
        "value": value,
        "threshold": threshold,
        "verdict": verdict if verdict is not None else (
            "PASS" if value is not None and value <= threshold else "FAIL"
        ),
        **extra,
    }


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            if rate <= 0:
                return None
            return wav.getnframes() / float(rate)
    except (OSError, wave.Error):
        return None


def ffprobe_duration_and_streams(path: Path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format=duration:stream=codec_type,codec_name", "-of", "json",
         str(path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None, None, result.stderr.strip()[-300:]
    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None, None, "ffprobe returned invalid JSON"
    duration = float((probe.get("format") or {}).get("duration") or 0) or None
    streams = [
        {"codec_type": s.get("codec_type"), "codec_name": s.get("codec_name")}
        for s in probe.get("streams") or []
    ]
    return duration, streams, None


def boundary_check(track: np.ndarray, sample_rate: int, entry: dict, entries: list):
    """Consistency of one placed window's boundaries with the actual audio.

    Speech must not start before placed_start nor extend past placed_end by
    more than the tolerance: the plan claims its timestamps were measured off
    the written audio, so a boundary with sound beyond it is drift. Boundaries
    covered by another segment's placed window (overflow overlap) are skipped —
    the sound there belongs to another window, not to this one. F5's own
    leading silence inside a window is a synthesis property, reported as
    onset_info, not drift.
    """
    rate = sample_rate
    threshold = max(float(np.percentile(np.abs(track), 10)) * 5.0, 1e-4)
    start = max(0, int(round(entry["placed_start"] * rate)))
    end = min(track.shape[-1], int(round(entry["placed_end"] * rate)))
    if end <= start:
        return None, 0.0, None, "empty window"
    window = track[start:end]
    presence = float(np.mean(np.abs(window) > threshold))

    def overlapped(lo: float, hi: float) -> bool:
        return any(
            other is not entry
            and other["placed_start"] < hi and other["placed_end"] > lo
            for other in entries
        )

    worst = 0.0
    onset_info = None
    # Start boundary: sound in the 50 ms before placed_start (when not covered
    # by an earlier window) means speech begins earlier than declared.
    pre_lo = max(0.0, entry["placed_start"] - TIMELINE_TOLERANCE_S)
    if not overlapped(pre_lo, entry["placed_start"]):
        pre = track[int(round(pre_lo * rate)):start]
        above = np.nonzero(np.abs(pre) > threshold)[0]
        if above.size:
            worst = max(worst, (start - int(above[-1])) / rate)
    # Speech must actually be present inside the window.
    first = int(np.argmax(np.abs(window[:rate]) > threshold)) if \
        (np.abs(window[:rate]) > threshold).any() else None
    if first is not None:
        onset_info = round(first / rate, 6)
    # End boundary: sound within 50 ms after placed_end (when not covered by a
    # later window) means speech extends beyond the declared end.
    post_hi = entry["placed_end"] + TIMELINE_TOLERANCE_S
    if post_hi * rate <= track.shape[-1] and not overlapped(entry["placed_end"], post_hi):
        post = track[end:int(round(post_hi * rate))]
        above = np.nonzero(np.abs(post) > threshold)[0]
        if above.size:
            worst = max(worst, (int(above[0]) + 1) / rate)
    return onset_info, presence, worst, None


def timeline_gate(workdir: Path, plan: dict, samples: int):
    track_path = workdir / "tts" / "tts.wav"
    if not track_path.is_file():
        return gate(None, TIMELINE_TOLERANCE_S, verdict="FAIL",
                    reason="tts/tts.wav missing", sampled=[])
    track, rate = sf.read(str(track_path), dtype="float32", always_2d=False)
    track = np.asarray(track, dtype="float32").reshape(-1)

    entries = plan["segments"]
    count = len(entries)
    picks = sorted({0, count - 1,
                    *{min(count - 1, round(i * (count - 1) / max(1, samples - 1)))
                      for i in range(samples)}})
    checked, worst, failures = [], 0.0, []
    for index in picks:
        entry = entries[index]
        onset, presence, violation, reason = boundary_check(track, rate, entry, entries)
        item = {"id": entry["id"], "placed_start": entry["placed_start"],
                "placed_end": entry["placed_end"],
                "onset_after_placed_start_s": onset,
                "speech_presence": round(presence, 4),
                "boundary_violation_s": round(violation, 6) if violation else 0.0}
        checked.append(item)
        if reason:
            failures.append({"id": entry["id"], "reason": reason})
            continue
        if presence < SPEECH_PRESENCE_MIN:
            failures.append({"id": entry["id"],
                             "reason": f"speech presence {presence:.2f} below floor"})
            continue
        worst = max(worst, violation)
    verdict = "PASS" if not failures and worst <= TIMELINE_TOLERANCE_S else "FAIL"
    return gate(round(worst, 6), TIMELINE_TOLERANCE_S, verdict=verdict,
                metric="max sound beyond a declared window boundary "
                       "(start or end), overlap-covered boundaries skipped",
                sampled=checked, failures=failures)


def overflow_gate(plan: dict):
    recomputed_s = round(sum(e["overflow_s"] for e in plan["segments"]), 6)
    recomputed_ratio = round(recomputed_s / plan["source_duration_s"], 6)
    reported = plan["overflow"]
    consistent = (
        abs(reported["total_s"] - recomputed_s) <= 1e-4
        and abs(reported["ratio"] - recomputed_ratio) <= 1e-4
        and reported["count"] == sum(1 for e in plan["segments"] if e["overflow_s"] > 0)
    )
    verdict = "PASS" if consistent and recomputed_ratio <= OVERFLOW_MAX_RATIO else "FAIL"
    overflow_count = sum(1 for e in plan["segments"] if e["overflow_s"] > 0)
    return gate(recomputed_ratio, OVERFLOW_MAX_RATIO, verdict=verdict,
                overflow_count=overflow_count, total_s=recomputed_s,
                plan_consistent=consistent)


def dub_gate(workdir: Path):
    dub_path = workdir / "output_dub.mp4"
    source_path = workdir / "media" / "source.mp4"
    for path in (dub_path, source_path):
        if not path.is_file():
            return gate(None, DUB_DRIFT_MAX_S, verdict="FAIL",
                        reason=f"{path.name} missing")
    dub_dur, dub_streams, err1 = ffprobe_duration_and_streams(dub_path)
    src_dur, _, err2 = ffprobe_duration_and_streams(source_path)
    if err1 or err2 or dub_dur is None or src_dur is None:
        return gate(None, DUB_DRIFT_MAX_S, verdict="FAIL",
                    reason=err1 or err2 or "ffprobe could not read duration")
    drift = round(abs(dub_dur - src_dur), 6)
    streams_ok = {"video", "audio"}.issubset(
        {s["codec_type"] for s in dub_streams})
    verdict = ("PASS" if streams_ok and drift <= DUB_DRIFT_MAX_S else "FAIL")
    return gate(drift, DUB_DRIFT_MAX_S, verdict=verdict,
                dub_duration_s=round(dub_dur, 3), source_duration_s=round(src_dur, 3),
                dub_streams=dub_streams, streams_ok=streams_ok)


def track_info(workdir: Path, plan: dict):
    """Informational cross-checks the validators also enforce; not a gate."""
    track_path = workdir / "tts" / "tts.wav"
    info = {"path": str(track_path)}
    if not track_path.is_file():
        info["error"] = "missing"
        return info
    try:
        meta = sf.info(str(track_path))
    except (OSError, RuntimeError) as exc:
        info["error"] = f"unreadable: {exc}"
        return info
    info.update({
        "sample_rate": meta.samplerate,
        "channels": meta.channels,
        "subtype": meta.subtype,
        "measured_duration_s": round(wav_duration(track_path) or 0.0, 6),
        "plan_track_duration_s": plan.get("track_duration_s"),
        "plan_sample_rate": plan.get("sample_rate"),
        "voice": plan.get("voice"),
    })
    info["matches_plan"] = (
        meta.samplerate == plan.get("sample_rate") == 24000
        and meta.channels == 1
        and abs(info["measured_duration_s"] - plan.get("track_duration_s", -1)) <= 0.01
    )
    return info


class VramSampler:
    """Poll total GPU memory used (MiB) once per second in the background."""

    def __init__(self, gpu_index: int):
        self.command = ["nvidia-smi", "-i", str(gpu_index),
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"]
        self.samples_mib = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.error = None

    def _loop(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run(self.command, capture_output=True, text=True)
                if result.returncode == 0:
                    self.samples_mib.append(float(result.stdout.strip().splitlines()[0]))
                else:
                    self.error = result.stderr.strip()[-200:]
            except OSError as exc:
                self.error = str(exc)
            self._stop.wait(1.0)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=3.0)
        return False

    def growth(self):
        """Sustained growth (MiB): p95 of the last third minus p95 of the
        first third, settling at the first-half peak so the one-off model
        load ramp is not counted as growth. Meaningful only when synthesis
        dominates the run (minutes of wall time); a run too short to judge
        returns None, which the gate treats as FAIL, never a silent PASS."""
        samples = self.samples_mib
        if len(samples) < 6:
            return None
        settle = int(np.argmax(samples[: len(samples) // 2]))
        settled = samples[settle:]
        if len(settled) < 6:
            return None
        third = max(1, len(settled) // 3)
        head = float(np.percentile(settled[:third], 95))
        tail = float(np.percentile(settled[-third:], 95))
        return round(tail - head, 1)


def timed_tts_gates(workdir: Path, config_path: Path, gpu_index: int):
    """Re-synthesize the run's translation in a scratch workdir, timed."""
    media_duration = wav_duration(workdir / "media" / "audio.wav")
    translation = workdir / "translation.json"
    if media_duration is None or not translation.is_file():
        return (
            gate(None, RTF_MAX, verdict="FAIL", reason="media/audio.wav or translation.json missing"),
            gate(None, VRAM_GROWTH_MAX_MB, verdict="FAIL", reason="no timed run"),
        )
    scratch = Path(tempfile.mkdtemp(prefix="tts-gate-"))
    (scratch / "media").mkdir()
    (scratch / "translation.json").write_bytes(translation.read_bytes())
    # The translation validator cross-checks the segments it came from.
    segments = workdir / "segments.json"
    if segments.is_file():
        (scratch / "segments.json").write_bytes(segments.read_bytes())
    (scratch / "media" / "audio.wav").write_bytes(
        (workdir / "media" / "audio.wav").read_bytes())
    # Cache off so this measures real synthesis, never a cache hit.
    from ruamel.yaml import YAML
    yaml = YAML(typ="safe")
    data = yaml.load(config_path.read_text(encoding="utf-8")) or {}
    data.setdefault("tts", {})["cache"] = False
    scratch_config = scratch / "config.yaml"
    with open(scratch_config, "w", encoding="utf-8") as handle:
        yaml.dump(data, handle)

    command = [sys.executable, "-m", "pipeline.runner",
               "--stages", "tts", "--workdir", str(scratch),
               "--config", str(scratch_config)]
    sampler = VramSampler(gpu_index)
    started = time.perf_counter()
    with sampler:
        result = subprocess.run(command, capture_output=True, text=True)
    wall = time.perf_counter() - started
    scratch_plan_path = scratch / "tts" / "dub_plan.json"
    try:
        scratch_plan = json.loads(scratch_plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        scratch_plan = None
    stderr_tail = (result.stderr or "").strip()[-800:]

    rtf = round(wall / media_duration, 6) if result.returncode == 0 else None
    rtf_report = gate(rtf, RTF_MAX, verdict="PASS" if rtf is not None and rtf <= RTF_MAX else "FAIL",
                      metric="wall_s / media_s, cold (includes model load), ADR-0004 convention",
                      wall_s=round(wall, 2), media_s=round(media_duration, 3),
                      exit_code=result.returncode,
                      scratch_synthesis_ok=result.returncode == 0 and scratch_plan is not None,
                      stderr_tail=stderr_tail)

    growth = sampler.growth() if result.returncode == 0 else None
    vram_report = gate(growth, VRAM_GROWTH_MAX_MB,
                       verdict="PASS" if growth is not None and growth <= VRAM_GROWTH_MAX_MB else "FAIL",
                       metric="p95(last third) - p95(first third) after the first-half load peak, MiB",
                       reason=None if growth is not None else
                       "fewer than 6 settled samples: run too short to judge a growth trend",
                       baseline_mib=sampler.samples_mib[0] if sampler.samples_mib else None,
                       peak_mib=max(sampler.samples_mib) if sampler.samples_mib else None,
                       sample_count=len(sampler.samples_mib),
                       sampler_error=sampler.error,
                       note="samples are total GPU memory.used (MiB), shared with other processes")

    shutil.rmtree(scratch, ignore_errors=True)
    return rtf_report, vram_report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--timed-tts", action="store_true",
                        help="re-synthesize in a scratch workdir for the RTF and VRAM gates")
    parser.add_argument("--samples", type=int, default=9,
                        help="segments to spot-check on the timeline")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--out", help="report path (default bench/results/tts-gate/<name>.json)")
    args = parser.parse_args()

    workdir = Path(args.workdir)
    config_path = Path(args.config)
    plan_path = workdir / "tts" / "dub_plan.json"
    if not plan_path.is_file():
        raise SystemExit(f"no dub plan at {plan_path}; run the tts stage first")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    report = {
        "schema_version": 1,
        "tool": "tts_gate",
        "workdir": str(workdir),
        "config": str(config_path),
        "gates": {
            "overflow_ratio": overflow_gate(plan),
            "timeline_spot_check_s": timeline_gate(workdir, plan, args.samples),
            "dub_duration_drift_s": dub_gate(workdir),
        },
        "track_info": track_info(workdir, plan),
    }
    if args.timed_tts:
        rtf, vram = timed_tts_gates(workdir, config_path, args.gpu_index)
        report["gates"]["tts_rtf"] = rtf
        report["gates"]["vram_growth_mb"] = vram
    else:
        report["gates"]["tts_rtf"] = {"value": None, "threshold": RTF_MAX,
                                      "verdict": "NOT_RUN", "note": "requires --timed-tts"}
        report["gates"]["vram_growth_mb"] = {"value": None, "threshold": VRAM_GROWTH_MAX_MB,
                                             "verdict": "NOT_RUN", "note": "requires --timed-tts"}

    verdicts = [g["verdict"] for g in report["gates"].values()]
    report["overall"] = ("PASS" if verdicts and all(v == "PASS" for v in verdicts)
                         else "FAIL")

    out = Path(args.out) if args.out else (
        Path("bench/results/tts-gate") / f"{workdir.name}-gate.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report: {out}", file=sys.stderr)
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
