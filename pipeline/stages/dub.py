"""dub stage: mux the synthesized dub track over the source video.

Consumes media/source.mp4 and tts/tts.wav and produces output_dub.mp4 with
the video stream copied untouched and the audio track replaced wholesale by
the tts track, encoded as AAC (ADR-0003): no background audio is retained
or mixed in, and every non-speech gap of the track stays silent. Subtitle
burning is not part of dub — output_dub.mp4 and output.mp4 (burn) are
independent artifacts.

The mux writes a temp file next to the output, validates it — both streams
present, duration within MAX_DURATION_DRIFT of the source video — and only
then atomically replaces the output. The track is never trimmed to the
video: a track that overshoots the source beyond the drift gate fails the
stage instead of clipping speech.
"""

import json
import os
import shutil
import subprocess

from pipeline.contracts import StageError

# Fixed contract, not config: the same drift gate burn enforces on output.mp4.
MAX_DURATION_DRIFT = 0.5
# The dub track is 24 kHz mono (validated against the plan); 96 kbit/s keeps
# AAC bits-per-frame under the encoder cap at that rate — 192k exceeds it
# and the native encoder rejects the mux outright.
DUB_AUDIO_BITRATE = "96k"


def _require_tool(name):
    path = shutil.which(name)
    if path is None:
        raise StageError(f"required tool not found on PATH: {name}")
    return path


def _run(command):
    try:
        return subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise StageError(f"cannot run {command[0]}: {exc}") from exc


def _tail(text):
    return (text or "").strip()[-1500:]


def _probe(path):
    result = _run(
        [
            _require_tool("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name",
            "-of",
            "json",
            str(path),
        ]
    )
    if result.returncode != 0:
        raise StageError(f"ffprobe cannot read {path}: {_tail(result.stderr)}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise StageError(f"ffprobe returned invalid JSON for {path}") from exc


def _streams(probe):
    streams = probe.get("streams")
    return streams if isinstance(streams, list) else []


def _duration(probe):
    try:
        duration = float((probe.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        raise StageError("media duration is unavailable") from None
    if duration <= 0:
        raise StageError("media duration must be positive")
    return duration


def _mux(source, track, temp):
    """Copy the source video stream, replace the audio with the dub track."""
    result = _run(
        [
            _require_tool("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-i",
            str(track),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            DUB_AUDIO_BITRATE,
            "-movflags",
            "+faststart",
            str(temp),
        ]
    )
    if result.returncode != 0 or not temp.is_file() or temp.stat().st_size == 0:
        if temp.exists():
            temp.unlink()
        raise StageError(f"cannot mux dub track: {_tail(result.stderr)}")


def run(ctx) -> None:
    source = ctx.path("source_video")
    track = ctx.path("tts_audio")
    output = ctx.path("output_dub")

    source_probe = _probe(source)
    stream_types = {stream.get("codec_type") for stream in _streams(source_probe)}
    if "video" not in stream_types:
        raise StageError(f"source video has no video stream: {source}")
    source_duration = _duration(source_probe)

    temp = output.with_name(".output_dub.tmp.mp4")
    if temp.exists():
        temp.unlink()
    _mux(source, track, temp)

    try:
        output_probe = _probe(temp)
        output_types = {stream.get("codec_type") for stream in _streams(output_probe)}
        if not {"video", "audio"}.issubset(output_types):
            raise StageError("dubbed video is missing a video or audio stream")
        output_duration = _duration(output_probe)
        if abs(output_duration - source_duration) > MAX_DURATION_DRIFT:
            raise StageError(
                f"dubbed duration {output_duration:.3f}s differs from source "
                f"{source_duration:.3f}s"
            )
        os.replace(temp, output)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
