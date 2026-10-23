"""video-transcribe entry: the six-stage pipeline in one command.

Wraps the stage runner with product defaults so the self-contained local
stack (native ASR engine + managed HY-MT2 translation LLM + rule-based
segmentation) runs without hand-editing config — zero cloud dependency:

    python video_transcribe.py /path/to/video.mp4
    python video_transcribe.py /path/to/video.mp4 --workdir runs/demo --stages ingest,transcribe

Artifacts land in the workdir (transcript/segments/translation JSON,
subs/*.srt, output.mp4); the workdir is resumable per stage.
"""

import argparse
import sys
import time
from pathlib import Path

from pipeline.runner import run

ALL_STAGES = "ingest,transcribe,segment,translate,subtitle,burn"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="video_transcribe",
        description="Translate one video into subtitles end to end "
                    "(zero cloud dependency).",
    )
    parser.add_argument("source", help="local video path or URL")
    parser.add_argument(
        "--workdir",
        help="artifact work directory (default: runs/<video name>)",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="read-only config path"
    )
    parser.add_argument(
        "--stages", default=ALL_STAGES,
        help=f"comma-separated stage ids (default: all six: {ALL_STAGES})",
    )
    args = parser.parse_args(argv)

    workdir = args.workdir or Path("runs") / Path(args.source).stem
    started = time.perf_counter()
    code = run(
        args.stages,
        workdir,
        args.config,
        source=args.source,
    )
    if code == 0:
        print(f"\ndone in {time.perf_counter() - started:.1f}s -> {workdir}")
    return code


if __name__ == "__main__":
    sys.exit(main())
