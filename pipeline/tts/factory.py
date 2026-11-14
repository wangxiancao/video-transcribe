"""TTS engine factory.

V1 ships exactly one local engine (ADR-0004); the seam exists so tests can
monkeypatch engine construction and so a future engine selection does not
touch the stage. The import stays inside the function: importing this
module must not load torch or f5_tts.
"""


def create_tts_engine(config):
    from pipeline.tts.engine import F5TTSEngine

    return F5TTSEngine()
