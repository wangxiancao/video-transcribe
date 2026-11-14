"""T6 translate stage tests: id-free ordered-array protocol (V4-7 O4).

The stage sends plain string arrays and maps results back to segment ids
by index; mocks below speak that protocol.
"""

import json
import shutil
import threading
from pathlib import Path

import pytest

from pipeline.runner import run

REPO_ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = REPO_ROOT / "tests/fixtures/segments/sample_6s.json"
EXAM = REPO_ROOT / "tests/fixtures/llm/translate_exam.json"
MISSING = REPO_ROOT / "tests/fixtures/llm/translate_exam_missing.json"


def _config(tmp_path, extra=""):
    path = tmp_path / "config.yaml"
    path.write_text(extra or "{}\n", encoding="utf-8")
    return path


def _workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    shutil.copyfile(SEGMENTS, workdir / "segments.json")
    return workdir


class ArrayClient:
    """Ordered-array protocol mock with per-call recording."""

    def __init__(self, transform=None):
        self.calls = 0
        self.requests = []
        self.lock = threading.Lock()
        self.transform = transform  # callable(texts, suffix) -> texts

    def complete(self, system, user):
        with self.lock:
            self.calls += 1
        request = json.loads(user)
        self.requests.append(request)
        texts = request.get("segments") or request.get("drafts")
        suffix = "faithful" if "faithfully" in system.lower() else "refined"
        if self.transform is not None:
            texts = self.transform(texts, suffix)
        return json.dumps({"translations": texts})


def test_translate_two_steps_and_preserves_timing(tmp_path):
    workdir = _workdir(tmp_path)

    assert run(
        "translate",
        workdir,
        _config(tmp_path),
        llm_fixture=EXAM,
    ) == 0

    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["source_language"] == "en"
    assert data["target_language"] == "简体中文"
    assert len(data["segments"]) == 2
    assert [segment["id"] for segment in data["segments"]] == [1, 2]
    assert data["segments"][0]["source"].startswith("This is a deliberately long")
    assert data["segments"][0]["target"] == "这是经过意译和反思后的自然中文版本。"
    assert data["segments"][1]["target"] == "短句。"

    segments = json.loads((workdir / "segments.json").read_text(encoding="utf-8"))
    for translated, source in zip(data["segments"], segments["segments"]):
        assert translated["start"] == source["start"]
        assert translated["end"] == source["end"]


def test_translate_heals_count_mismatch_via_bisection(tmp_path):
    """A batch returning the wrong count retries, then bisects to singles.

    The fixture drops the trailing entry for both full-batch attempts, then
    serves correct single-segment responses for the bisected halves.
    """
    workdir = _workdir(tmp_path)

    assert run(
        "translate",
        workdir,
        _config(tmp_path),
        llm_fixture=MISSING,
    ) == 0
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert [segment["target"] for segment in data["segments"]] == \
        ["第一句润色。", "第二句润色。"]


def test_translate_cache_avoids_second_llm_call(tmp_path, monkeypatch):
    import pipeline.stages.translate as translate_stage

    client = ArrayClient(transform=lambda texts, suffix:
                         [f"{suffix}-{index + 1}" for index in range(len(texts))])
    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: client,
    )
    workdir = _workdir(tmp_path)
    config = _config(tmp_path)

    assert run("translate", workdir, config) == 0
    assert client.calls == 2
    first = (workdir / "translation.json").read_text(encoding="utf-8")

    assert run("translate", workdir, config) == 0
    assert client.calls == 2
    assert (workdir / "translation.json").read_text(encoding="utf-8") == first


def test_translate_batches_segments(tmp_path, monkeypatch):
    import pipeline.stages.translate as translate_stage

    client = ArrayClient(transform=lambda texts, suffix:
                         [f"{suffix}-{index + 1}" for index in range(len(texts))])
    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: client,
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    segments = []
    for index in range(1, 10):
        start = float(index - 1)
        segments.append(
            {
                "id": index,
                "start": start,
                "end": start + 1.0,
                "text": f"Sentence {index}.",
                "words": [{"start": start, "end": start + 1.0, "text": f"Sentence {index}."}],
            }
        )
    (workdir / "segments.json").write_text(
        json.dumps({"schema_version": 1, "segments": segments}), encoding="utf-8"
    )
    config = _config(tmp_path, "translation:\n  batch_size: 2\nmax_workers: 2\n")

    assert run("translate", workdir, config) == 0
    assert client.calls == 10
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert len(data["segments"]) == 9
    # the mock numbers within each batch; the last batch is a single segment
    assert data["segments"][-1]["target"] == "refined-1"


def test_translate_term_map_normalizes_request_only(tmp_path, monkeypatch):
    import pipeline.stages.translate as translate_stage

    client = ArrayClient()
    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: client,
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    segments = [{
        "id": 1, "start": 0.0, "end": 2.0, "text": "平均时间是1.96匹秒",
        "words": [{"start": 0.0, "end": 2.0, "text": "平均时间是1.96匹秒"}],
    }]
    (workdir / "segments.json").write_text(
        json.dumps({"schema_version": 1, "segments": segments}), encoding="utf-8"
    )
    config = _config(
        tmp_path,
        "translation:\n  term_map:\n    匹秒: 皮秒\n  cache: false\n",
    )

    assert run("translate", workdir, config) == 0
    assert client.requests == [] or True  # ArrayClient records; assert below
    faithful = [call for call in client.requests if "segments" in call]
    assert faithful and faithful[0]["segments"] == ["平均时间是1.96皮秒"]
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["source"] == "平均时间是1.96匹秒"


class FlakyOnceClient(ArrayClient):
    """Drops the trailing entry on the first faithful call, then behaves."""

    def __init__(self):
        super().__init__()
        self.poisoned = True

    def complete(self, system, user):
        with self.lock:
            self.calls += 1
        request = json.loads(user)
        texts = request.get("segments") or request.get("drafts")
        suffix = "faithful" if "faithfully" in system.lower() else "refined"
        result = [f"{suffix}-{index + 1}" for index in range(len(texts))]
        if self.poisoned and suffix == "faithful":
            self.poisoned = False
            result = result[:-1]  # count drift -> contract failure
        return json.dumps({"translations": result})


def test_translate_retries_once_on_count_drift(tmp_path, monkeypatch):
    import pipeline.stages.translate as translate_stage

    client = FlakyOnceClient()
    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: client,
    )
    workdir = _workdir(tmp_path)
    config = _config(tmp_path, "translation:\n  cache: true\n")

    assert run("translate", workdir, config) == 0
    # faithful failed once (retried) + faithful ok + refine = 3 calls
    assert client.calls == 3
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["target"].startswith("refined-")


class UntranslatableClient:
    """Returns an empty string for one specific segment, both attempts."""

    def __init__(self, poisoned_index):
        self.poisoned_index = poisoned_index
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        request = json.loads(user)
        texts = request.get("segments") or request.get("drafts")
        suffix = "faithful" if "faithfully" in system.lower() else "refined"
        return json.dumps({"translations": [
            "" if (index == self.poisoned_index and suffix == "faithful")
            else f"{suffix}-{index + 1}"
            for index in range(len(texts))
        ]})


def test_translate_untranslatable_segment_fails_without_source_echo(tmp_path, monkeypatch, capsys):
    import pipeline.stages.translate as translate_stage

    client = UntranslatableClient(poisoned_index=0)
    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: client,
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    segments = [{
        "id": 1, "start": 0.0, "end": 1.0, "text": "电麦搜",
        "words": [{"start": 0.0, "end": 1.0, "text": "电麦搜"}],
    }]
    (workdir / "segments.json").write_text(
        json.dumps({"schema_version": 1, "segments": segments}), encoding="utf-8"
    )
    config = _config(tmp_path, "translation:\n  cache: false\n")

    assert run("translate", workdir, config) == 1
    assert not (workdir / "translation.json").exists()
    assert "no valid translation" in capsys.readouterr().err


def _single_segment_workdir(tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    segments = [{
        "id": 1, "start": 0.0, "end": 1.0, "text": "电麦搜",
        "words": [{"start": 0.0, "end": 1.0, "text": "电麦搜"}],
    }]
    (workdir / "segments.json").write_text(
        json.dumps({"schema_version": 1, "segments": segments}), encoding="utf-8"
    )
    return workdir


def test_translate_rejected_refinement_uses_faithful_draft(tmp_path, monkeypatch, capsys):
    import pipeline.stages.translate as translate_stage

    def transform(texts, suffix):
        if suffix == "refined":
            return ["将原文润色为自然流畅的简体中文表达。"]
        return ["合规忠实译文。"]

    client = ArrayClient(transform=transform)
    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: client,
    )
    workdir = _single_segment_workdir(tmp_path)

    assert run(
        "translate", workdir, _config(tmp_path, "translation:\n  cache: false\n")
    ) == 0
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["target"] == "合规忠实译文。"
    assert "using faithful drafts" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bad_response", ["将副标题翻译得符合目标语言表达习惯。", "长" * 89]
)
def test_translate_meta_or_overlong_faithful_response_fails(
    tmp_path, monkeypatch, bad_response
):
    import pipeline.stages.translate as translate_stage

    client = ArrayClient(transform=lambda texts, suffix: [bad_response])
    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: client,
    )
    workdir = _single_segment_workdir(tmp_path)

    assert run(
        "translate", workdir, _config(tmp_path, "translation:\n  cache: false\n")
    ) == 1
    assert not (workdir / "translation.json").exists()


def test_translate_tolerates_trailing_extra_json(tmp_path, monkeypatch):
    import pipeline.stages.translate as translate_stage

    class TrailingJunkClient:
        def complete(self, system, user):
            request = json.loads(user)
            texts = request.get("segments") or request.get("drafts")
            body = json.dumps({"translations": [f"t-{i + 1}" for i in range(len(texts))]})
            return body + '\n{"translations": ["junk"]}'

    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: TrailingJunkClient(),
    )
    workdir = _workdir(tmp_path)
    assert run("translate", workdir,
               _config(tmp_path, "translation:\n  cache: false\n")) == 0
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert data["segments"][0]["target"] == "t-1"


def test_translate_repairs_q4_degenerate_json_overflow(tmp_path, monkeypatch):
    """Replays the two real Q4 failures captured from the 66-min run.

    The runtime drops the final closing quote and imitates the two-string
    system-prompt example for single-segment batches; the stage repairs
    the envelope and keeps the first translation instead of failing.
    """
    import pipeline.stages.translate as translate_stage

    class DegenerateQ4Client:
        def complete(self, system, user):
            if "faithfully" in system.lower():
                # Real cached response for segment "呃coder" (batch=1).
                return '{"translations":["呃编码员。","呃编码员。]}'
            # Real cached paraphrase-pair shape, healed the same way.
            return '{"translations":["突然无法连接了。","突然连不上了。]}'

    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: DegenerateQ4Client(),
    )
    workdir = _workdir(tmp_path)
    assert run("translate", workdir,
               _config(tmp_path, "translation:\n  cache: false\n"
                                  "  batch_size: 1\n")) == 0
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    # The refined response is repaired the same way and wins over the
    # faithful draft, so the first refined string is the final target.
    assert data["segments"][0]["target"] == "突然无法连接了。"


def test_parse_json_repairs_real_q4_envelope_degenerations():
    from pipeline.stages.translate import _parse_json, StageError

    # Missing closing quote before "]" (real cached response).
    assert _parse_json('{"translations":["呃编码员。","呃编码员。]}') == {
        "translations": ["呃编码员。", "呃编码员。"]
    }
    # Curly-quote terminator (real cached response).
    assert _parse_json('{"translations":["对于ESNP来说，这是最大的。”]}') == {
        "translations": ["对于ESNP来说，这是最大的。"]
    }
    # Missing comma between strings (real cached response).
    assert _parse_json('{"translations":["全连接连接。" "全连接模式。"]}') == {
        "translations": ["全连接连接。", "全连接模式。"]
    }
    # Valid JSON and irreparable junk keep their current behavior.
    assert _parse_json('{"translations":["A"]}') == {"translations": ["A"]}
    with pytest.raises(StageError):
        _parse_json('{"translations":["A" "B。]}')


def test_translate_collapses_stutter_amplification(tmp_path, monkeypatch, capsys):
    """Real 66-min run failure: the stutter "更更更加可靠" translated
    faithfully as "更加更加更加可靠。" trips the low-information gate;
    the stage collapses the run instead of failing the whole output.
    """

    import pipeline.stages.translate as translate_stage

    class StutterAmplifierClient:
        def complete(self, system, user):
            request = json.loads(user)
            texts = request.get("segments") or request.get("drafts")
            return json.dumps(
                {"translations": ["更加更加更加可靠。" for _ in texts]}
            )

    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: StutterAmplifierClient(),
    )
    workdir = _workdir(tmp_path)
    assert run("translate", workdir,
               _config(tmp_path, "translation:\n  cache: false\n")) == 0
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert all(
        segment["target"] == "更加可靠。" for segment in data["segments"]
    )
    assert "target collapsed" in capsys.readouterr().err


def test_translate_rejects_refine_example_echo(tmp_path, monkeypatch, capsys):
    """Real 66-min run leak: the refine pass echoed the prompt example
    ("Refined first sentence.", "优化后的第一句翻译。") as the translation
    for 37 segments. The gate rejects the echo and falls back to the
    faithful draft.
    """
    import pipeline.stages.translate as translate_stage

    class LeakyRefineClient:
        def complete(self, system, user):
            request = json.loads(user)
            texts = request.get("segments") or request.get("drafts")
            if "faithfully" in system.lower():
                return json.dumps(
                    {"translations": [f"译-{i + 1}" for i in range(len(texts))]}
                )
            return json.dumps(
                {"translations": ["Refined first sentence." for _ in texts]}
            )

    monkeypatch.setattr(
        translate_stage, "create_llm_client",
        lambda _config, _fixture, _service=None: LeakyRefineClient(),
    )
    workdir = _workdir(tmp_path)
    assert run("translate", workdir,
               _config(tmp_path, "translation:\n  cache: false\n")) == 0
    data = json.loads((workdir / "translation.json").read_text(encoding="utf-8"))
    assert [segment["target"] for segment in data["segments"]] == ["译-1", "译-2"]
    assert "echoes the prompt example" in capsys.readouterr().err
