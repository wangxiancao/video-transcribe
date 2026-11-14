"""LLM response cache tests: single-file JSONL store (V4-8).

One responses.jsonl per workdir replaces one-file-per-response; legacy
per-key files stay readable so existing workdirs keep their warm cache.
"""

import json
import threading
from pathlib import Path

from pipeline.llm.cache import ResponseCache, cache_key


def test_cache_stores_entries_in_one_file(tmp_path):
    cache = ResponseCache(tmp_path / "cache")

    cache.set("k1", "翻译一")
    cache.set("k2", "translation two")

    assert cache.get("k1") == "翻译一"
    assert cache.get("k2") == "translation two"
    files = sorted(p.name for p in (tmp_path / "cache").iterdir())
    assert files == ["responses.jsonl"]
    reloaded = ResponseCache(tmp_path / "cache")
    assert reloaded.get("k1") == "翻译一"


def test_cache_reads_legacy_per_key_files(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / "legacy-key.json").write_text(
        json.dumps({"response": "旧缓存"}, ensure_ascii=False), encoding="utf-8"
    )

    cache = ResponseCache(directory)

    assert cache.get("legacy-key") == "旧缓存"
    assert cache.get("missing") is None


def test_cache_delete_evicts_store_and_legacy(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    (directory / "legacy-key.json").write_text(
        json.dumps({"response": "旧缓存"}, ensure_ascii=False), encoding="utf-8"
    )
    cache = ResponseCache(directory)
    cache.set("store-key", "新缓存")
    cache.set("both-key", "双份")
    (directory / "both-key.json").write_text(
        json.dumps({"response": "旧双份"}, ensure_ascii=False), encoding="utf-8"
    )

    cache.delete("legacy-key")
    cache.delete("store-key")
    cache.delete("both-key")

    assert cache.get("legacy-key") is None
    assert cache.get("store-key") is None
    assert cache.get("both-key") is None
    assert not (directory / "legacy-key.json").exists()
    assert not (directory / "both-key.json").exists()
    assert ResponseCache(directory).get("store-key") is None


def test_cache_tolerates_torn_tail_line(tmp_path):
    directory = tmp_path / "cache"
    directory.mkdir()
    store = directory / "responses.jsonl"
    store.write_text(
        json.dumps({"key": "k1", "response": "好"}, ensure_ascii=False) + "\n"
        + '{"key": "k2", "response": "断',
        encoding="utf-8",
    )

    cache = ResponseCache(directory)

    assert cache.get("k1") == "好"
    assert cache.get("k2") is None


def test_cache_concurrent_set_is_safe(tmp_path):
    cache = ResponseCache(tmp_path / "cache")
    keys = [f"key-{index}" for index in range(200)]

    def worker(prefix):
        for index, key in enumerate(keys):
            cache.set(key, f"{prefix}-{index}")

    threads = [
        threading.Thread(target=worker, args=(f"w{number}",))
        for number in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(cache.get(key) is not None for key in keys)
    reloaded = ResponseCache(tmp_path / "cache")
    assert all(reloaded.get(key) is not None for key in keys)


def test_cache_key_is_stable():
    assert cache_key("sys", "user") == cache_key("sys", "user")
    assert cache_key("sys", "user") != cache_key("sys", "other")
