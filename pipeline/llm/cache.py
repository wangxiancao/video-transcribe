import hashlib
import json
import os
import threading
from pathlib import Path

STORE_NAME = "responses.jsonl"


def cache_key(system, user):
    payload = (system + "\n" + user).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ResponseCache:
    """Single-file response store: one JSON line per cached response.

    The translate stage calls in from worker threads, so all access is
    serialized. Legacy per-key files (one JSON per response, the earlier
    layout) are still read on a miss so existing workdirs keep their warm
    cache; delete evicts both layouts.
    """

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._store_path = self.directory / STORE_NAME
        self._lock = threading.Lock()
        self._entries = {}
        self._load()

    def _load(self):
        try:
            lines = self._store_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn tail line from an interrupted append
            key = data.get("key")
            response = data.get("response")
            if isinstance(key, str) and isinstance(response, str):
                self._entries[key] = response

    def _legacy_path(self, key):
        return self.directory / f"{key}.json"

    def get(self, key):
        with self._lock:
            response = self._entries.get(key)
        if response is not None:
            return response
        try:
            data = json.loads(self._legacy_path(key).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        response = data.get("response")
        return response if isinstance(response, str) else None

    def set(self, key, response):
        line = json.dumps({"key": key, "response": response}, ensure_ascii=False)
        with self._lock:
            with self._store_path.open("a", encoding="utf-8") as store:
                store.write(line + "\n")
            self._entries[key] = response

    def delete(self, key):
        """Evict a cached response (e.g. one that failed contract parsing)."""
        with self._lock:
            if self._entries.pop(key, None) is not None:
                temp = self._store_path.with_name(f".{STORE_NAME}.tmp")
                temp.write_text(
                    "".join(
                        json.dumps(
                            {"key": cached_key, "response": cached_response},
                            ensure_ascii=False,
                        )
                        + "\n"
                        for cached_key, cached_response in self._entries.items()
                    ),
                    encoding="utf-8",
                )
                os.replace(temp, self._store_path)
        try:
            self._legacy_path(key).unlink()
        except FileNotFoundError:
            pass
