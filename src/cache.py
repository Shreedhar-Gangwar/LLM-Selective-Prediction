"""On-disk cache for model outputs.

Every model call is expensive on a laptop GPU, and evaluation gets re-run many times, so
outputs are keyed by (example_id, signal, params) and never recomputed blindly
(CLAUDE.md guardrail 5). `params` covers everything that could change the output — model
id, prompt version, label form, temperature, k — hashed into the key. A change to any of
them yields a different key rather than a stale hit.

Storage is one JSON-lines file per signal: append-only writes, last entry wins on load.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


def params_hash(params: dict[str, Any]) -> str:
    """Short stable digest of the parameters that determine a model output."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


class JsonlCache:
    """Append-only JSONL cache for one signal.

    Values must be JSON-serialisable (we store lists of floats and strings).
    """

    def __init__(self, signal: str, params: dict[str, Any], cache_dir: Path = CACHE_DIR):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.signal = signal
        self.phash = params_hash(params)
        self.path = cache_dir / f"{signal}_{self.phash}.jsonl"
        self._entries: dict[str, Any] = {}
        self._load()
        # Record the params next to the data so a cache file is self-describing.
        meta = cache_dir / f"{signal}_{self.phash}.params.json"
        if not meta.exists():
            meta.write_text(json.dumps(params, indent=2, sort_keys=True))

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._entries[rec["key"]] = rec["value"]

    def get(self, example_id: str) -> Any | None:
        return self._entries.get(example_id)

    def set(self, example_id: str, value: Any) -> None:
        self._entries[example_id] = value
        with self.path.open("a") as f:
            f.write(json.dumps({"key": example_id, "value": value}) + "\n")

    def __contains__(self, example_id: str) -> bool:
        return example_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)
