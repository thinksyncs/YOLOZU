from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Any, Iterable


@dataclass
class ReplayBuffer:
    """A tiny replay buffer with reservoir sampling.

    The buffer stores arbitrary JSON-serializable records (typically dataset records).

    - capacity=0 disables storage (memoryless mode).
    - Reservoir sampling keeps a uniform sample over all items seen so far.
    """

    capacity: int
    seed: int = 0
    items: list[dict[str, Any]] = field(default_factory=list)
    seen: int = 0

    def __post_init__(self) -> None:
        if int(self.capacity) < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = int(self.capacity)
        self._rng = random.Random(int(self.seed))

    def __len__(self) -> int:
        return int(len(self.items))

    def add(self, record: dict[str, Any]) -> None:
        if self.capacity <= 0:
            self.seen += 1
            return

        i = int(self.seen)
        self.seen += 1

        if len(self.items) < self.capacity:
            self.items.append(record)
            return

        # Reservoir: replace existing item with decreasing probability.
        j = self._rng.randint(0, i)
        if j < self.capacity:
            self.items[j] = record

    def add_many(self, records: Iterable[dict[str, Any]]) -> None:
        for rec in records:
            if isinstance(rec, dict):
                self.add(rec)

    def sample(self, k: int | None = None) -> list[dict[str, Any]]:
        if not self.items:
            return []
        if k is None:
            return list(self.items)
        k_i = int(k)
        if k_i <= 0:
            return []
        if k_i >= len(self.items):
            return list(self.items)
        return list(self._rng.sample(self.items, k_i))

    def summary(self) -> dict[str, Any]:
        def _safe_image_key(rec: dict[str, Any]) -> str | None:
            for key in ("image_path", "image"):
                value = rec.get(key)
                if isinstance(value, str) and value:
                    return value
            return None

        images = []
        for rec in self.items:
            p = _safe_image_key(rec)
            if p:
                images.append(p)

        return {
            "capacity": int(self.capacity),
            "seed": int(self.seed),
            "seen": int(self.seen),
            "size": int(len(self.items)),
            "images": images,
        }

