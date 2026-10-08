"""
Lab 7: The Collision Resolver -- starter.

Complete the three classes below. See
Lab_07_The_Collision_Resolver.md, Part B, for the full requirements.
"""

from typing import Generic, Hashable, List, Optional, Tuple, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")

_TOMBSTONE = object()  # sentinel marking a deleted open-addressing slot


class _ChainNode(Generic[K, V]):
    __slots__ = ("key", "value", "next")

    def __init__(self, key: K, value: V) -> None:
        self.key = key
        self.value = value
        self.next: Optional["_ChainNode[K, V]"] = None


class ChainedHashMap(Generic[K, V]):
    """Separate chaining: each bucket is a linked list of (key, value)."""

    def __init__(self, initial_size: int = 16) -> None:
        self._buckets: List[Optional[_ChainNode[K, V]]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Insert, or update in place if `key` already exists. Resize (double + rehash) once load factor > 0.75."""
        # TODO
        idx = hash(key) % len(self._buckets)
        node = self._buckets[idx]
        while node is not None:
            if node.key == key:
                node.value = value
                return
            node = node.next
        new = _ChainNode(key, value)
        new.next = self._buckets[idx]
        self._buckets[idx] = new
        self._count += 1
        if self._count / len(self._buckets) > 0.75:
            old = self._buckets
            self._buckets = [None] * (len(old) * 2)
            for first in old:  # relink every node into the doubled table
                cur = first
                while cur is not None:
                    nxt = cur.next
                    j = hash(cur.key) % len(self._buckets)
                    cur.next = self._buckets[j]
                    self._buckets[j] = cur
                    cur = nxt

    def get(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        # TODO
        node = self._buckets[hash(key) % len(self._buckets)]
        while node is not None:
            if node.key == key:
                return node.value
            node = node.next
        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove `key`. Raise KeyError if missing."""
        # TODO
        idx = hash(key) % len(self._buckets)
        prev = None
        node = self._buckets[idx]
        while node is not None:
            if node.key == key:
                if prev is None:
                    self._buckets[idx] = node.next
                else:
                    prev.next = node.next
                self._count -= 1
                return
            prev = node
            node = node.next
        raise KeyError(key)


class LinearProbingHashMap(Generic[K, V]):
    """Open addressing with linear probing and tombstone deletion."""

    def __init__(self, initial_size: int = 16) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Resize (double + rehash) once load factor > 0.7."""
        # TODO
        size = len(self._keys)
        if (self._count + 1) / size > 0.7:  # proactive: check BEFORE probing
            old_keys, old_values = self._keys, self._values
            new_size = size * 2
            self._keys = [None] * new_size
            self._values = [None] * new_size
            self._count = 0
            for k, v in zip(old_keys, old_values):
                if k is not None and k is not _TOMBSTONE:
                    self.insert(k, v)  # rehash into the bigger table (tombstones dropped)
            self.insert(key, value)
            return
        start = hash(key) % size
        free = -1  # first reusable slot (empty or tombstone) on the probe path
        for i in range(size):
            idx = (start + i) % size
            slot = self._keys[idx]
            if slot is None:
                if free == -1:
                    free = idx
                break
            if slot is _TOMBSTONE:
                if free == -1:
                    free = idx
            elif slot == key:
                self._values[idx] = value  # key exists: update in place
                return
        if free == -1:
            # probe sequence found no usable slot even though the table
            # is not full: grow and retry
            old_keys, old_values = self._keys, self._values
            new_size = size * 2
            self._keys = [None] * new_size
            self._values = [None] * new_size
            self._count = 0
            for k, v in zip(old_keys, old_values):
                if k is not None and k is not _TOMBSTONE:
                    self.insert(k, v)  # rehash into the bigger table (tombstones dropped)
            self.insert(key, value)
            return
        self._keys[free] = key
        self._values[free] = value
        self._count += 1

    def search(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        # TODO
        size = len(self._keys)
        start = hash(key) % size
        for i in range(size):
            idx = (start + i) % size
            slot = self._keys[idx]
            if slot is None:
                break  # a truly empty slot ends the probe sequence
            if slot is not _TOMBSTONE and slot == key:
                return self._values[idx]  # type: ignore[return-value]
        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove `key` using a tombstone (not None) so later probes don't stop early. Raise KeyError if missing."""
        # TODO
        size = len(self._keys)
        start = hash(key) % size
        for i in range(size):
            idx = (start + i) % size
            slot = self._keys[idx]
            if slot is None:
                break
            if slot is not _TOMBSTONE and slot == key:
                self._keys[idx] = _TOMBSTONE
                self._values[idx] = None
                self._count -= 1
                return
        raise KeyError(key)


class QuadraticProbingHashMap(Generic[K, V]):
    """
    Open addressing with quadratic probing and tombstone deletion.

    Pitfall to design around: with a power-of-2 table size, the probe
    sequence (idx + i^2) mod size does NOT reach every slot -- it can
    cycle through only about half of them, so the table can appear
    "full" and raise/loop forever even though empty slots exist
    elsewhere. Two standard fixes, pick one:
      (a) use a PRIME table size (so the quadratic sequence covers all
          slots whenever load factor < 1), or
      (b) resize proactively -- check load factor BEFORE attempting an
          insert's probe sequence, not only after a successful insert.
    Using both is safest.
    """

    def __init__(self, initial_size: int = 17) -> None:
        self._keys: List[object] = [None] * initial_size
        self._values: List[Optional[V]] = [None] * initial_size
        self._count = 0

    def __len__(self) -> int:
        return self._count

    def insert(self, key: K, value: V) -> None:
        """Resize (grow + rehash) once load factor > 0.7 -- see the pitfall note above."""
        # TODO
        size = len(self._keys)
        if (self._count + 1) / size > 0.7:  # proactive: check BEFORE probing
            old_keys, old_values = self._keys, self._values
            new_size = size * 2 + 1
            while any(new_size % d == 0 for d in range(2, int(new_size ** 0.5) + 1)):
                new_size += 1  # keep the table size PRIME
            self._keys = [None] * new_size
            self._values = [None] * new_size
            self._count = 0
            for k, v in zip(old_keys, old_values):
                if k is not None and k is not _TOMBSTONE:
                    self.insert(k, v)  # rehash into the bigger table (tombstones dropped)
            self.insert(key, value)
            return
        start = hash(key) % size
        free = -1  # first reusable slot (empty or tombstone) on the probe path
        for i in range(size):
            idx = (start + i * i) % size
            slot = self._keys[idx]
            if slot is None:
                if free == -1:
                    free = idx
                break
            if slot is _TOMBSTONE:
                if free == -1:
                    free = idx
            elif slot == key:
                self._values[idx] = value  # key exists: update in place
                return
        if free == -1:
            # probe sequence found no usable slot even though the table
            # is not full: grow and retry
            old_keys, old_values = self._keys, self._values
            new_size = size * 2 + 1
            while any(new_size % d == 0 for d in range(2, int(new_size ** 0.5) + 1)):
                new_size += 1  # keep the table size PRIME
            self._keys = [None] * new_size
            self._values = [None] * new_size
            self._count = 0
            for k, v in zip(old_keys, old_values):
                if k is not None and k is not _TOMBSTONE:
                    self.insert(k, v)  # rehash into the bigger table (tombstones dropped)
            self.insert(key, value)
            return
        self._keys[free] = key
        self._values[free] = value
        self._count += 1

    def search(self, key: K) -> V:
        """Return the value for `key`. Raise KeyError if missing."""
        # TODO
        size = len(self._keys)
        start = hash(key) % size
        for i in range(size):
            idx = (start + i * i) % size
            slot = self._keys[idx]
            if slot is None:
                break  # a truly empty slot ends the probe sequence
            if slot is not _TOMBSTONE and slot == key:
                return self._values[idx]  # type: ignore[return-value]
        raise KeyError(key)

    def delete(self, key: K) -> None:
        """Remove `key` using a tombstone. Raise KeyError if missing."""
        # TODO
        size = len(self._keys)
        start = hash(key) % size
        for i in range(size):
            idx = (start + i * i) % size
            slot = self._keys[idx]
            if slot is None:
                break
            if slot is not _TOMBSTONE and slot == key:
                self._keys[idx] = _TOMBSTONE
                self._values[idx] = None
                self._count -= 1
                return
        raise KeyError(key)
