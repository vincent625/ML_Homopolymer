from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .utils import ContigResolver, canonical_chrom


@dataclass(frozen=True)
class _ChromIntervals:
    starts: tuple[int, ...]
    ends: tuple[int, ...]


class IntervalSet:
    """Small dependency-free index for merged zero-based half-open BED intervals."""

    def __init__(self, intervals: dict[str, list[tuple[int, int]]]):
        indexed: dict[str, _ChromIntervals] = {}
        for chrom, values in intervals.items():
            merged: list[list[int]] = []
            for start, end in sorted(values):
                if end <= start:
                    continue
                if merged and start <= merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            indexed[canonical_chrom(chrom)] = _ChromIntervals(
                tuple(value[0] for value in merged),
                tuple(value[1] for value in merged),
            )
        self._indexed = indexed

    @classmethod
    def from_bed(cls, path: str | Path) -> "IntervalSet":
        intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        with Path(path).open() as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip() or line.startswith(('#', 'track', 'browser')):
                    continue
                fields = line.rstrip().split('\t')
                if len(fields) < 3:
                    raise ValueError(f"Invalid BED line {line_number} in {path}")
                intervals[canonical_chrom(fields[0])].append((int(fields[1]), int(fields[2])))
        return cls(intervals)

    def contains(self, chrom: str, start: int, end: int) -> bool:
        if end <= start:
            end = start + 1
        item = self._indexed.get(canonical_chrom(chrom))
        if item is None:
            return False
        index = bisect_right(item.starts, start) - 1
        return index >= 0 and item.ends[index] >= end

    def overlaps(self, chrom: str, start: int, end: int) -> bool:
        if end <= start:
            end = start + 1
        item = self._indexed.get(canonical_chrom(chrom))
        if item is None:
            return False
        index = bisect_right(item.starts, end - 1) - 1
        return index >= 0 and item.ends[index] > start

    def iter_intervals(self):
        """Yield merged intervals as canonical-contig, start, end tuples."""
        for chrom in sorted(
            self._indexed,
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
        ):
            item = self._indexed[chrom]
            yield from ((chrom, start, end) for start, end in zip(item.starts, item.ends))

    def write_bed(self, path: str | Path, contigs: Iterable[str]) -> None:
        """Write intervals using contig names present in an external file."""
        resolver = ContigResolver.from_names(contigs)
        with Path(path).open("w") as handle:
            for chrom, start, end in self.iter_intervals():
                resolved = resolver.resolve(chrom)
                if resolved is not None:
                    handle.write(f"{resolved}\t{start}\t{end}\n")
