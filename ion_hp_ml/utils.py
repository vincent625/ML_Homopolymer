from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np


DNA_BASES = frozenset("ACGTN")
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def canonical_chrom(chrom: str) -> str:
    """Return a build-independent contig token for chr/no-chr comparisons."""
    value = str(chrom).strip()
    if value.lower().startswith("chr"):
        value = value[3:]
    if value.upper() in {"M", "MT"}:
        return "MT"
    return value.upper() if value.upper() in {"X", "Y"} else value


@dataclass(frozen=True)
class ContigResolver:
    """Map chr1/1 and chrM/MT names onto names present in a file."""

    by_canonical: Mapping[str, str]

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "ContigResolver":
        mapping: dict[str, str] = {}
        for name in names:
            mapping.setdefault(canonical_chrom(name), str(name))
        return cls(mapping)

    def resolve(self, chrom: str) -> str | None:
        return self.by_canonical.get(canonical_chrom(chrom))


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def is_sequence_allele(allele: str | None) -> bool:
    return bool(allele) and set(str(allele).upper()) <= DNA_BASES


def summary_stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_sd": np.nan,
            f"{prefix}_iqr": np.nan,
        }
    q25, q75 = np.percentile(array, [25, 75])
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_median": float(np.median(array)),
        f"{prefix}_sd": float(np.std(array, ddof=0)),
        f"{prefix}_iqr": float(q75 - q25),
    }

