from __future__ import annotations

import math
from collections import Counter

from .utils import ContigResolver


def _minimal_difference(ref: str, alt: str) -> tuple[int, str, str]:
    prefix = 0
    limit = min(len(ref), len(alt))
    while prefix < limit and ref[prefix] == alt[prefix]:
        prefix += 1
    suffix = 0
    ref_left = len(ref) - prefix
    alt_left = len(alt) - prefix
    while suffix < min(ref_left, alt_left) and ref[-1 - suffix] == alt[-1 - suffix]:
        suffix += 1
    ref_end = len(ref) - suffix if suffix else len(ref)
    alt_end = len(alt) - suffix if suffix else len(alt)
    return prefix, ref[prefix:ref_end], alt[prefix:alt_end]


def _run_length_around(sequence: str, start: int, length: int, base: str) -> tuple[int, int, int]:
    left = start
    while left > 0 and sequence[left - 1] == base:
        left -= 1
    right = start + length
    while right < len(sequence) and sequence[right] == base:
        right += 1
    segment = sequence[start:start + length]
    if length and any(value != base for value in segment):
        return 0, start, start
    return right - left, left, right


def _entropy(sequence: str) -> float:
    bases = [base for base in sequence.upper() if base in "ACGT"]
    if not bases:
        return math.nan
    counts = Counter(bases)
    total = len(bases)
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def sequence_context_features(
    fasta,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    flank: int = 25,
) -> dict[str, object]:
    """Calculate reference-derived context for one normalized sequence indel."""
    resolver = ContigResolver.from_names(fasta.references)
    fasta_chrom = resolver.resolve(chrom)
    if fasta_chrom is None:
        raise ValueError(f"Contig {chrom!r} is not present in the FASTA")

    ref = ref.upper()
    alt = alt.upper()
    start0 = pos - 1
    end0 = start0 + len(ref)
    local_start = max(0, start0 - flank)
    local_end = end0 + flank
    local_context = fasta.fetch(fasta_chrom, local_start, local_end).upper()
    local_offset = start0 - local_start
    reference_match = local_context[local_offset:local_offset + len(ref)] == ref

    prefix_length, ref_difference, alt_difference = _minimal_difference(ref, alt)
    if len(alt) > len(ref):
        indel_type = "insertion"
        changed = alt_difference
    elif len(ref) > len(alt):
        indel_type = "deletion"
        changed = ref_difference
    else:
        indel_type = "complex"
        changed = ""
    indel_length = abs(len(alt) - len(ref))
    hp_base = changed[0] if changed and len(set(changed)) == 1 else None

    # A short context can end inside the homopolymer and silently truncate its
    # length. Use a wider sequence only for run detection; GC/entropy retain the
    # caller-requested local window.
    run_flank = max(flank, 100)
    fetch_start = max(0, start0 - run_flank)
    fetch_end = end0 + run_flank
    reference_context = fasta.fetch(fasta_chrom, fetch_start, fetch_end).upper()
    offset = start0 - fetch_start
    reference_match = reference_match and reference_context[offset:offset + len(ref)] == ref
    alternate_context = (
        reference_context[:offset] + alt + reference_context[offset + len(ref):]
    )
    edit_start = offset + prefix_length
    result: dict[str, object] = {
        "reference_match": reference_match,
        "hp_base": hp_base,
        "hp_ref_len": math.nan,
        "hp_alt_len": math.nan,
        "hp_delta": math.nan,
        "is_homopolymer_indel": False,
        "indel_type": indel_type,
        "indel_length": indel_length,
        "local_gc": math.nan,
        "local_entropy": _entropy(local_context),
        "_hp_ref_start0": start0,
        "_hp_ref_end0": end0,
    }
    acgt = [base for base in local_context if base in "ACGT"]
    if acgt:
        result["local_gc"] = float(sum(base in "GC" for base in acgt) / len(acgt))

    if not reference_match or hp_base is None:
        return result

    ref_length, ref_run_start, ref_run_end = _run_length_around(
        reference_context, edit_start, len(ref_difference), hp_base
    )
    alt_length, _, _ = _run_length_around(
        alternate_context, edit_start, len(alt_difference), hp_base
    )
    result.update(
        {
            "hp_ref_len": ref_length,
            "hp_alt_len": alt_length,
            "hp_delta": alt_length - ref_length,
            "is_homopolymer_indel": bool(ref_length > 0 and alt_length > 0),
            "_hp_ref_start0": fetch_start + ref_run_start,
            "_hp_ref_end0": fetch_start + ref_run_end,
        }
    )
    return result
