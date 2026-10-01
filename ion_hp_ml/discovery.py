from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bam_features import _usable_read
from .context import sequence_context_features
from .intervals import IntervalSet
from .utils import ContigResolver, is_sequence_allele
from .variants import VariantKey, normalize_variant_python


@dataclass
class _Evidence:
    support: int = 0
    forward: int = 0
    reverse: int = 0
    depth: int = 0
    mapq: list[float] = field(default_factory=list)

    def add(self, read) -> None:
        self.support += 1
        self.reverse += int(read.is_reverse)
        self.forward += int(not read.is_reverse)
        self.mapq.append(float(read.mapping_quality))


def read_indel_events(
    read,
    fasta,
    fasta_chrom: str,
    output_chrom: str,
    max_indel_length: int | None = 50,
    normalize: bool = True,
    homopolymer_only: bool = False,
) -> list[VariantKey]:
    """Return normalized sequence-indel alleles encoded by one read CIGAR.

    ``query_sequence`` follows the SAM alignment orientation, including for
    reverse-aligned reads. It is therefore traversed directly with the CIGAR;
    reverse-complementing here would corrupt insertion alleles.
    """
    query = (read.query_sequence or "").upper()
    query_cursor = 0
    reference_cursor = int(read.reference_start)
    events: list[VariantKey] = []
    for operation, length in read.cigartuples or []:
        if operation in {0, 7, 8}:  # M, =, X
            query_cursor += length
            reference_cursor += length
        elif operation == 1:  # insertion after the preceding reference base
            inserted = query[query_cursor:query_cursor + length]
            if (
                reference_cursor > 0
                and (max_indel_length is None or length <= max_indel_length)
                and len(inserted) == length
                and is_sequence_allele(inserted)
                and (not homopolymer_only or len(set(inserted)) == 1)
            ):
                anchor = fasta.fetch(
                    fasta_chrom, reference_cursor - 1, reference_cursor
                ).upper()
                if len(anchor) == 1 and is_sequence_allele(anchor):
                    raw = VariantKey.create(
                        output_chrom, reference_cursor, anchor, anchor + inserted
                    )
                    events.append(
                        normalize_variant_python(
                            fasta,
                            output_chrom,
                            raw.pos,
                            raw.ref,
                            raw.alt,
                            fasta_chrom=fasta_chrom,
                        )
                        if normalize
                        else raw
                    )
            query_cursor += length
        elif operation == 2:  # deletion, anchored on the preceding base
            if (
                reference_cursor > 0
                and (max_indel_length is None or length <= max_indel_length)
            ):
                deleted = fasta.fetch(
                    fasta_chrom, reference_cursor - 1, reference_cursor + length
                ).upper()
                if (
                    len(deleted) == length + 1
                    and is_sequence_allele(deleted)
                    and (not homopolymer_only or len(set(deleted[1:])) == 1)
                ):
                    raw = VariantKey.create(
                        output_chrom, reference_cursor, deleted, deleted[0]
                    )
                    events.append(
                        normalize_variant_python(
                            fasta,
                            output_chrom,
                            raw.pos,
                            raw.ref,
                            raw.alt,
                            fasta_chrom=fasta_chrom,
                        )
                        if normalize
                        else raw
                    )
            reference_cursor += length
        elif operation == 3:  # reference skip, not a sequence deletion call
            reference_cursor += length
        elif operation == 4:  # soft clip
            query_cursor += length
        elif operation in {5, 6}:  # hard clip, padding
            continue
    return events


def discover_indels(
    bam,
    fasta,
    targets: IntervalSet,
    min_mapq: int = 20,
    min_support: int = 2,
    max_indel_length: int | None = 50,
    min_hp_length: int | None = None,
    min_alt_fraction: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Blindly discover normalized indel alleles from BAM CIGAR events."""
    bam_resolver = ContigResolver.from_names(bam.references)
    fasta_resolver = ContigResolver.from_names(fasta.references)
    raw_evidence: dict[VariantKey, _Evidence] = {}
    metrics = {
        "target_intervals": 0,
        "unresolved_target_intervals": 0,
        "reads_fetched": 0,
        "reads_usable": 0,
        "reads_mapq_filtered": 0,
        "raw_indel_observations": 0,
        "alleles_before_support_filter": 0,
        "alleles_after_support_before_hp_filter": 0,
        "alleles_fraction_filtered": 0,
        "alleles_non_hp_filtered": 0,
        "alleles_short_hp_filtered": 0,
        "alleles_after_support_filter": 0,
    }
    for chrom, interval_start, interval_end in targets.iter_intervals():
        metrics["target_intervals"] += 1
        bam_chrom = bam_resolver.resolve(chrom)
        fasta_chrom = fasta_resolver.resolve(chrom)
        if bam_chrom is None or fasta_chrom is None:
            metrics["unresolved_target_intervals"] += 1
            continue
        coverage_difference = (
            np.zeros(interval_end - interval_start + 1, dtype=np.int32)
            if min_alt_fraction is not None
            else None
        )
        interval_keys: set[VariantKey] = set()
        for read in bam.fetch(bam_chrom, interval_start, interval_end):
            metrics["reads_fetched"] += 1
            if not _usable_read(read):
                continue
            metrics["reads_usable"] += 1
            if read.mapping_quality < min_mapq:
                metrics["reads_mapq_filtered"] += 1
                continue
            if coverage_difference is not None:
                covered_start = max(interval_start, int(read.reference_start))
                covered_end = min(interval_end, int(read.reference_end or covered_start))
                if covered_end > covered_start:
                    coverage_difference[covered_start - interval_start] += 1
                    coverage_difference[covered_end - interval_start] -= 1
            # Count each allele at most once per read. Restrict ownership to the
            # current merged interval so a long read fetched by two disjoint
            # targets cannot count the same event twice.
            seen: set[VariantKey] = set()
            for key in read_indel_events(
                read,
                fasta,
                fasta_chrom,
                chrom,
                max_indel_length=max_indel_length,
                normalize=False,
                homopolymer_only=min_hp_length is not None,
            ):
                start0 = key.pos - 1
                end0 = start0 + max(1, len(key.ref))
                if not (start0 < interval_end and end0 > interval_start):
                    continue
                if key in seen:
                    continue
                seen.add(key)
                interval_keys.add(key)
                metrics["raw_indel_observations"] += 1
                raw_evidence.setdefault(key, _Evidence()).add(read)

        if coverage_difference is not None and interval_keys:
            coverage = np.cumsum(coverage_difference[:-1])
            for key in interval_keys:
                offset = key.pos - 1 - interval_start
                if 0 <= offset < len(coverage):
                    raw_evidence[key].depth = max(
                        raw_evidence[key].depth, int(coverage[offset])
                    )

    # Normalization is intentionally delayed until after aggregation: deep Ion
    # BAMs repeat the same CIGAR allele thousands of times, and left-aligning
    # every observation is needlessly expensive. Merge shifted raw forms before
    # applying the support threshold.
    evidence: dict[VariantKey, _Evidence] = {}
    for raw_key, item in raw_evidence.items():
        fasta_chrom = fasta_resolver.resolve(raw_key.chrom)
        if fasta_chrom is None:
            continue
        key = normalize_variant_python(
            fasta,
            raw_key.chrom,
            raw_key.pos,
            raw_key.ref,
            raw_key.alt,
            fasta_chrom=fasta_chrom,
        )
        merged = evidence.setdefault(key, _Evidence())
        merged.support += item.support
        merged.forward += item.forward
        merged.reverse += item.reverse
        merged.depth = max(merged.depth, item.depth)
        merged.mapq.extend(item.mapq)

    metrics["alleles_before_support_filter"] = len(evidence)
    rows: list[dict[str, object]] = []
    for key, item in sorted(evidence.items()):
        if item.support < min_support:
            continue
        metrics["alleles_after_support_before_hp_filter"] += 1
        alt_fraction = item.support / item.depth if item.depth else math.nan
        if min_alt_fraction is not None and (
            not math.isfinite(alt_fraction) or alt_fraction < min_alt_fraction
        ):
            metrics["alleles_fraction_filtered"] += 1
            continue
        context: dict[str, object] = {}
        if min_hp_length is not None:
            context = sequence_context_features(
                fasta, key.chrom, key.pos, key.ref, key.alt
            )
            if not context["is_homopolymer_indel"]:
                metrics["alleles_non_hp_filtered"] += 1
                continue
            if max(float(context["hp_ref_len"]), float(context["hp_alt_len"])) < min_hp_length:
                metrics["alleles_short_hp_filtered"] += 1
                continue
        rows.append(
            {
                "chrom": key.chrom,
                "pos": key.pos,
                "ref": key.ref,
                "alt": key.alt,
                "locus_id": key.locus_id,
                "candidate_source": "BLIND_BAM",
                "bam_alt_support": item.support,
                "bam_alt_forward_reads": item.forward,
                "bam_alt_reverse_reads": item.reverse,
                "bam_discovery_depth": item.depth,
                "bam_alt_fraction": alt_fraction,
                "bam_alt_mapq_mean": float(np.mean(item.mapq)) if item.mapq else math.nan,
                "bam_alt_mapq_median": float(np.median(item.mapq)) if item.mapq else math.nan,
                **{name: value for name, value in context.items() if not name.startswith("_")},
                "_variant_key": key,
            }
        )
    metrics["alleles_after_support_filter"] = len(rows)
    return rows, metrics


def count_allele_support(
    bam,
    fasta,
    key: VariantKey,
    min_mapq: int = 20,
) -> dict[str, object]:
    """Force-count one normalized truth allele for diagnostic use only."""
    bam_resolver = ContigResolver.from_names(bam.references)
    fasta_resolver = ContigResolver.from_names(fasta.references)
    bam_chrom = bam_resolver.resolve(key.chrom)
    fasta_chrom = fasta_resolver.resolve(key.chrom)
    empty = {
        "coverage": 0,
        "alt_support": 0,
        "alt_forward_reads": 0,
        "alt_reverse_reads": 0,
        "alt_fraction": math.nan,
    }
    if bam_chrom is None or fasta_chrom is None:
        return empty
    start0 = key.pos - 1
    end0 = start0 + max(1, len(key.ref))
    coverage = support = forward = reverse = 0
    max_length = max(50, len(key.ref), len(key.alt))
    for read in bam.fetch(bam_chrom, max(0, start0 - 1), end0 + 1):
        if not _usable_read(read) or read.mapping_quality < min_mapq:
            continue
        if read.reference_start <= start0 and (read.reference_end or -1) >= end0:
            coverage += 1
        events = read_indel_events(
            read,
            fasta,
            fasta_chrom,
            key.chrom,
            max_indel_length=max_length,
        )
        if key in events:
            support += 1
            reverse += int(read.is_reverse)
            forward += int(not read.is_reverse)
    return {
        "coverage": coverage,
        "alt_support": support,
        "alt_forward_reads": forward,
        "alt_reverse_reads": reverse,
        "alt_fraction": support / coverage if coverage else math.nan,
    }
