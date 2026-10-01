from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Iterable

import numpy as np

from .flow import (
    candidate_query_position,
    find_candidate_flow,
    read_to_flow_table,
    reconstructed_sequence,
    sequencing_sequence,
)
from .utils import COMPLEMENT, ContigResolver, summary_stats


def _usable_read(read) -> bool:
    return not (
        read.is_unmapped
        or read.is_secondary
        or read.is_supplementary
        or read.is_duplicate
        or read.is_qcfail
    )


def _deterministic_subsample(reads: list, maximum: int | None) -> list:
    if maximum is None or len(reads) <= maximum:
        return reads
    scored = []
    for read in reads:
        token = f"{read.query_name}|{read.flag}|{read.reference_start}".encode()
        score = hashlib.blake2b(token, digest_size=8).digest()
        scored.append((score, read))
    return [read for _, read in sorted(scored, key=lambda item: item[0])[:maximum]]


def _softclip_fraction(read) -> tuple[float, bool]:
    softclipped = sum(length for operation, length in (read.cigartuples or []) if operation == 4)
    length = max(read.query_length or 0, 1)
    return softclipped / length, softclipped > 0


def _local_burdens(read, fasta, chrom: str, start: int, end: int) -> tuple[float, float]:
    query = (read.query_sequence or "").upper()
    query_cursor = 0
    reference_cursor = read.reference_start
    aligned_bases = 0
    mismatches = 0
    indel_bases = 0
    for operation, length in read.cigartuples or []:
        if operation in {0, 7, 8}:  # M, =, X
            overlap_start = max(reference_cursor, start)
            overlap_end = min(reference_cursor + length, end)
            if overlap_end > overlap_start:
                offset = overlap_start - reference_cursor
                count = overlap_end - overlap_start
                reference = fasta.fetch(chrom, overlap_start, overlap_end).upper()
                observed = query[query_cursor + offset:query_cursor + offset + count]
                aligned_bases += min(len(reference), len(observed))
                mismatches += sum(a != b for a, b in zip(observed, reference))
            query_cursor += length
            reference_cursor += length
        elif operation == 1:  # insertion
            if start <= reference_cursor < end:
                indel_bases += length
            query_cursor += length
        elif operation in {2, 3}:  # deletion or reference skip
            if operation == 2:
                indel_bases += max(0, min(reference_cursor + length, end) - max(reference_cursor, start))
            reference_cursor += length
        elif operation == 4:  # soft clipping
            query_cursor += length
        elif operation in {5, 6}:  # hard clipping, padding
            continue
    mismatch_burden = mismatches / aligned_bases if aligned_bases else math.nan
    denominator = aligned_bases + indel_bases
    indel_burden = indel_bases / denominator if denominator else math.nan
    return mismatch_burden, indel_burden


def _read_group_flow_orders(bam) -> dict[str, str]:
    result: dict[str, str] = {}
    for read_group in bam.header.to_dict().get("RG", []):
        if "ID" in read_group and "FO" in read_group:
            result[str(read_group["ID"])] = str(read_group["FO"])
    return result


def _empty_features(flow_window: int) -> dict[str, object]:
    result: dict[str, object] = {
        "bam_coverage": 0,
        "bam_reads_used": 0,
        "bam_forward_reads": 0,
        "bam_reverse_reads": 0,
        "bam_mapq_ge30_fraction": math.nan,
        "bam_softclip_read_fraction": math.nan,
        "flow_reads": 0,
        "flow_read_fraction": math.nan,
        "flow_reconstructed_fraction": math.nan,
        "flow_fraction_alt_closer": math.nan,
    }
    for prefix in (
        "bam_mapq", "bam_softclip_base_fraction", "bam_candidate_read_position",
        "bam_distance_to_read_end", "bam_local_mismatch_burden", "bam_local_indel_burden",
        "flow_candidate_signal", "flow_ref_residual", "flow_alt_residual",
        "flow_residual_difference", "flow_neighbor_noise", "flow_CF", "flow_IE", "flow_DR",
    ):
        result.update(summary_stats([], prefix))
    for offset in range(-flow_window, flow_window + 1):
        label = f"m{abs(offset)}" if offset < 0 else f"p{offset}"
        result.update(summary_stats([], f"flow_{label}_signal"))
    return result


def extract_bam_features(
    bam,
    fasta,
    chrom: str,
    hp_start0: int,
    hp_end0: int,
    hp_base: str | None,
    hp_ref_len: float,
    hp_alt_len: float,
    flow_window: int = 5,
    max_reads: int | None = None,
    include_flow: bool = True,
) -> dict[str, object]:
    result = _empty_features(flow_window)
    bam_resolver = ContigResolver.from_names(bam.references)
    fasta_resolver = ContigResolver.from_names(fasta.references)
    bam_chrom = bam_resolver.resolve(chrom)
    fasta_chrom = fasta_resolver.resolve(chrom)
    if bam_chrom is None or fasta_chrom is None:
        return result

    region_start = max(0, int(hp_start0))
    region_end = max(region_start + 1, int(hp_end0))
    reads = [
        read for read in bam.fetch(bam_chrom, region_start, region_end)
        if _usable_read(read)
        and read.reference_start <= region_start
        and (read.reference_end or -1) >= region_end
    ]
    result["bam_coverage"] = len(reads)
    result["bam_forward_reads"] = sum(not read.is_reverse for read in reads)
    result["bam_reverse_reads"] = sum(read.is_reverse for read in reads)
    reads = _deterministic_subsample(reads, max_reads)
    result["bam_reads_used"] = len(reads)
    if not reads:
        return result

    mapq: list[float] = []
    softclip_fraction: list[float] = []
    softclip_present: list[bool] = []
    read_positions: list[float] = []
    read_end_distances: list[float] = []
    mismatch_burdens: list[float] = []
    indel_burdens: list[float] = []
    candidate_signals: list[float] = []
    ref_residuals: list[float] = []
    alt_residuals: list[float] = []
    residual_differences: list[float] = []
    alt_closer: list[bool] = []
    neighbor_noise: list[float] = []
    zp_values: dict[str, list[float]] = {"CF": [], "IE": [], "DR": []}
    offset_signals: dict[int, list[float]] = defaultdict(list)
    flow_orders = _read_group_flow_orders(bam)
    reconstructed: list[bool] = []

    burden_start = max(0, region_start - 10)
    burden_end = region_end + 10
    for read in reads:
        mapq.append(float(read.mapping_quality))
        clipped_fraction, clipped = _softclip_fraction(read)
        softclip_fraction.append(clipped_fraction)
        softclip_present.append(clipped)
        query_position = candidate_query_position(read, region_start, region_end)
        if query_position is not None and read.query_length:
            read_positions.append(query_position / max(read.query_length - 1, 1))
            read_end_distances.append(min(query_position, read.query_length - 1 - query_position))
        mismatch, indel = _local_burdens(
            read, fasta, fasta_chrom, burden_start, burden_end
        )
        mismatch_burdens.append(mismatch)
        indel_burdens.append(indel)

        if read.has_tag("ZP"):
            values = list(read.get_tag("ZP"))
            for name, value in zip(("CF", "IE", "DR"), values):
                zp_values[name].append(float(value))

        if not include_flow or hp_base is None or query_position is None:
            continue
        if not (read.has_tag("RG") and read.has_tag("ZF") and read.has_tag("ZM")):
            continue
        flow_order = flow_orders.get(str(read.get_tag("RG")))
        if not flow_order:
            continue
        try:
            table = read_to_flow_table(read, flow_order)
        except (TypeError, ValueError):
            continue
        sequence = sequencing_sequence(read)
        reconstructed.append(reconstructed_sequence(table) == sequence)
        expected_base = hp_base.translate(COMPLEMENT) if read.is_reverse else hp_base
        table_index = find_candidate_flow(table, query_position, expected_base)
        if table_index is None:
            continue
        signal = table[table_index].signal
        ref_residual = abs(signal - float(hp_ref_len))
        alt_residual = abs(signal - float(hp_alt_len))
        candidate_signals.append(signal)
        ref_residuals.append(ref_residual)
        alt_residuals.append(alt_residual)
        residual_differences.append(ref_residual - alt_residual)
        alt_closer.append(alt_residual < ref_residual)
        local_residuals: list[float] = []
        for offset in range(-flow_window, flow_window + 1):
            neighbor_index = table_index + offset
            if not 0 <= neighbor_index < len(table):
                continue
            neighbor = table[neighbor_index]
            offset_signals[offset].append(neighbor.signal)
            if offset:
                local_residuals.append(neighbor.signal - neighbor.called_hp_length)
        if local_residuals:
            neighbor_noise.append(float(np.std(local_residuals, ddof=0)))

    result.update(summary_stats(mapq, "bam_mapq"))
    result["bam_mapq_ge30_fraction"] = float(np.mean(np.asarray(mapq) >= 30))
    result.update(summary_stats(softclip_fraction, "bam_softclip_base_fraction"))
    result["bam_softclip_read_fraction"] = float(np.mean(softclip_present))
    result.update(summary_stats(read_positions, "bam_candidate_read_position"))
    result.update(summary_stats(read_end_distances, "bam_distance_to_read_end"))
    result.update(summary_stats(mismatch_burdens, "bam_local_mismatch_burden"))
    result.update(summary_stats(indel_burdens, "bam_local_indel_burden"))

    result["flow_reads"] = len(candidate_signals)
    result["flow_read_fraction"] = len(candidate_signals) / len(reads)
    result["flow_reconstructed_fraction"] = (
        float(np.mean(reconstructed)) if reconstructed else math.nan
    )
    result.update(summary_stats(candidate_signals, "flow_candidate_signal"))
    result.update(summary_stats(ref_residuals, "flow_ref_residual"))
    result.update(summary_stats(alt_residuals, "flow_alt_residual"))
    result.update(summary_stats(residual_differences, "flow_residual_difference"))
    result["flow_fraction_alt_closer"] = float(np.mean(alt_closer)) if alt_closer else math.nan
    result.update(summary_stats(neighbor_noise, "flow_neighbor_noise"))
    for name, values in zp_values.items():
        result.update(summary_stats(values, f"flow_{name}"))
    for offset in range(-flow_window, flow_window + 1):
        label = f"m{abs(offset)}" if offset < 0 else f"p{offset}"
        result.update(summary_stats(offset_signals[offset], f"flow_{label}_signal"))
    return result

