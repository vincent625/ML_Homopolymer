from __future__ import annotations

import subprocess
from pathlib import Path

import pysam

from .context import sequence_context_features
from .intervals import IntervalSet
from .utils import is_sequence_allele
from .variants import VariantKey, normalize_variant_python


def run_bcftools_mpileup(
    bam_path: str | Path,
    fasta_path: str | Path,
    target_bed: str | Path,
    output_bcf: str | Path,
    bcftools: str = "bcftools",
    min_mapq: int = 20,
    max_depth: int = 1_000_000,
    threads: int = 4,
) -> None:
    """Generate a blind, allele-aware BAM pileup without consulting truth."""
    command = [
        bcftools,
        "mpileup",
        "--fasta-ref",
        str(fasta_path),
        "--regions-file",
        str(target_bed),
        "--max-depth",
        str(max_depth),
        "--min-MQ",
        str(min_mapq),
        "--annotate",
        "FORMAT/AD,FORMAT/ADF,FORMAT/ADR,FORMAT/DP",
        "--threads",
        str(threads),
        "--output-type",
        "b",
        "--output",
        str(output_bcf),
        str(bam_path),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"bcftools mpileup failed for {bam_path}:\n{completed.stderr.strip()}"
        )


def _allele_value(values, index: int) -> int:
    if values is None or index >= len(values) or values[index] is None:
        return 0
    return int(values[index])


def parse_mpileup_candidates(
    pileup_path: str | Path,
    fasta,
    targets: IntervalSet,
    min_support: int = 2,
    min_hp_length: int = 4,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Parse sequence-indel hypotheses proposed by bcftools mpileup."""
    candidates: dict[VariantKey, dict[str, object]] = {}
    metrics = {
        "pileup_records": 0,
        "sequence_indel_alleles": 0,
        "support_filtered": 0,
        "outside_target": 0,
        "non_hp_filtered": 0,
        "short_hp_filtered": 0,
        "normalized_duplicates": 0,
        "retained_candidates": 0,
    }
    with pysam.VariantFile(str(pileup_path)) as vcf:
        for record in vcf:
            metrics["pileup_records"] += 1
            if not record.alts:
                continue
            sample = next(iter(record.samples.values()), None)
            ad = sample.get("AD") if sample is not None and "AD" in sample else None
            adf = sample.get("ADF") if sample is not None and "ADF" in sample else None
            adr = sample.get("ADR") if sample is not None and "ADR" in sample else None
            dp = sample.get("DP") if sample is not None and "DP" in sample else None
            for alt_index, alt in enumerate(record.alts, start=1):
                if not is_sequence_allele(record.ref) or not is_sequence_allele(alt):
                    continue
                if len(record.ref) == len(alt):
                    continue
                metrics["sequence_indel_alleles"] += 1
                support = _allele_value(ad, alt_index)
                if support < min_support:
                    metrics["support_filtered"] += 1
                    continue
                key = normalize_variant_python(
                    fasta, record.chrom, record.pos, record.ref, alt
                )
                start0 = key.pos - 1
                end0 = start0 + max(1, len(key.ref))
                if not targets.overlaps(key.chrom, start0, end0):
                    metrics["outside_target"] += 1
                    continue
                context = sequence_context_features(
                    fasta, key.chrom, key.pos, key.ref, key.alt
                )
                if not context["is_homopolymer_indel"]:
                    metrics["non_hp_filtered"] += 1
                    continue
                if max(float(context["hp_ref_len"]), float(context["hp_alt_len"])) < min_hp_length:
                    metrics["short_hp_filtered"] += 1
                    continue
                depth = int(dp) if dp is not None else 0
                row = {
                    "chrom": key.chrom,
                    "pos": key.pos,
                    "ref": key.ref,
                    "alt": key.alt,
                    "locus_id": key.locus_id,
                    "candidate_source": "BLIND_BCFTOOLS_MPILEUP",
                    "bam_alt_support": support,
                    "bam_alt_forward_reads": _allele_value(adf, alt_index),
                    "bam_alt_reverse_reads": _allele_value(adr, alt_index),
                    "bam_discovery_depth": depth,
                    "bam_alt_fraction": support / depth if depth else None,
                    "bam_pileup_qual": record.qual,
                    **{name: value for name, value in context.items() if not name.startswith("_")},
                    "_variant_key": key,
                }
                previous = candidates.get(key)
                if previous is not None:
                    metrics["normalized_duplicates"] += 1
                    if int(previous["bam_alt_support"]) >= support:
                        continue
                candidates[key] = row
    metrics["retained_candidates"] = len(candidates)
    return list(candidates.values()), metrics
