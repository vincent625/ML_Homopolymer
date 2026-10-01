from __future__ import annotations

import gzip
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pysam

from .intervals import IntervalSet
from .utils import ContigResolver, canonical_chrom, is_sequence_allele


VCF_FIELDS = (
    "TYPE", "LEN", "AF", "AO", "FAO", "FDP", "FRO", "FSAF", "FSAR",
    "FSRF", "FSRR", "HRUN", "MLLD", "RBI", "REFB", "VARB", "FWDB",
    "REVB", "FXX", "STB", "STBP",
)


@dataclass(frozen=True, order=True)
class VariantKey:
    chrom: str
    pos: int
    ref: str
    alt: str

    @classmethod
    def create(cls, chrom: str, pos: int, ref: str, alt: str) -> "VariantKey":
        return cls(canonical_chrom(chrom), int(pos), ref.upper(), alt.upper())

    @property
    def locus_id(self) -> str:
        return f"{self.chrom}:{self.pos}:{self.ref}:{self.alt}"


def _open_text(path: str | Path):
    path = Path(path)
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def read_vcf_contigs(path: str | Path) -> list[str]:
    contigs: list[str] = []
    pattern = re.compile(r"^##contig=<ID=([^,>]+)")
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                break
            match = pattern.match(line)
            if match:
                contigs.append(match.group(1))
    return contigs


def sanitize_vcf(
    source: str | Path,
    destination: str | Path,
    reference_contigs: Iterable[str],
) -> dict[str, int]:
    """Make Ion Reporter VCF parseable without changing requested evidence fields.

    Ion Reporter FUNC values sometimes contain semicolons inside an unescaped value.
    FUNC is intentionally not a model feature, so it and only it is removed.
    """
    resolver = ContigResolver.from_names(reference_contigs)
    contig_pattern = re.compile(r"^(##contig=<ID=)([^,>]+)(.*)$")
    metrics = {"records": 0, "func_removed": 0, "unresolved_contigs": 0}
    has_gt_header = False
    destination = Path(destination)
    with _open_text(source) as input_handle, destination.open("w") as output_handle:
        for line in input_handle:
            if line.startswith("##FORMAT=<ID=GT,"):
                has_gt_header = True
            if line.startswith("#CHROM"):
                if not has_gt_header:
                    output_handle.write(
                        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                    )
                output_handle.write(line)
                continue
            match = contig_pattern.match(line.rstrip("\n"))
            if match:
                resolved = resolver.resolve(match.group(2))
                if resolved:
                    output_handle.write(f"{match.group(1)}{resolved}{match.group(3)}\n")
                else:
                    output_handle.write(line)
                continue
            if line.startswith("#"):
                output_handle.write(line)
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                raise ValueError(f"Malformed VCF record in {source}: {line[:120]!r}")
            metrics["records"] += 1
            resolved = resolver.resolve(fields[0])
            if resolved is None:
                metrics["unresolved_contigs"] += 1
            else:
                fields[0] = resolved
            marker = fields[7].find("FUNC=")
            if marker >= 0:
                prefix = fields[7][:marker]
                fields[7] = prefix[:-1] if prefix.endswith(";") else prefix
                fields[7] = fields[7] or "."
                metrics["func_removed"] += 1
            output_handle.write("\t".join(fields) + "\n")
    return metrics


def normalize_vcf(
    source: str | Path,
    destination: str | Path,
    fasta_path: str | Path,
    reference_contigs: Iterable[str],
    bcftools: str = "bcftools",
) -> dict[str, object]:
    destination = Path(destination)
    sanitized = destination.with_suffix(".sanitized.vcf")
    metrics = sanitize_vcf(source, sanitized, reference_contigs)
    command = [
        bcftools, "norm", "--fasta-ref", str(fasta_path), "--multiallelics", "-any",
        "--check-ref", "w", "--output-type", "v", "--output", str(destination),
        str(sanitized),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    metrics["bcftools_stderr"] = completed.stderr.strip()
    sanitized.unlink(missing_ok=True)
    if completed.returncode:
        raise RuntimeError(
            f"bcftools normalization failed for {source}:\n{completed.stderr.strip()}"
        )
    return metrics


def subset_vcf(
    source: str | Path,
    regions_bed: str | Path,
    destination: str | Path,
    bcftools: str = "bcftools",
) -> None:
    command = [
        bcftools, "view", "--regions-file", str(regions_bed), "--output-type", "v",
        "--output-file", str(destination), str(source),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"bcftools GIAB subsetting failed:\n{completed.stderr.strip()}")


def write_candidate_regions(
    keys: Iterable[VariantKey],
    destination: str | Path,
    source_contigs: Iterable[str],
    flank: int = 50,
) -> None:
    resolver = ContigResolver.from_names(source_contigs)
    intervals: list[tuple[str, int, int]] = []
    for key in set(keys):
        chrom = resolver.resolve(key.chrom)
        if chrom is None:
            continue
        start = max(0, key.pos - 1 - flank)
        end = key.pos - 1 + max(len(key.ref), 1) + flank
        intervals.append((chrom, start, end))
    order = {name: index for index, name in enumerate(source_contigs)}
    intervals.sort(key=lambda value: (order.get(value[0], 10**9), value[1], value[2]))
    merged: list[list[object]] = []
    for chrom, start, end in intervals:
        if merged and merged[-1][0] == chrom and start <= int(merged[-1][2]):
            merged[-1][2] = max(int(merged[-1][2]), end)
        else:
            merged.append([chrom, start, end])
    with Path(destination).open("w") as handle:
        for chrom, start, end in merged:
            handle.write(f"{chrom}\t{start}\t{end}\n")


def _plain_value(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        if not value:
            return None
        return _plain_value(value[0]) if len(value) == 1 else ",".join(map(str, value))
    return value


def _format_gt(sample) -> str | None:
    if sample is None or "GT" not in sample or sample["GT"] is None:
        return None
    separator = "|" if sample.phased else "/"
    return separator.join("." if value is None else str(value) for value in sample["GT"])


def parse_candidates(
    normalized_vcf: str | Path,
    sample_name: str,
    run_name: str,
    target_intervals: IntervalSet | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    candidates: dict[VariantKey, dict[str, object]] = {}
    metrics = {"sequence_indels": 0, "outside_target": 0, "duplicates": 0}
    with pysam.VariantFile(str(normalized_vcf)) as vcf:
        for record in vcf:
            if not record.alts:
                continue
            sample = next(iter(record.samples.values()), None)
            for alt in record.alts:
                if not is_sequence_allele(record.ref) or not is_sequence_allele(alt):
                    continue
                if len(record.ref) == len(alt):
                    continue
                metrics["sequence_indels"] += 1
                start0 = record.pos - 1
                end0 = start0 + max(1, len(record.ref))
                if target_intervals and not target_intervals.overlaps(record.chrom, start0, end0):
                    metrics["outside_target"] += 1
                    continue
                key = VariantKey.create(record.chrom, record.pos, record.ref, alt)
                filters = list(record.filter.keys())
                row: dict[str, object] = {
                    "sample": sample_name,
                    "run": run_name,
                    "chrom": record.chrom,
                    "pos": record.pos,
                    "ref": record.ref.upper(),
                    "alt": alt.upper(),
                    "locus_id": key.locus_id,
                    "vcf_QUAL": record.qual,
                    "vcf_FILTER": ";".join(filters) if filters else ".",
                    "vcf_GT": _format_gt(sample),
                    "_variant_key": key,
                }
                for field in VCF_FIELDS:
                    value = record.info.get(field)
                    if value is None and sample is not None and field in sample:
                        value = sample[field]
                    row[f"vcf_{field}"] = _plain_value(value)
                previous = candidates.get(key)
                if previous is not None:
                    metrics["duplicates"] += 1
                    previous_score = (
                        float(previous.get("vcf_FAO") or -1),
                        float(previous.get("vcf_QUAL") or -1),
                    )
                    score = (float(row.get("vcf_FAO") or -1), float(row.get("vcf_QUAL") or -1))
                    if score <= previous_score:
                        continue
                candidates[key] = row
    return list(candidates.values()), metrics


def parse_truth(normalized_vcf: str | Path, require_nonreference_gt: bool = False) -> set[VariantKey]:
    truth: set[VariantKey] = set()
    with pysam.VariantFile(str(normalized_vcf)) as vcf:
        for record in vcf:
            if not record.alts:
                continue
            sample = next(iter(record.samples.values()), None)
            gt = sample.get("GT") if sample is not None and "GT" in sample else None
            for alt_index, alt in enumerate(record.alts, start=1):
                if not is_sequence_allele(record.ref) or not is_sequence_allele(alt):
                    continue
                if require_nonreference_gt and gt is not None and alt_index not in gt:
                    continue
                truth.add(VariantKey.create(record.chrom, record.pos, record.ref, alt))
    return truth


def label_variant(
    key: VariantKey,
    aohc_truth: set[VariantKey],
    giab_truth: set[VariantKey],
    benchmark: IntervalSet,
) -> dict[str, object]:
    sources: list[str] = []
    if key in aohc_truth:
        sources.append("AOHC_SYNTHETIC")
    if key in giab_truth:
        sources.append("GIAB_HG002")
    if sources:
        return {"label": 1, "label_name": "TP", "truth_source": "+".join(sources)}
    start0 = key.pos - 1
    end0 = start0 + max(1, len(key.ref))
    if benchmark.contains(key.chrom, start0, end0):
        return {"label": 0, "label_name": "REFERENCE", "truth_source": "GIAB_CONFIDENT_REFERENCE"}
    return {"label": None, "label_name": "UNKNOWN", "truth_source": "OUTSIDE_GIAB_BENCHMARK"}


def normalize_variant_python(
    fasta,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    fasta_chrom: str | None = None,
) -> VariantKey:
    """Reference-aware normalization used for tests and validation cross-checks."""
    if fasta_chrom is None:
        resolver = ContigResolver.from_names(fasta.references)
        fasta_chrom = resolver.resolve(chrom)
    if fasta_chrom is None:
        raise ValueError(f"Contig {chrom!r} is absent from reference")
    pos = int(pos)
    ref, alt = ref.upper(), alt.upper()

    def trim(current_pos: int, current_ref: str, current_alt: str):
        while len(current_ref) > 1 and len(current_alt) > 1 and current_ref[-1] == current_alt[-1]:
            current_ref, current_alt = current_ref[:-1], current_alt[:-1]
        while len(current_ref) > 1 and len(current_alt) > 1 and current_ref[0] == current_alt[0]:
            current_ref, current_alt = current_ref[1:], current_alt[1:]
            current_pos += 1
        return current_pos, current_ref, current_alt

    pos, ref, alt = trim(pos, ref, alt)
    if len(ref) != len(alt):
        while pos > 1:
            old = (pos, ref, alt)
            previous = fasta.fetch(fasta_chrom, pos - 2, pos - 1).upper()
            pos, ref, alt = trim(pos - 1, previous + ref, previous + alt)
            if (pos, ref, alt) == old or pos >= old[0]:
                break
    return VariantKey.create(chrom, pos, ref, alt)
