from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import pandas as pd
import pysam

from .bam_features import extract_bam_features
from .context import sequence_context_features
from .discovery import count_allele_support
from .intervals import IntervalSet
from .pipeline import load_manifest
from .rescue import (
    _normalized_file,
    _write_frame,
    normalize_truth_profiles,
    original_call_positive,
    truth_sets_for_profile,
)
from .utils import is_sequence_allele
from .variants import (
    VCF_FIELDS,
    VariantKey,
    _format_gt,
    _plain_value,
    label_variant,
    parse_truth,
    read_vcf_contigs,
    subset_vcf,
)


LOGGER = logging.getLogger("ion_hp_ml.hotspot_rescue")

# Extra FORMAT fields useful for distinguishing raw read evidence from TVC's
# flow-adjusted evidence.  The original requested Oncomine fields remain in
# VCF_FIELDS; these are additive.
HOTSPOT_EXTRA_FIELDS = (
    "DP", "RO", "SAF", "SAR", "SRF", "SRR", "GQ",
)


def _allele_value(value, alt_index: int):
    """Return an allele-specific scalar while preserving non-allelic tuples."""
    if value is None:
        return None
    if isinstance(value, tuple):
        if not value:
            return None
        if len(value) == 1:
            return _plain_value(value)
        if 0 <= alt_index < len(value):
            return _plain_value(value[alt_index])
    return _plain_value(value)


def filter_reason_features(reason: object) -> dict[str, object]:
    """Convert Ion FR text into stable numeric flags and a shift magnitude."""
    text = "" if reason is None else str(reason)
    upper = text.upper()
    shift = re.search(r"PREDICTIONSHIFTX([0-9.+\-Ee]+)>", upper)
    return {
        "vcf_reason_prediction_shift": int("PREDICTIONSHIFT" in upper),
        "vcf_prediction_shift_value": float(shift.group(1)) if shift else math.nan,
        "vcf_reason_stringency": int("STRINGENCY" in upper),
        "vcf_reason_quality_score": int("QUALITYSCORE" in upper),
        "vcf_reason_hp_length": int("HPLEN" in upper or "HOMOPOLYMER" in upper),
        "vcf_reason_strand_bias": int("STRANDBIAS" in upper),
        "vcf_reason_read_quality": int("READQUALITY" in upper),
        "vcf_reason_low_coverage": int("COVERAGE" in upper),
    }


def _ratio(numerator: object, denominator: object) -> float:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
    except (TypeError, ValueError):
        return math.nan
    return numerator / denominator if denominator > 0 else math.nan


def parse_hotspot_indels(
    normalized_vcf: str | Path,
    sample_name: str,
    run_name: str,
    target_intervals: IntervalSet | None = None,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Read every normalized sequence indel marked HS, including 0/0/NOCALL."""
    rows: dict[VariantKey, dict[str, object]] = {}
    metrics = {
        "hotspot_sequence_indels": 0,
        "outside_target": 0,
        "duplicates": 0,
    }
    with pysam.VariantFile(str(normalized_vcf)) as vcf:
        for record in vcf:
            if not record.alts or not bool(record.info.get("HS", False)):
                continue
            sample = next(iter(record.samples.values()), None)
            filters = list(record.filter.keys())
            reason = _plain_value(record.info.get("FR"))
            for alt_index, alt in enumerate(record.alts):
                if not is_sequence_allele(record.ref) or not is_sequence_allele(alt):
                    continue
                if len(record.ref) == len(alt):
                    continue
                metrics["hotspot_sequence_indels"] += 1
                start0 = record.pos - 1
                end0 = start0 + max(1, len(record.ref))
                if target_intervals and not target_intervals.overlaps(
                    record.chrom, start0, end0
                ):
                    metrics["outside_target"] += 1
                    continue
                key = VariantKey.create(record.chrom, record.pos, record.ref, alt)
                row: dict[str, object] = {
                    "sample": sample_name,
                    "run": run_name,
                    "chrom": key.chrom,
                    "pos": key.pos,
                    "ref": key.ref,
                    "alt": key.alt,
                    "locus_id": key.locus_id,
                    "candidate_source": "FORCED_ONCOMINE_HOTSPOT",
                    "oncomine_filter_reason": reason,
                    "vcf_QUAL": record.qual,
                    "vcf_FILTER": ";".join(filters) if filters else ".",
                    "vcf_GT": _format_gt(sample),
                    "_variant_key": key,
                }
                for field in (*VCF_FIELDS, *HOTSPOT_EXTRA_FIELDS):
                    value = record.info.get(field) if field in vcf.header.info else None
                    if value is None and sample is not None and field in sample:
                        value = sample[field]
                    row[f"vcf_{field}"] = _allele_value(value, alt_index)
                row.update(filter_reason_features(reason))
                row["vcf_raw_af"] = _ratio(row.get("vcf_AO"), row.get("vcf_DP"))
                row["vcf_flow_af"] = _ratio(row.get("vcf_FAO"), row.get("vcf_FDP"))
                if math.isfinite(row["vcf_raw_af"]) and math.isfinite(row["vcf_flow_af"]):
                    row["vcf_raw_minus_flow_af"] = (
                        row["vcf_raw_af"] - row["vcf_flow_af"]
                    )
                else:
                    row["vcf_raw_minus_flow_af"] = math.nan

                previous = rows.get(key)
                if previous is not None:
                    metrics["duplicates"] += 1
                    old_score = (
                        float(previous.get("vcf_FAO") or -1),
                        float(previous.get("vcf_QUAL") or -1),
                    )
                    new_score = (
                        float(row.get("vcf_FAO") or -1),
                        float(row.get("vcf_QUAL") or -1),
                    )
                    if new_score <= old_score:
                        continue
                rows[key] = row
    return list(rows.values()), metrics


def _is_target_hp(context: dict[str, object], min_hp_length: int) -> bool:
    return bool(
        context["reference_match"]
        and context["is_homopolymer_indel"]
        and max(float(context["hp_ref_len"]), float(context["hp_alt_len"]))
        >= min_hp_length
    )


def _add_paired_na_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Add matched NA24385 median deltas without using labels or locus IDs."""
    if frame.empty or "truth_profile" not in frame:
        return frame
    evidence_columns: list[str] = []
    for column in frame.columns:
        if not column.startswith(("vcf_", "bam_", "flow_")):
            continue
        if column in {"vcf_FILTER", "vcf_GT", "vcf_TYPE"}:
            continue
        converted = pd.to_numeric(frame[column], errors="coerce")
        if converted.notna().any():
            frame[column] = converted
            evidence_columns.append(column)
    controls = frame.loc[frame["truth_profile"] == "HG002"]
    if controls.empty or not evidence_columns:
        return frame
    medians = controls.groupby("locus_id", sort=False)[evidence_columns].median()
    medians = medians.rename(columns=lambda value: f"na_median__{value}")
    frame = frame.join(medians, on="locus_id")
    deltas = {
        f"paired_delta__{column}": frame[column] - frame[f"na_median__{column}"]
        for column in evidence_columns
    }
    return pd.concat([frame, pd.DataFrame(deltas, index=frame.index)], axis=1)


def build_hotspot_rescue_dataset(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = (
        Path(args.normalization_cache).resolve()
        if args.normalization_cache
        else output_dir / "cache"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = output_dir / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    manifest = normalize_truth_profiles(load_manifest(args.manifest))
    fasta_path = Path(args.fasta).resolve()
    target_path = Path(args.target_bed).resolve()
    giab_bed_path = Path(args.giab_bed).resolve()
    aohc_truth_path = Path(args.aohc_truth).resolve()
    giab_truth_path = Path(args.giab_truth).resolve()
    targets = IntervalSet.from_bed(target_path)
    benchmark = IntervalSet.from_bed(giab_bed_path)
    qc_rows: list[dict[str, object]] = []
    all_features: list[pd.DataFrame] = []
    truth_rows: list[dict[str, object]] = []
    baseline_fp_rows: list[dict[str, object]] = []

    with pysam.FastaFile(str(fasta_path)) as fasta:
        normalized_aohc = _normalized_file(
            aohc_truth_path,
            cache_dir / "aohc_truth.normalized.vcf",
            fasta_path,
            fasta.references,
            args.bcftools,
            args.force_normalization,
        )
        giab_regions = cache_dir / "giab_target_regions.bed"
        targets.write_bed(giab_regions, read_vcf_contigs(giab_truth_path))
        giab_subset = cache_dir / "giab.target_subset.vcf"
        if args.force_normalization or not giab_subset.exists():
            subset_vcf(giab_truth_path, giab_regions, giab_subset, args.bcftools)
        normalized_giab = _normalized_file(
            giab_subset,
            cache_dir / "giab.target_subset.normalized.vcf",
            fasta_path,
            fasta.references,
            args.bcftools,
            args.force_normalization,
        )
        aohc_truth = parse_truth(normalized_aohc)
        giab_truth = parse_truth(normalized_giab, require_nonreference_gt=True)

        for manifest_row in manifest.itertuples(index=False):
            LOGGER.info("Preparing forced hotspots for %s", manifest_row.run)
            normalized_vcf = _normalized_file(
                manifest_row.vcf,
                cache_dir / f"{manifest_row.run}.oncomine.normalized.vcf",
                fasta_path,
                fasta.references,
                args.bcftools,
                args.force_normalization,
            )
            rows, metrics = parse_hotspot_indels(
                normalized_vcf,
                manifest_row.sample,
                manifest_row.run,
                targets,
            )
            run_aohc_truth, run_giab_truth = truth_sets_for_profile(
                manifest_row.truth_profile, aohc_truth, giab_truth
            )
            hp_rows: list[dict[str, object]] = []
            for row in rows:
                key = row["_variant_key"]
                context = sequence_context_features(
                    fasta, key.chrom, key.pos, key.ref, key.alt
                )
                label = label_variant(
                    key, run_aohc_truth, run_giab_truth, benchmark
                )
                row.update(label)
                row.update(
                    {name: value for name, value in context.items() if not name.startswith("_")}
                )
                row["truth_profile"] = manifest_row.truth_profile
                row["original_called"] = original_call_positive(row)
                row["targeted_hp"] = _is_target_hp(context, args.min_hp_length)
                row["evaluation_stratum"] = "HP" if row["targeted_hp"] else "NON_HP"

                public_row = {name: value for name, value in row.items() if name != "_variant_key"}
                if manifest_row.truth_profile == "AOHC" and key in aohc_truth:
                    truth_rows.append(public_row.copy())
                if label["label"] == 0 and row["original_called"]:
                    baseline_fp_rows.append(public_row.copy())
                if row["targeted_hp"]:
                    hp_rows.append(row)

            qc_rows.extend(
                {"run": manifest_row.run, "metric": name, "value": value}
                for name, value in metrics.items()
            )
            qc_rows.extend(
                [
                    {"run": manifest_row.run, "metric": "targeted_hp_records", "value": len(hp_rows)},
                    {
                        "run": manifest_row.run,
                        "metric": "targeted_hp_positive",
                        "value": sum(row["label"] == 1 for row in hp_rows),
                    },
                    {
                        "run": manifest_row.run,
                        "metric": "targeted_hp_reference",
                        "value": sum(row["label"] == 0 for row in hp_rows),
                    },
                    {
                        "run": manifest_row.run,
                        "metric": "targeted_hp_unknown",
                        "value": sum(row["label"] is None for row in hp_rows),
                    },
                ]
            )

            feature_cache = per_run_dir / f"{manifest_row.run}.hotspot_hp_features.parquet"
            if args.force_features or not feature_cache.exists():
                feature_rows: list[dict[str, object]] = []
                with pysam.AlignmentFile(
                    manifest_row.bam, "rb", threads=args.threads
                ) as bam:
                    for index, row in enumerate(hp_rows, start=1):
                        key = row["_variant_key"]
                        context = sequence_context_features(
                            fasta, key.chrom, key.pos, key.ref, key.alt
                        )
                        support = count_allele_support(
                            bam, fasta, key, min_mapq=args.min_mapq
                        )
                        bam_features = extract_bam_features(
                            bam,
                            fasta,
                            key.chrom,
                            int(context["_hp_ref_start0"]),
                            int(context["_hp_ref_end0"]),
                            context["hp_base"],
                            context["hp_ref_len"],
                            context["hp_alt_len"],
                            flow_window=args.flow_window,
                            max_reads=args.max_reads_per_candidate,
                            include_flow=not args.no_flow,
                        )
                        feature_row = {
                            name: value for name, value in row.items() if name != "_variant_key"
                        }
                        feature_row.update(
                            {f"bam_forced_{name}": value for name, value in support.items()}
                        )
                        feature_row.update(bam_features)
                        feature_rows.append(feature_row)
                        if index % 20 == 0 or index == len(hp_rows):
                            LOGGER.info(
                                "%s forced features: %d/%d",
                                manifest_row.run,
                                index,
                                len(hp_rows),
                            )
                run_frame = pd.DataFrame(feature_rows)
                run_frame.to_parquet(feature_cache, index=False)
            else:
                run_frame = pd.read_parquet(feature_cache)
            all_features.append(run_frame)

    candidate_frame = pd.concat(all_features, ignore_index=True)
    candidate_frame = _add_paired_na_deltas(candidate_frame)
    truth_frame = pd.DataFrame(truth_rows)
    baseline_fp_frame = pd.DataFrame(baseline_fp_rows)
    if "label" in candidate_frame:
        candidate_frame["label"] = pd.array(candidate_frame["label"], dtype="Int64")

    identity = ["sample", "run", "locus_id"]
    aohc_runs = int((manifest["truth_profile"] == "AOHC").sum())
    truth_loci = truth_frame["locus_id"].nunique() if not truth_frame.empty else 0
    checks = {
        "candidate_rows_nonempty": not candidate_frame.empty,
        "candidate_rows_unique": not candidate_frame.duplicated(identity).any(),
        "candidate_both_classes": candidate_frame["label"].dropna().nunique() == 2,
        "unknown_not_labeled": bool(
            candidate_frame.loc[
                candidate_frame["label_name"] == "UNKNOWN", "label"
            ].isna().all()
        ),
        "all_candidates_are_forced_hotspot_hp": bool(
            candidate_frame["targeted_hp"].fillna(False).all()
            and (candidate_frame["candidate_source"] == "FORCED_ONCOMINE_HOTSPOT").all()
        ),
        "aohc_truth_observations_complete": bool(
            truth_loci > 0 and len(truth_frame) == truth_loci * aohc_runs
        ),
        "original_calls_never_redefined": bool(
            candidate_frame.apply(
                lambda row: bool(row["original_called"])
                == original_call_positive(row.to_dict()),
                axis=1,
            ).all()
        ),
    }
    for name, passed in checks.items():
        qc_rows.append({"run": "ALL", "metric": f"validation_{name}", "value": passed})
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Hotspot rescue dataset validation failed: {', '.join(failed)}")

    candidate_stem = output_dir / "hotspot_hp_features"
    truth_stem = output_dir / "aohc_hotspot_truth_observations"
    fp_stem = output_dir / "baseline_hotspot_false_positive_calls"
    _write_frame(candidate_frame, candidate_stem)
    _write_frame(truth_frame, truth_stem)
    _write_frame(baseline_fp_frame, fp_stem)
    qc_path = output_dir / "hotspot_rescue_qc.tsv"
    pd.DataFrame(qc_rows).to_csv(qc_path, sep="\t", index=False)
    marker = {
        "validation_passed": True,
        "checks": checks,
        "candidate_rows": len(candidate_frame),
        "candidate_labeled_rows": int(candidate_frame["label"].notna().sum()),
        "candidate_positive_rows": int((candidate_frame["label"] == 1).sum()),
        "candidate_negative_rows": int((candidate_frame["label"] == 0).sum()),
        "candidate_unknown_rows": int(candidate_frame["label"].isna().sum()),
        "aohc_hotspot_truth_observations": len(truth_frame),
        "aohc_hotspot_truth_loci": int(truth_loci),
        "parameters": {
            "min_hp_length": args.min_hp_length,
            "min_mapq": args.min_mapq,
            "flow_window": args.flow_window,
            "max_reads_per_candidate": args.max_reads_per_candidate,
            "flow_enabled": not args.no_flow,
        },
    }
    (output_dir / "HOTSPOT_RESCUE_DATA_VALIDATED.json").write_text(
        json.dumps(marker, indent=2) + "\n"
    )
    return (
        candidate_stem.with_suffix(".parquet"),
        truth_stem.with_suffix(".parquet"),
        qc_path,
    )


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build forced Oncomine-hotspot HP features for add-on rescue."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--aohc-truth", required=True)
    parser.add_argument("--giab-truth", required=True)
    parser.add_argument("--giab-bed", required=True)
    parser.add_argument("--target-bed", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output-dir", default="output/hotspot_rescue")
    parser.add_argument("--normalization-cache")
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--min-hp-length", type=int, default=4)
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--flow-window", type=int, default=5)
    parser.add_argument("--max-reads-per-candidate", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument("--force-features", action="store_true")
    parser.add_argument("--force-normalization", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for path in build_hotspot_rescue_dataset(args):
        LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
