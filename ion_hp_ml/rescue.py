from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import pandas as pd
import pysam

from .bam_features import extract_bam_features
from .context import sequence_context_features
from .discovery import count_allele_support, discover_indels
from .intervals import IntervalSet
from .pileup_candidates import parse_mpileup_candidates, run_bcftools_mpileup
from .pipeline import load_manifest
from .variants import (
    VariantKey,
    label_variant,
    normalize_vcf,
    parse_candidates,
    parse_truth,
    read_vcf_contigs,
    subset_vcf,
)


LOGGER = logging.getLogger("ion_hp_ml.rescue")

TRUTH_PROFILE_ALIASES = {
    "AOHC": "AOHC",
    "HG002": "HG002",
    "NA24385": "HG002",
    "GIAB": "HG002",
}


def normalize_truth_profiles(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return a manifest with an explicit, validated truth profile per run.

    Legacy AOHC-only manifests did not have this column, so AOHC remains the
    backwards-compatible default. Combined experiments should always specify it.
    """
    normalized = manifest.copy()
    if "truth_profile" not in normalized:
        normalized["truth_profile"] = "AOHC"
    profiles = normalized["truth_profile"].fillna("").str.strip().str.upper()
    unknown = sorted(set(profiles) - set(TRUTH_PROFILE_ALIASES))
    if unknown:
        raise ValueError(
            "Unsupported truth_profile value(s): "
            f"{', '.join(unknown)}; expected AOHC or HG002"
        )
    normalized["truth_profile"] = profiles.map(TRUTH_PROFILE_ALIASES)
    return normalized


def truth_sets_for_profile(
    truth_profile: str,
    aohc_truth: set[VariantKey],
    giab_truth: set[VariantKey],
) -> tuple[set[VariantKey], set[VariantKey]]:
    """Select truths that are biologically present in a sample profile."""
    profile = TRUTH_PROFILE_ALIASES.get(str(truth_profile).strip().upper())
    if profile == "AOHC":
        return aohc_truth, giab_truth
    if profile == "HG002":
        return set(), giab_truth
    raise ValueError(f"Unsupported truth_profile: {truth_profile!r}")


def original_call_positive(row: dict[str, object] | None) -> bool:
    if row is None or str(row.get("vcf_FILTER")) != "PASS":
        return False
    gt = row.get("vcf_GT")
    if gt is None or pd.isna(gt):
        return False
    tokens = str(gt).replace("|", "/").split("/")
    return any(token.isdigit() and int(token) > 0 for token in tokens)


def _write_frame(frame: pd.DataFrame, stem: Path) -> None:
    frame.to_csv(stem.with_suffix(".tsv"), sep="\t", index=False, na_rep="NA")
    frame.to_parquet(stem.with_suffix(".parquet"), index=False)


def _context_for_truth(
    fasta,
    keys: set[VariantKey],
    aohc_truth: set[VariantKey],
    giab_truth: set[VariantKey],
    targets: IntervalSet,
    min_hp_length: int,
) -> dict[VariantKey, dict[str, object]]:
    catalog: dict[VariantKey, dict[str, object]] = {}
    for key in sorted(keys):
        if len(key.ref) == len(key.alt):
            continue
        start0 = key.pos - 1
        end0 = start0 + max(1, len(key.ref))
        if not targets.overlaps(key.chrom, start0, end0):
            continue
        context = sequence_context_features(fasta, key.chrom, key.pos, key.ref, key.alt)
        if not context["reference_match"] or not context["is_homopolymer_indel"]:
            continue
        if max(float(context["hp_ref_len"]), float(context["hp_alt_len"])) < min_hp_length:
            continue
        sources = []
        if key in aohc_truth:
            sources.append("AOHC_SYNTHETIC")
        if key in giab_truth:
            sources.append("GIAB_HG002")
        catalog[key] = {
            "truth_source": "+".join(sources),
            **{name: value for name, value in context.items() if not name.startswith("_")},
        }
    return catalog


def _candidate_key(row) -> VariantKey:
    return VariantKey.create(row.chrom, row.pos, row.ref, row.alt)


def _normalized_file(
    source: str | Path,
    destination: Path,
    fasta_path: Path,
    fasta_contigs,
    bcftools: str,
    force: bool,
) -> Path:
    if force or not destination.exists():
        normalize_vcf(source, destination, fasta_path, fasta_contigs, bcftools)
    return destination


def build_rescue_dataset(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir).resolve()
    cache_dir = output_dir / "cache"
    per_run_dir = output_dir / "per_run"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
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

    with pysam.FastaFile(str(fasta_path)) as fasta:
        normalized_aohc = _normalized_file(
            aohc_truth_path,
            cache_dir / "aohc_truth.normalized.vcf",
            fasta_path,
            fasta.references,
            args.bcftools,
            args.force,
        )
        giab_regions = cache_dir / "giab_target_regions.bed"
        targets.write_bed(giab_regions, read_vcf_contigs(giab_truth_path))
        giab_subset = cache_dir / "giab.target_subset.vcf"
        if args.force or not giab_subset.exists():
            subset_vcf(giab_truth_path, giab_regions, giab_subset, args.bcftools)
        normalized_giab = _normalized_file(
            giab_subset,
            cache_dir / "giab.target_subset.normalized.vcf",
            fasta_path,
            fasta.references,
            args.bcftools,
            args.force,
        )
        aohc_truth = parse_truth(normalized_aohc)
        giab_truth = parse_truth(normalized_giab, require_nonreference_gt=True)
        truth_catalogs: dict[str, dict[VariantKey, dict[str, object]]] = {}
        for truth_profile in sorted(manifest["truth_profile"].unique()):
            profile_aohc, profile_giab = truth_sets_for_profile(
                truth_profile, aohc_truth, giab_truth
            )
            truth_catalogs[truth_profile] = _context_for_truth(
                fasta,
                profile_aohc | profile_giab,
                profile_aohc,
                profile_giab,
                targets,
                args.min_hp_length,
            )
            if not truth_catalogs[truth_profile]:
                raise RuntimeError(
                    "No targeted homopolymer truth indels passed the experiment "
                    f"definition for truth profile {truth_profile}"
                )
            qc_rows.extend(
                [
                    {
                        "run": "ALL",
                        "metric": f"truth_hp_loci_{truth_profile.lower()}",
                        "value": len(truth_catalogs[truth_profile]),
                    },
                    {
                        "run": "ALL",
                        "metric": f"truth_aohc_hp_loci_{truth_profile.lower()}",
                        "value": sum(
                            "AOHC_SYNTHETIC" in value["truth_source"]
                            for value in truth_catalogs[truth_profile].values()
                        ),
                    },
                    {
                        "run": "ALL",
                        "metric": f"truth_giab_hp_loci_{truth_profile.lower()}",
                        "value": sum(
                            "GIAB_HG002" in value["truth_source"]
                            for value in truth_catalogs[truth_profile].values()
                        ),
                    },
                ]
            )

        all_feature_frames: list[pd.DataFrame] = []
        truth_observations: list[dict[str, object]] = []
        baseline_false_positives: list[dict[str, object]] = []

        for manifest_row in manifest.itertuples(index=False):
            LOGGER.info("Preparing blind rescue candidates for %s", manifest_row.run)
            run_aohc_truth, run_giab_truth = truth_sets_for_profile(
                manifest_row.truth_profile, aohc_truth, giab_truth
            )
            run_truth_catalog = truth_catalogs[manifest_row.truth_profile]
            qc_rows.append(
                {
                    "run": manifest_row.run,
                    "metric": "truth_profile",
                    "value": manifest_row.truth_profile,
                }
            )
            normalized_oncomine = _normalized_file(
                manifest_row.vcf,
                cache_dir / f"{manifest_row.run}.oncomine.normalized.vcf",
                fasta_path,
                fasta.references,
                args.bcftools,
                args.force,
            )
            oncomine_rows, oncomine_metrics = parse_candidates(
                normalized_oncomine, manifest_row.sample, manifest_row.run, targets
            )
            oncomine_by_key = {row["_variant_key"]: row for row in oncomine_rows}
            for name, value in oncomine_metrics.items():
                qc_rows.append({"run": manifest_row.run, "metric": f"oncomine_{name}", "value": value})

            for key, row in oncomine_by_key.items():
                if not original_call_positive(row):
                    continue
                context = sequence_context_features(fasta, key.chrom, key.pos, key.ref, key.alt)
                if (
                    not context["reference_match"]
                    or not context["is_homopolymer_indel"]
                    or max(float(context["hp_ref_len"]), float(context["hp_alt_len"])) < args.min_hp_length
                ):
                    continue
                label = label_variant(key, run_aohc_truth, run_giab_truth, benchmark)
                if label["label"] == 0:
                    baseline_false_positives.append(
                        {
                            "sample": manifest_row.sample,
                            "run": manifest_row.run,
                            "truth_profile": manifest_row.truth_profile,
                            "chrom": key.chrom,
                            "pos": key.pos,
                            "ref": key.ref,
                            "alt": key.alt,
                            "locus_id": key.locus_id,
                            "original_called": True,
                            **label,
                            **{name: value for name, value in context.items() if not name.startswith("_")},
                        }
                    )

            run_target_bed = cache_dir / f"{manifest_row.run}.targets.bed"
            with pysam.AlignmentFile(manifest_row.bam, "rb") as header_bam:
                targets.write_bed(run_target_bed, header_bam.references)
            basic_cache = per_run_dir / f"{manifest_row.run}.blind_candidates.parquet"
            if args.force or not basic_cache.exists():
                if args.candidate_generator == "mpileup":
                    pileup_path = cache_dir / f"{manifest_row.run}.mpileup.bcf"
                    if args.force or not pileup_path.exists():
                        run_bcftools_mpileup(
                            manifest_row.bam,
                            fasta_path,
                            run_target_bed,
                            pileup_path,
                            bcftools=args.bcftools,
                            min_mapq=args.min_mapq,
                            max_depth=args.max_depth,
                            threads=args.threads,
                        )
                    candidates, discovery_metrics = parse_mpileup_candidates(
                        pileup_path,
                        fasta,
                        targets,
                        min_support=args.min_support,
                        min_hp_length=args.min_hp_length,
                    )
                else:
                    with pysam.AlignmentFile(
                        manifest_row.bam, "rb", threads=args.threads
                    ) as discovery_bam:
                        candidates, discovery_metrics = discover_indels(
                            discovery_bam,
                            fasta,
                            targets,
                            min_mapq=args.min_mapq,
                            min_support=args.min_support,
                            max_indel_length=args.max_indel_length,
                            min_hp_length=args.min_hp_length,
                            min_alt_fraction=args.min_alt_fraction,
                        )
                basic_rows = []
                unknown = 0
                for candidate in candidates:
                    key = candidate["_variant_key"]
                    candidate.update(
                        label_variant(key, run_aohc_truth, run_giab_truth, benchmark)
                    )
                    if candidate["label"] is None:
                        unknown += 1
                        continue
                    original = oncomine_by_key.get(key)
                    candidate.update(
                        {
                            "sample": manifest_row.sample,
                            "run": manifest_row.run,
                            "truth_profile": manifest_row.truth_profile,
                            "oncomine_vcf_present": original is not None,
                            "original_called": original_call_positive(original),
                            "oncomine_filter": original.get("vcf_FILTER") if original else None,
                            "oncomine_gt": original.get("vcf_GT") if original else None,
                        }
                    )
                    candidate.pop("_variant_key", None)
                    basic_rows.append(candidate)
                basic = pd.DataFrame(basic_rows)
                if "label" in basic:
                    basic["label"] = basic["label"].astype(int)
                basic.to_parquet(basic_cache, index=False)
                for name, value in discovery_metrics.items():
                    qc_rows.append({"run": manifest_row.run, "metric": f"discovery_{name}", "value": value})
                qc_rows.append({"run": manifest_row.run, "metric": "discovery_unknown_excluded", "value": unknown})
            else:
                basic = pd.read_parquet(basic_cache)
            basic["truth_profile"] = manifest_row.truth_profile

            # A cache may have been generated with a more permissive CIGAR
            # fraction during a scale run. Reapply the requested floor before
            # any expensive per-candidate BAM/flow extraction. This is safe only
            # in the stricter direction; fresh output directories should be used
            # when lowering a previously cached threshold.
            if (
                args.candidate_generator == "cigar"
                and args.min_alt_fraction is not None
                and "bam_alt_fraction" in basic
            ):
                basic = basic.loc[
                    pd.to_numeric(basic["bam_alt_fraction"], errors="coerce")
                    >= args.min_alt_fraction
                ].reset_index(drop=True)

            basic_keys = {
                VariantKey.create(row.chrom, row.pos, row.ref, row.alt)
                for row in basic.itertuples(index=False)
            }
            feature_cache = per_run_dir / f"{manifest_row.run}.features.parquet"
            if args.force or not feature_cache.exists():
                feature_rows: list[dict[str, object]] = []
                with pysam.AlignmentFile(manifest_row.bam, "rb", threads=args.threads) as bam:
                    for index, row in enumerate(basic.itertuples(index=False), start=1):
                        candidate = row._asdict()
                        context = sequence_context_features(
                            fasta, row.chrom, row.pos, row.ref, row.alt
                        )
                        candidate.update(
                            extract_bam_features(
                                bam,
                                fasta,
                                row.chrom,
                                int(context["_hp_ref_start0"]),
                                int(context["_hp_ref_end0"]),
                                context["hp_base"],
                                context["hp_ref_len"],
                                context["hp_alt_len"],
                                flow_window=args.flow_window,
                                max_reads=args.max_reads_per_candidate,
                                include_flow=not args.no_flow,
                            )
                        )
                        feature_rows.append(candidate)
                        if index % 250 == 0 or index == len(basic):
                            LOGGER.info("%s features: %d/%d", manifest_row.run, index, len(basic))
                features = pd.DataFrame(feature_rows)
                features.to_parquet(feature_cache, index=False)
            else:
                features = pd.read_parquet(feature_cache)
            features["truth_profile"] = manifest_row.truth_profile
            all_feature_frames.append(features)

            with pysam.AlignmentFile(manifest_row.bam, "rb", threads=args.threads) as bam:
                for key, truth_meta in run_truth_catalog.items():
                    context = sequence_context_features(fasta, key.chrom, key.pos, key.ref, key.alt)
                    forced_support = count_allele_support(bam, fasta, key, min_mapq=args.min_mapq)
                    forced_features = extract_bam_features(
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
                    original = oncomine_by_key.get(key)
                    truth_observations.append(
                        {
                            "sample": manifest_row.sample,
                            "run": manifest_row.run,
                            "truth_profile": manifest_row.truth_profile,
                            "chrom": key.chrom,
                            "pos": key.pos,
                            "ref": key.ref,
                            "alt": key.alt,
                            "locus_id": key.locus_id,
                            "diagnostic_only": True,
                            "original_vcf_present": original is not None,
                            "original_called": original_call_positive(original),
                            "blind_candidate_present": key in basic_keys,
                            **truth_meta,
                            **{f"forced_{name}": value for name, value in forced_support.items()},
                            **{f"forced_{name}": value for name, value in forced_features.items()},
                        }
                    )
            qc_rows.extend(
                [
                    {"run": manifest_row.run, "metric": "labeled_blind_candidates", "value": len(basic)},
                    {"run": manifest_row.run, "metric": "positive_blind_candidates", "value": int((basic["label"] == 1).sum()) if not basic.empty else 0},
                    {"run": manifest_row.run, "metric": "negative_blind_candidates", "value": int((basic["label"] == 0).sum()) if not basic.empty else 0},
                ]
            )

    candidates_frame = pd.concat(all_feature_frames, ignore_index=True) if all_feature_frames else pd.DataFrame()
    truth_frame = pd.DataFrame(truth_observations)
    baseline_fp_frame = pd.DataFrame(baseline_false_positives)
    identity = ["sample", "run", "locus_id"]
    expected_truth_observations = sum(
        len(truth_catalogs[row.truth_profile])
        for row in manifest.itertuples(index=False)
    )
    checks = {
        "candidate_rows_nonempty": not candidates_frame.empty,
        "candidate_rows_unique": not candidates_frame.duplicated(identity).any() if not candidates_frame.empty else False,
        "candidate_both_classes": candidates_frame["label"].nunique() == 2 if "label" in candidates_frame else False,
        "truth_observations_complete": len(truth_frame) == expected_truth_observations,
        "forced_rows_diagnostic_only": bool(truth_frame["diagnostic_only"].all()) if not truth_frame.empty else False,
    }
    for name, passed in checks.items():
        qc_rows.append({"run": "ALL", "metric": f"validation_{name}", "value": passed})
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Rescue dataset validation failed: {', '.join(failed)}")

    candidate_stem = output_dir / "blind_candidate_features"
    truth_stem = output_dir / "truth_observations"
    baseline_stem = output_dir / "baseline_false_positive_calls"
    _write_frame(candidates_frame, candidate_stem)
    _write_frame(truth_frame, truth_stem)
    _write_frame(baseline_fp_frame, baseline_stem)
    qc_path = output_dir / "rescue_qc.tsv"
    pd.DataFrame(qc_rows).to_csv(qc_path, sep="\t", index=False)
    marker = {
        "validation_passed": True,
        "checks": checks,
        "candidate_rows": len(candidates_frame),
        "candidate_positive_rows": int((candidates_frame["label"] == 1).sum()),
        "candidate_negative_rows": int((candidates_frame["label"] == 0).sum()),
        "truth_observations": len(truth_frame),
        "truth_loci": int(truth_frame["locus_id"].nunique()),
        "truth_profiles": {
            profile: int((manifest["truth_profile"] == profile).sum())
            for profile in sorted(manifest["truth_profile"].unique())
        },
        "parameters": {
            "candidate_generator": args.candidate_generator,
            "min_mapq": args.min_mapq,
            "min_support": args.min_support,
            "min_alt_fraction": args.min_alt_fraction,
            "min_hp_length": args.min_hp_length,
            "max_depth": args.max_depth,
            "flow_window": args.flow_window,
            "max_reads_per_candidate": args.max_reads_per_candidate,
            "flow_enabled": not args.no_flow,
        },
    }
    (output_dir / "RESCUE_DATA_VALIDATED.json").write_text(json.dumps(marker, indent=2) + "\n")
    return candidate_stem.with_suffix(".parquet"), truth_stem.with_suffix(".parquet"), qc_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build blind BAM homopolymer rescue candidates")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--aohc-truth", required=True)
    parser.add_argument("--giab-truth", required=True)
    parser.add_argument("--giab-bed", required=True)
    parser.add_argument("--target-bed", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output-dir", default="output/rescue")
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument(
        "--candidate-generator", choices=("cigar", "mpileup"), default="cigar"
    )
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-alt-fraction", type=float, default=0.10)
    parser.add_argument("--min-hp-length", type=int, default=4)
    parser.add_argument("--max-indel-length", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=1_000_000)
    parser.add_argument("--flow-window", type=int, default=5)
    parser.add_argument("--max-reads-per-candidate", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-flow", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    for path in build_rescue_dataset(args):
        LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
