from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
import pysam

from .bam_features import extract_bam_features
from .context import sequence_context_features
from .intervals import IntervalSet
from .variants import (
    VariantKey,
    label_variant,
    normalize_vcf,
    parse_candidates,
    parse_truth,
    read_vcf_contigs,
    subset_vcf,
    write_candidate_regions,
)


LOGGER = logging.getLogger("ion_hp_ml")


def _resolve_path(value: str, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path).resolve()
    manifest = pd.read_csv(path, sep="\t", dtype=str)
    required = {"sample", "run", "bam", "vcf"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
    for column in ("bam", "vcf"):
        manifest[column] = manifest[column].map(lambda value: str(_resolve_path(value, path.parent)))
    if manifest[["sample", "run"]].duplicated().any():
        raise ValueError("Manifest sample/run pairs must be unique")
    for column in ("bam", "vcf"):
        missing_paths = [value for value in manifest[column] if not Path(value).exists()]
        if missing_paths:
            raise FileNotFoundError(f"Missing {column} inputs: {missing_paths}")
    return manifest


def _qc_row(section: str, metric: str, value, sample: str = "ALL", run: str = "ALL"):
    return {"sample": sample, "run": run, "section": section, "metric": metric, "value": value}


def _validate_dataset(frame: pd.DataFrame) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str = ""):
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("dataset_nonempty", not frame.empty, f"rows={len(frame)}")
    identity = ["sample", "run", "locus_id"]
    duplicate_count = int(frame.duplicated(identity).sum()) if not frame.empty else 0
    check("candidate_rows_unique", duplicate_count == 0, f"duplicates={duplicate_count}")
    valid_label_names = {"TP", "REFERENCE", "UNKNOWN"}
    observed_names = set(frame.get("label_name", pd.Series(dtype=str)).dropna())
    check("label_names_valid", observed_names <= valid_label_names, str(sorted(observed_names)))
    unknown = frame.get("label_name", pd.Series(dtype=str)) == "UNKNOWN"
    unknown_has_label = int(frame.loc[unknown, "label"].notna().sum()) if "label" in frame else 0
    check("unknown_labels_excluded", unknown_has_label == 0, f"labeled_unknown={unknown_has_label}")
    positive_count = int((frame.get("label", pd.Series(dtype=float)) == 1).sum())
    check("positive_truth_matches_present", positive_count > 0, f"positives={positive_count}")
    check(
        "metadata_not_prefixed_as_features",
        not any(column.startswith(("vcf_", "bam_", "flow_")) for column in identity),
        "sample/run/locus_id retained as metadata only",
    )
    return checks


def build_dataset(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    fasta_path = Path(args.fasta).resolve()
    aohc_truth_path = Path(args.aohc_truth).resolve()
    giab_truth_path = Path(args.giab_truth).resolve()
    giab_bed_path = Path(args.giab_bed).resolve()
    target_bed_path = Path(args.target_bed).resolve()
    for path in (fasta_path, aohc_truth_path, giab_truth_path, giab_bed_path, target_bed_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if not Path(str(fasta_path) + ".fai").exists():
        raise FileNotFoundError(f"FASTA index is missing: {fasta_path}.fai")

    target_intervals = IntervalSet.from_bed(target_bed_path)
    benchmark_intervals = IntervalSet.from_bed(giab_bed_path)
    qc_rows: list[dict[str, object]] = []
    qc_rows.extend(
        [
            _qc_row("configuration", "flow_window", args.flow_window),
            _qc_row("configuration", "truth_flank", args.truth_flank),
            _qc_row("configuration", "max_candidates_per_run", args.max_candidates),
            _qc_row("configuration", "max_reads_per_candidate", args.max_reads_per_candidate),
            _qc_row("configuration", "flow_enabled", not args.no_flow),
        ]
    )
    candidates_by_run: dict[str, list[dict[str, object]]] = {}

    with pysam.FastaFile(str(fasta_path)) as fasta, tempfile.TemporaryDirectory(
        prefix="ion_hp_ml_", dir=output_dir
    ) as temporary_directory:
        temporary = Path(temporary_directory)
        normalized_aohc = temporary / "aohc_truth.normalized.vcf"
        truth_metrics = normalize_vcf(
            aohc_truth_path, normalized_aohc, fasta_path, fasta.references, args.bcftools
        )
        for metric, value in truth_metrics.items():
            if metric != "bcftools_stderr":
                qc_rows.append(_qc_row("normalization", f"aohc_{metric}", value))

        all_keys: list[VariantKey] = []
        for manifest_row in manifest.itertuples(index=False):
            LOGGER.info("Normalizing %s/%s candidates", manifest_row.sample, manifest_row.run)
            normalized = temporary / f"{manifest_row.run}.normalized.vcf"
            normalize_metrics = normalize_vcf(
                manifest_row.vcf, normalized, fasta_path, fasta.references, args.bcftools
            )
            candidates, candidate_metrics = parse_candidates(
                normalized, manifest_row.sample, manifest_row.run, target_intervals
            )
            candidates.sort(key=lambda row: row["_variant_key"])
            if args.max_candidates is not None:
                limit = min(args.max_candidates, len(candidates))
                if limit == 1:
                    candidates = candidates[:1]
                elif limit:
                    # A smoke run should span the panel rather than selecting
                    # only the first chromosome, which can omit every truth TP.
                    indices = [
                        round(index * (len(candidates) - 1) / (limit - 1))
                        for index in range(limit)
                    ]
                    candidates = [candidates[index] for index in indices]
            candidates_by_run[manifest_row.run] = candidates
            all_keys.extend(row["_variant_key"] for row in candidates)
            for metric, value in {**normalize_metrics, **candidate_metrics}.items():
                if metric != "bcftools_stderr":
                    qc_rows.append(
                        _qc_row("candidates", metric, value, manifest_row.sample, manifest_row.run)
                    )
            qc_rows.append(
                _qc_row(
                    "candidates", "retained_target_indels", len(candidates),
                    manifest_row.sample, manifest_row.run,
                )
            )

        if not all_keys:
            raise RuntimeError("No sequence indel candidates remained after target filtering")

        giab_contigs = read_vcf_contigs(giab_truth_path)
        regions = temporary / "candidate_regions.giab.bed"
        write_candidate_regions(all_keys, regions, giab_contigs, flank=args.truth_flank)
        giab_subset = temporary / "giab.subset.vcf"
        subset_vcf(giab_truth_path, regions, giab_subset, args.bcftools)
        normalized_giab = temporary / "giab.normalized.vcf"
        giab_metrics = normalize_vcf(
            giab_subset, normalized_giab, fasta_path, fasta.references, args.bcftools
        )
        for metric, value in giab_metrics.items():
            if metric != "bcftools_stderr":
                qc_rows.append(_qc_row("normalization", f"giab_subset_{metric}", value))

        aohc_truth = parse_truth(normalized_aohc)
        giab_truth = parse_truth(normalized_giab, require_nonreference_gt=True)
        qc_rows.extend(
            [
                _qc_row("truth", "aohc_normalized_alleles", len(aohc_truth)),
                _qc_row("truth", "giab_subset_normalized_alleles", len(giab_truth)),
            ]
        )

        output_rows: list[dict[str, object]] = []
        manifest_by_run = {row.run: row for row in manifest.itertuples(index=False)}
        for run_name, candidates in candidates_by_run.items():
            manifest_row = manifest_by_run[run_name]
            LOGGER.info("Extracting %d candidates from %s", len(candidates), run_name)
            with pysam.AlignmentFile(manifest_row.bam, "rb") as bam:
                for index, candidate in enumerate(candidates, start=1):
                    key = candidate["_variant_key"]
                    candidate.update(label_variant(key, aohc_truth, giab_truth, benchmark_intervals))
                    context = sequence_context_features(
                        fasta, candidate["chrom"], candidate["pos"], candidate["ref"], candidate["alt"]
                    )
                    candidate.update({name: value for name, value in context.items() if not name.startswith("_")})
                    bam_features = extract_bam_features(
                        bam=bam,
                        fasta=fasta,
                        chrom=candidate["chrom"],
                        hp_start0=int(context["_hp_ref_start0"]),
                        hp_end0=int(context["_hp_ref_end0"]),
                        hp_base=context["hp_base"],
                        hp_ref_len=context["hp_ref_len"],
                        hp_alt_len=context["hp_alt_len"],
                        flow_window=args.flow_window,
                        max_reads=args.max_reads_per_candidate,
                        include_flow=not args.no_flow,
                    )
                    candidate.update(bam_features)
                    candidate.pop("_variant_key", None)
                    output_rows.append(candidate)
                    if index % 100 == 0 or index == len(candidates):
                        LOGGER.info("%s: %d/%d candidates", run_name, index, len(candidates))

    frame = pd.DataFrame(output_rows)
    if "label" in frame:
        frame["label"] = pd.array(frame["label"], dtype="Int64")
    sort_columns = [column for column in ("sample", "run", "chrom", "pos", "ref", "alt") if column in frame]
    frame = frame.sort_values(sort_columns).reset_index(drop=True)
    preferred = [
        "sample", "run", "chrom", "pos", "ref", "alt", "locus_id",
        "label", "label_name", "truth_source", "reference_match", "hp_base",
        "hp_ref_len", "hp_alt_len", "hp_delta", "is_homopolymer_indel",
        "indel_type", "indel_length", "local_gc", "local_entropy",
    ]
    frame = frame[[column for column in preferred if column in frame] + [
        column for column in frame.columns if column not in preferred
    ]]

    checks = _validate_dataset(frame)
    for item in checks:
        qc_rows.append(
            _qc_row("validation", item["check"], f"{item['passed']}: {item['detail']}")
        )
    for (sample, run), group in frame.groupby(["sample", "run"], dropna=False):
        qc_rows.extend(
            [
                _qc_row("labels", "rows", len(group), sample, run),
                _qc_row("labels", "tp", int((group["label"] == 1).sum()), sample, run),
                _qc_row("labels", "reference", int((group["label"] == 0).sum()), sample, run),
                _qc_row("labels", "unknown", int(group["label"].isna().sum()), sample, run),
                _qc_row(
                    "context", "reference_mismatches",
                    int((~group["reference_match"].fillna(False).astype(bool)).sum()), sample, run,
                ),
                _qc_row(
                    "context", "homopolymer_indels",
                    int(group["is_homopolymer_indel"].fillna(False).astype(bool).sum()), sample, run,
                ),
                _qc_row(
                    "flow", "candidates_with_flow", int((group["flow_reads"] > 0).sum()), sample, run
                ),
            ]
        )

    tsv_path = output_dir / "candidate_features.tsv"
    parquet_path = output_dir / "candidate_features.parquet"
    qc_path = output_dir / "qc_summary.tsv"
    frame.to_csv(tsv_path, sep="\t", index=False, na_rep="NA")
    frame.to_parquet(parquet_path, index=False)
    pd.DataFrame(qc_rows).to_csv(qc_path, sep="\t", index=False)

    validation_passed = all(item["passed"] for item in checks)
    marker = {
        "validation_passed": validation_passed,
        "checks": checks,
        "rows": len(frame),
        "training_rows": int(frame["label"].notna().sum()),
        "unknown_rows": int(frame["label"].isna().sum()),
        "parameters": {
            "max_candidates_per_run": args.max_candidates,
            "max_reads_per_candidate": args.max_reads_per_candidate,
            "flow_window": args.flow_window,
            "truth_flank": args.truth_flank,
            "flow_enabled": not args.no_flow,
        },
    }
    (output_dir / "VALIDATION_PASSED.json").write_text(json.dumps(marker, indent=2) + "\n")
    if not validation_passed:
        failed = [item["check"] for item in checks if not item["passed"]]
        raise RuntimeError(f"Dataset validation failed: {', '.join(failed)}")
    return tsv_path, parquet_path, qc_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build candidate-level Ion Torrent homopolymer-indel features."
    )
    parser.add_argument("--manifest", required=True, help="TSV: sample, run, bam, vcf")
    parser.add_argument("--aohc-truth", required=True)
    parser.add_argument("--giab-truth", required=True)
    parser.add_argument("--giab-bed", required=True)
    parser.add_argument("--target-bed", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--bcftools", default="bcftools")
    parser.add_argument("--flow-window", type=int, default=5)
    parser.add_argument("--truth-flank", type=int, default=200)
    parser.add_argument("--max-reads-per-candidate", type=int)
    parser.add_argument("--max-candidates", type=int, help="Smoke-test limit per run")
    parser.add_argument("--no-flow", action="store_true", help="Skip ZM flow extraction")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    paths = build_dataset(args)
    for path in paths:
        LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
