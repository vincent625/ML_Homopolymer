from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from .training import CATEGORICAL_FEATURES, CONTEXT_FEATURES, _make_model


LOGGER = logging.getLogger("ion_hp_ml.rescue_training")


def confusion_without_tn(
    truth_called: pd.Series,
    negative_called: pd.Series,
) -> dict[str, float | int]:
    """Compute call metrics where the genomic true-negative universe is undefined."""
    truth_values = truth_called.fillna(False).astype(bool)
    negative_values = negative_called.fillna(False).astype(bool)
    tp = int(truth_values.sum())
    fn = int((~truth_values).sum())
    fp = int(negative_values.sum())
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "truth_total": tp + fn,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
    }


def _cluster_bootstrap_rate(
    truth: pd.DataFrame,
    called_column: str,
    iterations: int,
    random_state: int,
) -> tuple[float, float]:
    per_locus = truth.groupby("locus_id", sort=True)[called_column].agg(["sum", "count"])
    if per_locus.empty:
        return float("nan"), float("nan")
    values = per_locus[["sum", "count"]].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    rates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        rates[index] = sampled[:, 0].sum() / sampled[:, 1].sum()
    return tuple(map(float, np.quantile(rates, [0.025, 0.975])))


def _cluster_bootstrap_gain(
    truth: pd.DataFrame,
    model_column: str,
    iterations: int,
    random_state: int,
) -> tuple[float, float]:
    working = truth.assign(
        _difference=truth[model_column].astype(int) - truth["original_called"].astype(int)
    )
    per_locus = working.groupby("locus_id", sort=True)["_difference"].agg(["sum", "count"])
    values = per_locus[["sum", "count"]].to_numpy(dtype=float)
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(random_state)
    gains = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = values[rng.integers(0, len(values), size=len(values))]
        gains[index] = sampled[:, 0].sum() / sampled[:, 1].sum()
    return tuple(map(float, np.quantile(gains, [0.025, 0.975])))


def _feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    context = [name for name in CONTEXT_FEATURES if name in frame]
    discovery = sorted(
        name
        for name in frame
        if name.startswith(("bam_alt_", "bam_discovery_", "bam_pileup_"))
    )
    bam = sorted(name for name in frame if name.startswith("bam_"))
    flow = sorted(name for name in frame if name.startswith("flow_"))
    return {
        "R0_DISCOVERY": discovery + context,
        "R1_BAM": bam + context,
        "R2_BAM_FLOW": bam + flow + context,
    }


def _load_inputs(args: argparse.Namespace):
    candidate_path = Path(args.candidates).resolve()
    truth_path = Path(args.truth_observations).resolve()
    marker_path = candidate_path.parent / "RESCUE_DATA_VALIDATED.json"
    if not marker_path.exists() or not json.loads(marker_path.read_text()).get("validation_passed"):
        raise RuntimeError(f"Missing passing rescue validation marker: {marker_path}")
    candidates = pd.read_parquet(candidate_path)
    truth = pd.read_parquet(truth_path)
    baseline_path = candidate_path.parent / "baseline_false_positive_calls.parquet"
    baseline_fp = pd.read_parquet(baseline_path) if baseline_path.exists() else pd.DataFrame()
    return candidates, truth, baseline_fp


def _end_to_end_ap(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    probability_column: str,
) -> float:
    probabilities = candidates[probability_column].to_numpy(dtype=float)
    labels = candidates["label"].to_numpy(dtype=int)
    candidate_ids = set(
        zip(candidates["sample"], candidates["run"], candidates["locus_id"])
    )
    missing = sum(
        (row.sample, row.run, row.locus_id) not in candidate_ids
        for row in truth.itertuples(index=False)
    )
    if missing:
        probabilities = np.concatenate([probabilities, np.zeros(missing)])
        labels = np.concatenate([labels, np.ones(missing, dtype=int)])
    return float(average_precision_score(labels, probabilities))


def _threshold_for_fp_budget(
    candidates: pd.DataFrame,
    probability_column: str,
    fp_budget: int,
) -> float:
    """Post-hoc threshold maximizing emitted TPs under a tied-score FP budget."""
    grouped = (
        candidates.assign(_probability=candidates[probability_column].astype(float))
        .groupby("_probability", sort=False)["label"]
        .agg(positive=lambda values: int((values == 1).sum()),
             negative=lambda values: int((values == 0).sum()))
        .sort_index(ascending=False)
    )
    grouped["cum_positive"] = grouped["positive"].cumsum()
    grouped["cum_negative"] = grouped["negative"].cumsum()
    eligible = grouped.loc[grouped["cum_negative"] <= fp_budget]
    if eligible.empty:
        return float(np.nextafter(candidates[probability_column].max(), np.inf))
    best_positive = eligible["cum_positive"].max()
    best = eligible.loc[eligible["cum_positive"] == best_positive]
    best_negative = best["cum_negative"].min()
    best = best.loc[best["cum_negative"] == best_negative]
    return float(best.index.max())


def train_rescue_ablation(args: argparse.Namespace) -> Path:
    candidates, truth, baseline_fp = _load_inputs(args)
    candidates = candidates.copy().reset_index(drop=True)
    candidates["label"] = candidates["label"].astype(int)
    truth = truth.copy()
    truth["original_called"] = truth["original_called"].fillna(False).astype(bool)
    if args.positive_source:
        candidate_is_source = candidates["truth_source"].astype(str).str.contains(
            args.positive_source, regex=False, na=False
        )
        candidates = candidates.loc[
            (candidates["label"] == 0) | candidate_is_source
        ].reset_index(drop=True)
        truth = truth.loc[
            truth["truth_source"].astype(str).str.contains(
                args.positive_source, regex=False, na=False
            )
        ].reset_index(drop=True)
    if candidates.empty or candidates["label"].nunique() != 2:
        raise ValueError("Rescue training requires emitted candidates from both classes")

    feature_sets = _feature_sets(candidates)
    if not feature_sets["R0_DISCOVERY"] or not feature_sets["R2_BAM_FLOW"]:
        raise ValueError("Rescue candidates are missing discovery or flow features")
    all_features = sorted(set().union(*feature_sets.values()))
    for column in all_features:
        if column in CATEGORICAL_FEATURES:
            candidates[column] = candidates[column].astype("string")
        else:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce")

    # In a mixed AOHC/HG002 experiment, a synthetic AOHC allele is positive in
    # AOHC but may correctly be reference in HG002. Grouping still keeps every
    # observation of that normalized allele in one fold, preventing leakage.
    group_labels = candidates.groupby("locus_id")["label"].nunique()
    mixed_label_loci = int((group_labels > 1).sum())
    positive_loci = candidates.loc[candidates["label"] == 1, "locus_id"].nunique()
    negative_loci = candidates.loc[candidates["label"] == 0, "locus_id"].nunique()
    if positive_loci < args.min_positive_loci:
        raise ValueError(
            f"Only {positive_loci} positive loci were emitted; at least "
            f"{args.min_positive_loci} are required for a model comparison"
        )
    folds = min(args.folds, positive_loci, negative_loci)
    if folds < 2:
        raise ValueError("At least two positive and two negative emitted loci are required")
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=args.random_state
    )
    splits = list(
        splitter.split(candidates, candidates["label"], groups=candidates["locus_id"])
    )
    fold_assignment = np.full(len(candidates), -1, dtype=int)
    for fold, (_, test_indices) in enumerate(splits, start=1):
        fold_assignment[test_indices] = fold
    candidates["oof_fold"] = fold_assignment
    if (fold_assignment < 1).any():
        raise AssertionError("Every candidate must receive exactly one held-out fold")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, object]] = []
    for model_name, features in feature_sets.items():
        LOGGER.info("Generating locus-grouped OOF predictions for %s", model_name)
        probabilities = np.full(len(candidates), np.nan, dtype=float)
        for fold, (train_indices, test_indices) in enumerate(splits, start=1):
            train_groups = set(candidates.iloc[train_indices]["locus_id"])
            test_groups = set(candidates.iloc[test_indices]["locus_id"])
            if train_groups & test_groups:
                raise AssertionError("Locus leakage detected in rescue folds")
            if candidates.iloc[train_indices]["label"].nunique() != 2:
                raise ValueError(f"Training fold {fold} does not contain both classes")
            model = _make_model(features, candidates.iloc[train_indices]["label"])
            model.fit(
                candidates.iloc[train_indices][features],
                candidates.iloc[train_indices]["label"],
            )
            probabilities[test_indices] = model.predict_proba(
                candidates.iloc[test_indices][features]
            )[:, 1]
            joblib.dump(model, output_dir / f"{model_name}.fold{fold}.joblib")
        if not np.isfinite(probabilities).all():
            raise AssertionError(f"Incomplete OOF probabilities for {model_name}")
        probability_column = f"probability_{model_name}"
        call_column = f"called_{model_name}"
        candidates[probability_column] = probabilities
        candidates[call_column] = probabilities >= args.threshold

        identity = ["sample", "run", "locus_id"]
        truth = truth.merge(
            candidates[identity + [probability_column]],
            on=identity,
            how="left",
            validate="one_to_one",
        )
        truth[call_column] = truth[probability_column].fillna(0.0) >= args.threshold
        negative_calls = candidates.loc[candidates["label"] == 0, call_column]
        metrics = confusion_without_tn(truth[call_column], negative_calls)
        ci_low, ci_high = _cluster_bootstrap_rate(
            truth, call_column, args.bootstrap_iterations, args.random_state
        )
        gain_low, gain_high = _cluster_bootstrap_gain(
            truth, call_column, args.bootstrap_iterations, args.random_state
        )
        rescued = int((~truth["original_called"] & truth[call_column]).sum())
        lost = int((truth["original_called"] & ~truth[call_column]).sum())
        result_rows.append(
            {
                "system": model_name,
                "feature_count": len(features),
                "threshold": args.threshold,
                **metrics,
                "sensitivity_ci95_low_locus_bootstrap": ci_low,
                "sensitivity_ci95_high_locus_bootstrap": ci_high,
                "sensitivity_gain_vs_original": metrics["sensitivity"] - truth["original_called"].mean(),
                "gain_ci95_low_locus_bootstrap": gain_low,
                "gain_ci95_high_locus_bootstrap": gain_high,
                "rescued_original_fn": rescued,
                "lost_original_tp": lost,
                "fp_per_run": metrics["FP"] / truth["run"].nunique(),
                "candidate_pr_auc": float(
                    average_precision_score(candidates["label"], probabilities)
                ),
                "end_to_end_pr_auc_missing_truth_zero": _end_to_end_ap(
                    candidates, truth, probability_column
                ),
            }
        )

    baseline_negative = pd.Series(
        np.ones(len(baseline_fp), dtype=bool), dtype=bool
    )
    baseline_metrics = confusion_without_tn(truth["original_called"], baseline_negative)
    base_ci_low, base_ci_high = _cluster_bootstrap_rate(
        truth, "original_called", args.bootstrap_iterations, args.random_state
    )
    result_rows.insert(
        0,
        {
            "system": "ORIGINAL_ONCOMINE",
            "feature_count": 0,
            "threshold": float("nan"),
            **baseline_metrics,
            "sensitivity_ci95_low_locus_bootstrap": base_ci_low,
            "sensitivity_ci95_high_locus_bootstrap": base_ci_high,
            "sensitivity_gain_vs_original": 0.0,
            "gain_ci95_low_locus_bootstrap": 0.0,
            "gain_ci95_high_locus_bootstrap": 0.0,
            "rescued_original_fn": 0,
            "lost_original_tp": 0,
            "fp_per_run": baseline_metrics["FP"] / truth["run"].nunique(),
            "candidate_pr_auc": float("nan"),
            "end_to_end_pr_auc_missing_truth_zero": float("nan"),
        },
    )

    metrics_frame = pd.DataFrame(result_rows)
    baseline_fp_per_run = float(metrics_frame.iloc[0]["fp_per_run"])
    metrics_frame["additional_fp_per_run_vs_original"] = (
        metrics_frame["fp_per_run"] - baseline_fp_per_run
    )
    metrics_frame["meets_exploratory_success_gate"] = (
        (metrics_frame["sensitivity_gain_vs_original"] >= 0.10)
        & (metrics_frame["additional_fp_per_run_vs_original"] <= 2.0)
    )

    by_source_rows: list[dict[str, object]] = []
    for truth_source, source_truth in truth.groupby("truth_source", sort=True):
        for system, call_column in [
            ("ORIGINAL_ONCOMINE", "original_called"),
            *[(name, f"called_{name}") for name in feature_sets],
        ]:
            called = source_truth[call_column].fillna(False).astype(bool)
            by_source_rows.append(
                {
                    "truth_source": truth_source,
                    "system": system,
                    "TP": int(called.sum()),
                    "FN": int((~called).sum()),
                    "truth_total": len(called),
                    "sensitivity": float(called.mean()),
                }
            )

    # Exploratory only: choose an operating point after seeing OOF labels, with
    # the total FP budget fixed to original FP + two additional FPs per run.
    fp_budget = int(baseline_metrics["FP"] + 2 * truth["run"].nunique())
    posthoc_rows: list[dict[str, object]] = []
    for model_name in feature_sets:
        probability_column = f"probability_{model_name}"
        threshold = _threshold_for_fp_budget(candidates, probability_column, fp_budget)
        candidate_calls = candidates[probability_column] >= threshold
        identity = ["sample", "run", "locus_id"]
        posthoc_truth = truth[identity + ["original_called"]].merge(
            candidates[identity + [probability_column]],
            on=identity,
            how="left",
            validate="one_to_one",
        )
        posthoc_truth["called"] = posthoc_truth[probability_column].fillna(0.0) >= threshold
        negative_calls = candidate_calls[candidates["label"] == 0]
        metrics = confusion_without_tn(posthoc_truth["called"], negative_calls)
        posthoc_rows.append(
            {
                "system": model_name,
                "selection": "POSTHOC_MAX_TP_WITH_FP_BUDGET",
                "fp_budget_total": fp_budget,
                "threshold": threshold,
                **metrics,
                "rescued_original_fn": int(
                    (~posthoc_truth["original_called"] & posthoc_truth["called"]).sum()
                ),
                "lost_original_tp": int(
                    (posthoc_truth["original_called"] & ~posthoc_truth["called"]).sum()
                ),
                "fp_per_run": metrics["FP"] / truth["run"].nunique(),
            }
        )

    per_run_rows: list[dict[str, object]] = []
    for run, run_truth in truth.groupby("run", sort=True):
        run_baseline_fp = (
            pd.Series(np.ones(int((baseline_fp.get("run", pd.Series(dtype=str)) == run).sum()), dtype=bool))
            if not baseline_fp.empty
            else pd.Series(dtype=bool)
        )
        per_run_rows.append(
            {"run": run, "system": "ORIGINAL_ONCOMINE", **confusion_without_tn(run_truth["original_called"], run_baseline_fp)}
        )
        for model_name in feature_sets:
            call_column = f"called_{model_name}"
            run_negatives = candidates.loc[
                (candidates["run"] == run) & (candidates["label"] == 0), call_column
            ]
            per_run_rows.append(
                {"run": run, "system": model_name, **confusion_without_tn(run_truth[call_column], run_negatives)}
            )

    per_sample_rows: list[dict[str, object]] = []
    for sample, sample_truth in truth.groupby("sample", sort=True):
        sample_baseline_fp = (
            pd.Series(
                np.ones(
                    int(
                        (
                            baseline_fp.get("sample", pd.Series(dtype=str))
                            == sample
                        ).sum()
                    ),
                    dtype=bool,
                )
            )
            if not baseline_fp.empty
            else pd.Series(dtype=bool)
        )
        per_sample_rows.append(
            {
                "sample": sample,
                "system": "ORIGINAL_ONCOMINE",
                **confusion_without_tn(
                    sample_truth["original_called"], sample_baseline_fp
                ),
            }
        )
        for model_name in feature_sets:
            call_column = f"called_{model_name}"
            sample_negatives = candidates.loc[
                (candidates["sample"] == sample) & (candidates["label"] == 0),
                call_column,
            ]
            per_sample_rows.append(
                {
                    "sample": sample,
                    "system": model_name,
                    **confusion_without_tn(
                        sample_truth[call_column], sample_negatives
                    ),
                }
            )

    metrics_path = output_dir / "end_to_end_metrics.tsv"
    metrics_frame.to_csv(metrics_path, sep="\t", index=False, na_rep="NA")
    pd.DataFrame(by_source_rows).to_csv(
        output_dir / "metrics_by_truth_source.tsv", sep="\t", index=False
    )
    pd.DataFrame(posthoc_rows).to_csv(
        output_dir / "posthoc_fp_budget_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(per_run_rows).to_csv(
        output_dir / "per_run_metrics.tsv", sep="\t", index=False, na_rep="NA"
    )
    pd.DataFrame(per_sample_rows).to_csv(
        output_dir / "per_sample_metrics.tsv", sep="\t", index=False, na_rep="NA"
    )
    candidates.to_csv(
        output_dir / "candidate_oof_predictions.tsv", sep="\t", index=False, na_rep="NA"
    )
    truth.to_csv(
        output_dir / "truth_end_to_end_predictions.tsv", sep="\t", index=False, na_rep="NA"
    )
    assignments = candidates[["locus_id", "oof_fold"]].drop_duplicates()
    assignments.to_csv(output_dir / "locus_fold_assignments.tsv", sep="\t", index=False)
    (output_dir / "feature_sets.json").write_text(json.dumps(feature_sets, indent=2) + "\n")
    (output_dir / "RESCUE_EXPERIMENT_COMPLETE.json").write_text(
        json.dumps(
            {
                "complete": True,
                "locus_leakage": False,
                "folds": folds,
                "threshold": args.threshold,
                "truth_observations": len(truth),
                "candidate_rows": len(candidates),
                "positive_source": args.positive_source,
                "mixed_label_loci": mixed_label_loci,
            },
            indent=2,
        )
        + "\n"
    )
    return metrics_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate blind homopolymer rescue end to end")
    parser.add_argument("--candidates", default="output/rescue/blind_candidate_features.parquet")
    parser.add_argument("--truth-observations", default="output/rescue/truth_observations.parquet")
    parser.add_argument("--output-dir", default="output/rescue/models")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min-positive-loci", type=int, default=5)
    parser.add_argument(
        "--positive-source",
        help="Optional truth_source substring; excludes other positive sources but retains negatives",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    LOGGER.info("Wrote %s", train_rescue_ablation(args))


if __name__ == "__main__":
    main()
