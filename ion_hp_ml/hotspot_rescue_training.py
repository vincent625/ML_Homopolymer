from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .training import CATEGORICAL_FEATURES, CONTEXT_FEATURES, _make_model


LOGGER = logging.getLogger("ion_hp_ml.hotspot_rescue_training")


def grouped_locus_folds(
    frame: pd.DataFrame,
    requested_folds: int = 5,
    random_state: int = 17,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create folds with every normalized allele confined to exactly one fold."""
    locus_positive = frame.groupby("locus_id", sort=True)["label"].max().astype(bool)
    positive_loci = locus_positive.index[locus_positive].to_numpy(copy=True)
    negative_loci = locus_positive.index[~locus_positive].to_numpy(copy=True)
    folds = min(requested_folds, len(positive_loci), len(negative_loci))
    if folds < 2:
        raise ValueError("At least two positive and two negative loci are required")
    rng = np.random.default_rng(random_state)
    rng.shuffle(positive_loci)
    rng.shuffle(negative_loci)
    assignment: dict[str, int] = {}
    for index, locus in enumerate(positive_loci):
        assignment[str(locus)] = index % folds
    for index, locus in enumerate(negative_loci):
        assignment[str(locus)] = index % folds
    row_fold = frame["locus_id"].astype(str).map(assignment).to_numpy(dtype=int)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(folds):
        test = np.flatnonzero(row_fold == fold)
        train = np.flatnonzero(row_fold != fold)
        train_loci = set(frame.iloc[train]["locus_id"].astype(str))
        test_loci = set(frame.iloc[test]["locus_id"].astype(str))
        if train_loci & test_loci:
            raise AssertionError("A normalized allele crossed a model fold")
        if frame.iloc[train]["label"].nunique() != 2:
            raise ValueError(f"Training fold {fold + 1} lacks one class")
        splits.append((train, test))
    return splits


def select_addon_threshold(
    labels: pd.Series,
    original_called: pd.Series,
    probabilities: np.ndarray | pd.Series,
    fp_budget: int = 0,
) -> float:
    """Maximize rescued FNs under an incremental-FP budget on training data."""
    table = pd.DataFrame(
        {
            "label": pd.to_numeric(labels, errors="raise").astype(int),
            "original_called": original_called.fillna(False).astype(bool),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )
    table = table.loc[~table["original_called"]]
    if table.empty:
        return float("inf")
    grouped = (
        table.groupby("probability", sort=True)["label"]
        .agg(
            rescued=lambda values: int((values == 1).sum()),
            added_fp=lambda values: int((values == 0).sum()),
        )
        .sort_index(ascending=False)
    )
    grouped["cum_rescued"] = grouped["rescued"].cumsum()
    grouped["cum_added_fp"] = grouped["added_fp"].cumsum()
    eligible = grouped.loc[grouped["cum_added_fp"] <= fp_budget]
    if eligible.empty:
        return float(np.nextafter(table["probability"].max(), np.inf))
    best = eligible.loc[eligible["cum_rescued"] == eligible["cum_rescued"].max()]
    best = best.loc[best["cum_added_fp"] == best["cum_added_fp"].min()]
    return float(best.index.max())


def addon_calls(
    original_called: pd.Series,
    probabilities: np.ndarray | pd.Series,
    thresholds: np.ndarray | pd.Series | float,
) -> tuple[pd.Series, pd.Series]:
    """Return add-on and final calls; final calls can never lose an original TP."""
    original = original_called.fillna(False).astype(bool).reset_index(drop=True)
    probability = np.asarray(probabilities, dtype=float)
    threshold = np.asarray(thresholds, dtype=float)
    rescue = pd.Series((~original.to_numpy()) & (probability >= threshold))
    final = original | rescue
    return rescue, final


def _feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    context = [column for column in CONTEXT_FEATURES if column in frame]
    vcf = sorted(
        column
        for column in frame
        if column.startswith("vcf_") and column != "vcf_HS"
    )
    bam = sorted(column for column in frame if column.startswith("bam_"))
    flow = sorted(column for column in frame if column.startswith("flow_"))
    paired = sorted(column for column in frame if column.startswith("paired_delta__"))
    return {
        "A1_VCF_CONTEXT": vcf + context,
        "A2_VCF_BAM": vcf + bam + context,
        "A3_VCF_BAM_FLOW": vcf + bam + flow + context,
        "A4_PAIRED_NA": vcf + bam + flow + context + paired,
    }


def _inner_oof_probabilities(
    frame: pd.DataFrame,
    features: list[str],
    requested_folds: int,
    random_state: int,
) -> np.ndarray:
    splits = grouped_locus_folds(frame, requested_folds, random_state)
    probabilities = np.full(len(frame), np.nan, dtype=float)
    for train, test in splits:
        model = _make_model(features, frame.iloc[train]["label"])
        model.fit(frame.iloc[train][features], frame.iloc[train]["label"])
        probabilities[test] = model.predict_proba(frame.iloc[test][features])[:, 1]
    if not np.isfinite(probabilities).all():
        raise AssertionError("Inner grouped predictions are incomplete")
    return probabilities


def _call_metrics(truth_called: pd.Series, false_positive_count: int) -> dict[str, object]:
    called = truth_called.fillna(False).astype(bool)
    tp = int(called.sum())
    fn = int((~called).sum())
    fp = int(false_positive_count)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "truth_total": tp + fn,
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
    }


def _stratum_mask(frame: pd.DataFrame, stratum: str) -> pd.Series:
    if stratum == "ALL":
        return pd.Series(True, index=frame.index)
    return frame["evaluation_stratum"].eq(stratum)


def train_hotspot_addon(args: argparse.Namespace) -> Path:
    candidates_path = Path(args.candidates).resolve()
    output_root = candidates_path.parent
    marker_path = output_root / "HOTSPOT_RESCUE_DATA_VALIDATED.json"
    if not marker_path.exists() or not json.loads(marker_path.read_text()).get(
        "validation_passed"
    ):
        raise RuntimeError(f"Missing passing hotspot validation marker: {marker_path}")
    truth_path = Path(args.truth_observations).resolve()
    baseline_fp_path = output_root / "baseline_hotspot_false_positive_calls.parquet"
    candidates = pd.read_parquet(candidates_path)
    truth = pd.read_parquet(truth_path)
    baseline_fp = (
        pd.read_parquet(baseline_fp_path)
        if baseline_fp_path.exists()
        else pd.DataFrame(columns=["run", "evaluation_stratum"])
    )
    # The primary endpoint is the four AOHC runs. NA24385 is a matched
    # specificity/training control, not part of the AOHC PPV denominator.
    if "sample" in baseline_fp:
        baseline_fp = baseline_fp.loc[baseline_fp["sample"] == "AOHC"].copy()

    candidates = candidates.loc[candidates["label"].notna()].copy().reset_index(drop=True)
    candidates["label"] = candidates["label"].astype(int)
    candidates["original_called"] = candidates["original_called"].fillna(False).astype(bool)
    truth = truth.copy()
    truth["original_called"] = truth["original_called"].fillna(False).astype(bool)
    if candidates["label"].nunique() != 2:
        raise ValueError("Hotspot add-on training requires both truth classes")

    feature_sets = _feature_sets(candidates)
    if not feature_sets["A1_VCF_CONTEXT"] or not any(
        name.startswith("flow_") for name in feature_sets["A3_VCF_BAM_FLOW"]
    ):
        raise ValueError("Forced-hotspot candidates lack a requested feature modality")
    all_features = sorted(set().union(*feature_sets.values()))
    for column in all_features:
        if column in CATEGORICAL_FEATURES:
            candidates[column] = candidates[column].astype("string")
        else:
            candidates[column] = pd.to_numeric(candidates[column], errors="coerce")

    outer_splits = grouped_locus_folds(
        candidates, args.folds, args.random_state
    )
    fold_assignment = np.full(len(candidates), -1, dtype=int)
    for fold, (_, test) in enumerate(outer_splits, start=1):
        fold_assignment[test] = fold
    if (fold_assignment < 1).any():
        raise AssertionError("Each candidate must receive one held-out prediction")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = candidates[
        [
            "sample", "run", "truth_profile", "chrom", "pos", "ref", "alt",
            "locus_id", "label", "truth_source", "original_called",
            "evaluation_stratum",
        ]
    ].copy()
    predictions["oof_fold"] = fold_assignment
    threshold_rows: list[dict[str, object]] = []

    for model_index, (model_name, features) in enumerate(feature_sets.items()):
        LOGGER.info("Nested locus-grouped evaluation for %s", model_name)
        probabilities = np.full(len(candidates), np.nan, dtype=float)
        thresholds = np.full(len(candidates), np.nan, dtype=float)
        for fold, (outer_train, outer_test) in enumerate(outer_splits, start=1):
            development = candidates.iloc[outer_train].reset_index(drop=True)
            inner_probabilities = _inner_oof_probabilities(
                development,
                features,
                args.inner_folds,
                args.random_state + 100 * model_index + fold,
            )
            threshold = select_addon_threshold(
                development["label"],
                development["original_called"],
                inner_probabilities,
                args.inner_fp_budget,
            )
            model = _make_model(features, candidates.iloc[outer_train]["label"])
            model.fit(
                candidates.iloc[outer_train][features],
                candidates.iloc[outer_train]["label"],
            )
            probabilities[outer_test] = model.predict_proba(
                candidates.iloc[outer_test][features]
            )[:, 1]
            thresholds[outer_test] = threshold
            joblib.dump(model, output_dir / f"{model_name}.outer_fold{fold}.joblib")
            threshold_rows.append(
                {
                    "system": model_name,
                    "outer_fold": fold,
                    "selection_data": "INNER_LOCUS_GROUPED_OOF_ONLY",
                    "incremental_fp_budget": args.inner_fp_budget,
                    "threshold": threshold,
                    "outer_train_loci": development["locus_id"].nunique(),
                    "outer_test_loci": candidates.iloc[outer_test]["locus_id"].nunique(),
                }
            )
        if not np.isfinite(probabilities).all() or not np.isfinite(thresholds).all():
            raise AssertionError(f"Incomplete nested predictions for {model_name}")
        rescue, final = addon_calls(
            candidates["original_called"], probabilities, thresholds
        )
        predictions[f"probability_{model_name}"] = probabilities
        predictions[f"threshold_{model_name}"] = thresholds
        predictions[f"rescued_{model_name}"] = rescue.to_numpy()
        predictions[f"final_called_{model_name}"] = final.to_numpy()

    identity = ["sample", "run", "locus_id"]
    truth_predictions = truth.copy()
    for model_name in feature_sets:
        columns = identity + [
            f"probability_{model_name}",
            f"threshold_{model_name}",
            f"rescued_{model_name}",
        ]
        truth_predictions = truth_predictions.merge(
            predictions[columns], on=identity, how="left", validate="one_to_one"
        )
        rescue_column = f"rescued_{model_name}"
        truth_predictions[rescue_column] = (
            truth_predictions[rescue_column].fillna(False).astype(bool)
        )
        truth_predictions[f"final_called_{model_name}"] = (
            truth_predictions["original_called"] | truth_predictions[rescue_column]
        )

    metric_rows: list[dict[str, object]] = []
    systems = ["ORIGINAL_ONCOMINE", *feature_sets]
    for stratum in ("ALL", "HP", "NON_HP"):
        truth_subset = truth_predictions.loc[_stratum_mask(truth_predictions, stratum)]
        fp_subset = baseline_fp.loc[_stratum_mask(baseline_fp, stratum)]
        baseline_count = int(len(fp_subset))
        for system in systems:
            if system == "ORIGINAL_ONCOMINE":
                calls = truth_subset["original_called"]
                incremental_fp = 0
                rescued = 0
            else:
                calls = truth_subset[f"final_called_{system}"]
                negative_mask = (
                    (candidates["sample"] == "AOHC")
                    & (candidates["label"] == 0)
                    & (~candidates["original_called"])
                )
                if stratum != "ALL":
                    negative_mask &= candidates["evaluation_stratum"].eq(stratum)
                incremental_fp = int(
                    predictions.loc[negative_mask, f"rescued_{system}"].sum()
                )
                rescued = int(
                    (
                        ~truth_subset["original_called"]
                        & truth_subset[f"final_called_{system}"]
                    ).sum()
                )
            metric_rows.append(
                {
                    "system": system,
                    "stratum": stratum,
                    **_call_metrics(calls, baseline_count + incremental_fp),
                    "baseline_FP": baseline_count,
                    "additional_FP": incremental_fp,
                    "rescued_original_FN": rescued,
                    "lost_original_TP": 0,
                    "candidate_pr_auc": (
                        float("nan")
                        if system == "ORIGINAL_ONCOMINE"
                        else float(
                            average_precision_score(
                                candidates["label"],
                                predictions[f"probability_{system}"],
                            )
                        )
                    ),
                }
            )

    per_run_rows: list[dict[str, object]] = []
    for run, run_truth in truth_predictions.groupby("run", sort=True):
        run_baseline = int((baseline_fp.get("run", pd.Series(dtype=str)) == run).sum())
        for system in systems:
            if system == "ORIGINAL_ONCOMINE":
                calls = run_truth["original_called"]
                incremental = 0
            else:
                negative_mask = (
                    (candidates["run"] == run)
                    & (candidates["label"] == 0)
                    & (~candidates["original_called"])
                )
                incremental = int(
                    predictions.loc[negative_mask, f"rescued_{system}"].sum()
                )
                calls = run_truth[f"final_called_{system}"]
            per_run_rows.append(
                {
                    "run": run,
                    "system": system,
                    **_call_metrics(calls, run_baseline + incremental),
                    "baseline_FP": run_baseline,
                    "additional_FP": incremental,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    metrics["sensitivity_gain_vs_original"] = metrics.groupby("stratum")[
        "sensitivity"
    ].transform(lambda values: values - values.iloc[0])
    metrics["meets_predefined_gate"] = (
        (metrics["stratum"] == "ALL")
        & (metrics["rescued_original_FN"] >= args.required_rescues)
        & (metrics["additional_FP"] <= args.allowed_additional_fp)
    )
    metrics.to_csv(output_dir / "addon_ablation_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(per_run_rows).to_csv(
        output_dir / "addon_per_run_metrics.tsv", sep="\t", index=False
    )
    pd.DataFrame(threshold_rows).to_csv(
        output_dir / "nested_fold_thresholds.tsv", sep="\t", index=False
    )
    predictions.to_csv(
        output_dir / "hotspot_hp_oof_predictions.tsv", sep="\t", index=False
    )
    truth_predictions.to_csv(
        output_dir / "aohc_all_hotspot_predictions.tsv", sep="\t", index=False
    )
    (output_dir / "feature_sets.json").write_text(
        json.dumps(feature_sets, indent=2) + "\n"
    )
    marker = {
        "experiment_complete": True,
        "evaluation": "nested_locus_grouped_addon",
        "original_calls_retained": True,
        "unknown_rows_excluded": True,
        "outer_folds": len(outer_splits),
        "inner_fp_budget": args.inner_fp_budget,
        "predefined_gate": {
            "required_rescued_original_fn": args.required_rescues,
            "allowed_additional_fp": args.allowed_additional_fp,
        },
    }
    (output_dir / "HOTSPOT_ADDON_EXPERIMENT_COMPLETE.json").write_text(
        json.dumps(marker, indent=2) + "\n"
    )
    return output_dir / "addon_ablation_metrics.tsv"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Nested locus-grouped XGBoost add-on for forced hotspot HP alleles."
    )
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--truth-observations", required=True)
    parser.add_argument("--output-dir", default="output/hotspot_rescue/models")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--inner-fp-budget", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--required-rescues", type=int, default=8)
    parser.add_argument("--allowed-additional-fp", type=int, default=1)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    path = train_hotspot_addon(args)
    LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()
