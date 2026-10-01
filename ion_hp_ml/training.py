from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


LOGGER = logging.getLogger("ion_hp_ml.training")
CONTEXT_FEATURES = (
    "hp_base", "hp_ref_len", "hp_alt_len", "hp_delta", "indel_type",
    "indel_length", "local_gc", "local_entropy",
)
CATEGORICAL_FEATURES = {"vcf_TYPE", "vcf_FILTER", "vcf_GT", "hp_base", "indel_type"}


def grouped_stratified_split(
    labels: pd.Series,
    groups: pd.Series,
    test_size: float = 0.25,
    random_state: int = 17,
) -> tuple[np.ndarray, np.ndarray]:
    """Stratify unique genomic loci, then expand them back to rows."""
    table = pd.DataFrame({"label": labels.to_numpy(), "group": groups.to_numpy()})
    per_group = table.groupby("group", sort=True)["label"].agg(lambda values: sorted(set(values)))
    inconsistent = per_group[per_group.map(len) != 1]
    if not inconsistent.empty:
        raise ValueError(f"Labels differ across runs for loci: {list(inconsistent.index[:5])}")
    group_labels = per_group.map(lambda values: values[0])
    rng = np.random.default_rng(random_state)
    test_groups: set[str] = set()
    for label, class_groups in group_labels.groupby(group_labels):
        values = class_groups.index.to_numpy(copy=True)
        if len(values) < 2:
            raise ValueError(f"Class {label} has fewer than two unique loci")
        rng.shuffle(values)
        count = min(len(values) - 1, max(1, round(len(values) * test_size)))
        test_groups.update(map(str, values[:count]))
    is_test = groups.astype(str).isin(test_groups).to_numpy()
    train_indices = np.flatnonzero(~is_test)
    test_indices = np.flatnonzero(is_test)
    train_groups = set(groups.iloc[train_indices].astype(str))
    heldout_groups = set(groups.iloc[test_indices].astype(str))
    if train_groups & heldout_groups:
        raise AssertionError("A genomic locus appears in both training and test sets")
    return train_indices, test_indices


def _load_validated_dataset(path: Path) -> pd.DataFrame:
    marker_path = path.parent / "VALIDATION_PASSED.json"
    if not marker_path.exists():
        raise RuntimeError(f"Missing extraction validation marker: {marker_path}")
    marker = json.loads(marker_path.read_text())
    if not marker.get("validation_passed"):
        raise RuntimeError(f"Extraction validation did not pass: {marker_path}")
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t", na_values=["NA"])


def _feature_sets(frame: pd.DataFrame, include_context: bool) -> dict[str, list[str]]:
    vcf = sorted(column for column in frame if column.startswith("vcf_"))
    bam = sorted(column for column in frame if column.startswith("bam_"))
    flow = sorted(column for column in frame if column.startswith("flow_"))
    context = [column for column in CONTEXT_FEATURES if column in frame] if include_context else []
    return {
        "M1_VCF": vcf + context,
        "M2_VCF_BAM": vcf + bam + context,
        "M3_VCF_BAM_FLOW": vcf + bam + flow + context,
    }


def _make_model(features: list[str], y_train: pd.Series) -> Pipeline:
    categorical = [column for column in features if column in CATEGORICAL_FEATURES]
    numeric = [column for column in features if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric)
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())
    classifier = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=negatives / max(positives, 1),
        random_state=17,
        n_jobs=4,
        tree_method="hist",
    )
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("classifier", classifier)])


def _precision_at_recall(y_true: np.ndarray, probabilities: np.ndarray, target: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    eligible = precision[recall >= target]
    return float(np.max(eligible)) if len(eligible) else float("nan")


def train_ablation(args: argparse.Namespace) -> Path:
    dataset_path = Path(args.dataset).resolve()
    frame = _load_validated_dataset(dataset_path)
    frame = frame.loc[frame["label"].notna()].copy()
    frame["label"] = frame["label"].astype(int)
    if "reference_match" in frame:
        frame = frame.loc[frame["reference_match"].fillna(False).astype(bool)]
    if "is_homopolymer_indel" in frame:
        frame = frame.loc[frame["is_homopolymer_indel"].fillna(False).astype(bool)]
    if args.min_hp_length is not None:
        frame = frame.loc[frame[["hp_ref_len", "hp_alt_len"]].max(axis=1) >= args.min_hp_length]
    if frame.empty or frame["label"].nunique() != 2:
        raise ValueError("Training requires validated homopolymer rows from both classes")

    feature_sets = _feature_sets(frame, args.include_context)
    if not feature_sets["M1_VCF"] or not any(
        column.startswith("flow_") for column in feature_sets["M3_VCF_BAM_FLOW"]
    ):
        raise ValueError("Dataset is missing one or more requested feature modalities")
    for features in feature_sets.values():
        for column in features:
            if column not in CATEGORICAL_FEATURES:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                frame[column] = frame[column].astype("string")

    frame = frame.reset_index(drop=True)
    train_indices, test_indices = grouped_stratified_split(
        frame["label"], frame["locus_id"], args.test_size, args.random_state
    )
    y_train = frame.iloc[train_indices]["label"]
    y_test = frame.iloc[test_indices]["label"].to_numpy()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = frame[["sample", "run", "chrom", "pos", "ref", "alt", "locus_id", "label"]].copy()
    predictions["split"] = "train"
    predictions.iloc[test_indices, predictions.columns.get_loc("split")] = "test"
    result_rows: list[dict[str, object]] = []
    test_probabilities: dict[str, np.ndarray] = {}
    for model_name, features in feature_sets.items():
        LOGGER.info("Training %s with %d features", model_name, len(features))
        model = _make_model(features, y_train)
        model.fit(frame.iloc[train_indices][features], y_train)
        probability = model.predict_proba(frame.iloc[test_indices][features])[:, 1]
        predicted = (probability >= args.threshold).astype(int)
        test_probabilities[model_name] = probability
        predictions[f"probability_{model_name}"] = np.nan
        predictions.loc[test_indices, f"probability_{model_name}"] = probability
        result_rows.append(
            {
                "model": model_name,
                "feature_count": len(features),
                "train_rows": len(train_indices),
                "test_rows": len(test_indices),
                "train_loci": frame.iloc[train_indices]["locus_id"].nunique(),
                "test_loci": frame.iloc[test_indices]["locus_id"].nunique(),
                "pr_auc": average_precision_score(y_test, probability),
                "precision_at_threshold": precision_score(y_test, predicted, zero_division=0),
                "recall_at_threshold": recall_score(y_test, predicted, zero_division=0),
                "threshold": args.threshold,
            }
        )
        joblib.dump(model, output_dir / f"{model_name}.joblib")

    baseline = next(row for row in result_rows if row["model"] == "M1_VCF")
    matched_recall = (
        args.matched_recall
        if args.matched_recall is not None
        else float(baseline["recall_at_threshold"])
    )
    for row in result_rows:
        row["matched_recall_target"] = matched_recall
        row["precision_at_matched_recall"] = _precision_at_recall(
            y_test, test_probabilities[str(row["model"])], matched_recall
        )

    results = pd.DataFrame(result_rows)
    results.to_csv(output_dir / "ablation_results.tsv", sep="\t", index=False)
    predictions.to_csv(output_dir / "predictions.tsv", sep="\t", index=False, na_rep="NA")
    assignments = predictions[["locus_id", "split"]].drop_duplicates().sort_values("locus_id")
    assignments.to_csv(output_dir / "split_assignments.tsv", sep="\t", index=False)
    (output_dir / "feature_sets.json").write_text(json.dumps(feature_sets, indent=2) + "\n")
    return output_dir / "ablation_results.tsv"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Locus-grouped XGBoost modality ablation")
    parser.add_argument("--dataset", default="output/candidate_features.parquet")
    parser.add_argument("--output-dir", default="output/models")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--matched-recall", type=float)
    parser.add_argument("--min-hp-length", type=int, default=2)
    parser.add_argument(
        "--include-context", action="store_true",
        help="Add sequence context to all models; default is exactly VCF / +BAM / +flow",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    path = train_ablation(args)
    LOGGER.info("Wrote %s", path)


if __name__ == "__main__":
    main()

