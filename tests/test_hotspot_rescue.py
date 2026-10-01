import numpy as np
import pandas as pd

from ion_hp_ml.hotspot_rescue import filter_reason_features
from ion_hp_ml.hotspot_rescue_training import (
    addon_calls,
    grouped_locus_folds,
    select_addon_threshold,
)


def test_filter_reason_features_extracts_tvc_failure_modes():
    features = filter_reason_features(
        ".&PREDICTIONSHIFTx0.243571>0.2&STRINGENCY&QualityScore<12"
    )
    assert features["vcf_reason_prediction_shift"] == 1
    assert features["vcf_prediction_shift_value"] == 0.243571
    assert features["vcf_reason_stringency"] == 1
    assert features["vcf_reason_quality_score"] == 1


def test_addon_rule_never_loses_original_calls():
    original = pd.Series([True, False, False])
    rescue, final = addon_calls(original, [0.01, 0.9, 0.1], 0.5)
    assert list(rescue) == [False, True, False]
    assert list(final) == [True, True, False]


def test_threshold_honors_zero_incremental_fp_budget():
    labels = pd.Series([1, 1, 0, 0])
    original = pd.Series([False, False, False, False])
    threshold = select_addon_threshold(labels, original, [0.95, 0.8, 0.7, 0.2], 0)
    assert threshold == 0.8
    calls = np.asarray([0.95, 0.8, 0.7, 0.2]) >= threshold
    assert int(((labels == 0) & calls).sum()) == 0
    assert int(((labels == 1) & calls).sum()) == 2


def test_nested_fold_helper_has_no_locus_leakage():
    frame = pd.DataFrame(
        {
            "locus_id": [
                "p1", "p1", "p2", "p2", "p3", "p3",
                "n1", "n1", "n2", "n2", "n3", "n3",
            ],
            "label": [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        }
    )
    splits = grouped_locus_folds(frame, requested_folds=3, random_state=3)
    assert len(splits) == 3
    for train, test in splits:
        assert set(frame.iloc[train].locus_id).isdisjoint(set(frame.iloc[test].locus_id))
        assert 1 in set(frame.iloc[test].label)
