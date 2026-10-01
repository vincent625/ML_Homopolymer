import pandas as pd

from ion_hp_ml.rescue import (
    normalize_truth_profiles,
    original_call_positive,
    truth_sets_for_profile,
)
from ion_hp_ml.rescue_training import confusion_without_tn
from ion_hp_ml.variants import VariantKey


def test_original_call_requires_pass_and_nonreference_gt():
    assert original_call_positive({"vcf_FILTER": "PASS", "vcf_GT": "0/1"})
    assert not original_call_positive({"vcf_FILTER": "PASS", "vcf_GT": "0/0"})
    assert not original_call_positive({"vcf_FILTER": "NOCALL", "vcf_GT": "0/1"})
    assert not original_call_positive(None)


def test_end_to_end_confusion_counts_missing_truth_as_false_negative():
    truth_called = pd.Series([True, False, False])
    negative_called = pd.Series([True, False])
    metrics = confusion_without_tn(truth_called, negative_called)
    assert metrics["TP"] == 1
    assert metrics["FN"] == 2
    assert metrics["FP"] == 1
    assert metrics["sensitivity"] == 1 / 3
    assert metrics["precision"] == 0.5


def test_truth_profiles_are_explicit_and_sample_specific():
    synthetic = VariantKey.create("1", 101, "A", "AA")
    native = VariantKey.create("2", 202, "CC", "C")
    manifest = normalize_truth_profiles(
        pd.DataFrame(
            {
                "sample": ["AOHC", "NA24385"],
                "run": ["AOHC1", "NA24385_1"],
                "bam": ["a.bam", "n.bam"],
                "vcf": ["a.vcf", "n.vcf"],
                "truth_profile": ["AOHC", "NA24385"],
            }
        )
    )
    assert list(manifest["truth_profile"]) == ["AOHC", "HG002"]
    assert truth_sets_for_profile("AOHC", {synthetic}, {native}) == (
        {synthetic},
        {native},
    )
    assert truth_sets_for_profile("HG002", {synthetic}, {native}) == (
        set(),
        {native},
    )


def test_legacy_manifest_defaults_to_aohc_truth_profile():
    manifest = normalize_truth_profiles(
        pd.DataFrame({"sample": ["AOHC"], "run": ["AOHC1"]})
    )
    assert manifest.loc[0, "truth_profile"] == "AOHC"
