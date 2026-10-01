import pandas as pd

from ion_hp_ml.training import grouped_stratified_split


def test_grouped_split_has_no_locus_leakage_and_keeps_both_classes():
    groups = pd.Series(["p1", "p1", "p2", "p2", "n1", "n1", "n2", "n2"])
    labels = pd.Series([1, 1, 1, 1, 0, 0, 0, 0])
    train, test = grouped_stratified_split(labels, groups, test_size=0.5, random_state=7)
    assert set(groups.iloc[train]).isdisjoint(set(groups.iloc[test]))
    assert set(labels.iloc[train]) == {0, 1}
    assert set(labels.iloc[test]) == {0, 1}

