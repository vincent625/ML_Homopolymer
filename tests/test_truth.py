from ion_hp_ml.intervals import IntervalSet
from ion_hp_ml.variants import VariantKey, label_variant, normalize_variant_python


class FakeFasta:
    references = ("chr1",)

    def __init__(self, sequence: str):
        self.sequence = sequence

    def fetch(self, chrom: str, start: int, end: int) -> str:
        assert chrom == "chr1"
        return self.sequence[start:end]


def test_homopolymer_representations_normalize_to_same_allele():
    fasta = FakeFasta("CAAAAAG")
    left = normalize_variant_python(fasta, "chr1", 1, "CA", "C")
    shifted = normalize_variant_python(fasta, "1", 3, "AA", "A")
    assert left == shifted


def test_aohc_synthetic_truth_is_positive():
    key = VariantKey.create("chr10", 89690818, "TTA", "T")
    label = label_variant(key, {key}, set(), IntervalSet({"10": [(0, 100_000_000)]}))
    assert label == {"label": 1, "label_name": "TP", "truth_source": "AOHC_SYNTHETIC"}


def test_giab_confident_reference_is_negative():
    key = VariantKey.create("chr1", 151, "A", "AT")
    label = label_variant(key, set(), set(), IntervalSet({"1": [(100, 200)]}))
    assert label["label"] == 0
    assert label["label_name"] == "REFERENCE"


def test_outside_giab_benchmark_is_unknown():
    key = VariantKey.create("1", 251, "A", "AT")
    label = label_variant(key, set(), set(), IntervalSet({"chr1": [(100, 200)]}))
    assert label["label"] is None
    assert label["label_name"] == "UNKNOWN"


def test_interval_set_exposes_merged_intervals():
    intervals = IntervalSet({"chr1": [(10, 20), (15, 30)], "2": [(5, 8)]})
    assert list(intervals.iter_intervals()) == [("1", 10, 30), ("2", 5, 8)]
