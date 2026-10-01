from ion_hp_ml.context import sequence_context_features


class FakeFasta:
    references = ("chr1",)

    def __init__(self, sequence: str):
        self.sequence = sequence

    def fetch(self, chrom: str, start: int, end: int) -> str:
        return self.sequence[start:end]


def test_reference_derived_homopolymer_lengths():
    fasta = FakeFasta("CCCAAAAAGGG")
    features = sequence_context_features(fasta, "1", 3, "CA", "C", flank=3)
    assert features["reference_match"] is True
    assert features["hp_base"] == "A"
    assert features["hp_ref_len"] == 5
    assert features["hp_alt_len"] == 4
    assert features["hp_delta"] == -1
    assert features["indel_type"] == "deletion"

