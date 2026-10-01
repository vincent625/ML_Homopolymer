import pysam

from ion_hp_ml.discovery import discover_indels, read_indel_events
from ion_hp_ml.intervals import IntervalSet
from ion_hp_ml.variants import VariantKey


class FakeFasta:
    references = ("chr1",)

    def __init__(self, sequence: str):
        self.sequence = sequence

    def fetch(self, chrom: str, start: int, end: int) -> str:
        assert chrom == "chr1"
        return self.sequence[start:end]


def make_read(sequence: str, cigar: str, reverse: bool = False):
    read = pysam.AlignedSegment()
    read.query_name = "discovery-test"
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = 0
    read.mapping_quality = 60
    read.cigarstring = cigar
    # SAM stores reverse-aligned SEQ in alignment/reference orientation. CIGAR
    # traversal must therefore not reverse-complement it.
    read.query_sequence = sequence
    return read


def test_read_indel_events_builds_insertion_allele():
    fasta = FakeFasta("AACGTTA")
    read = make_read("AACTGTTA", "3M1I4M")
    events = read_indel_events(read, fasta, "chr1", "chr1")
    assert events == [VariantKey.create("chr1", 3, "C", "CT")]


def test_read_indel_events_builds_deletion_allele():
    fasta = FakeFasta("AACGTTA")
    read = make_read("AATTA", "2M2D3M")
    events = read_indel_events(read, fasta, "chr1", "chr1")
    assert events == [VariantKey.create("chr1", 2, "ACG", "A")]


def test_reverse_read_cigar_allele_remains_reference_oriented():
    fasta = FakeFasta("AACGTTA")
    read = make_read("AACTGTTA", "3M1I4M", reverse=True)
    events = read_indel_events(read, fasta, "chr1", "chr1")
    assert events == [VariantKey.create("chr1", 3, "C", "CT")]


class FakeBam:
    references = ("chr1",)

    def __init__(self, reads):
        self.reads = reads

    def fetch(self, chrom, start, end):
        assert chrom == "chr1"
        return iter(self.reads)


def test_discovery_applies_candidate_allele_fraction():
    fasta = FakeFasta("AACGTTA")
    alt_reads = [make_read("AACTGTTA", "3M1I4M") for _ in range(2)]
    ref_reads = [make_read("AACGTTA", "7M") for _ in range(2)]
    rows, _ = discover_indels(
        FakeBam(alt_reads + ref_reads),
        fasta,
        IntervalSet({"1": [(0, 7)]}),
        min_mapq=20,
        min_support=2,
        min_alt_fraction=0.5,
    )
    assert len(rows) == 1
    assert rows[0]["bam_discovery_depth"] == 4
    assert rows[0]["bam_alt_fraction"] == 0.5

    filtered, metrics = discover_indels(
        FakeBam(alt_reads + ref_reads),
        fasta,
        IntervalSet({"1": [(0, 7)]}),
        min_mapq=20,
        min_support=2,
        min_alt_fraction=0.51,
    )
    assert filtered == []
    assert metrics["alleles_fraction_filtered"] == 1
