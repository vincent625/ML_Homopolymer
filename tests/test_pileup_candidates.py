from ion_hp_ml.intervals import IntervalSet
from ion_hp_ml.pileup_candidates import parse_mpileup_candidates


class FakeFasta:
    references = ("chr1",)

    def __init__(self, sequence: str):
        self.sequence = sequence

    def fetch(self, chrom: str, start: int, end: int) -> str:
        assert chrom == "chr1"
        return self.sequence[start:end]


def test_parse_mpileup_keeps_two_read_hp_candidate(tmp_path):
    vcf = tmp_path / "pileup.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=7>\n"
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
        '##FORMAT=<ID=ADF,Number=R,Type=Integer,Description="Forward depths">\n'
        '##FORMAT=<ID=ADR,Number=R,Type=Integer,Description="Reverse depths">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS\n"
        "chr1\t1\t.\tT\tTC\t20\t.\t.\tAD:ADF:ADR:DP\t98,2:97,1:1,1:100\n"
    )
    rows, metrics = parse_mpileup_candidates(
        vcf,
        FakeFasta("TCCCCAG"),
        IntervalSet({"1": [(0, 7)]}),
        min_support=2,
        min_hp_length=4,
    )
    assert metrics["retained_candidates"] == 1
    assert rows[0]["bam_alt_support"] == 2
    assert rows[0]["bam_alt_forward_reads"] == 1
    assert rows[0]["bam_alt_reverse_reads"] == 1
    assert rows[0]["bam_alt_fraction"] == 0.02
    assert rows[0]["hp_ref_len"] == 4
    assert rows[0]["hp_alt_len"] == 5
