from array import array

import pysam

from ion_hp_ml.flow import read_to_flow_table, reconstructed_sequence, sequencing_sequence
from ion_hp_ml.utils import reverse_complement


FLOW_ORDER = "TACGTACGTACGTACG"
SEQUENCE = "AACCCGTT"
ZF = 1


def make_read(reverse: bool = False):
    read = pysam.AlignedSegment()
    read.query_name = "flow-test"
    read.flag = 16 if reverse else 0
    read.reference_id = 0
    read.reference_start = 100
    read.mapping_quality = 60
    read.cigarstring = f"{len(SEQUENCE)}M"
    read.query_sequence = reverse_complement(SEQUENCE) if reverse else SEQUENCE
    read.set_tag("ZF", ZF, value_type="i")
    signals = [0] * len(FLOW_ORDER)
    signals[1] = 2 * 256
    signals[2] = 3 * 256
    signals[3] = 1 * 256
    signals[4] = 2 * 256
    read.set_tag("ZM", array("h", signals))
    return read


def test_forward_read_reconstructs_sequence():
    read = make_read(reverse=False)
    table = read_to_flow_table(read, FLOW_ORDER)
    assert reconstructed_sequence(table) == SEQUENCE
    assert sequencing_sequence(read) == SEQUENCE
    assert [row.called_hp_length for row in table[:4]] == [2, 3, 1, 2]


def test_zf_is_first_template_flow():
    table = read_to_flow_table(make_read(), FLOW_ORDER)
    assert table[0].flow_index == ZF
    assert table[0].flow_base == FLOW_ORDER[ZF]
    assert table[0].query_start == 0


def test_reverse_read_uses_sequencing_orientation():
    read = make_read(reverse=True)
    assert read.query_sequence == reverse_complement(SEQUENCE)
    assert sequencing_sequence(read) == SEQUENCE
    table = read_to_flow_table(read, FLOW_ORDER)
    assert reconstructed_sequence(table) == SEQUENCE

