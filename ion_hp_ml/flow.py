from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .utils import reverse_complement


@dataclass(frozen=True)
class FlowTableRow:
    flow_index: int
    flow_base: str
    signal: float
    called_hp_length: int
    query_start: int
    query_end: int


def sequencing_sequence(read) -> str:
    """Return bases in the physical sequencing/flow orientation."""
    if hasattr(read, "get_forward_sequence"):
        sequence = read.get_forward_sequence()
        if sequence is not None:
            return sequence.upper()
    sequence = (read.query_sequence or "").upper()
    return reverse_complement(sequence) if read.is_reverse else sequence


def read_to_flow_table(
    read,
    flow_order: str,
    signals: Sequence[int] | None = None,
) -> list[FlowTableRow]:
    """Convert FO/ZF/ZM and the called read into one row per template flow.

    FO and ZM are indexed from the start of the run. ZF is the zero-based first
    template flow. ZM is stored in 1/256 signal units. Query coordinates in the
    returned table always refer to sequencing orientation, including reverse-
    aligned reads.
    """
    if signals is None:
        if not read.has_tag("ZM"):
            raise ValueError("Read is missing ZM")
        signals = read.get_tag("ZM")
    if not read.has_tag("ZF"):
        raise ValueError("Read is missing ZF")
    zf = int(read.get_tag("ZF"))
    if zf < 0:
        raise ValueError(f"ZF must be non-negative, got {zf}")
    if zf >= len(flow_order) or zf >= len(signals):
        raise ValueError(f"ZF {zf} is outside FO/ZM bounds")

    sequence = sequencing_sequence(read)
    cursor = 0
    table: list[FlowTableRow] = []
    upper_bound = min(len(flow_order), len(signals))
    for flow_index in range(zf, upper_bound):
        base = flow_order[flow_index].upper()
        query_start = cursor
        while cursor < len(sequence) and sequence[cursor] == base:
            cursor += 1
        table.append(
            FlowTableRow(
                flow_index=flow_index,
                flow_base=base,
                signal=float(signals[flow_index]) / 256.0,
                called_hp_length=cursor - query_start,
                query_start=query_start,
                query_end=cursor,
            )
        )
        if cursor == len(sequence):
            break
    return table


def reconstructed_sequence(table: Sequence[FlowTableRow]) -> str:
    return "".join(row.flow_base * row.called_hp_length for row in table)


def candidate_query_position(read, reference_start: int, reference_end: int) -> int | None:
    """Locate a reference homopolymer in sequencing-oriented query coordinates."""
    positions = read.get_reference_positions(full_length=True)
    hits = [
        query_position
        for query_position, reference_position in enumerate(positions)
        if reference_position is not None and reference_start <= reference_position < reference_end
    ]
    if not hits:
        center = (reference_start + reference_end - 1) / 2.0
        available = [
            (abs(reference_position - center), query_position)
            for query_position, reference_position in enumerate(positions)
            if reference_position is not None
        ]
        if not available:
            return None
        hits = [min(available)[1]]
    stored_query_position = hits[len(hits) // 2]
    query_length = len(positions)
    if read.is_reverse:
        return query_length - 1 - stored_query_position
    return stored_query_position


def find_candidate_flow(
    table: Sequence[FlowTableRow],
    query_position: int,
    expected_base: str,
) -> int | None:
    expected_base = expected_base.upper()
    for index, row in enumerate(table):
        if (
            row.flow_base == expected_base
            and row.query_start <= query_position < row.query_end
        ):
            return index
    matching: list[tuple[float, int]] = []
    for index, row in enumerate(table):
        if row.flow_base != expected_base:
            continue
        if query_position < row.query_start:
            distance = row.query_start - query_position
        elif query_position >= row.query_end:
            distance = query_position - row.query_end + 1
        else:
            distance = 0
        matching.append((distance, index))
    if not matching:
        return None
    distance, index = min(matching)
    return index if distance <= 3 else None

