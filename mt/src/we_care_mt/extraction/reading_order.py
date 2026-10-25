"""Reading-order reconstruction for single- and two-column pages."""

from __future__ import annotations

from we_care_mt.schema import ContentBlock


def sort_blocks(blocks: list[ContentBlock], two_column: bool, page_width: float) -> list[ContentBlock]:
    if not blocks:
        return blocks
    if not two_column:
        ordered = sorted(blocks, key=lambda b: (round(b.bbox[1], 1), round(b.bbox[0], 1)))
    else:
        mid = page_width / 2.0
        left = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2.0 < mid]
        right = [b for b in blocks if (b.bbox[0] + b.bbox[2]) / 2.0 >= mid]
        left = sorted(left, key=lambda b: (round(b.bbox[1], 1), round(b.bbox[0], 1)))
        right = sorted(right, key=lambda b: (round(b.bbox[1], 1), round(b.bbox[0], 1)))
        ordered = left + right
    for i, block in enumerate(ordered):
        block.reading_order = i
    return ordered


def detect_two_column(block_bboxes: list[list[float]], page_width: float, min_gap: float) -> bool:
    if len(block_bboxes) < 6:
        return False
    mid = page_width / 2.0
    left = [b for b in block_bboxes if b[2] < mid - min_gap / 2]
    right = [b for b in block_bboxes if b[0] > mid + min_gap / 2]
    spanning = [b for b in block_bboxes if b[0] < mid < b[2] and (b[2] - b[0]) > page_width * 0.6]
    if spanning:
        return False
    return len(left) >= 3 and len(right) >= 3
