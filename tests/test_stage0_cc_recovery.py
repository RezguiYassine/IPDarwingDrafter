"""
Regression tests for Stage 0's OCR-miss numeral recovery (`_recover_missed_numerals`).

easyocr localizes only ~60 % of numerals on patent line-art; the recovery pass
adds back the rest using size-banded, isolated, digit-like connected components.
The contract these tests lock in:

  1. Isolated numeral-sized digits sitting in whitespace ARE recovered.
  2. A same-sized digit embedded in dense geometry is NOT recovered (the
     isolation ring-check protects connected geometry from being removed).
  3. Recovery is a no-op when OCR found nothing (no calibration ⇒ no safe
     recovery — never runs on a drawing with no references).

No easyocr / GPU dependency: the OCR hits are faked, recovery runs on a
synthetic ink mask.

Runs under pytest, or standalone:  python tests/test_stage0_cc_recovery.py
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage0_handling_references"))
import stage0_handle_references as s0  # noqa: E402


_H, _W = 1000, 1400
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _digit(ink, text, org, scale=1.6, thick=3):
    cv2.putText(ink, text, org, _FONT, scale, 1, thick, cv2.LINE_8)


def _build_ink():
    ink = np.zeros((_H, _W), np.uint8)
    # three isolated numerals in whitespace (periphery)
    _digit(ink, "12", (120, 150))
    _digit(ink, "3", (1200, 200))
    _digit(ink, "45", (200, 850))
    # a same-size numeral buried in dense geometry (cross-hatched block)
    bx, by = 650, 500
    for off in range(-40, 80, 6):
        cv2.line(ink, (bx - 60, by + off), (bx + 120, by + off - 60), 1, 2)
    _digit(ink, "9", (bx, by))
    return ink


def _cfg():
    return {"ocr_cc_recovery": True, "ocr_cc_min_h_frac": 0.012,
            "ocr_cc_max_h_frac": 0.06, "ocr_cc_fill_lo": 0.05,
            "ocr_cc_fill_hi": 0.85, "ocr_cc_ring_max": 0.08, "ocr_max_aspect": 4.0}


def _fake_ocr_label():
    # one confirmed numeral so recovery is allowed to run (calibration gate)
    return [{"bbox": [120, 110, 60, 45], "centroid": [150, 132],
             "components": [[120, 110, 60, 45]], "ink_area": 800,
             "kind": "ocr_numeral", "text": "12", "confidence": 0.9,
             "leader_lines": []}]


def test_recovers_isolated_numerals():
    ink = _build_ink()
    rec = s0._recover_missed_numerals(ink, _fake_ocr_label(), _cfg())
    # the three whitespace digits ("3", "45", and the "12" region if not masked)
    # should be recovered; expect at least the two clearly outside the OCR box.
    cxs = [r["centroid"][0] for r in rec]
    assert any(c > 1100 for c in cxs), "isolated '3' (top-right) not recovered"
    assert any(c < 400 for c in cxs), "isolated '45' (bottom-left) not recovered"
    assert len(rec) >= 2


def test_rejects_numeral_embedded_in_geometry():
    ink = _build_ink()
    rec = s0._recover_missed_numerals(ink, _fake_ocr_label(), _cfg())
    # the "9" buried in the hatched block (~x650,y500) must NOT be recovered
    for r in rec:
        cx, cy = r["centroid"]
        assert not (560 < cx < 800 and 420 < cy < 540), \
            f"recovered a numeral embedded in geometry at ({cx:.0f},{cy:.0f})"


def test_no_recovery_without_ocr_hits():
    ink = _build_ink()
    assert s0._recover_missed_numerals(ink, [], _cfg()) == []


# ── line fragments must not be recovered as numerals ────────────────────────

def _dashed_line(ink, x0, y0, x1, y1, dash=26, gap=18, thick=3):
    """A dashed line: every dash is its own isolated, numeral-sized component."""
    import math as _m
    length = _m.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    at = 0.0
    while at < length:
        end = min(length, at + dash)
        cv2.line(ink, (int(x0 + ux * at), int(y0 + uy * at)),
                 (int(x0 + ux * end), int(y0 + uy * end)), 1, thick, cv2.LINE_8)
        at = end + gap


def test_rejects_dashed_line_segments():
    """Dashes pass every size/fill/isolation filter, so shape has to reject them."""
    ink = np.zeros((_H, _W), np.uint8)
    _digit(ink, "12", (120, 150))                   # calibration numeral
    _dashed_line(ink, 500, 300, 500, 800)           # vertical dashed line
    _dashed_line(ink, 700, 300, 1100, 700)          # diagonal, so the axis-aligned
    rec = s0._recover_missed_numerals(ink, _fake_ocr_label(), _cfg())
    on_line = [r for r in rec
               if 470 < r["centroid"][0] < 530
               or (650 < r["centroid"][0] < 1150 and 250 < r["centroid"][1] < 750)]
    assert not on_line, f"recovered {len(on_line)} dash(es) as reference numerals"


def test_elongation_gate_can_be_disabled():
    """Turning the gate off restores the old behaviour, dashes and all.

    The diagonal line is the one to check: an axis-aligned vertical dash fills
    its bounding box almost completely and the pre-existing fill window
    already rejects it, so it proves nothing about this gate.
    """
    ink = np.zeros((_H, _W), np.uint8)
    _digit(ink, "12", (120, 150))
    _dashed_line(ink, 700, 300, 1100, 700)
    def on_diagonal(labels):
        return [r for r in labels if 650 < r["centroid"][0] < 1150
                and 250 < r["centroid"][1] < 750]
    assert not on_diagonal(s0._recover_missed_numerals(ink, _fake_ocr_label(), _cfg()))
    cfg = dict(_cfg(), ocr_cc_max_elongation=0)
    assert on_diagonal(s0._recover_missed_numerals(ink, _fake_ocr_label(), cfg))


def test_round_digits_survive_the_elongation_gate():
    """The gate must not reach glyphs: '0' and '8' are nowhere near a bar."""
    ink = np.zeros((_H, _W), np.uint8)
    _digit(ink, "12", (120, 150))
    _digit(ink, "08", (900, 850))
    rec = s0._recover_missed_numerals(ink, _fake_ocr_label(), _cfg())
    assert any(c["centroid"][0] > 850 and c["centroid"][1] > 750 for c in rec), \
        "a round digit was rejected as a line fragment"


# ── clipped-fragment merging ────────────────────────────────────────────────

def _glyph_boxes(ink):
    """Exact component boxes, so a fixture label matches the glyph it stands for."""
    n, _, stats, cents = cv2.connectedComponentsWithStats(ink, 8)
    boxes = [([int(v) for v in stats[i][:4]], [float(c) for c in cents[i]],
              int(stats[i][4])) for i in range(1, n)]
    return sorted(boxes, key=lambda b: b[0][0])


def _label_for(box, centroid, area, text="5"):
    return {"bbox": list(box), "centroid": list(centroid), "components": [list(box)],
            "ink_area": area, "kind": "ocr_numeral", "text": text,
            "confidence": 0.9, "leader_lines": []}


def test_fragment_abutting_a_read_label_is_merged_not_relabelled():
    """The other half of a token joins the label that was read, keeping one text."""
    ink = np.zeros((_H, _W), np.uint8)
    _digit(ink, "5", (400, 500))                    # the part OCR boxed
    _digit(ink, "0", (438, 500))                    # the part it clipped off
    (box, centroid, area), *_ = _glyph_boxes(ink)
    label = _label_for(box, centroid, area)
    width_before = box[2]
    rec = s0._recover_missed_numerals(ink, [label], _cfg())
    assert not any(430 < r["centroid"][0] < 490 and 450 < r["centroid"][1] < 520
                   for r in rec), "the clipped half stayed a separate unknown label"
    assert label["bbox"][2] > width_before, "the label was not widened over the fragment"
    assert len(label["components"]) == 2
    assert label["text"] == "5", "merging must not disturb the text that was read"


def test_merge_is_refused_when_it_would_swallow_ink():
    """A gap with geometry in it is not intra-token whitespace."""
    ink = np.zeros((_H, _W), np.uint8)
    _digit(ink, "5", (400, 500))
    _digit(ink, "0", (500, 500))                    # further away
    (box, centroid, area), *_ = _glyph_boxes(ink)
    cv2.line(ink, (470, 400), (470, 600), 1, 3)     # geometry between them
    label = _label_for(box, centroid, area)
    before = list(label["bbox"])
    rec = s0._recover_missed_numerals(ink, [label], _cfg())
    assert label["bbox"] == before, "widened the label across intervening geometry"
    assert any(480 < r["centroid"][0] < 560 for r in rec), \
        "the far component should remain its own candidate"


def _run_standalone():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); passed += 1
        except AssertionError as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_standalone() else 1)
