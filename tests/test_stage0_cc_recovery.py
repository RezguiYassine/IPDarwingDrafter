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
