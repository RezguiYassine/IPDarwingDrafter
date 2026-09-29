"""The leader audit is R1's acceptance test, so it is checked first.

Three cases with known answers: a straight leader masked by its chord leaves
nothing behind; a curved leader masked by its chord leaves most of its ink;
the same curved leader masked along its path leaves nothing. If the audit
could not tell these apart it could not judge the fix.
"""
import numpy as np
import cv2

from tools import leader_audit as la


def canvas():
    return np.zeros((300, 300), np.uint8)


def arc_points():
    t = np.linspace(0, np.pi / 2, 200)
    return np.column_stack([60 + 150 * np.sin(t), 240 - 150 * (1 - np.cos(t))]).astype(np.int32)


def draw(img, points, thickness=2):
    cv2.polylines(img, [points.reshape(-1, 1, 2)], False, 255, thickness)
    return img > 0


def test_straight_leader_masked_by_its_chord_leaves_nothing():
    a, b = (40, 40), (240, 200)
    ink = draw(canvas(), np.array([a, b], np.int32))
    left = ink.copy()
    cv2.line(left.view(np.uint8), a, b, 0, 5)          # remove along the chord
    r = la.audit_leader(ink, left > 0, {"p1": list(a), "p2": list(b)})
    assert r["chord_on_ink"] > 0.95
    assert r["sinuosity"] < 1.05
    assert r["ink_left"] < 0.05


def test_curved_leader_masked_by_its_chord_leaves_most_of_its_ink():
    """The release-1 defect: the chord misses the curve."""
    pts = arc_points()
    ink = draw(canvas(), pts)
    left = ink.astype(np.uint8)
    a, b = tuple(pts[0]), tuple(pts[-1])
    cv2.line(left, a, b, 0, 5)                          # chord mask, as release 1 did
    r = la.audit_leader(ink, left > 0, {"p1": list(a), "p2": list(b)})
    assert r["chord_on_ink"] < la.CURVED_CHORD_ON_INK, "an arc must read as curved"
    assert r["sinuosity"] > 1.05
    assert r["ink_left"] > 0.6


def test_curved_leader_masked_along_its_path_leaves_nothing():
    """What L2 has to achieve."""
    pts = arc_points()
    ink = draw(canvas(), pts)
    left = ink.astype(np.uint8)
    cv2.polylines(left, [pts.reshape(-1, 1, 2)], False, 0, 5)   # path mask
    r = la.audit_leader(ink, left > 0, {"p1": list(pts[0]), "p2": list(pts[-1])})
    assert r["ink_left"] < 0.05


def test_disconnected_endpoints_are_left_out_not_guessed():
    ink = draw(canvas(), np.array([(40, 40), (100, 40)], np.int32))
    assert la.audit_leader(ink, ink, {"p1": [40, 40], "p2": [260, 260]}) is None


def test_short_leaders_are_skipped():
    ink = draw(canvas(), np.array([(40, 40), (50, 45)], np.int32))
    assert la.audit_leader(ink, ink, {"p1": [40, 40], "p2": [50, 45]}) is None
