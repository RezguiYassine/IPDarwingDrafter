"""Tests for the patent-sheet compositor (Tier-2 merge engine v1).

Pins the exact-ground-truth invariants that make composited sheets usable as
training data: every emitted keypoint lands on object skeleton (pure
translation preserves GT), no keypoint is mislabeled as annotation/border, and
the semantic layer separates object / text / border.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import patent_sheet_compositor as psc  # noqa: E402


def _panel(seed=0):
    """A small synthetic panel: a rectangle skeleton + its 4 corner keypoints."""
    sk = np.zeros((512, 512), np.uint8)
    import cv2
    cv2.rectangle(sk, (120, 120), (390, 360), 255, 1)
    kps = np.array([[120, 120, psc.SEM_BG], [390, 120, 2],
                    [120, 360, 2], [390, 360, 2]], np.int32)
    kps[:, 2] = 2  # corner type
    return sk, kps


_CFG = dict(cell=512, margin=40, gap=30, label_h=34,
            numerals=True, border=True, degrade=False)


def test_kps_land_on_object_skeleton():
    rng = np.random.default_rng(0)
    sk, kps, sem = psc.compose_sheet([_panel(), _panel()], rng, _CFG)
    assert len(kps) == 8, f"expected 8 kps (2 panels x 4), got {len(kps)}"
    on = sum(1 for x, y, t in kps if sk[y, x] > 0)
    assert on == len(kps), f"{len(kps) - on} kps off the skeleton"


def test_no_kp_labeled_as_annotation():
    rng = np.random.default_rng(1)
    sk, kps, sem = psc.compose_sheet([_panel(), _panel(), _panel()], rng, _CFG)
    bad = sum(1 for x, y, t in kps if sem[y, x] in (psc.SEM_TEXT, psc.SEM_BORDER))
    assert bad == 0, f"{bad} object keypoints wrongly marked as text/border"


def test_semantic_layers_present():
    rng = np.random.default_rng(2)
    _, _, sem = psc.compose_sheet([_panel(), _panel()], rng, _CFG)
    present = set(np.unique(sem).tolist())
    assert psc.SEM_OBJECT in present and psc.SEM_TEXT in present \
        and psc.SEM_BORDER in present


def test_no_border_no_numerals():
    rng = np.random.default_rng(3)
    cfg = dict(_CFG, numerals=False, border=False)
    sk, kps, sem = psc.compose_sheet([_panel(), _panel()], rng, cfg)
    assert psc.SEM_BORDER not in np.unique(sem)
    # FIG labels still present -> some text; but no reference numerals/leaders
    assert psc.SEM_OBJECT in np.unique(sem)


def test_translation_preserves_kp_count():
    rng = np.random.default_rng(4)
    _, kps, _ = psc.compose_sheet([_panel()], rng, _CFG)
    assert len(kps) == 4  # single-panel sheet keeps all 4 corners
