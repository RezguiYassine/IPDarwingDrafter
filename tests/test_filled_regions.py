"""Filled regions: what is an area, what is still a line, and what is removed.

The detector's thresholds were set against real figures (0 of 85 usable audit
figures fire). These tests hold the shapes that set them: a fill is found and
kept whole, a bold stroke and a bold ring are not fills, a thin line touching
a fill stays a line, and nothing is removed from the skeleton that the
exported polygon does not cover.
"""
import json

import cv2
import numpy as np

from stage1_preprocessing import filled_regions as fr

CFG = fr.settings({"stage1": {"filled_regions": {"enabled": True}}})


def _blank(h=400, w=600):
    return np.zeros((h, w), np.uint8)


def _areas(ink):
    return [r for r in fr.detect(ink > 0, CFG) if r.kind == "area"]


def _polygon_raster(region, shape):
    raster = np.zeros(shape, np.uint8)
    rings = [np.asarray(ring, np.int32).reshape(-1, 1, 2) for ring in region.rings]
    cv2.fillPoly(raster, rings, 1)          # even-odd: holes stay empty
    return raster > 0


def test_off_by_default():
    assert fr.settings({})["enabled"] is False


def test_a_solid_block_is_one_area_with_a_four_corner_outline():
    ink = _blank()
    cv2.rectangle(ink, (100, 100), (259, 219), 255, -1)
    [region] = _areas(ink)
    assert len(region.rings) == 1
    assert len(region.rings[0]) == 4, "a rectangle simplifies to its corners"
    routed = fr.routed_mask([region], ink.shape)
    assert routed[100:220, 100:260].all(), "corners included: the opening's rounding is undone"
    assert routed.sum() == (ink > 0).sum()


def test_a_bold_stroke_is_recorded_but_not_routed():
    ink = _blank(300, 900)
    cv2.line(ink, (50, 150), (850, 150), 255, 18)
    regions = fr.detect(ink > 0, CFG)
    assert [r.kind for r in regions] == ["bold_stroke"]
    assert not fr.routed_mask(regions, ink.shape).any(), "a line stays with the skeleton"


def test_a_bold_ring_is_not_a_filled_disc():
    ink = _blank(400, 400)
    cv2.circle(ink, (200, 200), 110, 255, 24)
    assert _areas(ink) == [], "solidity: the ring's outline encloses paper"


def test_thin_ink_never_fires():
    ink = _blank()
    for x in range(20, 580, 6):                    # dense thin hatching, 3 px lines
        cv2.line(ink, (x, 20), (x, 380), 255, 3)
    assert fr.detect(ink > 0, CFG) == []


def test_a_thin_line_touching_a_fill_stays_a_line():
    ink = _blank()
    cv2.rectangle(ink, (100, 100), (259, 219), 255, -1)
    cv2.line(ink, (260, 160), (560, 160), 255, 3)
    [region] = _areas(ink)
    routed = fr.routed_mask([region], ink.shape)
    assert not routed[:, 265:].any(), "the line beyond the fill's edge is not taken"
    assert (ink[:, 265:] > 0).sum() > 800


def test_a_label_box_inside_a_fill_is_a_hole_and_specks_are_not():
    ink = _blank()
    cv2.rectangle(ink, (100, 100), (339, 299), 255, -1)
    cv2.rectangle(ink, (180, 170), (239, 209), 0, -1)       # white label box
    ink[130, 130] = ink[260, 300] = 0                        # scan specks
    [region] = _areas(ink)
    assert len(region.rings) == 2, "outer ring plus the label box"
    raster = _polygon_raster(region, ink.shape)
    assert not raster[185:195, 195:225].any(), "the box stays paper"
    assert raster[130, 130] and raster[260, 300], "specks are filled, not holes"


def test_the_polygon_covers_what_is_removed_from_the_skeleton():
    ink = _blank()
    pts = np.array([[80, 300], [200, 60], [420, 110], [520, 330], [300, 370]], np.int32)
    cv2.fillPoly(ink, [pts], 255)
    [region] = _areas(ink)
    routed = fr.routed_mask([region], ink.shape)
    raster = cv2.dilate(_polygon_raster(region, ink.shape).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    assert (routed & ~raster).sum() / routed.sum() < 0.01
    assert (raster & ~(ink > 0)).sum() / raster.sum() < 0.03, "the fill lies on ink"


def test_ink_near_a_fill_but_not_joined_to_it_is_left_alone():
    ink = _blank()
    cv2.rectangle(ink, (100, 100), (259, 219), 255, -1)
    cv2.line(ink, (100, 226), (259, 226), 255, 2)            # 6 px below, not touching
    [region] = _areas(ink)
    assert not fr.routed_mask([region], ink.shape)[226].any()


def test_write_records_the_mask_it_removed(tmp_path):
    ink = _blank()
    cv2.rectangle(ink, (100, 100), (259, 219), 255, -1)
    cv2.line(ink, (50, 350), (550, 350), 255, 20)
    regions = fr.detect(ink > 0, CFG)
    routed = fr.routed_mask(regions, ink.shape)
    document = fr.write(regions, routed, CFG, tmp_path / "r.json", tmp_path / "r.png")
    saved = json.loads((tmp_path / "r.json").read_text())
    assert saved == document
    assert saved["schema"] == fr.SCHEMA
    assert len(saved["regions"]) == 1 and len(saved["bold_strokes"]) == 1
    reread = cv2.imread(str(tmp_path / "r.png"), cv2.IMREAD_GRAYSCALE) > 0
    assert fr.mask_digest(reread) == saved["routed_mask"]["sha256"]
    assert saved["routed_mask"]["pixels"] == int(routed.sum())
