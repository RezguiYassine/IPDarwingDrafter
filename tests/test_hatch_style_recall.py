"""N2 scoring: recall is measured on an area's ink, per style."""
import numpy as np

from tools import hatch_style_recall as hsr


def test_recall_counts_ink_inside_the_area_not_paper():
    ink = np.zeros((100, 100), bool)
    ink[20:80:4, 20:80] = True                     # hatch strokes inside the area
    prediction = np.zeros_like(ink)
    prediction[20:50, :] = True                    # the detector marks the top half
    square = [[20, 20], [80, 20], [80, 80], [20, 80]]
    [row] = hsr.area_scores(prediction, ink, [square], ["bold"])
    assert row["style"] == "bold"
    assert 0.45 < row["recall"] < 0.55


def test_areas_without_a_style_are_thin_and_empty_areas_are_skipped():
    ink = np.zeros((50, 50), bool)
    ink[10, 10:40] = True
    rows = hsr.area_scores(ink, ink, [[[5, 5], [45, 5], [45, 15], [5, 15]],
                                      [[30, 30], [40, 30], [40, 40]]], [])
    assert [r["style"] for r in rows] == ["thin"]


def test_summary_weights_recall_by_ink():
    rows = [{"figure": "A", "style": "bold", "ink_px": 900, "recall": 0.9},
            {"figure": "B", "style": "bold", "ink_px": 100, "recall": 0.1},
            {"figure": "B", "style": "shading", "ink_px": 50, "recall": 0.2}]
    summary = hsr.summarise(rows)
    assert summary["bold"]["recall_ink_weighted"] == 0.82
    assert summary["bold"]["areas_at_target"] == 1 and summary["bold"]["figures"] == 2
    assert summary["shading"]["areas"] == 1
