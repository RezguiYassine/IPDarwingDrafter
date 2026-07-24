"""Regression tests for Stage-2 dashed-line / bolt-circle grouping.

Dashed centre-lines and dashed bolt-circles arrive as many short disconnected
segments that inflate micro_edge_ratio and trip the fragmentation gate on valid
drawings (pilot-v3 benchmark: EP3499690A1/F0006 was gated at micro=0.51).
`_group_dashed_edges` merges them into single logical edges. These tests pin:
  * a dashed straight line collapses to ONE open dashed edge;
  * a dashed circle collapses to ONE closed dashed edge;
  * two parallel dashed lines stay SEPARATE (nearby != same line);
  * a solid long line and a lone stray dash are left untouched;
  * the pass is a no-op unless explicitly enabled.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2_strokeextraction"))
import stage2_stroke_extract as s2  # noqa: E402

_CFG = {"dashed_grouping": True}


def _seg(x0, y0, x1, y1, step=1.0):
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
    return [[x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n] for t in range(n + 1)]


class _Builder:
    """Accumulate edges with auto-created endpoint nodes."""
    def __init__(self):
        self.nodes, self.edges, self._nid, self._eid = [], [], 0, 0

    def add(self, pixels, is_closed=False):
        s, t = self._nid, self._nid + 1
        self.nodes += [{"id": s, "x": pixels[0][0], "y": pixels[0][1], "type": "endpoint"},
                       {"id": t, "x": pixels[-1][0], "y": pixels[-1][1], "type": "endpoint"}]
        self._nid += 2
        self.edges.append({"id": self._eid, "source": s, "target": t,
                           "pixels": [[float(a), float(b)] for a, b in pixels],
                           "is_closed": is_closed})
        self._eid += 1


def _dashed_line(y, x_start, n_dashes, dash=6, gap=6):
    b = []
    x = x_start
    segs = []
    for _ in range(n_dashes):
        segs.append(_seg(x, y, x + dash, y))
        x += dash + gap
    return segs


def _dashed_circle(cx, cy, r, n_dashes=8, dash_deg=14):
    segs = []
    for k in range(n_dashes):
        a0 = math.radians(360.0 * k / n_dashes)
        a1 = a0 + math.radians(dash_deg)
        segs.append([[cx + r * math.cos(a), cy + r * math.sin(a)]
                     for a in (a0, (a0 + a1) / 2, a1)])
    return segs


def _fixture():
    b = _Builder()
    for s in _dashed_line(100, 10, 5):      # dashed line A
        b.add(s)
    for s in _dashed_line(160, 10, 5):      # dashed line B, 60px away
        b.add(s)
    for s in _dashed_circle(400, 400, 60):  # dashed circle (8 arcs)
        b.add(s)
    b.add(_seg(10, 300, 130, 300))          # solid long line (not a candidate)
    b.add(_seg(500, 50, 505, 55))           # lone stray dash (ungroupable)
    return b


def test_dashed_line_and_circle_group():
    b = _fixture()
    n_before = len(b.edges)
    nodes, edges, n_groups = s2._group_dashed_edges(b.nodes, b.edges, _CFG)
    # line A, line B, circle -> 3 groups
    assert n_groups == 3, f"expected 3 groups, got {n_groups}"
    dashed = [e for e in edges if e.get("is_dashed")]
    assert len(dashed) == 3
    assert sum(1 for e in dashed if e.get("is_closed")) == 1      # the circle
    assert sum(1 for e in dashed if not e.get("is_closed")) == 2  # the two lines
    # 15 short dash edges (3 groups x 5) collapse; solid + stray survive
    assert len(edges) < n_before
    assert n_before - len(edges) == (5 + 5 + 8) - 3


def test_parallel_lines_not_merged():
    b = _fixture()
    _, edges, _ = s2._group_dashed_edges(b.nodes, b.edges, _CFG)
    open_dashed = [e for e in edges if e.get("is_dashed") and not e.get("is_closed")]
    ys = sorted(round(e["pixels"][0][1]) for e in open_dashed)
    assert ys == [100, 160], f"parallel dashed lines merged? rows={ys}"


def test_solid_line_and_stray_dash_untouched():
    b = _fixture()
    _, edges, _ = s2._group_dashed_edges(b.nodes, b.edges, _CFG)
    # the 120px solid line survives, not marked dashed
    solid = [e for e in edges
             if not e.get("is_dashed") and not e.get("is_closed")
             and s2._chain_length([(int(p[0]), int(p[1])) for p in e["pixels"]]) > 100]
    assert len(solid) == 1
    # the lone stray dash survives ungrouped
    stray = [e for e in edges if not e.get("is_dashed")
             and 5 <= s2._chain_length([(int(p[0]), int(p[1])) for p in e["pixels"]]) <= 12]
    assert len(stray) == 1


def test_disabled_is_noop():
    b = _fixture()
    n_before = len(b.edges)
    nodes, edges, n_groups = s2._group_dashed_edges(b.nodes, b.edges, {})
    assert n_groups == 0 and len(edges) == n_before


def test_circle_center_recovered():
    b = _fixture()
    _, edges, _ = s2._group_dashed_edges(b.nodes, b.edges, _CFG)
    circ = [e for e in edges if e.get("is_dashed") and e.get("is_closed")][0]
    cx, cy, r = circ["circle"]
    assert abs(cx - 400) < 3 and abs(cy - 400) < 3 and abs(r - 60) < 3


# ── floating-noise prune ─────────────────────────────────────────────────────

def _noise_fixture():
    b = _Builder()
    b.add(_seg(10, 10, 11, 10))        # 1px floating speckle
    b.add(_seg(50, 50, 52, 50))        # 2px floating speckle
    b.add(_seg(80, 80, 83, 80))        # 3px floating speckle (at boundary)
    b.add(_seg(0, 0, 100, 0))          # long real line (floating but large)
    # a connected pair sharing a node -> NOT floating, must survive
    s = b._nid
    b.nodes += [{"id": s}, {"id": s + 1}, {"id": s + 2}]
    b.edges += [
        {"id": b._eid,     "source": s,     "target": s + 1,
         "pixels": [[200.0, 200.0], [202.0, 200.0]], "is_closed": False},
        {"id": b._eid + 1, "source": s + 1, "target": s + 2,
         "pixels": [[202.0, 200.0], [204.0, 200.0]], "is_closed": False},
    ]
    return b


def test_floating_noise_pruned():
    b = _noise_fixture()
    cfg = {"prune_floating_noise": True, "floating_noise_max_len": 3.0}
    _, edges, n, pix = s2._prune_floating_noise(b.nodes, b.edges, cfg)
    assert n == 3, f"expected 3 speckle removed, got {n}"
    assert len(pix) > 0                      # returns removed pixels for iso-ignore
    # long line + the two connected (degree-2 shared node) short edges survive
    assert len(edges) == 3


def test_noise_prune_spares_connected_short_edges():
    b = _noise_fixture()
    cfg = {"prune_floating_noise": True, "floating_noise_max_len": 3.0}
    _, edges, _, _ = s2._prune_floating_noise(b.nodes, b.edges, cfg)
    lens = sorted(round(s2._chain_length([(int(p[0]), int(p[1])) for p in e["pixels"]]))
                  for e in edges)
    assert lens == [2, 2, 100], f"connected short pair or long line lost: {lens}"


def test_noise_prune_disabled_is_noop():
    b = _noise_fixture()
    n_before = len(b.edges)
    _, edges, n, pix = s2._prune_floating_noise(b.nodes, b.edges, {})
    assert n == 0 and len(edges) == n_before and pix == set()
