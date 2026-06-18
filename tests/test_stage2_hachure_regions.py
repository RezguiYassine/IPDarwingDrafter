"""
Regression tests for the Stage 2 parametric hatch-region detector
(`_aggregate_hachure_regions` + `_hatch_region_params`).

Hachures used to be vectorized line-by-line (≈45 % of all primitives; cross-hatch
shattered into fragment chaos at every intersection). The detector collapses a
block of removed hatch lines into ONE region descriptor — boundary + angle(s) +
spacing — for native HATCH export. These tests lock in:

  1. A single-angle fill → one region, correct angle & spacing, not cross-hatch.
  2. A cross-hatch fill → one region flagged double with two angles.
  3. Two spatially separate fills → two distinct regions (no over-merge).

Runs under pytest, or standalone:  python tests/test_stage2_hachure_regions.py
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage2_strokeextraction"))
import stage2_stroke_extract as s2  # noqa: E402

_CFG = {"hachure_region_dilate": 5, "hachure_region_min_lines": 6,
        "hachure_crosshatch_ratio": 0.45}


def _line(x0, y0, x1, y1, step=1.0):
    n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step))
    return [[x0 + (x1 - x0) * t / n, y0 + (y1 - y0) * t / n] for t in range(n + 1)]


def _family(angle_deg, spacing, count, length, origin=(40, 40)):
    """A set of `count` parallel lines at `angle_deg`, `spacing` apart."""
    th = math.radians(angle_deg)
    dx, dy = math.cos(th), math.sin(th)
    px, py = -dy, dx
    ox, oy = origin
    lines = []
    for k in range(count):
        bx, by = ox + px * k * spacing, oy + py * k * spacing
        lines.append(_line(bx, by, bx + dx * length, by + dy * length))
    return [{"pixels": ln} for ln in lines]


def test_single_angle_region():
    rh = _family(45.0, 10.0, 14, 120, origin=(40, 40))
    regs = s2._aggregate_hachure_regions(rh, (256, 256), _CFG)
    assert regs, "no region detected"
    r = max(regs, key=lambda x: x["n_lines"])
    assert not r["double"], f"single hatch flagged as cross-hatch: {r['angles']}"
    assert s2._angle_delta_deg(r["angles"][0], 45.0) < 10.0, r["angles"]
    assert abs(r["spacing"] - 10.0) < 3.0, r["spacing"]


def test_cross_hatch_region():
    # Both families must cover the SAME square so they cross densely (one region).
    # _family offsets 90° lines toward -x, so start them at the right edge.
    rh = _family(0.0, 12.0, 12, 140, origin=(40, 40)) + \
         _family(90.0, 12.0, 12, 140, origin=(180, 40))
    regs = s2._aggregate_hachure_regions(rh, (256, 256), _CFG)
    assert regs
    r = max(regs, key=lambda x: x["n_lines"])
    assert r["double"], f"cross-hatch not flagged double: {r['angles']}"
    got = sorted(r["angles"])
    assert any(s2._angle_delta_deg(a, 0.0) < 12 for a in got), got
    assert any(s2._angle_delta_deg(a, 90.0) < 12 for a in got), got


def test_two_separate_regions_not_merged():
    rh = _family(45.0, 10.0, 12, 80, origin=(30, 30)) + \
         _family(45.0, 10.0, 12, 80, origin=(500, 500))
    regs = s2._aggregate_hachure_regions(rh, (700, 700), _CFG)
    big = [r for r in regs if r["n_lines"] >= 6]
    assert len(big) >= 2, f"expected 2 separate regions, got {len(big)}"


def _run():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    ok = 0
    for t in tests:
        try:
            t(); print(f"  PASS  {t.__name__}"); ok += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{ok}/{len(tests)} passed")
    return ok == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
