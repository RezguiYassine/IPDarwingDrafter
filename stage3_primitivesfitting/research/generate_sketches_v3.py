"""
generate_sketches_v3.py
=======================
AP3 Vectorization Pipeline — Free2CAD Per-Edge Training Data Generator (v3)

This is a ground-up rewrite of the data generator following Priority 1
("per-edge training") and Priority 3 ("CIRCLE realism") from the Free2CAD
investigation roadmap.

KEY DIFFERENCE FROM v2
----------------------
v2 produced multi-stroke sketches with a sequence of commands per sample.
The model was trained on (sequence_of_strokes → sequence_of_commands) but
infers on a single edge at a time, creating a train/inference distribution
mismatch that caused systematic underperformance vs RANSAC.

v3 produces ONE STROKE + ONE COMMAND per sample. The shape distribution
matches the real Stage-2 input distribution (measured on
Picture1_skeleton_graph.json: 817 edges):

  2-point edges          : 69 %  →  emitted as LINE
  3-5 point edges        : 17 %  →  mostly LINE, some POLYLINE
  6-15 point edges       :  3 %  →  LINE / ARC / POLYLINE mix
  16+ point edges        : 11 %  →  CIRCLE (closed) or LINE / ARC / POLYLINE

OUTPUT FORMAT (one JSON file per sample)
----------------------------------------
{
  "stroke":  [[x0,y0], [x1,y1], ...],          # single stroke
  "command": {"type": "LINE",   "start": [x,y], "end": [x,y]}
           | {"type": "ARC",    "center": [x,y], "radius": r,
                                "start_angle": deg, "end_angle": deg}
           | {"type": "CIRCLE", "center": [x,y], "radius": r}
           | {"type": "POLYLINE"}              # parameters meaningless
}

All coordinates normalised to the [0, 1] canvas. Strokes fill the canvas
(centred, with a 10 % margin) — this matches how the inference wrapper
encodes a single edge.

CIRCLE REALISM (Priority 3)
---------------------------
Real closed loops from Stage-2 skeletons have:
  • 30-60 points on a small radius (median ~50 pts on 20 px span)
  • Non-uniform angular spacing (skeletonisation creates clusters and gaps)
  • 5-10 % radial wobble of the radius (vs 0.5-1.5 % in v2)

This generator reproduces all three properties via _gen_realistic_circle.

Author : Yassine Rezgui — HAW Landshut / IP DrawingDrafter
"""

from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

import numpy as np


# ─── Distribution: target shares per stroke length bucket ─────────────────────
#
# Each "type" corresponds to one of the synthetic shape families below.
# The shares are chosen to MATCH the real Stage-2 input distribution.
# The model trained on this should see, at training time, the same mix of
# stroke lengths and primitive types it encounters at inference.
#
# Class summary (after the per-edge rebalance):
#   LINE     ≈ 60 %     (40% short + 12% mid + 8% long)
#   ARC      ≈ 12 %
#   CIRCLE   ≈ 13 %
#   POLYLINE ≈ 15 %     (curves + zigzags)

EDGE_KINDS = {
    "line_short":      0.40,   # 2-point exact lines
    "line_mid":        0.12,   #  3-8 point near-straight lines (with noise)
    "line_long":       0.08,   #  9-25 point straight lines (with noise)
    "arc_mid":         0.06,   #  6-15 point arcs
    "arc_long":        0.06,   #  16-40 point arcs
    "circle_realistic":0.13,   #  30-60 point closed loops
    "polyline_curve":  0.10,   #  smooth polylines
    "polyline_zigzag": 0.05,   #  sharp polylines
}
_KINDS = list(EDGE_KINDS.keys())
_PROBS = list(EDGE_KINDS.values())
assert abs(sum(_PROBS) - 1.0) < 1e-6, "Edge-kind probabilities must sum to 1.0"


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def _add_noise(pts: np.ndarray, std: float,
               rng: np.random.Generator) -> np.ndarray:
    """Additive isotropic Gaussian noise on 2D points."""
    return pts + rng.normal(0.0, std, pts.shape)


def _sample_line(p1, p2, n_pts: int) -> np.ndarray:
    """Sample n_pts uniformly along a line p1 → p2."""
    t = np.linspace(0, 1, n_pts)
    return np.array(p1)[None] + t[:, None] * (np.array(p2) - np.array(p1))


def _sample_arc(cx: float, cy: float, r: float,
                sa_deg: float, ea_deg: float, n_pts: int) -> np.ndarray:
    """Sample n_pts uniformly along a circular arc (angles in degrees)."""
    angles = np.linspace(np.radians(sa_deg), np.radians(ea_deg), n_pts)
    return np.column_stack([cx + r * np.cos(angles),
                            cy + r * np.sin(angles)])


def _normalise_to_canvas(stroke: np.ndarray,
                         margin: float = 0.10) -> tuple[np.ndarray, dict]:
    """
    Centre the stroke on (0.5, 0.5) and isotropically scale it so the
    larger axis spans (1 - 2*margin). Returns the normalised stroke and
    the transform parameters (so command parameters can be remapped too).
    """
    mn = stroke.min(axis=0)
    mx = stroke.max(axis=0)
    span = float((mx - mn).max())
    if span < 1e-9:
        # Degenerate stroke (single point repeated). Place at canvas centre.
        return np.full_like(stroke, 0.5), {
            "centre": (mn + mx) / 2.0,
            "scale":  1.0,
            "target": 0.5,
        }
    target_span = 1.0 - 2.0 * margin
    scale       = target_span / span
    centre_in   = (mn + mx) / 2.0
    centred     = (stroke - centre_in) * scale + 0.5
    return centred, {
        "centre": centre_in,
        "scale":  scale,
        "target": 0.5,
    }


def _remap_point(pt, transform) -> list:
    """Apply the same canvas transform to a point lying in stroke space."""
    p = (np.array(pt) - transform["centre"]) * transform["scale"] + transform["target"]
    return [round(float(p[0]), 5), round(float(p[1]), 5)]


def _remap_length(v: float, transform) -> float:
    """Apply the canvas scale to a scalar length (e.g. a radius)."""
    return round(float(v) * transform["scale"], 5)


# ─── Per-kind generators ──────────────────────────────────────────────────────
#
# Each generator returns (stroke_pts_in_unit_square, command_dict) BEFORE
# canvas-normalisation. Canvas-normalisation is applied uniformly afterwards
# in _generate_sample. This keeps each generator focused on shape geometry.

def _gen_line_short(rng: np.random.Generator,
                    noise_std: float) -> tuple[np.ndarray, dict]:
    """2-point exact line. The dominant edge type in real Stage-2 input."""
    p1 = rng.uniform(0.10, 0.90, 2)
    p2 = rng.uniform(0.10, 0.90, 2)
    # Ensure a minimum length to avoid trivial degenerate samples
    while np.linalg.norm(p2 - p1) < 0.10:
        p2 = rng.uniform(0.10, 0.90, 2)
    pts = np.stack([p1, p2])
    cmd = {"type": "LINE",
           "start": p1.tolist(),
           "end":   p2.tolist()}
    return pts, cmd


def _gen_line_mid(rng: np.random.Generator,
                  noise_std: float) -> tuple[np.ndarray, dict]:
    """3-8 point near-straight line with light noise."""
    n_pts = int(rng.integers(3, 9))
    p1 = rng.uniform(0.08, 0.92, 2)
    p2 = rng.uniform(0.08, 0.92, 2)
    while np.linalg.norm(p2 - p1) < 0.20:
        p2 = rng.uniform(0.08, 0.92, 2)
    pts = _sample_line(p1, p2, n_pts)
    pts = _add_noise(pts, noise_std, rng)
    # Command params reflect the IDEAL geometry, not the noisy stroke
    cmd = {"type": "LINE",
           "start": p1.tolist(),
           "end":   p2.tolist()}
    return pts, cmd


def _gen_line_long(rng: np.random.Generator,
                   noise_std: float) -> tuple[np.ndarray, dict]:
    """9-25 point straight line with realistic stroke noise."""
    n_pts = int(rng.integers(9, 26))
    p1 = rng.uniform(0.05, 0.95, 2)
    p2 = rng.uniform(0.05, 0.95, 2)
    while np.linalg.norm(p2 - p1) < 0.40:
        p2 = rng.uniform(0.05, 0.95, 2)
    pts = _sample_line(p1, p2, n_pts)
    pts = _add_noise(pts, noise_std * 1.5, rng)
    cmd = {"type": "LINE",
           "start": p1.tolist(),
           "end":   p2.tolist()}
    return pts, cmd


def _gen_arc_mid(rng: np.random.Generator,
                 noise_std: float) -> tuple[np.ndarray, dict]:
    """6-15 point arc — typical for curved fillets and short bends."""
    n_pts = int(rng.integers(6, 16))
    cx, cy = rng.uniform(0.30, 0.70, 2)
    r      = float(rng.uniform(0.15, 0.30))
    sa     = float(rng.uniform(0, 360))
    sweep  = float(rng.uniform(45, 270)) * rng.choice([-1, 1])
    ea     = sa + sweep
    pts = _sample_arc(cx, cy, r, sa, ea, n_pts)
    pts = _add_noise(pts, noise_std, rng)
    cmd = {"type":        "ARC",
           "center":      [cx, cy],
           "radius":      r,
           "start_angle": sa,
           "end_angle":   ea}
    return pts, cmd


def _gen_arc_long(rng: np.random.Generator,
                  noise_std: float) -> tuple[np.ndarray, dict]:
    """16-40 point arc — longer curved profiles."""
    n_pts = int(rng.integers(16, 41))
    cx, cy = rng.uniform(0.30, 0.70, 2)
    r      = float(rng.uniform(0.20, 0.35))
    sa     = float(rng.uniform(0, 360))
    sweep  = float(rng.uniform(60, 300)) * rng.choice([-1, 1])
    ea     = sa + sweep
    pts = _sample_arc(cx, cy, r, sa, ea, n_pts)
    pts = _add_noise(pts, noise_std * 1.2, rng)
    cmd = {"type":        "ARC",
           "center":      [cx, cy],
           "radius":      r,
           "start_angle": sa,
           "end_angle":   ea}
    return pts, cmd


def _gen_realistic_circle(rng: np.random.Generator,
                          noise_std: float) -> tuple[np.ndarray, dict]:
    """
    Closed-loop generator that mimics Stage-2 skeleton output.

    Three properties of real closed loops that v2 missed:

      1. DENSITY        : 30-60 points on the loop (was 12-30 in v2)
      2. ANGULAR SAMPLING: non-uniform spacing — clusters and gaps
                            (was uniform spacing in v2)
      3. RADIAL WOBBLE   : 5-10 % of radius (was 0.5-1.5 % in v2)

    This is the Priority 3 fix from the roadmap.
    """
    # 1 — DENSITY: Sample 30–60 points (median 50, matching real input)
    n_pts = int(rng.integers(30, 61))

    cx, cy = rng.uniform(0.35, 0.65, 2)
    r      = float(rng.uniform(0.18, 0.35))

    # 2 — ANGULAR SAMPLING: jitter angles to create clusters and gaps
    base_angles  = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    angle_jitter = rng.normal(0, 2 * np.pi / n_pts * 0.6, n_pts)
    angles = np.sort(base_angles + angle_jitter)

    # 3 — RADIAL WOBBLE: each point's radius perturbed by 5–10 % of r
    wobble_amplitude = r * float(rng.uniform(0.05, 0.10))
    radii            = r + rng.normal(0, wobble_amplitude, n_pts)

    pts = np.column_stack([cx + radii * np.cos(angles),
                           cy + radii * np.sin(angles)])

    # Light additional noise on top to break perfect smoothness
    pts = _add_noise(pts, noise_std * 1.5, rng)

    # Optionally close the loop visually by appending the first point.
    # We do NOT do this at training time because real skeletons trace open
    # paths around the loop; the topology (`is_closed=True`) is signalled
    # to the model by other means at inference (geometric proximity of
    # endpoints in the encoder representation).

    cmd = {"type":   "CIRCLE",
           "center": [cx, cy],
           "radius": r}
    return pts, cmd


def _gen_polyline_curve(rng: np.random.Generator,
                        noise_std: float) -> tuple[np.ndarray, dict]:
    """Smooth curve that is not a circle / arc — e.g. cubic-Bézier-like."""
    n_pts = int(rng.integers(8, 30))

    # Build a smooth curve by sampling a Bézier-like path
    p0 = rng.uniform(0.10, 0.30, 2)
    p3 = rng.uniform(0.70, 0.90, 2)
    p1 = rng.uniform(0.10, 0.90, 2)
    p2 = rng.uniform(0.10, 0.90, 2)
    t  = np.linspace(0, 1, n_pts)[:, None]
    pts = (
        (1 - t) ** 3 * p0
        + 3 * (1 - t) ** 2 * t * p1
        + 3 * (1 - t) * t ** 2 * p2
        + t ** 3 * p3
    )
    pts = _add_noise(pts, noise_std, rng)

    # POLYLINE has no canonical parameter form
    cmd = {"type": "POLYLINE"}
    return pts, cmd


def _gen_polyline_zigzag(rng: np.random.Generator,
                         noise_std: float) -> tuple[np.ndarray, dict]:
    """Sharp zig-zag polyline, 4-10 vertices, modest noise."""
    n_verts = int(rng.integers(4, 11))
    verts = [rng.uniform(0.10, 0.90, 2)]
    for _ in range(n_verts - 1):
        nxt = verts[-1] + rng.uniform(-0.30, 0.30, 2)
        nxt = np.clip(nxt, 0.05, 0.95)
        verts.append(nxt)
    verts = np.stack(verts)

    # Densify each segment slightly so the stroke has 6-25 total points
    densified = []
    for i in range(len(verts) - 1):
        seg = _sample_line(verts[i], verts[i + 1], int(rng.integers(2, 4)))
        densified.append(seg if i == 0 else seg[1:])
    pts = np.vstack(densified)
    pts = _add_noise(pts, noise_std, rng)
    cmd = {"type": "POLYLINE"}
    return pts, cmd


# Registry of all kind → generator function
_KIND_FNS = {
    "line_short":      _gen_line_short,
    "line_mid":        _gen_line_mid,
    "line_long":       _gen_line_long,
    "arc_mid":         _gen_arc_mid,
    "arc_long":        _gen_arc_long,
    "circle_realistic":_gen_realistic_circle,
    "polyline_curve":  _gen_polyline_curve,
    "polyline_zigzag": _gen_polyline_zigzag,
}


# ─── Sample assembly ──────────────────────────────────────────────────────────

def _generate_sample(rng: np.random.Generator,
                     noise_std: float) -> tuple[dict, str]:
    """
    Generate a single (stroke, command) sample.

    Returns:
      sample : {"stroke": [...], "command": {...}}
      kind   : the edge kind string (for distribution reporting)
    """
    kind = rng.choice(_KINDS, p=_PROBS)
    pts, cmd = _KIND_FNS[kind](rng, noise_std)

    # Canvas-normalise stroke; remap command params with the same transform
    norm_pts, transform = _normalise_to_canvas(pts)

    if cmd["type"] == "LINE":
        norm_cmd = {
            "type":  "LINE",
            "start": _remap_point(cmd["start"], transform),
            "end":   _remap_point(cmd["end"],   transform),
        }
    elif cmd["type"] == "ARC":
        norm_cmd = {
            "type":        "ARC",
            "center":      _remap_point(cmd["center"], transform),
            "radius":      _remap_length(cmd["radius"], transform),
            "start_angle": round(float(cmd["start_angle"]) % 360, 2),
            "end_angle":   round(float(cmd["end_angle"])   % 360, 2),
        }
    elif cmd["type"] == "CIRCLE":
        norm_cmd = {
            "type":   "CIRCLE",
            "center": _remap_point(cmd["center"], transform),
            "radius": _remap_length(cmd["radius"], transform),
        }
    else:  # POLYLINE
        norm_cmd = {"type": "POLYLINE"}

    sample = {
        "stroke":  [[round(float(x), 5), round(float(y), 5)]
                    for x, y in norm_pts],
        "command": norm_cmd,
    }
    return sample, kind


# ─── Class-share validator ────────────────────────────────────────────────────

# Per-class minimum shares of the command vocabulary in EACH split.
# The training run aborts if any split fails these floors so that we never
# accidentally start training on a degenerate distribution.

_DEFAULT_MIN_SHARES = {
    "LINE":     0.45,   # dominant class, but never above ~70 %
    "ARC":      0.08,
    "CIRCLE":   0.08,
    "POLYLINE": 0.10,
}


def _report_distribution(samples_dir: Path) -> tuple[int, Counter, Counter]:
    """Read all samples in samples_dir and tally command type + edge kind."""
    type_counts = Counter()
    kind_counts = Counter()
    n = 0
    for path in samples_dir.glob("*.json"):
        with open(path) as f:
            sample = json.load(f)
        type_counts[sample["command"]["type"]] += 1
        # The "kind" field is optional metadata for debugging
        if "kind" in sample:
            kind_counts[sample["kind"]] += 1
        n += 1
    return n, type_counts, kind_counts


def _print_distribution(split: str, n_samples: int,
                        type_counts: Counter,
                        kind_counts: Counter) -> None:
    print(f"\n  {split.upper():<5}  total = {n_samples}")
    print(f"  {'─'*55}")
    print(f"  {'TYPE':<10}  {'count':>7}   {'share':>7}")
    for cmd_type, count in sorted(type_counts.items(),
                                  key=lambda x: -x[1]):
        share = count / max(n_samples, 1)
        print(f"  {cmd_type:<10}  {count:>7}   {share:>6.1%}")
    if kind_counts:
        print(f"\n  {'KIND':<18}  {'count':>7}   {'share':>7}")
        for kind, count in sorted(kind_counts.items(), key=lambda x: -x[1]):
            share = count / max(n_samples, 1)
            print(f"  {kind:<18}  {count:>7}   {share:>6.1%}")


def _validate_shares(split: str, n_samples: int,
                     type_counts: Counter,
                     min_shares: dict) -> list[str]:
    """Return a list of human-readable failure messages (empty if all pass)."""
    failures = []
    for cls, floor in min_shares.items():
        share = type_counts.get(cls, 0) / max(n_samples, 1)
        if share < floor:
            failures.append(
                f"  ✗ [{split}] {cls}: {share:.1%} < required {floor:.1%}")
    return failures


# ─── Top-level dataset generation ─────────────────────────────────────────────

def generate_dataset(output_dir: str, n_samples: int, noise_std: float,
                     splits: str, seed: int,
                     min_shares: dict) -> None:
    """
    Generate n_samples per-edge training examples and split them
    into train/val/test by the given ratios.
    """
    out = Path(output_dir)
    for split in ["train", "val", "test"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    ratios = [float(x) for x in splits.split(",")]
    assert abs(sum(ratios) - 1.0) < 1e-6, \
        f"Splits must sum to 1.0, got {sum(ratios):.4f}"

    n_train = int(n_samples * ratios[0])
    n_val   = int(n_samples * ratios[1])
    n_test  = n_samples - n_train - n_val
    sizes = {"train": n_train, "val": n_val, "test": n_test}

    rng = np.random.default_rng(seed)

    print(f"Generating {n_samples} per-edge samples")
    print(f"  splits      : train={n_train}  val={n_val}  test={n_test}")
    print(f"  noise_std   : {noise_std}")
    print(f"  output_dir  : {out}")
    print(f"  shape kinds : {len(_KINDS)}")

    written = 0
    for split, count in sizes.items():
        print(f"\n  Writing {split} ({count} samples) ...")
        for i in range(count):
            sample, kind = _generate_sample(rng, noise_std)
            sample["kind"] = kind   # debug metadata; the trainer ignores it
            dest = out / split / f"sample_{written:06d}.json"
            with open(dest, "w") as f:
                json.dump(sample, f)
            written += 1
            if (i + 1) % 5000 == 0:
                print(f"    [{split}] {i+1}/{count}")

    # ── Validate distributions ────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print(" Dataset distribution report")
    print("═" * 60)

    all_failures = []
    for split in ["train", "val", "test"]:
        n_split, type_counts, kind_counts = _report_distribution(out / split)
        _print_distribution(split, n_split, type_counts, kind_counts)
        failures = _validate_shares(split, n_split, type_counts, min_shares)
        all_failures.extend(failures)

    print("\n" + "═" * 60)
    if all_failures:
        print(" ✗ DISTRIBUTION VALIDATION FAILED")
        print("═" * 60)
        for f in all_failures:
            print(f)
        print("\nClass shares fell below the required floors. The training set")
        print("would be unbalanced. Re-run with a different seed or with more")
        print("samples (--n_samples), or relax the floor with --min_shares.")
        raise SystemExit(1)
    else:
        print(" ✓ All class floors met in every split")
        print("═" * 60)
        print(f"\nDataset written to: {out}")
        print(f"Total files       : {written}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _parse_min_shares(arg: str) -> dict:
    """Parse 'LINE=0.40,ARC=0.10' into a dict, falling back to defaults."""
    out = dict(_DEFAULT_MIN_SHARES)
    if not arg:
        return out
    for pair in arg.split(","):
        cls, val = pair.split("=")
        out[cls.strip().upper()] = float(val)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-edge synthetic data for Free2CAD (v3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", type=str,
                        default="data/free2cad_training_v3")
    parser.add_argument("--n_samples",  type=int,   default=30000)
    parser.add_argument("--noise_std",  type=float, default=0.005)
    parser.add_argument("--splits",     type=str,   default="0.85,0.10,0.05")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument(
        "--min_shares", type=str, default="",
        help="Override per-class minimum shares as 'LINE=0.40,ARC=0.10,...' "
             "Defaults: LINE>=45%, ARC>=8%, CIRCLE>=8%, POLYLINE>=10%")
    args = parser.parse_args()

    min_shares = _parse_min_shares(args.min_shares)

    generate_dataset(
        output_dir = args.output_dir,
        n_samples  = args.n_samples,
        noise_std  = args.noise_std,
        splits     = args.splits,
        seed       = args.seed,
        min_shares = min_shares,
    )