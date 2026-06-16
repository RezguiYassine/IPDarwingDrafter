#!/usr/bin/env python3
"""No-GT structural diagnostic for the stroke/corner problems on PatentData.

PatentData has no ground truth, so we cannot measure Chamfer. Instead we measure
*structural* symptoms that are objectively computable from the Stage-2 graph and
the Stage-3 primitives, and rank sketches by how badly each symptom fires. The
goal is to tell which of {corners, long strokes, curved strokes, chamfers} is the
single biggest driver before we fix anything.

Signals (all per-sketch, then aggregated):
  * collinear_breaks  - degree-2 nodes where the two incident edges are both
                        straight and nearly collinear (turn < COLLINEAR_DEG).
                        => a long stroke that got chopped into pieces.
  * corner_clusters   - groups of >=2 nodes packed within CLUSTER_FRAC of the
                        median edge length. => one real corner fragmented into
                        several vertices.
  * rounded_corners   - degree-2 "corner" nodes (turn in [CORNER_LO,CORNER_HI])
                        whose two edges are short arcs/polylines rather than
                        lines. => a sharp corner that got rounded off.
  * polyline_ratio    - fraction of primitives emitted as polylines (the "could
                        not fit a clean line/arc" fallback). => curve trouble.
  * jagged_polylines  - polylines whose turning roughness is high. => curves
                        traced as zig-zags.
  * chamfer_candidates- short line segments bridging two longer near-45deg lines
                        (present = good; we count them to spot where they vanish).

Usage:  python tools/stroke_corner_diag.py latest_run [--top 15] [--out diag.json]
"""
from __future__ import annotations
import argparse, glob, json, math, os, sys
from collections import Counter, defaultdict

COLLINEAR_DEG = 18.0      # turn below this at a deg-2 node => false break
CORNER_LO, CORNER_HI = 30.0, 150.0   # genuine corner turn range
CLUSTER_FRAC = 0.15       # node-cluster radius as fraction of median edge len
JAGGED_ROUGH = 0.6        # rad of total turning per unit length => jagged
CHAMFER_MAXLEN_FRAC = 0.4 # chamfer line shorter than this * median edge len


def _ang(p, q):
    return math.atan2(q[1] - p[1], q[0] - p[0])


def _turn_deg(d1, d2):
    """Angle between two *outgoing* direction vectors, folded to [0,180]."""
    a = math.degrees(abs(d1 - d2)) % 360.0
    if a > 180.0:
        a = 360.0 - a
    return a


def _edge_dir_at(edge, node_xy):
    """Outgoing direction (radians) of an edge leaving the given node."""
    pts = edge.get("smooth_pts") or edge.get("pixels")
    if not pts or len(pts) < 2:
        return None
    a, b = pts[0], pts[-1]
    da = (a[0] - node_xy[0]) ** 2 + (a[1] - node_xy[1]) ** 2
    db = (b[0] - node_xy[0]) ** 2 + (b[1] - node_xy[1]) ** 2
    if da <= db:                       # node is at pts[0], look toward pts[1]
        return _ang(pts[0], pts[1])
    return _ang(pts[-1], pts[-2])      # node is at pts[-1], look toward pts[-2]


def _poly_len(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))


def _poly_roughness(pts):
    """Total absolute turning (rad) per unit length over a polyline."""
    if len(pts) < 3:
        return 0.0
    tot = 0.0
    for i in range(1, len(pts) - 1):
        d1 = _ang(pts[i - 1], pts[i])
        d2 = _ang(pts[i], pts[i + 1])
        t = abs(d2 - d1)
        if t > math.pi:
            t = 2 * math.pi - t
        tot += t
    L = _poly_len(pts)
    return tot / L if L > 1e-6 else 0.0


def analyze(graph, prims):
    nodes = {int(n["id"]): n for n in graph["nodes"]}
    edges = graph["edges"]
    med_edge = graph.get("metrics", {}).get("median_edge_length") or 1.0
    # edge_id -> primitive type
    ptype = {}
    plist = prims.get("primitives", [])
    for pr in plist:
        if pr.get("edge_id") is not None:
            ptype[int(pr["edge_id"])] = pr["type"]
    # incidence
    inc = defaultdict(list)        # node id -> [edge,...]
    for e in edges:
        inc[int(e["source"])].append(e)
        inc[int(e["target"])].append(e)

    collinear_breaks = rounded = genuine_corners = 0
    for nid, es in inc.items():
        if len(es) != 2:
            continue
        nxy = (nodes[nid]["x"], nodes[nid]["y"])
        d1 = _edge_dir_at(es[0], nxy)
        d2 = _edge_dir_at(es[1], nxy)
        if d1 is None or d2 is None:
            continue
        # outgoing dirs both point away from node; straight-through => ~180
        turn = 180.0 - _turn_deg(d1, d2)
        t0, t1 = ptype.get(int(es[0]["id"]), "line"), ptype.get(int(es[1]["id"]), "line")
        if abs(turn) < COLLINEAR_DEG and t0 == "line" and t1 == "line":
            collinear_breaks += 1
        elif CORNER_LO <= abs(turn) <= CORNER_HI:
            genuine_corners += 1
            if t0 in ("arc", "polyline") or t1 in ("arc", "polyline"):
                rounded += 1

    # corner clustering: junctions + corner-ish deg2 nodes packed together
    pts = [(n["x"], n["y"]) for n in graph["nodes"]
           if n["type"] == "junction" or len(inc[int(n["id"])]) >= 3]
    rad = max(2.0, CLUSTER_FRAC * med_edge)
    used = [False] * len(pts)
    clusters = 0
    for i in range(len(pts)):
        if used[i]:
            continue
        members = [i]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if not used[j] and math.dist(pts[i], pts[j]) <= rad:
                used[j] = True
                members.append(j)
        if len(members) >= 2:
            clusters += 1

    # curve fallbacks
    n_prim = max(1, len(plist))
    polys = [pr for pr in plist if pr["type"] == "polyline"]
    jagged = sum(1 for pr in polys
                 if _poly_roughness(pr.get("points", [])) > JAGGED_ROUGH)

    # chamfer candidates: short lines whose both neighbours are longer lines at angle
    lines = [pr for pr in plist if pr["type"] == "line"]
    chamfers = 0
    maxlen = CHAMFER_MAXLEN_FRAC * med_edge
    for pr in lines:
        L = math.dist(pr["p1"], pr["p2"])
        if 2.0 < L <= maxlen:
            chamfers += 1

    return {
        "n_prim": len(plist),
        "n_nodes": len(graph["nodes"]),
        "collinear_breaks": collinear_breaks,
        "genuine_corners": genuine_corners,
        "rounded_corners": rounded,
        "corner_clusters": clusters,
        "polyline_ratio": round(len(polys) / n_prim, 4),
        "jagged_polylines": jagged,
        "chamfer_candidates": chamfers,
        "micro_edge_ratio": graph.get("metrics", {}).get("micro_edge_ratio", 0.0),
        "short_edge_ratio": graph.get("metrics", {}).get("short_edge_ratio", 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = []
    for gpath in sorted(glob.glob(os.path.join(a.run_dir, "*", "graphs", "*_graph.json"))):
        sk = os.path.basename(gpath).replace("_graph.json", "")
        patent = gpath.split(os.sep)[-3]
        ppath = gpath.replace("graphs", "primitives").replace("_graph.json", "_primitives.json")
        if not os.path.exists(ppath):
            continue
        try:
            g = json.load(open(gpath)); p = json.load(open(ppath))
            r = analyze(g, p); r["id"] = f"{patent}/{sk}"
            rows.append(r)
        except Exception as exc:
            print(f"skip {patent}/{sk}: {exc}", file=sys.stderr)

    if not rows:
        print("no sketches found", file=sys.stderr); sys.exit(1)

    n = len(rows)
    agg = Counter()
    for r in rows:
        for k in ("collinear_breaks", "genuine_corners", "rounded_corners",
                  "corner_clusters", "jagged_polylines", "chamfer_candidates", "n_prim"):
            agg[k] += r[k]
    poly_ratio = sum(r["polyline_ratio"] for r in rows) / n

    print(f"\n=== STROKE/CORNER STRUCTURAL DIAGNOSTIC  ({n} sketches) ===\n")
    print(f"  total primitives ............ {agg['n_prim']:>6}")
    print(f"  polyline fallback ratio ..... {poly_ratio:>6.1%}  (curves not cleanly fit)")
    print(f"  jagged polylines (total) .... {agg['jagged_polylines']:>6}   "
          f"({agg['jagged_polylines']/max(1,n):.1f}/sketch)")
    print()
    print(f"  collinear breaks (total) .... {agg['collinear_breaks']:>6}   "
          f"({agg['collinear_breaks']/n:.1f}/sketch)  << long strokes over-segmented")
    print(f"  genuine corners ............. {agg['genuine_corners']:>6}")
    print(f"  rounded corners ............. {agg['rounded_corners']:>6}   "
          f"({agg['rounded_corners']/max(1,agg['genuine_corners']):.1%} of corners)")
    print(f"  corner clusters (total) ..... {agg['corner_clusters']:>6}   "
          f"({agg['corner_clusters']/n:.1f}/sketch)  << corners fragmented")
    print(f"  chamfer candidates .......... {agg['chamfer_candidates']:>6}")

    def rank(key, label):
        print(f"\n  -- worst by {label} --")
        for r in sorted(rows, key=lambda x: -x[key])[:a.top]:
            print(f"    {r[key]:>4}  {r['id']:<24} "
                  f"(prim={r['n_prim']}, poly={r['polyline_ratio']:.0%}, "
                  f"clusters={r['corner_clusters']}, breaks={r['collinear_breaks']})")
    rank("collinear_breaks", "collinear breaks (long-stroke fragmentation)")
    rank("corner_clusters", "corner clusters (corner fragmentation)")
    rank("jagged_polylines", "jagged polylines (curve quality)")

    if a.out:
        json.dump({"params": {"COLLINEAR_DEG": COLLINEAR_DEG, "CLUSTER_FRAC": CLUSTER_FRAC,
                              "JAGGED_ROUGH": JAGGED_ROUGH}, "rows": rows},
                  open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
