#!/usr/bin/env python3
"""Post-vectorization quality gate for training-data selection.

The pre-filter (tools/filter_patent_data.py) predicts "will this vectorize
well" from raw-image features and has hit its accuracy ceiling: residual junk
(flowcharts, block diagrams, plots, document scans) still slips through as
"drawing", and real hatched/orthographic drawings get wrongly discarded,
because coarse global image features cannot separate those classes.

This gate takes the opposite, more reliable approach: it RUNS the pipeline and
measures whether each figure ACTUALLY vectorized like a clean line drawing,
using the Stage-2 metrics already stored in the batch_run results DB. Junk
vectorizes into a characteristic mess — many micro-edges (text glyphs, bond
notation, plot ticks) with short median stroke length — while real drawings
produce long, coherent strokes.

Calibration (2026-07-23, 100-fig patent pilot, visually audited):
  - JUNK anchors (DNA/protein/chemistry listings, flowcharts, block/network
    diagrams): micro_edge_ratio 0.10-0.27 with median_edge_len 14-32.
  - REAL drawings (incl. the one real drawing at micro 0.110): median_edge_len
    >= 42, and the vast majority have micro_edge_ratio 0.00-0.07.
  The conjunction (high micro AND short median) flagged 6/7 audited junk
  figures and spared the lone real drawing at the boundary — cleaner than
  either metric alone.

This is a SECOND layer, applied after the pre-filter, for selecting the clean
training corpus. It does not modify raw data; it writes a manifest of the
figures that both passed the pre-filter and vectorized cleanly, plus a report.

    python -m tools.postvec_quality_gate \
        --results-db output/pilot_phaseA/results.db \
        --out output/PatentData/train_clean_manifest.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path


def _rows(db_path: Path) -> list[dict]:
    con = sqlite3.connect(db_path)
    cols = [c[1] for c in con.execute("PRAGMA table_info(results)").fetchall()]
    out = [dict(zip(cols, r)) for r in con.execute("SELECT * FROM results").fetchall()]
    con.close()
    return out


def _text_load(db_path: Path, patent_id: str, sketch_id: str) -> tuple[int, float]:
    """Read Stage-0 references JSON (co-located with the results DB) and return
    (n_labels, cc_recovered_frac).

    cc_recovered labels are the small text-like connected components Stage-0
    strips. Clean-line non-drawings (flowcharts, block/network/UML diagrams,
    tables, charts, UI mock-ups) are dominated by them, because their words
    fragment into many small CCs. This is the ONE signal that ranks that
    content class -- which the micro/isolation fragmentation gate cannot see,
    because such diagrams vectorize CLEANLY (long orthogonal strokes, low micro,
    low isolation) and therefore pass. It is a *ranking*, not a clean
    classifier: real curve/hatch-annotated drawings also score high, so it is
    used only to ROUTE-TO-REVIEW (advisory), never to hard-reject. See the
    2026-07-24 accepted-sample audit: a threshold here recovers ~14 diagram/
    chart/table false-negatives that passed the fragmentation gate as "clean".
    """
    refs = db_path.parent / patent_id / "references" / f"{sketch_id}_references.json"
    if not refs.exists():
        return -1, -1.0
    try:
        d = json.load(open(refs))
    except Exception:
        return -1, -1.0
    labs = d.get("reference_labels", [])
    n = len(labs)
    if n == 0:
        return 0, 0.0
    cc = sum(1 for l in labs if l.get("kind") == "cc_recovered")
    return n, cc / n


def gate_verdict(row: dict, cfg: dict) -> tuple[str, str]:
    """Return (verdict, reason).

    verdict in {clean, low_quality, not_ok, review_text_heavy}.
    """
    if row.get("status") != "ok":
        return "not_ok", f"pipeline_status:{row.get('status')}"

    micro = row.get("s2_micro_edge_ratio")
    med   = row.get("s2_median_edge_len")
    iso   = row.get("s2_isolation")
    nedg  = row.get("s2_n_edges") or 0

    # Primary junk signature: many micro-edges AND short median stroke.
    if (micro is not None and med is not None
            and micro >= cfg["micro_max"] and med < cfg["median_min"]):
        return "low_quality", f"fragmented_short_strokes(micro={micro:.3f},med={med:.1f})"

    # Secondary: extreme micro alone (very text-dense) regardless of median.
    if micro is not None and micro >= cfg["micro_hard_max"]:
        return "low_quality", f"very_high_micro(micro={micro:.3f})"

    # Coverage failure: almost nothing captured relative to the ink present.
    if iso is not None and iso >= cfg["isolation_max"]:
        return "low_quality", f"high_isolation(iso={iso:.3f})"

    # Degenerate: essentially empty vectorization.
    if nedg < cfg["min_edges"]:
        return "low_quality", f"too_few_edges(n={nedg})"

    # Advisory: text-heavy -> likely a clean-line diagram/table/chart the
    # fragmentation gate can't catch. Route to review, NOT a hard reject
    # (real curve/hatch drawings can score here too). Only meaningful when the
    # references sidecar was found (ccf >= 0).
    ccf = row.get("cc_recovered_frac")
    nlab = row.get("n_ref_labels")
    if (ccf is not None and ccf >= 0 and nlab is not None
            and ccf >= cfg["textload_cc_frac_min"]
            and nlab >= cfg["textload_min_labels"]):
        return "review_text_heavy", f"text_heavy(cc_frac={ccf:.2f},n_labels={nlab})"

    return "clean", "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-db", type=Path, nargs="+", required=True,
                    help="one or more batch_run results DBs to gate")
    ap.add_argument("--out", type=Path, required=True,
                    help="output clean-training manifest CSV")
    ap.add_argument("--pre-filter", type=Path, default=None,
                    help="optional filter_patent_data manifest; only figures "
                         "labeled 'drawing' there are eligible")
    ap.add_argument("--micro-max", type=float, default=0.10,
                    help="micro_edge_ratio at/above this + short median = junk")
    ap.add_argument("--median-min", type=float, default=35.0,
                    help="median_edge_len below this + high micro = junk")
    ap.add_argument("--micro-hard-max", type=float, default=0.20,
                    help="micro_edge_ratio at/above this = junk regardless of median")
    ap.add_argument("--isolation-max", type=float, default=0.40,
                    help="isolation_ratio at/above this = coverage failure")
    ap.add_argument("--min-edges", type=int, default=4,
                    help="fewer edges than this = degenerate vectorization")
    ap.add_argument("--textload-cc-frac-min", type=float, default=0.80,
                    help="cc_recovered fraction at/above this + many labels "
                         "= text-heavy -> route to review (advisory, not reject)")
    ap.add_argument("--textload-min-labels", type=int, default=20,
                    help="min Stage-0 labels for the text-heavy review flag")
    a = ap.parse_args()

    cfg = dict(micro_max=a.micro_max, median_min=a.median_min,
               micro_hard_max=a.micro_hard_max, isolation_max=a.isolation_max,
               min_edges=a.min_edges,
               textload_cc_frac_min=a.textload_cc_frac_min,
               textload_min_labels=a.textload_min_labels)

    pre_ok = None
    if a.pre_filter and a.pre_filter.exists():
        pre_ok = {r["path"] for r in csv.DictReader(open(a.pre_filter))
                  if r.get("label") == "drawing"}
        print(f"pre-filter: {len(pre_ok)} figures eligible ('drawing')")

    # cleanliness rank: prefer to remember the WORSE verdict for a figure across
    # configs, except text-heavy which is a Stage-0 (config-independent) property.
    rank = {"not_ok": 0, "low_quality": 1, "review_text_heavy": 2, "clean": 3}
    seen: dict[tuple, dict] = {}
    for db in a.results_db:
        for row in _rows(db):
            key = (row["patent_id"], row["sketch_id"])
            nlab, ccf = _text_load(db, row["patent_id"], row["sketch_id"])
            row["n_ref_labels"], row["cc_recovered_frac"] = nlab, ccf
            v, reason = gate_verdict(row, cfg)
            rec = {"patent_id": row["patent_id"], "sketch_id": row["sketch_id"],
                   "input_path": row.get("input_path", ""),
                   "verdict": v, "reason": reason,
                   "micro": row.get("s2_micro_edge_ratio"),
                   "median_edge_len": row.get("s2_median_edge_len"),
                   "isolation": row.get("s2_isolation"),
                   "n_edges": row.get("s2_n_edges"),
                   "n_ref_labels": nlab, "cc_recovered_frac": round(ccf, 3)}
            # Keep the cleanest verdict across configs (a figure that vectorizes
            # cleanly in ANY config is a usable positive); but never let a
            # config's "clean" override the config-independent text-heavy flag.
            prev = seen.get(key)
            if prev is None or rank[v] > rank[prev["verdict"]]:
                seen[key] = rec
            if v == "review_text_heavy" and seen[key]["verdict"] == "clean":
                seen[key]["verdict"], seen[key]["reason"] = v, reason

    records = list(seen.values())
    if pre_ok is not None:
        for r in records:
            if r["input_path"] and r["input_path"] not in pre_ok:
                r["verdict"], r["reason"] = "pre_filtered", "not_drawing_in_pre_filter"

    # ── write clean manifest ──────────────────────────────────────────────
    a.out.parent.mkdir(parents=True, exist_ok=True)
    clean = [r for r in records if r["verdict"] == "clean"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(clean)

    # full audit CSV (all verdicts) next to the clean one
    audit_path = a.out.with_suffix(".audit.csv")
    with open(audit_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    # ── report ────────────────────────────────────────────────────────────
    import collections
    vc = collections.Counter(r["verdict"] for r in records)
    print(f"\n=== post-vectorization quality gate ===")
    print(f"total figures: {len(records)}")
    for v, n in vc.most_common():
        print(f"  {v:<14} {n:>6}  ({n/len(records)*100:.1f}%)")
    lowq = [r for r in records if r["verdict"] == "low_quality"]
    rc = collections.Counter(r["reason"].split("(")[0] for r in lowq)
    print("\nlow_quality reasons:")
    for reason, n in rc.most_common():
        print(f"  {reason:<28} {n:>6}")
    n_review = vc.get("review_text_heavy", 0)
    if n_review:
        print(f"\nreview_text_heavy: {n_review} figures routed to manual review "
              f"(likely clean-line diagram/table/chart the fragmentation gate "
              f"cannot see; advisory only -- excluded from the clean manifest, "
              f"NOT deleted).")
    print(f"\nclean training manifest: {len(clean)} figures -> {a.out}")
    print(f"full audit (all verdicts) -> {audit_path}")


if __name__ == "__main__":
    main()
