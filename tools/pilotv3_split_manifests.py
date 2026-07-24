#!/usr/bin/env python3
"""Split the pilot-v3 100-figure set into the four training manifests defined in
PATENTVEC_PILOT_V3_RESULTS_NEXT_STEPS.md (Step 2), attaching per-figure failure
labels (Step 3).

Buckets (the critical rule: NEVER merge invalid-content with valid-but-hard):
  A. positive_clean   -> normal real positives (clean vectorization)
  B. positive_hard    -> VALID patent drawings that over-fragment (Group 2)
  C. negative_invalid -> not vectorization targets (Group 1: flowchart/table/
                         block/network/UML/UI/plot/chart/chem/shaded render)
  D. borderline       -> valid but uncertain / gate slightly aggressive (Group 3)
                         + text-heavy figures routed to review

Source of truth = the postvec gate audit CSV (verdict + metrics + text-load)
overlaid with this session's manual visual content audit (the CONTENT dict).
Figures not individually audited inherit a provisional role from the gate
verdict and are marked audited=False, so the accepted-sample false-negative
audit (doc Step 4) can be completed later without losing the confirmed labels.
"""
from __future__ import annotations
import argparse, csv, collections
from pathlib import Path

# ── manual visual content audit (2026-07-24 session) ─────────────────────────
# value = (content_class, is_valid_drawing, failure_family, notes)
INVALID = "negative_invalid"; HARD = "positive_hard"
BORDER  = "borderline";       CLEAN = "positive_clean"

CONTENT: dict[str, tuple] = {
    # ---- flagged-15: Group 1 invalid content ----
    "EP3329273B1/F0003": ("shaded_render",  False, "shaded_render", "photorealistic 3D render"),
    "EP2398750B1/F0002": ("plot_or_chart",  False, "plot_or_chart", "contour/line plots"),
    "EP3493040A1/A0001": ("ui_mockup",      False, "ui_mockup",     "message/UI + block"),
    "EP3313051B1/F0004": ("block_diagram",  False, "block_diagram", "boxes + dashed connectors"),
    "EP2765815B1/F0003": ("block_diagram",  False, "block_diagram", "signal-flow diagram"),
    # ---- flagged-15: Group 2 valid-but-fragmented (HARD POSITIVES) ----
    "EP2020016B1/F0006": ("mech_3d_pcb",    True,  "dense_detail",              "PCB/coil 3D, numerals"),
    "EP3492404A1/F0003": ("mech_hatched",   True,  "hatch_boundary_fragmentation", "hatched parts"),
    "EP3077846B1/F0010": ("mech_dimension", True,  "dimension_arc_confusion",   "angle/dimension arcs"),
    "EP3499690B1/F0006": ("mech_bolt_circle", True, "dashed_line_fragmentation", "dashed bolt-circle centerlines"),
    "EP3499690A1/F0006": ("mech_bolt_circle", True, "dashed_line_fragmentation", "dashed bolt-circle centerlines"),
    "EP3493068A1/F0002": ("mech_section_box", True, "text_box_isolation",       "hatched section + boxed labels"),
    # ---- flagged-15: Group 3 borderline valid ----
    "EP2795396B1/F0001": ("mech_simple",    True,  "borderline_valid", "clean box + glyph"),
    "EP3046835B1/F0009": ("geom_construction", True, "borderline_valid", "sphere construction lines"),
    "EP3492685A1/A0001": ("mech_coil",      True,  "borderline_valid", "detailed coil, iso boundary"),
    "EP3494953A1/F0003": ("mech_shaded_panel", True, "borderline_valid", "one shaded panel"),
    "EP2994479B1/F0003": ("gel_cassette",   False, "specialized_dense", "gel/cassette lane diagram"),
    # ---- accepted-set false-negatives (non-drawings that passed as clean) ----
    "EP3496339A1/F0007": ("table",          False, "table",         "disaster-info data table"),
    "EP3101949B1/F0018": ("flowchart",      False, "flowchart",     "handover flowchart"),
    "EP3272039B1/F0012": ("block_diagram",  False, "block_diagram", "EM-phase block diagram"),
    "EP3503515A1/F0028": ("ui_mockup",      False, "ui_mockup",     "UI screenshots"),
    "EP3454625A3/A0001": ("block_diagram",  False, "block_diagram", "commlabs network diagram"),
    "EP3503631A1/F0003": ("flowchart",      False, "flowchart",     "WUR-setup flowchart"),
    "EP3493138A1/A0001": ("flowchart",      False, "flowchart",     "determine/recommend flowchart"),
    "EP2807227B1/F0026": ("chem_shaded",    False, "shaded_render", "Na/K synthesis + fiber render"),
    "EP3048613B1/F0001": ("uml_pid",        False, "block_diagram", "UML class diagram + P&ID"),
    "EP3256873B1/F0002": ("network_diagram", False, "block_diagram", "WWAN/WLAN network diagram"),
    "EP1763241B1/F0001": ("table",          False, "table",         "A/B/C/D grid (prior art)"),
    "EP3270639B1/F0005": ("block_diagram",  False, "block_diagram", "access-apparatus block"),
    "EP3496009A1/A0001": ("block_diagram",  False, "block_diagram", "numbered dataflow boxes"),
    "EP2903619B1/F0007": ("plot_or_chart",  False, "plot_or_chart", "AA-ME/AA-H bar charts"),
    "EP3100880B1/F0008": ("plot_or_chart",  False, "plot_or_chart", "damping-force line plot"),
    # ---- audited real drawings that score high text-load (review-flag FPs) ----
    "EP2795747B1/F0002": ("optics",         True,  "", "fiber optics circles (real)"),
    "EP3269113B1/F0010": ("optics",         True,  "", "geometric optics (real)"),
    "EP2694884B1/F0003": ("geom_scientific", True, "", "hemisphere dot diagrams (real)"),
    "EP3495573A1/A0001": ("mech_hatched",   True,  "", "hatched latch cross-section (real)"),
    "EP2979610B1/F0003": ("mech_tool",      True,  "", "hand-tool assembly (real)"),
    "EP3046249B1/F0001": ("mixed",          True,  "borderline_valid", "real sensor + block panels"),
    "EP3496069A1/F0006": ("mixed",          True,  "borderline_valid", "road diagrams + photo panel"),
    # ---- audited real drawings, clean ----
    "EP2789736B1/F0003": ("mech", True, "", "pipe coupling cross-section"),
    "EP3503702A1/F0002": ("mech", True, "", "cross-sections"),
    "EP3492282A1/F0003": ("mech", True, "", "component + dimensions"),
    "EP2385583B1/F0003": ("mech", True, "", "layered structure"),
    "EP3502579A1/F0008": ("geom", True, "", "3D prism"),
    "EP2942310B1/F0004": ("mech", True, "", "dense assembly"),
    "EP2724832B1/F0010": ("mech", True, "", "apparatus"),
    "EP3188187B1/F0004": ("mech", True, "", "device housing"),
    "EP2508802B1/F0003": ("mech", True, "", "curved part"),
    "EP2099533B1/F0006": ("mech", True, "", "corrugated cross-section"),
    "EP3215734B1/F0006": ("mech", True, "", "mechanical linkage"),
    "EP2792300B1/F0003": ("geom", True, "", "staircase diagram"),
    "EP3492390A1/F0001": ("mech", True, "", "helicopter/aircraft"),
    "EP2434121B1/F0002": ("mech", True, "", "detailed assembly"),
    "EP3501346A1/F0009": ("mech", True, "", "cross-sections"),
    "EP2757380B1/F0001": ("schematic", True, "borderline_valid", "lock-in amplifier schematic"),
    # ---- 47 provisional positive_clean, fully audited 2026-07-24 ----
    # invalid content found among the provisional set:
    "EP3197444B1/F0003": ("plot_or_chart", False, "plot_or_chart", "transmittance spectrum plot"),
    "EP3060522B1/F0064": ("plot_or_chart", False, "plot_or_chart", "concentration line plot"),
    "EP3493090A1/F0003": ("flowchart",     False, "flowchart",     "S301-S312 flowchart"),
    "EP3495359A1/A0001": ("chemistry",     False, "chemistry",     "molecular structure (I)"),
    "EP3498367A1/F0002": ("plot_or_chart", False, "plot_or_chart", "mm-vs-pH line plots"),
    "EP3503313A1/F0001": ("block_diagram", False, "block_diagram", "control block diagram"),
    # borderline among the provisional set:
    "EP3090501B1/F0004": ("schematic", True, "borderline_valid", "circuit/signal-flow schematic"),
    "EP3209516B1/F0002": ("mixed",     True, "borderline_valid", "vehicle + projection geometry"),
    "EP3492735A1/F0002": ("mixed",     True, "borderline_valid", "rotating-disc diagram + sine plot"),
    "EP3501750A1/F0006": ("mixed",     True, "borderline_valid", "spring diagrams + data table"),
    "EP3502514A1/F0001": ("schematic", True, "borderline_valid", "hydraulic/pneumatic circuit schematic"),
}

# The remaining 36 provisional figures were audited as genuine clean drawings.
AUDITED_CLEAN_KEYS = {
    "EP2000273B1/F0020","EP2375003B1/F0002","EP1835879B1/F0008","EP2277019B1/F0005",
    "EP1930717B1/F0031","EP2558785B1/F0007","EP2466634B1/F0004","EP2519299B1/F0002",
    "EP2464865B1/F0005","EP2730496B1/F0005","EP2806261B1/F0004","EP2863449B1/F0001",
    "EP2811932B1/F0003","EP2828888B1/F0001","EP2978096B1/F0001","EP2987931B1/F0005",
    "EP3045591B1/F0004","EP2971551B1/F0002","EP3156693B1/F0002","EP3062407B1/F0002",
    "EP3270850B1/F0009","EP3295061B1/F0008","EP3492425A1/F0002","EP3492785A1/F0001",
    "EP3492864A1/F0002","EP3495689A1/F0011","EP3498592A1/F0003","EP3499154A1/A0001",
    "EP3498954A1/F0003","EP3502393A1/F0005","EP3501322A1/F0005","EP3502341A1/F0003",
    "EP3496522A2/F0005","EP3502490A1/F0013","EP3503123A1/A0001","EP3503059A1/F0003",
}
for _k in AUDITED_CLEAN_KEYS:
    CONTENT.setdefault(_k, ("drawing", True, "", "audited clean"))

ROLE_BY_CLASS = {}  # filled below from is_valid + failure_family


def bucket_for(verdict: str, audited: bool, entry) -> tuple[str, str, str, bool]:
    """Return (bucket, failure_family, notes, is_valid)."""
    if entry is not None:
        content, is_valid, fam, notes = entry
        if not is_valid:
            return INVALID, fam, notes, False
        # valid drawing
        if fam in ("borderline_valid",) or content == "mixed":
            return BORDER, fam, notes, True
        if fam:  # a real fragmentation failure family -> hard positive
            return HARD, fam, notes, True
        # valid, no failure family: clean positive (even if review-flagged)
        return CLEAN, "", notes, True
    # not manually audited -> provisional role from gate verdict
    if verdict == "clean":
        return CLEAN, "", "provisional (unaudited)", None
    if verdict == "review_text_heavy":
        return BORDER, "text_heavy", "provisional (unaudited, text-heavy)", None
    if verdict in ("low_quality", "not_ok"):
        # gate flagged it but we didn't audit content -> review, not a label
        return BORDER, "gate_flagged", "provisional (unaudited, gate-flagged)", None
    return BORDER, "", "provisional", None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path,
                    default=Path("output/PatentData/pilotv3_train_clean.audit.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("benchmarks/pilotv3"))
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(a.audit)))
    out_rows = collections.defaultdict(list)
    labeled = []
    for r in rows:
        key = f"{r['patent_id']}/{r['sketch_id']}"
        entry = CONTENT.get(key)
        audited = entry is not None
        bucket, fam, notes, is_valid = bucket_for(r["verdict"], audited, entry)
        rec = {
            "patent_id": r["patent_id"], "sketch_id": r["sketch_id"],
            "input_path": r["input_path"],
            "gate_verdict": r["verdict"], "gate_reason": r["reason"],
            "content_class": entry[0] if entry else "",
            "is_valid_drawing": "" if is_valid is None else int(bool(is_valid)),
            "bucket": bucket, "failure_family": fam,
            "audited": int(audited), "notes": notes,
            "micro": r.get("micro"), "median_edge_len": r.get("median_edge_len"),
            "isolation": r.get("isolation"), "n_edges": r.get("n_edges"),
            "n_ref_labels": r.get("n_ref_labels"),
            "cc_recovered_frac": r.get("cc_recovered_frac"),
        }
        out_rows[bucket].append(rec)
        labeled.append(rec)

    files = {CLEAN: "positive_clean.csv", HARD: "positive_hard.csv",
             INVALID: "negative_invalid.csv", BORDER: "borderline.csv"}
    fields = list(labeled[0].keys())
    for bucket, fn in files.items():
        recs = out_rows[bucket]
        with open(a.out_dir / fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(recs)
    # combined failure-labeled audit
    with open(a.out_dir / "pilotv3_labeled_audit.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(labeled)

    # ── report ──
    print("=== pilot-v3 4-manifest split ===")
    for bucket, fn in files.items():
        recs = out_rows[bucket]
        aud = sum(r["audited"] for r in recs)
        print(f"  {bucket:16s} {len(recs):3d}  (audited={aud}, provisional={len(recs)-aud})  -> {fn}")
    print(f"  total            {len(labeled):3d}")
    fam = collections.Counter(r["failure_family"] for r in out_rows[HARD])
    print("\nhard-positive failure families:")
    for k, n in fam.most_common():
        print(f"  {k:32s} {n}")
    print(f"\nwritten to {a.out_dir}/")


if __name__ == "__main__":
    main()
