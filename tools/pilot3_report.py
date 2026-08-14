"""3-way pilot A/B/C report: CN-fusion vs Full-CNN-phaseA vs Full-CNN-phaseB(SG+D2C mix0.40)."""
import sqlite3
import numpy as np

DBS = {
    "CN-fusion":         "output/pilot_cnfusion/results.db",
    "Full-CNN phaseA":   "output/pilot_phaseA/results.db",
    "Full-CNN phaseB (SG+D2C mix0.40)": "output/pilot_phaseB/results.db",
}

def load(db):
    con = sqlite3.connect(db)
    cols = [c[1] for c in con.execute("PRAGMA table_info(results)").fetchall()]
    return [dict(zip(cols, r)) for r in con.execute("SELECT * FROM results").fetchall()]

data = {k: load(v) for k, v in DBS.items()}
ok = lambda rows: [r for r in rows if r["status"] == "ok"]
names = list(DBS.keys())

print(f"{'metric':<32}" + "".join(f"{n:>22}" for n in names))
print("-" * (32 + 22 * len(names)))

def row(label, fn, fmt="{:.2f}"):
    vals = []
    for n in names:
        try:
            v = fn(data[n])
        except Exception:
            v = None
        vals.append(v)
    def f(v):
        if v is None: return "-"
        return fmt.format(v) if isinstance(v, float) else str(v)
    print(f"{label:<32}" + "".join(f"{f(v):>22}" for v in vals))

row("n total", lambda r: len(r), "{}")
row("n status=ok", lambda r: len(ok(r)), "{}")
for st in ("stage0", "stage1", "stage2", "stage3", "stage4"):
    row(f"  errors in {st}", lambda r, s=st: sum(1 for x in r if x["status"] == s), "{}")
row("  quality-gate stops", lambda r: sum(1 for x in r if str(x["status"]).startswith("quality_gate")), "{}")
print()
row("s2 keypoint src (mode)",
    lambda r: max(set(x["s2_keypoint_src"] for x in ok(r)), key=[x["s2_keypoint_src"] for x in ok(r)].count) if ok(r) else None,
    "{}")
row("s2 edges (mean)", lambda r: np.mean([x["s2_n_edges"] for x in ok(r)]))
row("s2 edges (median)", lambda r: float(np.median([x["s2_n_edges"] for x in ok(r)])))
row("s2 micro-edge ratio (mean)", lambda r: np.mean([x["s2_micro_edge_ratio"] or 0 for x in ok(r)]), "{:.3f}")
row("s2 isolation (mean)", lambda r: np.mean([x["s2_isolation"] or 0 for x in ok(r)]), "{:.3f}")
row("s2 flagged", lambda r: sum(x["s2_flagged"] or 0 for x in r), "{}")
print()
row("s3 primitives (mean)", lambda r: np.mean([x["s3_n_primitives"] for x in ok(r)]))
row("s3 mean confidence", lambda r: np.mean([x["s3_mean_conf"] or 0 for x in ok(r)]), "{:.3f}")
row("s3 low-conf ratio (mean)", lambda r: np.mean([x["s3_low_conf_ratio"] or 0 for x in ok(r)]), "{:.3f}")
print()
row("total time/fig s (mean)", lambda r: np.mean([x["total_time"] or 0 for x in ok(r)]))
row("total time/fig s (p95)", lambda r: float(np.percentile([x["total_time"] or 0 for x in ok(r)], 95)))

# paired per-figure comparison on the intersection of "ok" across all 3
keys = [set((r["patent_id"], r["sketch_id"]) for r in ok(data[n])) for n in names]
shared = sorted(set.intersection(*keys))
print(f"\nfigures 'ok' in ALL 3 configs: {len(shared)} / 99")
if shared:
    idx = {n: {(r["patent_id"], r["sketch_id"]): r for r in ok(data[n])} for n in names}
    edges = {n: np.array([idx[n][k]["s2_n_edges"] for k in shared]) for n in names}
    print(f"\n{'':<20}" + "".join(f"{n:>22}" for n in names))
    print(f"{'s2 edges (paired mean)':<20}" + "".join(f"{edges[n].mean():>22.1f}" for n in names))
    base = names[0]
    for n in names[1:]:
        d = edges[n] - edges[base]
        print(f"  {n} vs {base}: mean delta {d.mean():+.1f}  "
              f"fewer={int((d<0).sum())} more={int((d>0).sum())} same={int((d==0).sum())}")
