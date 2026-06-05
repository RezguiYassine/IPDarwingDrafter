"""Compare two d2c_results DBs on their overlapping (sample_id, view) set."""
import sqlite3, sys
import numpy as np

def load(db):
    c = sqlite3.connect(db)
    rows = c.execute(
        "SELECT sample_id,view,status,n_strokes_gt,n_prims_out,iou_pixel,"
        "iou_skeleton,precision_pixel,recall_pixel,chamfer_sym,chamfer_p95_sym,"
        "s2_time,total_time FROM d2c_results"
    ).fetchall()
    d = {}
    for r in rows:
        d[(r[0], r[1])] = r
    return d

def col(d, keys, idx, ok_only=True):
    out = []
    for k in keys:
        r = d[k]
        if ok_only and r[2] != "ok":
            continue
        v = r[idx]
        if v is not None:
            out.append(v)
    return np.array(out, float)

def stat(a):
    if len(a) == 0:
        return "—"
    return f"{a.mean():.3f} (p50 {np.percentile(a,50):.3f} p95 {np.percentile(a,95):.3f})"

def main(db_a, db_b):
    A, B = load(db_a), load(db_b)
    keys = sorted(set(A) & set(B))
    # only where both ok
    keys = [k for k in keys if A[k][2] == "ok" and B[k][2] == "ok"]
    print(f"A = {db_a}\nB = {db_b}\noverlap (both ok): {len(keys)}\n")
    fields = [("n_prims_out",4),("iou_pixel",5),("iou_skeleton",6),
              ("precision_pixel",7),("recall_pixel",8),("chamfer_sym",9),
              ("chamfer_p95_sym",10),("s2_time",11),("total_time",12)]
    print(f"{'metric':<16} {'A (before)':<34} {'B (after)':<34}")
    for name, idx in fields:
        a = col(A, keys, idx); b = col(B, keys, idx)
        print(f"{name:<16} {stat(a):<34} {stat(b):<34}")
    # fragmentation rate
    def frag(d):
        n = sum(1 for k in keys if d[k][4] is not None and d[k][3] is not None
                and d[k][4] > d[k][3])
        ratios = [d[k][4]/max(d[k][3],1) for k in keys
                  if d[k][4] is not None and d[k][3] is not None]
        return n, np.mean(ratios)
    na, ra = frag(A); nb, rb = frag(B)
    print(f"\nfrag cases (prims>gt):  A {na} ({100*na/len(keys):.1f}%)   "
          f"B {nb} ({100*nb/len(keys):.1f}%)")
    print(f"mean prim/stroke ratio: A {ra:.2f}   B {rb:.2f}")
    # paired iou delta
    ia = col(A, keys, 5, False); ib = col(B, keys, 5, False)
    print(f"\npaired IoU mean delta (B-A): {(ib-ia).mean():+.4f}")
    ca = col(A, keys, 9, False); cb = col(B, keys, 9, False)
    print(f"paired Chamfer mean delta (B-A): {(cb-ca).mean():+.4f} px")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
