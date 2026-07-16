"""Grid search over lr × pos_weight for the Phase 2 hatch-region CNN.

Runs 9 training configurations (3 learning rates × 3 pos_weight values),
saves every checkpoint and per-epoch history, then promotes the best model
to models/hatch_unet.pth.

All runs share the same train/val/test split (same seed) so results are
directly comparable.  Everything needed for a paper is saved:

    models/hatch_grid/
        splits.json                  figure-level train/val/test assignment
        run_lr{lr}_pw{pw}.pth        checkpoint for each configuration
        run_lr{lr}_pw{pw}_result.json  per-epoch loss + IoU history
        summary.csv                  ranked table (one row per run)
        summary.json                 same + full history for each run
    models/hatch_unet.pth            best model (ready for inference)

Usage:
    python -m tools.hatch_grid_search \
        --gt output/PatentData/hatch_gt \
        --tif-root data/PatentData/ReorganisedData \
        --out-dir models/hatch_grid \
        --best-model models/hatch_unet.pth \
        --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from itertools import product

import torch

from tools.hatch_dataset import load_figures, split_figures
from tools.hatch_train_cnn import run_training

# ─── Grid definition ─────────────────────────────────────────────────────────

LR_VALUES         = [3e-4, 1e-4, 3e-5]
POS_WEIGHT_VALUES = [3.0,  5.0,  8.0]

# Fixed across all runs
FIXED = dict(
    epochs        = 40,
    batch         = 32,
    patch         = 512,
    stride        = 256,
    samples       = 2000,
    workers       = 8,
    val_frac      = 0.1,
    test_frac     = 0.1,
    seed          = 42,
    unfreeze_after = None,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_id(lr: float, pw: float) -> str:
    return f"run_lr{lr:.0e}_pw{pw:.1f}"


def _save_splits(train_figs, val_figs, test_figs, path: str) -> None:
    def _rec(figs):
        return [{"patent": r.patent, "sketch": r.sketch,
                 "is_positive": r.is_positive} for r in figs]
    json.dump({"train": _rec(train_figs),
               "val":   _rec(val_figs),
               "test":  _rec(test_figs)},
              open(path, "w"), indent=2)


def _save_summary(results: list[dict], out_dir: str) -> None:
    """Write summary.csv and summary.json, sorted by best val_iou_pos."""
    ranked = sorted(results, key=lambda r: r["best_val_iou_pos"], reverse=True)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "summary.csv")
    fields = ["rank", "run_id", "lr", "pos_weight",
              "best_epoch", "best_val_iou_pos", "best_val_iou_all",
              "test_iou_pos", "test_iou_all"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rank, r in enumerate(ranked, 1):
            w.writerow({
                "rank":             rank,
                "run_id":           r["run_id"],
                "lr":               r["config"]["lr"],
                "pos_weight":       r["config"]["pos_weight"],
                "best_epoch":       r["best_epoch"],
                "best_val_iou_pos": round(r["best_val_iou_pos"], 4),
                "best_val_iou_all": round(r["best_val_iou_all"], 4),
                "test_iou_pos":     round(r["test_iou_pos"], 4),
                "test_iou_all":     round(r["test_iou_all"], 4),
            })

    # ── JSON ─────────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "summary.json")
    json.dump({
        "grid":        {"lr": LR_VALUES, "pos_weight": POS_WEIGHT_VALUES},
        "fixed_params": FIXED,
        "runs_ranked": ranked,
    }, open(json_path, "w"), indent=2)

    return ranked, csv_path, json_path


def _print_table(ranked: list[dict]) -> None:
    print(f"\n{'Rank':>4}  {'Run ID':<30}  {'lr':>8}  {'pos_w':>6}  "
          f"{'best_ep':>7}  {'val_iou_pos':>11}  {'test_iou_pos':>12}")
    print("─" * 90)
    for rank, r in enumerate(ranked, 1):
        marker = " ◀ BEST" if rank == 1 else ""
        print(f"{rank:>4}  {r['run_id']:<30}  "
              f"{r['config']['lr']:>8.0e}  {r['config']['pos_weight']:>6.1f}  "
              f"{r['best_epoch']:>7}  {r['best_val_iou_pos']:>11.4f}  "
              f"{r['test_iou_pos']:>12.4f}{marker}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",          default="output/PatentData/hatch_gt")
    ap.add_argument("--tif-root",    default="data/PatentData/ReorganisedData")
    ap.add_argument("--out-dir",     default="models/hatch_grid",
                     help="directory for per-run checkpoints and histories")
    ap.add_argument("--best-model",  default="models/hatch_unet.pth",
                     help="path to copy the best checkpoint to")
    ap.add_argument("--device",
                     default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume",      action="store_true",
                     help="skip runs whose result JSON already exists")
    ap.add_argument("--epochs",      type=int, default=FIXED["epochs"])
    ap.add_argument("--batch",       type=int, default=FIXED["batch"])
    ap.add_argument("--workers",     type=int, default=FIXED["workers"])
    ap.add_argument("--samples",     type=int, default=FIXED["samples"])
    ap.add_argument("--seed",        type=int, default=FIXED["seed"])
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.best_model)), exist_ok=True)

    fixed = dict(FIXED)
    fixed.update(epochs=a.epochs, batch=a.batch,
                 workers=a.workers, samples=a.samples, seed=a.seed)

    combinations = list(product(LR_VALUES, POS_WEIGHT_VALUES))
    n_total = len(combinations)

    # ── Load and split data ONCE (same split for all runs) ───────────────────
    print("=" * 70)
    print(f"Hatch CNN Grid Search  —  {n_total} configurations")
    print(f"Grid: lr={LR_VALUES}  pos_weight={POS_WEIGHT_VALUES}")
    print(f"Fixed: {fixed}")
    print("=" * 70)

    print("\nLoading ground-truth masks …")
    records = load_figures(a.gt, a.tif_root)
    train_figs, val_figs, test_figs = split_figures(
        records, fixed["val_frac"], fixed["test_frac"], fixed["seed"])

    splits_path = os.path.join(a.out_dir, "splits.json")
    _save_splits(train_figs, val_figs, test_figs, splits_path)
    print(f"Split saved → {splits_path}\n")

    # ── Run grid ─────────────────────────────────────────────────────────────
    all_results: list[dict] = []
    t_grid_start = time.time()

    for run_idx, (lr, pw) in enumerate(combinations, 1):
        rid      = _run_id(lr, pw)
        ckpt     = os.path.join(a.out_dir, f"{rid}.pth")
        hist_out = os.path.join(a.out_dir, f"{rid}_result.json")

        print(f"\n{'═'*70}")
        print(f"[{run_idx}/{n_total}]  {rid}   lr={lr:.0e}  pos_weight={pw}")
        print(f"{'═'*70}")

        # Resume: skip if result already saved
        if a.resume and os.path.exists(hist_out):
            print(f"  → already done, loading from {hist_out}")
            result = json.load(open(hist_out))
            result["run_id"] = rid
            all_results.append(result)
            continue

        result = run_training(
            gt=a.gt, tif_root=a.tif_root, out=ckpt,
            lr=lr, pos_weight=pw,
            train_figs=train_figs, val_figs=val_figs, test_figs=test_figs,
            device=a.device, verbose=True, **fixed,
        )
        result["run_id"] = rid

        json.dump(result, open(hist_out, "w"), indent=2)
        print(f"  History → {hist_out}")
        all_results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────────
    ranked, csv_path, json_path = _save_summary(all_results, a.out_dir)

    print(f"\n{'='*70}")
    print("GRID SEARCH COMPLETE")
    print(f"Total time: {(time.time()-t_grid_start)/60:.1f} min")
    _print_table(ranked)

    # Promote best model
    best = ranked[0]
    best_ckpt = best["checkpoint"]
    shutil.copy(best_ckpt, a.best_model)

    # Save best config separately for easy reference
    best_cfg_path = os.path.join(a.out_dir, "best_config.json")
    json.dump({
        "run_id":           best["run_id"],
        "config":           best["config"],
        "best_epoch":       best["best_epoch"],
        "best_val_iou_pos": best["best_val_iou_pos"],
        "best_val_iou_all": best["best_val_iou_all"],
        "test_iou_pos":     best["test_iou_pos"],
        "test_iou_all":     best["test_iou_all"],
        "checkpoint":       a.best_model,
    }, open(best_cfg_path, "w"), indent=2)

    print(f"\n  Best model → {a.best_model}")
    print(f"  Best config → {best_cfg_path}")
    print(f"  Summary CSV → {csv_path}")
    print(f"  Summary JSON → {json_path}")
    print(f"\nFor plotting (Python):")
    print(f"  import json")
    print(f"  s = json.load(open('{json_path}'))")
    print(f"  # s['runs_ranked'][i]['history'] → loss/IoU curves per run")


if __name__ == "__main__":
    main()
