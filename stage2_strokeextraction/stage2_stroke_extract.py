"""
stage2_stroke_extract.py
========================
AP3 Vectorization Pipeline — Stage 2: Stroke Extraction

Transforms a 1px binary skeleton (output of Stage 1) into a stroke graph:
  - Nodes : endpoints, junctions, sharp corners  (pixel coordinates)
  - Edges : ordered pixel chains connecting nodes (raw + smoothed coords)

Three internal layers:
  Layer 1 — Keypoint detection
      Puhachov et al. stacked-hourglass CNN if weights are available,
      otherwise classical crossing-number (CN) classification as fallback.

  Layer 2 — Topology extraction
      Dijkstra shortest-path on the skeleton pixel graph connects keypoints
      into edges. Closed loops (circles, rectangles) are detected separately.

  Layer 3 — Curve smoothing
      Ramer-Douglas-Peucker simplification followed by scipy B-spline fitting
      produces sub-pixel-accurate smooth_pts for each edge.

Output per sketch
  output/graphs/<sketch_id>_graph.json   ← stroke graph (nodes + edges)

Confidence signal
  isolation_ratio : float in [0, 1]
      Fraction of foreground pixels not captured by any edge.
      High value → missed junctions → flag for manual review.

Author : Yassine Rezgui — HAW Landshut / IP DrawingDrafter
"""

from __future__ import annotations

import json
import hashlib
import heapq
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
from skimage.morphology import skeletonize as _skeletonize
import networkx as nx
import numpy as np
from rdp import rdp
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


# ─── Output contract ────────────────────────────────────────────────────────

@dataclass
class Stage2Result:
    sketch_id: str
    graph_path: Path
    isolation_ratio: float      # confidence signal in [0, 1]
    flagged: bool
    processing_time_s: float
    keypoint_source: str        # "cnn" | "classical"
    n_nodes: int
    n_edges: int
    n_closed_edges: int = 0
    n_hachure_edges_removed: int = 0
    median_edge_length: float = 0.0
    micro_edge_ratio: float = 0.0   # open edges shorter than 6 px
    short_edge_ratio: float = 0.0   # open edges shorter than 15 px
    n_unclaimed_noncycle_components: int = 0
    max_unclaimed_noncycle_pixels: int = 0


# ─── Keypoint type constants ─────────────────────────────────────────────────

KP_ENDPOINT  = "endpoint"
KP_JUNCTION  = "junction"
KP_CORNER    = "corner"
KP_LOOP_ANCHOR = "loop_anchor"


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1 — KEYPOINT DETECTION
# ═══════════════════════════════════════════════════════════════════════════

class PuhachovKeypointDetector:
    """
    Wrapper around the Puhachov et al. stacked-hourglass keypoint CNN.

    Input : grayscale uint8 skeleton image
    Output: list of dicts {x, y, type, confidence}

    Raises ModelNotAvailableError if weights are missing or PyTorch
    is not installed — caller falls back to classical detection.
    """

    def __init__(self, weights_path: Optional[str], device: str = "cuda"):
        self._model  = None
        self._device = device
        self._ready  = False

        if not weights_path:
            raise ModelNotAvailableError("No weights path specified in config.")

        weights_file = Path(weights_path)
        if not weights_file.exists():
            raise ModelNotAvailableError(
                f"Puhachov weights not found at: {weights_file}\n"
                "  → Clone https://github.com/ivanpuhachov/"
                "line-drawing-vectorization-polyvector-flow\n"
                "  → Download best_model_checkpoint.pth from releases\n"
                "  → Set puhachov.weights in config.yaml"
            )

        self._load(weights_file, device)

    def _load(self, weights_file: Path, device: str) -> None:
        try:
            import torch
            # The Puhachov model is a stacked hourglass; load the checkpoint
            # as provided by the original repo (state_dict or full checkpoint).
            checkpoint = torch.load(weights_file, map_location=device, weights_only=False)
            state_dict = (checkpoint.get("state_dict")
                          or checkpoint.get("model_state_dict")
                          or checkpoint)
            model = _build_stacked_hourglass()

            # Be deliberately strict about compatibility even though the final
            # load uses strict=False.  Older revisions silently accepted the
            # shipped checkpoint although none of its tensor names matched this
            # lightweight wrapper, yielding effectively random heatmaps while
            # still reporting "cnn" as the keypoint source.
            model_state = model.state_dict()

            def _strip_known_prefixes(sd: dict) -> dict:
                out = {}
                for key, value in sd.items():
                    k = key
                    for prefix in ("module.", "model.", "net."):
                        if k.startswith(prefix):
                            k = k[len(prefix):]
                    out[k] = value
                return out

            state_dict = _strip_known_prefixes(state_dict)
            matched = [
                k for k, v in state_dict.items()
                if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
            ]
            min_required = max(10, int(0.20 * len(model_state)))
            if len(matched) < min_required:
                raise ModelNotAvailableError(
                    "Puhachov checkpoint is incompatible with the in-repo "
                    f"hourglass wrapper: matched {len(matched)}/{len(model_state)} "
                    f"model tensors from {weights_file}. Refusing to run an "
                    "effectively untrained keypoint detector."
                )

            load_result = model.load_state_dict(state_dict, strict=False)
            if load_result.missing_keys:
                logger.warning(
                    "Puhachov checkpoint loaded partially: %d missing, %d unexpected keys.",
                    len(load_result.missing_keys), len(load_result.unexpected_keys),
                )
            model.eval()
            model.to(device)
            self._model  = model
            self._device = device
            self._ready  = True
            logger.info(f"Puhachov keypoint CNN loaded on {device}")
        except ImportError:
            raise ModelNotAvailableError("PyTorch not installed.")
        except Exception as exc:
            raise ModelNotAvailableError(f"Failed to load Puhachov model: {exc}")

    def detect(
        self,
        skeleton: np.ndarray,
        conf_threshold: float = 0.5,
        nms_radius: int = 5,
    ) -> list[dict]:
        """
        Run the CNN on a binary skeleton image and return keypoints.

        Returns list of {x, y, type, confidence}.
        """
        import torch

        H, W = skeleton.shape
        # Normalise to [0, 1] float tensor (1, 1, H, W)
        img = skeleton.astype(np.float32) / 255.0

        # The stacked-hourglass architecture downsamples by 64× (stem /4 × depth-4
        # hourglass /16). Inputs whose dims are not multiples of 64 cause an
        # encoder/decoder shape mismatch on the upsampling skip-connection (e.g.
        # 23 vs 22 at dim 3). Pad to the next multiple, then crop back.
        STRIDE = 64
        pad_h = (STRIDE - H % STRIDE) % STRIDE
        pad_w = (STRIDE - W % STRIDE) % STRIDE
        if pad_h or pad_w:
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="constant")

        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            heatmaps = self._model(tensor)          # (1, 3, H_pad, W_pad)
            heatmaps = torch.sigmoid(heatmaps)
            heatmaps = heatmaps.squeeze(0).cpu().numpy()   # (3, H_pad, W_pad)

        # Crop padding back off
        heatmaps = heatmaps[:, :H, :W]

        # Channel 0 = endpoints, 1 = junctions, 2 = corners
        channel_types = [KP_ENDPOINT, KP_JUNCTION, KP_CORNER]
        keypoints = []
        for ch, kp_type in enumerate(channel_types):
            kps = _extract_peaks(heatmaps[ch], conf_threshold, nms_radius)
            for x, y, conf in kps:
                keypoints.append({"x": int(x), "y": int(y),
                                   "type": kp_type, "confidence": float(conf)})
        return keypoints

    def detect_tiled(
        self,
        skeleton: np.ndarray,
        conf_threshold: float = 0.5,
        nms_radius: int = 5,
        patch_size: int = 512,
        stride: int = 256,
    ) -> list[dict]:
        """
        Sliding-window keypoint detection for images far larger than the CNN's
        ~512 px training resolution.

        The stacked hourglass was trained on ~512 px sketches; run whole on a
        2000–2700 px patent TIF it sees features at the wrong scale and
        over-segments massively (10–300 k edges vs an expected 200–2 k). Here the
        CNN runs on `patch_size` tiles with 50 % overlap, the three keypoint
        heatmaps are averaged across the seams, and peaks are extracted once on
        the stitched full-resolution map — so the detector stays inside its
        trained receptive field while the graph is built at native resolution.
        Cross-seam duplicates are removed for free because NMS runs on the
        stitched map, not per tile. Mirrors the HatchUNet sliding-window path.

        `patch_size` must be a multiple of 64 (the hourglass downsamples 64×).
        """
        import torch

        H, W = skeleton.shape
        img  = skeleton.astype(np.float32) / 255.0

        n_ch = 3
        acc  = np.zeros((n_ch, H, W), np.float32)
        cnt  = np.zeros((H, W), np.float32)

        ys = list(range(0, max(1, H - patch_size), stride)) + [max(0, H - patch_size)]
        xs = list(range(0, max(1, W - patch_size), stride)) + [max(0, W - patch_size)]

        with torch.no_grad():
            for y0 in dict.fromkeys(ys):
                for x0 in dict.fromkeys(xs):
                    y1, x1 = y0 + patch_size, x0 + patch_size
                    tile   = img[y0:y1, x0:x1]
                    th, tw = tile.shape
                    if th < patch_size or tw < patch_size:
                        pad = np.zeros((patch_size, patch_size), np.float32)
                        pad[:th, :tw] = tile
                        tile = pad
                    t  = torch.from_numpy(tile[None, None]).to(self._device)
                    hm = torch.sigmoid(self._model(t))[0].cpu().numpy()   # (3, P, P)
                    acc[:, y0:y0 + th, x0:x0 + tw] += hm[:, :th, :tw]
                    cnt[y0:y0 + th, x0:x0 + tw]     += 1.0

        acc /= np.maximum(cnt, 1e-6)[None]

        channel_types = [KP_ENDPOINT, KP_JUNCTION, KP_CORNER]
        keypoints: list[dict] = []
        for ch, kp_type in enumerate(channel_types):
            for x, y, conf in _extract_peaks(acc[ch], conf_threshold, nms_radius):
                keypoints.append({"x": int(x), "y": int(y),
                                   "type": kp_type, "confidence": float(conf)})
        return keypoints


def _extract_peaks(
    heatmap: np.ndarray,
    threshold: float,
    nms_radius: int,
) -> list[tuple[int, int, float]]:
    """
    Extract local maxima from a heatmap above a confidence threshold.
    Non-maximum suppression within nms_radius.
    Returns list of (x, y, confidence).
    """
    h, w = heatmap.shape
    peaks = []
    # 2D max-pool approximation via dilation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * nms_radius + 1, 2 * nms_radius + 1)
    )
    dilated = cv2.dilate(heatmap.astype(np.float32), kernel)
    local_max = (heatmap == dilated) & (heatmap >= threshold)
    ys, xs = np.where(local_max)
    for y, x in zip(ys, xs):
        peaks.append((int(x), int(y), float(heatmap[y, x])))
    return peaks


class ModelNotAvailableError(Exception):
    pass


# ─── Stacked hourglass model skeleton ────────────────────────────────────────

def _build_stacked_hourglass():
    """
    Minimal stacked hourglass architecture matching the Puhachov repo.

    Input : (B, 1, H, W)
    Output: (B, 3, H, W) — one channel per keypoint class
    """
    try:
        import torch
        import torch.nn as nn

        def _conv_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        class ResBlock(nn.Module):
            def __init__(self, ch):
                super().__init__()
                self.net = nn.Sequential(
                    _conv_bn_relu(ch, ch // 2, k=1, p=0),
                    _conv_bn_relu(ch // 2, ch // 2),
                    nn.Conv2d(ch // 2, ch, 1, bias=False),
                    nn.BatchNorm2d(ch),
                )
                self.relu = nn.ReLU(inplace=True)

            def forward(self, x):
                return self.relu(x + self.net(x))

        class Hourglass(nn.Module):
            def __init__(self, depth, ch):
                super().__init__()
                self.depth = depth
                self.down   = nn.MaxPool2d(2, stride=2)
                self.up     = nn.Upsample(scale_factor=2, mode="bilinear",
                                           align_corners=False)
                self.pre    = ResBlock(ch)
                self.lower  = ResBlock(ch)
                self.inner  = ResBlock(ch) if depth == 1 else Hourglass(depth-1, ch)
                self.after  = ResBlock(ch)
                self.skip   = ResBlock(ch)

            def forward(self, x):
                up    = self.skip(x)
                low   = self.lower(self.down(self.pre(x)))
                low   = self.inner(low)
                low   = self.after(low)
                return up + self.up(low)

        class StackedHourglass(nn.Module):
            """Two stacked hourglass modules with intermediate supervision."""
            def __init__(self, n_classes=3, ch=256):
                super().__init__()
                # Stem
                self.stem = nn.Sequential(
                    _conv_bn_relu(1, 64, k=7, s=2, p=3),
                    ResBlock(64),
                    nn.MaxPool2d(2, stride=2),
                    _conv_bn_relu(64, 128, k=1, p=0),
                    _conv_bn_relu(128, ch, k=1, p=0),
                )
                # Two hourglass modules
                self.hg1 = Hourglass(4, ch)
                self.hg2 = Hourglass(4, ch)
                # Intermediate output
                self.out1  = nn.Conv2d(ch, n_classes, 1)
                self.remap = nn.Conv2d(n_classes, ch, 1)
                self.merge = nn.Conv2d(ch, ch, 1)
                # Final output
                self.out2  = nn.Conv2d(ch, n_classes, 1)
                # Upsample back to input size (stem did /4)
                self.up4   = nn.Upsample(scale_factor=4, mode="bilinear",
                                          align_corners=False)

            def forward(self, x):
                feat  = self.stem(x)
                feat1 = self.hg1(feat)
                hm1   = self.out1(feat1)
                feat2 = self.hg2(feat + self.merge(feat1) + self.remap(hm1))
                hm2   = self.out2(feat2)
                return self.up4(hm2)      # return final heatmaps only

        return StackedHourglass()

    except ImportError:
        raise ModelNotAvailableError("PyTorch not installed.")


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 1 — CLASSICAL FALLBACK (crossing-number based)
# ═══════════════════════════════════════════════════════════════════════════

def _cn_map_vectorized(binary: np.ndarray) -> np.ndarray:
    """
    Compute the crossing-number map for every foreground pixel using NumPy
    slice operations instead of a pixel-level Python loop.

    CN(p) = (Σ_{i=0}^{7} |n_i − n_{(i+1)%8}|) / 2
    where n_0…n_7 are the 8 clockwise ring neighbours.

    Typical speedup: ~1 000× at 1 000 px resolution vs the Python loop.
    Returns int16 array shaped (H, W); background pixels are zero.
    """
    b  = binary.astype(np.int16)
    H, W = b.shape

    # Ring neighbours in clockwise order: N, NE, E, SE, S, SW, W, NW
    ring = [
        b[:-2, 1:-1],   # N
        b[:-2, 2:],     # NE
        b[1:-1, 2:],    # E
        b[2:,  2:],     # SE
        b[2:,  1:-1],   # S
        b[2:,  :-2],    # SW
        b[1:-1, :-2],   # W
        b[:-2, :-2],    # NW
    ]
    cn_inner = sum(np.abs(ring[i] - ring[(i + 1) % 8]) for i in range(8)) // 2

    cn = np.zeros((H, W), dtype=np.int16)
    cn[1:-1, 1:-1] = cn_inner
    cn[binary == 0] = 0   # ensure background is zero
    return cn


def _classical_keypoints(skeleton: np.ndarray) -> list[dict]:
    """
    Classify foreground pixels by crossing number (CN):
      CN = 1  → endpoint
      CN >= 3 → junction
    Returns list of {x, y, type, confidence}.

    Junction pixels are clustered (connected components → centroid).
    Uses the vectorised CN map for speed.
    """
    binary = (skeleton > 0).astype(np.uint8)
    cn     = _cn_map_vectorized(binary)

    endpoint_mask = ((cn == 1) & (binary == 1)).astype(np.uint8) * 255
    junction_mask = ((cn >= 3) & (binary == 1)).astype(np.uint8) * 255

    keypoints = []

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(endpoint_mask)
    for i in range(1, n):
        cx, cy = centroids[i]
        keypoints.append({
            "x": int(round(cx)), "y": int(round(cy)),
            "type": KP_ENDPOINT, "confidence": 1.0,
        })

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(junction_mask)
    for i in range(1, n):
        cx, cy = centroids[i]
        keypoints.append({
            "x": int(round(cx)), "y": int(round(cy)),
            "type": KP_JUNCTION, "confidence": 1.0,
        })

    return keypoints


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2 — TOPOLOGY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def _build_pixel_graph(skeleton: np.ndarray) -> nx.Graph:
    """
    Build a NetworkX graph where every foreground pixel is a node.
    Edges connect 8-connected neighbours with cost = 1.0 (uniform).
    Using adjacency list avoids dense matrix allocation for large images.
    """
    binary = (skeleton > 0)
    ys, xs = np.where(binary)
    G = nx.Graph()

    for y, x in zip(ys, xs):
        G.add_node((x, y))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if (0 <= ny < skeleton.shape[0]
                        and 0 <= nx_ < skeleton.shape[1]
                        and binary[ny, nx_]):
                    # Diagonal edges cost sqrt(2) for geometric accuracy
                    cost = 1.414 if (dx != 0 and dy != 0) else 1.0
                    G.add_edge((x, y), (nx_, ny), weight=cost)

    return G


def _cn_keypoint_clusters(skeleton: np.ndarray) -> list[dict]:
    """
    Classical CN keypoint clusters: connected components of CN==1 (endpoints)
    then CN>=3 (junctions), each carrying its core skeleton pixels as
    ``{x, y, type, confidence, pixels:[(x,y), …]}``.

    This is the default seeding consumed by ``_extract_topology``. Emitting
    explicit clusters — instead of letting ``_extract_topology`` recompute CN
    internally — is the contract a learned keypoint detector also targets
    (Phase 3 of the Puhachov roadmap): produce the same cluster list and
    topology extraction is byte-for-byte identical.
    """
    binary = (skeleton > 0).astype(np.uint8)
    cn_map = _cn_map_vectorized(binary)
    clusters: list[dict] = []
    for mask_bool, kp_type in (((cn_map == 1), KP_ENDPOINT),
                               ((cn_map >= 3), KP_JUNCTION)):
        mask = mask_bool.astype(np.uint8)
        n, labels = cv2.connectedComponents(mask)
        if n <= 1:
            continue
        # Single pass: all foreground pixels and their cluster labels, sorted by
        # label so same-cluster pixels are contiguous (never np.where per cluster).
        ys_all, xs_all = np.where(labels > 0)
        if len(ys_all) == 0:
            continue
        lv     = labels[ys_all, xs_all]
        order  = np.argsort(lv, kind="stable")
        xs_s   = xs_all[order]; ys_s = ys_all[order]; lv_s = lv[order]
        splits = np.where(np.diff(lv_s))[0] + 1
        starts = np.concatenate([[0], splits])
        ends   = np.concatenate([splits, [len(lv_s)]])
        for s, e in zip(starts.tolist(), ends.tolist()):
            xs_g = xs_s[s:e]; ys_g = ys_s[s:e]
            clusters.append({
                "x": int(np.mean(xs_g)), "y": int(np.mean(ys_g)),
                "type": kp_type, "confidence": 1.0,
                "pixels": list(zip(xs_g.tolist(), ys_g.tolist())),
            })
    return clusters


def _clusters_from_points(
    keypoints: list[dict], skeleton: np.ndarray, snap_radius: int = 3
) -> list[dict]:
    """
    Build keypoint clusters from bare point detections (the learned-detector
    path). Each ``{x, y, type}`` is snapped to the nearest skeleton foreground
    pixel within ``snap_radius``; that single pixel becomes the cluster core and
    the 1-px halo in ``_materialize_keypoint_clusters`` absorbs the rest. Points
    snapping to the same pixel are de-duplicated.

    NOTE: minimal plumbing for the Puhachov CNN path — not exercised while
    ``puhachov.weights`` is empty. Phase 3 proper may grow richer cores (e.g.
    the local CN-connected blob) so CNN junctions match the multi-pixel
    clusters the CN path produces.
    """
    binary = (skeleton > 0)
    H, W = binary.shape
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return []
    fg = np.stack([xs, ys], axis=1).astype(np.int64)   # (N,2) as (x,y)
    used: dict[tuple, dict] = {}
    for kp in keypoints:
        px, py = int(round(kp["x"])), int(round(kp["y"]))
        if 0 <= py < H and 0 <= px < W and binary[py, px]:
            sx, sy = px, py
        else:
            d2 = (fg[:, 0] - px) ** 2 + (fg[:, 1] - py) ** 2
            j = int(np.argmin(d2))
            if d2[j] > snap_radius ** 2:
                continue
            sx, sy = int(fg[j, 0]), int(fg[j, 1])
        key = (sx, sy)
        if key in used:
            continue
        used[key] = {
            "x": sx, "y": sy,
            "type": kp.get("type", KP_JUNCTION),
            "confidence": float(kp.get("confidence", 1.0)),
            "pixels": [(sx, sy)],
        }
    return list(used.values())


def _fuse_cn_cnn_clusters(
    skeleton: np.ndarray, cnn_keypoints: list[dict], min_corner_dist: float = 5.0
) -> list[dict]:
    """
    Fusion seeding: keep the CN endpoints + junctions (near-exact on clean
    skeletons via crossing number) and add ONLY the CNN's *corners* — genuine
    `CN==2` points the crossing number cannot detect. CNN corners within
    `min_corner_dist` px of an existing CN endpoint/junction are dropped as
    redundant (so a corner next to a real junction doesn't spawn a tiny spur).
    """
    cn = _cn_keypoint_clusters(skeleton)
    base = [c for c in cn if c["type"] in (KP_ENDPOINT, KP_JUNCTION)]
    corner_pts = [k for k in cnn_keypoints if k.get("type") == KP_CORNER]
    corners = _clusters_from_points(corner_pts, skeleton)
    if base and corners:
        bx = np.array([[c["x"], c["y"]] for c in base], dtype=np.float64)
        d2_min = float(min_corner_dist) ** 2
        corners = [c for c in corners
                   if ((bx[:, 0] - c["x"]) ** 2 + (bx[:, 1] - c["y"]) ** 2).min() > d2_min]
    return base + corners


def _hatch_bridge_junction_clusters(
    skeleton: np.ndarray,
    hatch_mask: np.ndarray,
    structural_clusters: list[dict],
    dedup_radius: float = 5.0,
) -> list[dict]:
    """Return temporary CN junctions needed to cross hatch ink.

    A hatch-suppressed detector intentionally does not emit the raster
    junctions created where hatch lines cross an object contour.  The pixel
    tracer still needs a temporary node at those crossings so graph
    simplification can pair the straight-through object branches.  Only CN
    junctions touching the learned hatch region are admitted; hatch endpoints
    stay out of the main graph and are retained by the separate hatch prepass.
    """
    if hatch_mask.shape != skeleton.shape:
        raise ValueError(
            f"hatch mask shape {hatch_mask.shape} != skeleton {skeleton.shape}"
        )
    if dedup_radius < 0:
        raise ValueError("hatch bridge dedup radius must be non-negative")

    structural_xy = np.asarray(
        [[cluster["x"], cluster["y"]] for cluster in structural_clusters],
        dtype=np.float64,
    ).reshape(-1, 2)
    maximum_squared = float(dedup_radius) ** 2
    bridges: list[dict] = []
    for cluster in _cn_keypoint_clusters(skeleton):
        if cluster.get("type") != KP_JUNCTION:
            continue
        if not any(
            hatch_mask[int(y), int(x)] > 0
            for x, y in cluster.get("pixels", [])
        ):
            continue
        point = np.asarray([cluster["x"], cluster["y"]], dtype=np.float64)
        if len(structural_xy):
            distance_squared = np.sum((structural_xy - point) ** 2, axis=1)
            if float(np.min(distance_squared)) <= maximum_squared:
                continue
        bridge = dict(cluster)
        bridge["structural_seed"] = False
        bridge["topology_origin"] = "hatch_bridge_cn"
        bridges.append(bridge)
    return bridges


def _junction_arm_features(
    skeleton: np.ndarray,
    mask: np.ndarray,
    cluster: dict,
    *,
    support_length: float = 10.0,
    core_radius: int = 1,
    minimum_pixels: int = 4,
) -> list[dict]:
    """Measure direction and thin-mask support for each CN junction arm.

    Removing a one-pixel halo around the CN core separates the local arms. A
    short direction-preserving trace then measures each arm independently,
    avoiding a flood fill that could reconnect through the surrounding hatch
    lattice. The result is used only to decide whether a junction mixes hatch
    and structural ink; it does not alter topology itself.
    """
    if skeleton.shape != mask.shape:
        raise ValueError("skeleton and junction mask shapes must match")
    if support_length <= 0:
        raise ValueError("junction arm support length must be positive")
    if core_radius < 0:
        raise ValueError("junction arm core radius must be non-negative")
    if minimum_pixels <= 0:
        raise ValueError("junction arm minimum pixels must be positive")

    binary = skeleton > 0
    height, width = binary.shape
    core = np.zeros_like(skeleton, dtype=np.uint8)
    for x, y in cluster.get("pixels", []):
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height and binary[y, x]:
            core[y, x] = 1
    if not np.any(core):
        x, y = int(cluster["x"]), int(cluster["y"])
        if not (0 <= x < width and 0 <= y < height and binary[y, x]):
            return []
        core[y, x] = 1

    if core_radius > 0:
        size = core_radius * 2 + 1
        blocked = cv2.dilate(
            core, np.ones((size, size), dtype=np.uint8)
        ).astype(bool)
    else:
        blocked = core.astype(bool)
    blocked &= binary

    boundary = (
        binary
        & ~blocked
        & (cv2.dilate(blocked.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    )
    component_count, labels = cv2.connectedComponents(
        boundary.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return []

    centre = np.asarray(
        [float(cluster["x"]), float(cluster["y"])], dtype=np.float64
    )
    features: list[dict] = []

    def neighbours8(point: tuple[int, int]):
        x, y = point
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < width and 0 <= ny_ < height and binary[ny_, nx_]:
                    yield (nx_, ny_)

    blocked_pixels = {
        (int(x), int(y))
        for y, x in zip(*np.where(blocked))
    }
    for component_id in range(1, component_count):
        ys, xs = np.where(labels == component_id)
        if len(xs) == 0:
            continue
        points = [(int(x), int(y)) for x, y in zip(xs, ys)]
        start = max(
            points,
            key=lambda point: (
                float(np.sum((np.asarray(point, dtype=float) - centre) ** 2)),
                -point[1],
                -point[0],
            ),
        )
        initial = np.asarray(start, dtype=np.float64) - centre
        initial_norm = float(np.linalg.norm(initial))
        if initial_norm <= 1e-9:
            continue
        initial /= initial_norm

        chain = [start]
        visited = set(blocked_pixels)
        visited.add(start)
        travelled = 0.0
        previous_direction = initial
        while travelled < support_length:
            current = chain[-1]
            candidates = []
            for candidate in neighbours8(current):
                if candidate in visited:
                    continue
                label = int(labels[candidate[1], candidate[0]])
                if label > 0 and label != component_id:
                    continue
                step = np.asarray(candidate, dtype=np.float64) - np.asarray(
                    current, dtype=np.float64
                )
                step_norm = float(np.linalg.norm(step))
                if step_norm <= 1e-9:
                    continue
                step_direction = step / step_norm
                radial_gain = float(
                    np.linalg.norm(np.asarray(candidate, dtype=float) - centre)
                    - np.linalg.norm(np.asarray(current, dtype=float) - centre)
                )
                candidates.append((
                    float(np.dot(previous_direction, step_direction)),
                    radial_gain,
                    float(np.dot(initial, step_direction)),
                    -candidate[1],
                    -candidate[0],
                    candidate,
                    step_direction,
                    step_norm,
                ))
            if not candidates:
                break
            selected = max(candidates)
            candidate = selected[-3]
            previous_direction = selected[-2]
            step_norm = selected[-1]
            if travelled + step_norm > support_length + 1e-9:
                break
            chain.append(candidate)
            visited.add(candidate)
            travelled += step_norm

        if len(chain) < minimum_pixels:
            continue
        delta = np.asarray(chain[-1], dtype=float) - np.asarray(
            chain[0], dtype=float
        )
        if float(np.linalg.norm(delta)) <= 1e-9:
            continue
        features.append({
            "direction_deg": float(
                np.degrees(np.arctan2(delta[1], delta[0])) % 360.0
            ),
            "angle_deg": float(
                np.degrees(np.arctan2(delta[1], delta[0])) % 180.0
            ),
            "mask_fraction": float(np.mean([
                bool(mask[y, x]) for x, y in chain
            ])),
            "n_pixels": len(chain),
        })
    return features


def _junction_arm_mask_fractions(
    skeleton: np.ndarray,
    mask: np.ndarray,
    cluster: dict,
    *,
    support_length: float = 10.0,
    core_radius: int = 1,
    minimum_pixels: int = 4,
) -> list[float]:
    """Compatibility view of local junction-arm thin-mask coverage."""
    return [
        float(feature["mask_fraction"])
        for feature in _junction_arm_features(
            skeleton,
            mask,
            cluster,
            support_length=support_length,
            core_radius=core_radius,
            minimum_pixels=minimum_pixels,
        )
    ]


def _mixed_hatch_junction_clusters(
    skeleton: np.ndarray,
    hatch_stroke_mask: np.ndarray,
    candidates: list[dict],
    cfg: dict,
    hough_families: list[dict] | None = None,
) -> list[dict]:
    """Keep only junctions joining Hough hatch arms to non-hatch arms."""
    minimum_hatch_fraction = float(
        cfg.get("hachure_hough_bridge_min_hatch_arm_frac", 0.55)
    )
    maximum_structural_fraction = float(
        cfg.get("hachure_hough_bridge_max_structural_arm_frac", 0.25)
    )
    if not (0.0 <= maximum_structural_fraction < minimum_hatch_fraction <= 1.0):
        raise ValueError("invalid mixed Hough junction arm thresholds")
    angle_tolerance = float(
        cfg.get("hachure_hough_bridge_family_angle_tolerance", 15.0)
    )
    minimum_aligned_fraction = float(
        cfg.get("hachure_hough_bridge_min_aligned_arm_frac", 0.10)
    )
    bbox_margin = float(
        cfg.get("hachure_hough_bridge_family_bbox_margin", 4.0)
    )
    if angle_tolerance <= 0 or angle_tolerance > 90:
        raise ValueError("invalid mixed Hough junction angle tolerance")
    if not (0.0 <= minimum_aligned_fraction <= 1.0):
        raise ValueError("invalid aligned Hough arm support threshold")
    pair_tolerance = float(
        cfg.get("hachure_hough_bridge_pair_angle_tolerance", 35.0)
    )
    if pair_tolerance <= 0 or pair_tolerance > 90:
        raise ValueError("invalid mixed Hough arm-pair tolerance")

    def has_opposite_pair(features: list[dict], indices: list[int]) -> bool:
        for first_offset, first in enumerate(indices):
            for second in indices[first_offset + 1:]:
                delta = abs(
                    float(features[first]["direction_deg"])
                    - float(features[second]["direction_deg"])
                ) % 360.0
                delta = min(delta, 360.0 - delta)
                if abs(180.0 - delta) <= pair_tolerance:
                    return True
        return False

    selected: list[dict] = []
    for candidate in candidates:
        features = _junction_arm_features(
            skeleton,
            hatch_stroke_mask,
            candidate,
            support_length=float(
                cfg.get("hachure_hough_bridge_arm_length", 10.0)
            ),
            core_radius=int(
                cfg.get("hachure_hough_bridge_arm_core_radius", 1)
            ),
            minimum_pixels=int(
                cfg.get("hachure_hough_bridge_min_arm_pixels", 4)
            ),
        )
        local_angles: list[float] = []
        for family in hough_families or []:
            bbox = family.get("bbox")
            if bbox and len(bbox) == 4:
                left, top, width, height = (float(value) for value in bbox)
                if not (
                    left - bbox_margin <= float(candidate["x"])
                    <= left + width - 1 + bbox_margin
                    and top - bbox_margin <= float(candidate["y"])
                    <= top + height - 1 + bbox_margin
                ):
                    continue
            local_angles.append(float(family["angle_deg"]))

        if local_angles:
            aligned = [
                min(
                    _angle_delta_deg(feature["angle_deg"], family_angle)
                    for family_angle in local_angles
                ) <= angle_tolerance
                for feature in features
            ]
            hatch_indices = [
                index for index, is_aligned in enumerate(aligned)
                if is_aligned
            ]
            structural_indices = [
                index for index, (feature, is_aligned) in enumerate(
                    zip(features, aligned)
                )
                if not is_aligned
                and feature["mask_fraction"] <= maximum_structural_fraction
            ]
            has_supported_hatch_arm = any(
                is_aligned
                and feature["mask_fraction"] >= minimum_aligned_fraction
                for feature, is_aligned in zip(features, aligned)
            )
            structural_pair = has_opposite_pair(
                features, structural_indices
            )
            if len(features) == 3:
                solvable = (
                    len(hatch_indices) == 1
                    and len(structural_indices) == 2
                    and structural_pair
                )
            elif len(features) == 4:
                solvable = (
                    len(hatch_indices) == 2
                    and len(structural_indices) == 2
                    and structural_pair
                    and has_opposite_pair(features, hatch_indices)
                )
            else:
                solvable = False
            has_hatch_arm = has_supported_hatch_arm and solvable
            has_structural_arm = structural_pair and solvable
        else:
            has_hatch_arm = any(
                feature["mask_fraction"] >= minimum_hatch_fraction
                for feature in features
            )
            has_structural_arm = any(
                feature["mask_fraction"] <= maximum_structural_fraction
                for feature in features
            )
        if has_hatch_arm and has_structural_arm:
            selected.append(candidate)
    return selected


def _structural_cn_endpoint_clusters(
    skeleton: np.ndarray,
    hatch_stroke_mask: np.ndarray,
    structural_clusters: list[dict],
    dedup_radius: float = 5.0,
) -> list[dict]:
    """Recover missed open-stroke endpoints outside verified hatch ink."""
    if hatch_stroke_mask.shape != skeleton.shape:
        raise ValueError(
            "hatch stroke mask shape "
            f"{hatch_stroke_mask.shape} != skeleton {skeleton.shape}"
        )
    if dedup_radius < 0:
        raise ValueError("endpoint recovery dedup radius must be non-negative")

    structural_xy = np.asarray(
        [[cluster["x"], cluster["y"]] for cluster in structural_clusters],
        dtype=np.float64,
    ).reshape(-1, 2)
    maximum_squared = float(dedup_radius) ** 2
    recovered: list[dict] = []
    for cluster in _cn_keypoint_clusters(skeleton):
        if cluster.get("type") != KP_ENDPOINT:
            continue
        if any(
            hatch_stroke_mask[int(y), int(x)] > 0
            for x, y in cluster.get("pixels", [])
        ):
            continue
        point = np.asarray([cluster["x"], cluster["y"]], dtype=np.float64)
        if len(structural_xy):
            distance_squared = np.sum((structural_xy - point) ** 2, axis=1)
            if float(np.min(distance_squared)) <= maximum_squared:
                continue
        endpoint = dict(cluster)
        endpoint["structural_seed"] = True
        endpoint["topology_origin"] = "structural_cn_endpoint_recovery"
        recovered.append(endpoint)
    return recovered


def _hatch_cn_endpoint_clusters(
    skeleton: np.ndarray,
    hatch_stroke_mask: np.ndarray,
    existing_clusters: list[dict],
    dedup_radius: float = 5.0,
) -> list[dict]:
    """Return temporary CN endpoints belonging to a thin hatch stroke mask."""
    if hatch_stroke_mask.shape != skeleton.shape:
        raise ValueError(
            "hatch stroke mask shape "
            f"{hatch_stroke_mask.shape} != skeleton {skeleton.shape}"
        )
    if dedup_radius < 0:
        raise ValueError("hatch endpoint dedup radius must be non-negative")

    existing_xy = np.asarray(
        [[cluster["x"], cluster["y"]] for cluster in existing_clusters],
        dtype=np.float64,
    ).reshape(-1, 2)
    maximum_squared = float(dedup_radius) ** 2
    endpoints: list[dict] = []
    for cluster in _cn_keypoint_clusters(skeleton):
        if cluster.get("type") != KP_ENDPOINT:
            continue
        if not any(
            hatch_stroke_mask[int(y), int(x)] > 0
            for x, y in cluster.get("pixels", [])
        ):
            continue
        point = np.asarray([cluster["x"], cluster["y"]], dtype=np.float64)
        if len(existing_xy):
            distance_squared = np.sum((existing_xy - point) ** 2, axis=1)
            if float(np.min(distance_squared)) <= maximum_squared:
                continue
        endpoint = dict(cluster)
        endpoint["structural_seed"] = False
        endpoint["topology_origin"] = "hatch_endpoint_cn"
        endpoints.append(endpoint)
    return endpoints


def _materialize_keypoint_clusters(
    clusters: list[dict], binary: np.ndarray, H: int, W: int
) -> tuple[list[dict], dict, dict]:
    """
    Turn a keypoint-cluster list into (kp_info, kp_map, kp_pixels) for the walk.

    Registers each cluster's core pixels (cluster id = list index), then extends
    every cluster by 1px via a single bulk dilation of a float label image
    (max-pool: each extended pixel takes the highest adjacent kid+1; higher ID
    wins ties — fine, any adjacent cluster stops the walk). This is the legacy
    extension step, factored out unchanged.
    """
    kp_info: list[dict] = []
    kp_map: dict = {}
    kp_pixels: dict = {}
    for cl in clusters:
        kid = len(kp_info)
        node = {
            "id": kid,
            "x": int(cl["x"]), "y": int(cl["y"]),
            "type": cl["type"], "confidence": float(cl.get("confidence", 1.0)),
        }
        for attribute in ("structural_seed", "topology_origin"):
            if attribute in cl:
                node[attribute] = cl[attribute]
        kp_info.append(node)
        kp_pixels[kid] = set()
        for x, y in cl["pixels"]:
            kp_map[(int(x), int(y))] = kid
            kp_pixels[kid].add((int(x), int(y)))

    if kp_info:
        label_img = np.zeros((H, W), dtype=np.float32)
        for (x, y), kid in kp_map.items():
            label_img[y, x] = float(kid + 1)
        dilated = cv2.dilate(label_img, np.ones((3, 3), np.uint8))
        ext_ys, ext_xs = np.where((dilated > 0) & (binary > 0) & (label_img == 0))
        for y, x in zip(ext_ys.tolist(), ext_xs.tolist()):
            kid = int(dilated[y, x]) - 1
            if (x, y) not in kp_map:
                kp_map[(x, y)] = kid
                kp_pixels[kid].add((x, y))
    return kp_info, kp_map, kp_pixels


def _residual_chains(pixels):
    """Partition digital adjacencies into maximal degree-2 chains, without chords."""
    points = {tuple(map(int, point)) for point in pixels}
    adjacency = {}
    for x, y in points:
        neighbours = []
        for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0),
                       (1, 0), (-1, 1), (0, 1), (1, 1)):
            neighbour = (x + dx, y + dy)
            if neighbour not in points:
                continue
            if dx and dy and ((x + dx, y) in points or (x, y + dy) in points):
                continue
            neighbours.append(neighbour)
        adjacency[(x, y)] = sorted(neighbours)
    visited = set()
    chains = []

    def link(a, b):
        return (a, b) if a < b else (b, a)

    def walk(start, following):
        chain = [start, following]
        visited.add(link(start, following))
        previous, current = start, following
        while len(adjacency[current]) == 2 and current != start:
            following = next(point for point in adjacency[current] if point != previous)
            if link(current, following) in visited:
                break
            visited.add(link(current, following))
            chain.append(following)
            previous, current = current, following
        chains.append(chain)

    for point in sorted(points):
        neighbours = adjacency[point]
        if not neighbours:
            chains.append([point])
        if len(neighbours) == 2:
            continue
        for neighbour in neighbours:
            if link(point, neighbour) not in visited:
                walk(point, neighbour)
    # Components containing only degree-2 pixels are actual cycles.
    for point in sorted(points):
        for neighbour in adjacency[point]:
            if link(point, neighbour) not in visited:
                walk(point, neighbour)
    return adjacency, chains


def repair_noncycle_residuals(nodes, edges):
    """Replace unclaimed pseudo-loops with ordered traces; retain all source pixels.

    This also supports frozen graph replay. Untouched edges keep their IDs;
    additional chains receive new IDs and retain their parent's provenance.
    """
    output_nodes = list(nodes)
    output_edges = []
    repaired = False
    next_node = max((node["id"] for node in nodes), default=-1) + 1
    next_edge = max((edge["id"] for edge in edges), default=-1) + 1
    for edge in edges:
        if (edge.get("topology_origin") != "unclaimed_component"
                or edge.get("is_simple_cycle") is not False
                or len(edge.get("pixels", [])) > 100_000):
            output_edges.append(edge)
            continue
        adjacency, chains = _residual_chains(edge["pixels"])
        repaired = True
        anchors = {}

        def anchor(point, closed):
            nonlocal next_node
            if point not in anchors:
                anchors[point] = next_node
                degree = len(adjacency[point])
                kind = KP_JUNCTION if degree > 2 else KP_LOOP_ANCHOR if closed else KP_ENDPOINT
                output_nodes.append({"id": next_node, "x": point[0], "y": point[1],
                                     "type": kind, "confidence": 1.0})
                next_node += 1
            return anchors[point]

        for index, chain in enumerate(chains):
            closed = len(chain) > 2 and chain[0] == chain[-1]
            source, target = anchor(chain[0], closed), anchor(chain[-1], closed)
            output_edges.append({
                "id": edge["id"] if index == 0 else next_edge,
                "source": source, "target": target,
                "pixels": [list(point) for point in chain], "smooth_pts": [],
                "is_closed": closed, "is_simple_cycle": closed,
                "topology_origin": "recovered_residual",
                "residual_parent_edge_ids": [edge["id"]],
            })
            if index:
                next_edge += 1
        recovered = {point for chain in chains for point in chain}
        if recovered != set(adjacency):
            raise RuntimeError("Residual repair lost source pixels")
    if not repaired:
        return nodes, edges
    used_nodes = {edge[key] for edge in output_edges for key in ("source", "target")}
    return [node for node in output_nodes if node["id"] in used_nodes], output_edges


def _coverage_spans(mask):
    """Lossless row runs [y, first_x, exclusive_end_x] for source accounting."""
    spans = []
    for y in np.flatnonzero(np.any(mask, axis=1)):
        changes = np.diff(np.r_[False, mask[y], False].astype(np.int8))
        spans.extend([int(y), int(a), int(b)] for a, b in
                     zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    return spans


class _CoverageLedger:
    """Track actual source pixels, never node proximity or inferred hatch fill."""

    def __init__(self, skeleton):
        self.source = np.asarray(skeleton, dtype=bool).copy()
        self.previous = self.source.copy()
        self.stages = []

    def record(self, operation, edges, hatches=(), *, active_skeleton=None):
        represented = _edge_support_mask(edges, self.source.shape).astype(bool)
        represented |= _edge_support_mask(hatches, self.source.shape).astype(bool)
        if active_skeleton is not None:
            represented |= np.asarray(active_skeleton) > 0
        represented &= self.source
        lost = self.previous & ~represented
        self.stages.append({
            "operation": operation,
            "represented_source_pixels": int(represented.sum()),
            "missing_source_pixels": int((self.source & ~represented).sum()),
            "newly_missing_pixels": int(lost.sum()),
            "restored_pixels": int((represented & ~self.previous).sum()),
            "newly_missing_spans": _coverage_spans(lost),
        })
        self.previous = represented

    def report(self, recovery, *, scale=1.0, original_shape=None):
        total = int(self.source.sum())
        missing = self.source & ~self.previous
        residual = np.zeros_like(missing)
        for y, start, end in recovery["unresolved_spans"]:
            residual[y, start:end] = True
        if not np.array_equal(residual, missing):
            raise RuntimeError("Coverage residuals do not match the final graph")
        return {
            "schema": "ap3-stage2-coverage-v1", "coordinate_frame": "stage2_pixels",
            "image_shape": list(self.source.shape), "stage2_scale": scale,
            "original_image_shape": list(original_shape or self.source.shape),
            "original_grid_preserved": scale == 1.0,
            "source_mask_sha256": hashlib.sha256(np.packbits(self.source).tobytes()).hexdigest(),
            "stages": self.stages, "recovery": recovery,
            "status": "review" if missing.any() else "pass",
            "source_pixels": total,
            "represented_source_pixels": int(self.previous.sum()),
            "residual_source_pixels": int(missing.sum()),
            "unaccounted_source_pixels": 0,
            "represented_fraction": float(self.previous.sum()/total) if total else 1.0,
            "accounted_fraction": 1.0,
        }


def recover_source_coverage(skeleton, nodes, edges, hatches=(), *,
                            max_component_pixels=100_000, max_new_edges=5_000):
    """Recover only unrepresented source ink; retain ambiguous singletons for review.

    A one-pixel source halo supplies real contacts at existing strokes. No gap
    is dilated into invented ink, and already owned hatch strokes stay separate.
    This is geometric recovery, not semantic merging of existing graph edges.
    """
    if max_component_pixels < 1 or max_new_edges < 1:
        raise ValueError("Coverage recovery limits must be positive")
    source = np.asarray(skeleton) > 0
    support = (_edge_support_mask(edges, source.shape) |
               _edge_support_mask(hatches, source.shape)).astype(bool)
    missing = source & ~support
    count, labels, stats, _ = cv2.connectedComponentsWithStats(missing.astype(np.uint8), connectivity=8)
    output_nodes, output_edges = list(nodes), list(edges)
    next_node = max((n["id"] for n in nodes), default=-1)+1
    next_edge = max((e["id"] for e in edges), default=-1)+1
    anchors = {}
    for edge in edges:
        if edge.get("pixels"):
            for point, key in ((edge["pixels"][0], "source"), (edge["pixels"][-1], "target")):
                anchors.setdefault(tuple(point), edge[key])
    components, new_edges = [], 0
    for label in range(1, count):
        x, y, w, h, size = map(int, stats[label])
        entry = {"id": label-1, "bbox": [x, y, w, h], "source_pixels": size, "edge_ids": []}
        components.append(entry)
        if size > max_component_pixels:
            entry.update(status="review", reason="component_budget")
            continue
        if new_edges >= max_new_edges:
            entry.update(status="review", reason="edge_budget")
            continue
        left, top, right, bottom = max(0, x-1), max(0, y-1), min(source.shape[1], x+w+1), min(source.shape[0], y+h+1)
        component = labels[top:bottom, left:right] == label
        context = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        context &= source[top:bottom, left:right]
        context &= component | support[top:bottom, left:right]
        cy, cx = np.nonzero(context)
        points = list(zip((cx+left).tolist(), (cy+top).tolist()))
        my, mx = np.nonzero(component)
        owned = set(zip((mx+left).tolist(), (my+top).tolist()))
        adjacency, chains = _residual_chains(points)
        chains = [chain for chain in chains if len(chain) >= 2 and any(p in owned for p in chain)]
        if not chains:
            entry.update(status="review", reason="isolated_source_pixel")
            continue
        if new_edges + len(chains) > max_new_edges:
            entry.update(status="review", reason="edge_budget")
            continue

        def anchor(point):
            nonlocal next_node
            if point not in anchors:
                anchors[point] = next_node
                output_nodes.append({"id": next_node, "x": point[0], "y": point[1],
                                     "type": KP_JUNCTION if len(adjacency[point]) > 2 else KP_ENDPOINT,
                                     "confidence": 0.3, "topology_origin": "coverage_recovery"})
                next_node += 1
            return anchors[point]

        for chain in chains:
            closed = len(chain) > 2 and chain[0] == chain[-1]
            edge = {"id": next_edge, "source": anchor(chain[0]), "target": anchor(chain[-1]),
                    "pixels": [list(p) for p in chain], "smooth_pts": [],
                    "is_closed": closed, "is_simple_cycle": closed,
                    "topology_origin": "coverage_recovery", "coverage_component_id": label-1}
            output_edges.append(edge)
            entry["edge_ids"].append(next_edge)
            next_edge += 1
            new_edges += 1
        entry.update(status="recovered", reason="source_only_trace")
    after = (_edge_support_mask(output_edges, source.shape) |
             _edge_support_mask(hatches, source.shape)).astype(bool)
    remaining = source & ~after
    return output_nodes, output_edges, {
        "max_component_pixels": max_component_pixels, "max_new_edges": max_new_edges,
        "before_missing_pixels": int(missing.sum()), "recovered_pixels": int((missing & after).sum()),
        "new_edges": new_edges, "components": components,
        "unresolved_pixels": int(remaining.sum()), "unresolved_spans": _coverage_spans(remaining),
    }


def integrate_recovered_connections(skeleton, nodes, edges, hatches=()):
    """Splice measured recovery into strokes, without pruning or guessing branches.

    Preserve every input pixel and record input-to-output ownership. Short local
    insertions keep the original stroke's point order; endpoint joins require a
    degree-two source contact, including the hatch layer in that decision.
    """
    source = np.asarray(skeleton) > 0
    originals = {e["id"]: e for e in edges}
    work = {i: dict(e) for i, e in originals.items()}
    if len(work) != len(edges):
        raise ValueError("Coverage integration requires unique main edge IDs")
    points = {i: [tuple(p) for p in e["pixels"]] for i, e in work.items()}
    parents = {i: {i} for i in work}
    recovery_ids = {i for i, e in work.items() if e.get("topology_origin") == "coverage_recovery"}
    hatch_points = {tuple(p) for e in hatches for p in e.get("pixels", [])}
    original_points = {p for i, chain in points.items() if i not in recovery_ids for p in chain}
    before_points = {p for chain in points.values() for p in chain}
    free = before_points - original_points - hatch_points
    next_edge = max(work, default=-1) + 1
    inserted = retired = splits = joins = 0

    def link(a, b):
        return (a, b) if a < b else (b, a)

    recovery_links = {}
    adjacency = {}
    for i in sorted(recovery_ids):
        for a, b in zip(points[i], points[i][1:]):
            if a == b or max(abs(a[0]-b[0]), abs(a[1]-b[1])) > 1:
                continue
            if not all(0 <= p[0] < source.shape[1] and 0 <= p[1] < source.shape[0]
                       and source[p[1], p[0]] for p in (a, b)):
                continue
            recovery_links.setdefault(link(a, b), set()).add(i)
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)

    def insertion(a, b):
        # Bounded local repair, not a search for a shortcut across the drawing.
        distance = float(np.hypot(b[0]-a[0], b[1]-a[1]))
        if a == b or distance > 3 or a not in adjacency or b not in adjacency:
            return None
        candidates = []
        stack = [(a, [a], 0.0)]
        while stack:
            current, chain, length = stack.pop()
            for following in sorted(adjacency[current]):
                if following in chain:
                    continue
                new_length = length + float(np.hypot(following[0]-current[0], following[1]-current[1]))
                if new_length > distance + 1.0 or len(chain) >= 5:
                    continue
                if following == b:
                    if len(chain) > 1:
                        candidates.append(chain + [b])
                        if len(candidates) > 1:
                            return None
                elif following in free:
                    stack.append((following, chain + [following], new_length))
        return candidates[0] if candidates else None

    changed = set()
    for i in sorted(set(work) - recovery_ids):
        if work[i].get("is_dashed"):
            continue
        chain = points[i]
        if not chain:
            continue
        result = [chain[0]]
        for a, b in zip(chain, chain[1:]):
            patch = insertion(a, b)
            if patch is None:
                result.append(b)
                continue
            result.extend(patch[1:])
            inserted += len(patch)-2
            for x, y in zip(patch, patch[1:]):
                parents[i].update(recovery_links[link(x, y)])
            changed.add(i)
        points[i] = result

    # Only retire recovery links actually traversed by an existing stroke.
    # Pixel proximity, bounding boxes and fitted curves are not ownership.
    owners = {}
    for i in sorted(set(work) - recovery_ids):
        for a, b in zip(points[i], points[i][1:]):
            key = link(a, b)
            if key in recovery_links:
                owners.setdefault(key, set()).add(i)
    for i in sorted(recovery_ids):
        runs, run = [], []
        for a, b in zip(points[i], points[i][1:]):
            covered = owners.get(link(a, b), set())
            if covered:
                for owner in covered:
                    parents[owner].add(i)
                    changed.add(owner)
                if run:
                    runs.append(run)
                    run = []
            else:
                if not run:
                    run = [a]
                run.append(b)
        if run:
            runs.append(run)
        if runs == [points[i]] or len(points[i]) < 2:
            continue
        template = work.pop(i)
        points.pop(i)
        parents.pop(i)
        changed.discard(i)
        retired += not runs
        splits += max(0, len(runs)-1)
        for index, run in enumerate(runs):
            ident = i if index == 0 else next_edge
            next_edge += index > 0
            work[ident] = dict(template, id=ident, is_closed=run[0] == run[-1],
                               is_simple_cycle=run[0] == run[-1])
            points[ident], parents[ident] = run, {i}
            changed.add(ident)

    # Coordinate contacts, not keypoint IDs: a keypoint cluster can contain
    # distinct endpoints. Interior contacts and hatch junctions remain branches.
    endpoints, interior = {}, set()
    for i, chain in points.items():
        interior.update(chain[1:-1])
        if len(chain) < 2 or work[i].get("is_closed") or work[i].get("is_dashed"):
            interior.update(chain)
            continue
        for p in (chain[0], chain[-1]):
            endpoints.setdefault(p, set()).add(i)

    def degree_two(p):
        x, y = p
        if not (0 <= x < source.shape[1] and 0 <= y < source.shape[0] and source[y, x]):
            return False
        local = [(xx, yy) for yy in range(max(0, y-1), min(source.shape[0], y+2))
                 for xx in range(max(0, x-1), min(source.shape[1], x+2)) if source[yy, xx]]
        neighbours, _ = _residual_chains(local)
        return len(neighbours[p]) == 2

    for p in sorted(endpoints):
        ids = endpoints[p]
        if len(ids) != 2 or p in interior or p in hatch_points or not degree_two(p):
            continue
        a, b = sorted(ids)
        if not ((parents[a] | parents[b]) & recovery_ids):
            continue
        if work[a].get("style") != work[b].get("style"):
            continue
        first = points[a] if points[a][-1] == p else points[a][::-1]
        second = points[b] if points[b][0] == p else points[b][::-1]
        overlap = set(first) & set(second)
        closed = first[0] == second[-1]
        if overlap != ({p, first[0]} if closed else {p}):
            continue
        combined = first + second[1:]
        if closed and (len(set(combined[:-1])) != len(combined)-1 or
                       any(max(abs(x[0]-y[0]), abs(x[1]-y[1])) > 1
                           for x, y in zip(combined, combined[1:]))):
            continue
        # A shared endpoint must also be an actual consecutive source step.
        if any(max(abs(x[0]-y[0]), abs(x[1]-y[1])) > 1
               for x, y in ((first[-2], p), (p, second[1]))):
            continue
        for ident in (a, b):
            for endpoint in (points[ident][0], points[ident][-1]):
                endpoints[endpoint].discard(ident)
        points[a] = combined
        parents[a].update(parents.pop(b))
        work[a].update(is_closed=closed, is_simple_cycle=closed)
        work.pop(b)
        points.pop(b)
        changed.discard(b)
        changed.add(a)
        if not closed:
            for endpoint in (combined[0], combined[-1]):
                endpoints[endpoint].add(a)
        interior.update(combined[1:-1])
        joins += 1

    output_nodes = list(nodes)
    node_at = {(n["x"], n["y"]): n["id"] for n in nodes}
    next_node = max((n["id"] for n in nodes), default=-1)+1

    def anchor(p):
        nonlocal next_node
        if p not in node_at:
            node_at[p] = next_node
            output_nodes.append({"id": next_node, "x": p[0], "y": p[1],
                                 "type": KP_ENDPOINT, "confidence": 0.3,
                                 "topology_origin": "coverage_integration"})
            next_node += 1
        return node_at[p]

    mapping = {i: [] for i in sorted(e["id"] for e in edges)}
    output_edges = []
    for i in sorted(work):
        e = work[i]
        if i in changed:
            ancestry = set(parents[i])
            for parent in parents[i]:
                ancestry.update(originals[parent].get("coverage_parent_edge_ids", []))
            raw = [list(p) for p in points[i]]
            e.update(pixels=raw,
                     smooth_pts=[] if raw != originals.get(i, {}).get("pixels") else e.get("smooth_pts", []),
                     source=anchor(points[i][0]), target=anchor(points[i][-1]),
                     coverage_parent_edge_ids=sorted(ancestry))
            if i not in recovery_ids or len(parents[i]) > 1:
                e["topology_origin"] = "coverage_integration"
        output_edges.append(e)
        for parent in parents[i]:
            mapping[parent].append(i)
    after_points = {p for chain in points.values() for p in chain}
    if after_points != before_points:
        raise RuntimeError("Coverage integration changed the represented main pixel set")
    # Per-input ownership includes fully absorbed and partially split recovery.
    for e in edges:
        owned = {p for i in mapping[e["id"]] for p in points[i]}
        if not set(map(tuple, e["pixels"])) <= owned:
            raise RuntimeError("Coverage integration lost an input stroke's source ownership")
    return output_nodes, output_edges, {
        "schema": "ap3-stage2-coverage-integration-v1", "status": "pass",
        "before_edges": len(edges), "after_edges": len(output_edges),
        "inserted_pixel_occurrences": inserted, "retired_recovery_edges": int(retired),
        "recovery_splits": splits, "endpoint_joins": joins,
        "main_pixel_set_preserved": True, "hatch_edges_unchanged": True,
        "input_to_output_edge_ids": {str(i): ids for i, ids in mapping.items()},
    }


def _extract_topology(
    skeleton: np.ndarray,
    kp_clusters: list[dict] | None = None,   # keypoint seeds; None → classical CN
    max_search_radius: int = 60,  # unused — walk terminates at extended kp regions
    unclaimed_mode: str = "all",
    directional_walk: bool = False,
    directional_walk_baseline: float = 6.0,
) -> tuple[list[dict], list[dict]]:
    """
    Build a stroke graph using vectorised CN-cluster skeleton tracing.

    Algorithm (no Dijkstra, no Gurobi):
      1. Vectorised CN map (NumPy shifts, ~1 000× faster than a Python loop).
      2. Cluster CN=1 pixels -> endpoints; cluster CN>=3 pixels -> junctions.
      3. Extended kp_map: every keypoint pixel + its 8-connected foreground
         neighbours map to that keypoint's ID.  The 1-px halo absorbs
         Zhang-Suen staircase artefacts so walks stop correctly at junctions
         even when approaching from a diagonal pixel.
      4. Walk outward from each keypoint cluster; one edge per unique
         (src, dst) pair.
      5. Unclaimed CCs >= min_loop_pixels -> ordered paths or genuine cycles.
         ``closed_only``
         retains only components whose induced pixel graph contains no open
         endpoint; ``none`` omits all unclaimed components.

    Returns (nodes, edges) as plain dicts for JSON serialisation.

    `kp_clusters` are the keypoint seeds that drive topology: a list of
    {x, y, type, confidence, pixels:[(x,y),…]} where `pixels` are the core
    skeleton pixels of each keypoint. When None, classical CN seeding is used
    (`_cn_keypoint_clusters`), reproducing the legacy behaviour exactly. A
    learned keypoint detector supplies the same structure instead, so swapping
    detectors changes only the seeds, not the walk.
    """
    binary = (skeleton > 0).astype(np.uint8)
    H, W   = binary.shape
    if unclaimed_mode not in {"all", "closed_only", "none"}:
        raise ValueError(f"invalid unclaimed topology mode: {unclaimed_mode!r}")
    if directional_walk_baseline <= 0:
        raise ValueError("directional walk baseline must be positive")

    # ── Steps 1–2: keypoint seeds → kp_info / kp_map / kp_pixels ─────────
    # Keypoints now drive topology. None → classical CN seeding (CN==1
    # endpoints, CN>=3 junctions), which reproduces the legacy behaviour
    # exactly; a learned detector supplies the same cluster contract instead.
    if kp_clusters is None:
        kp_clusters = _cn_keypoint_clusters(skeleton)
    kp_info, kp_map, kp_pixels = _materialize_keypoint_clusters(
        kp_clusters, binary, H, W
    )

    # ── Step 3: walk from each keypoint cluster outward ─────────────────
    def _neighbours8(x, y):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= ny_ < H and 0 <= nx_ < W and binary[ny_, nx_]:
                    yield (nx_, ny_)

    edges_raw    = []   # list of (sid, did, pixel_chain)
    all_edge_pix = set()

    def _incoming_direction(chain):
        if len(chain) < 2:
            return None
        end = np.asarray(chain[-1], dtype=np.float64)
        distance = 0.0
        anchor = chain[0]
        for index in range(len(chain) - 2, -1, -1):
            anchor = chain[index]
            distance += float(np.hypot(
                chain[index + 1][0] - chain[index][0],
                chain[index + 1][1] - chain[index][1],
            ))
            if distance >= directional_walk_baseline:
                break
        vector = end - np.asarray(anchor, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-9 else None

    def _next_pixel(chain, neighbours):
        if not directional_walk or len(neighbours) <= 1:
            return neighbours[0]
        incoming = _incoming_direction(chain)
        if incoming is None:
            return neighbours[0]
        current = np.asarray(chain[-1], dtype=np.float64)

        def score(point):
            vector = np.asarray(point, dtype=np.float64) - current
            norm = float(np.linalg.norm(vector))
            continuation = (
                float(np.dot(incoming, vector / norm)) if norm > 1e-9 else -1.0
            )
            # Stable coordinate tie-break keeps output deterministic.
            return (continuation, -int(point[1]), -int(point[0]))

        return max(neighbours, key=score)

    for src_kp in kp_info:
        sid     = src_kp["id"]
        src_pxs = kp_pixels[sid]   # O(1) reverse-map lookup

        for sp in list(src_pxs):
            for entry in _neighbours8(*sp):
                if kp_map.get(entry) == sid:
                    continue   # same cluster, skip
                # Skip entry pixels already claimed by a previously stored edge.
                # This deduplicates reverse-direction walks (J1→J0 after J0→J1
                # was already stored) while still allowing genuine parallel edges
                # (two distinct arcs between the same two nodes use disjoint
                # pixel sets, so their entry pixels are always unclaimed).
                if (entry[0], entry[1]) in all_edge_pix:
                    continue
                chain   = [sp, entry]
                visited = set(src_pxs) | {entry}
                cx, cy  = entry
                found   = None

                if entry in kp_map:
                    found = kp_map[entry]
                else:
                    for _ in range(5000):
                        nbs = [p for p in _neighbours8(cx, cy)
                               if p not in visited]
                        if not nbs:
                            break
                        nxt = _next_pixel(chain, nbs)
                        chain.append(nxt)
                        if nxt in kp_map:
                            found = kp_map[nxt]
                            break
                        visited.add(nxt)
                        cx, cy = nxt

                if found is not None and found != sid:
                    edges_raw.append((sid, found, chain))
                    for p in chain:
                        all_edge_pix.add((p[0], p[1]))

    # ── Step 4: build final node/edge lists ──────────────────────────────
    nodes   = list(kp_info)
    edges   = []
    edge_id = 0

    for sid, did, chain in edges_raw:
        edges.append({
            "id": edge_id,
            "source": sid,
            "target": did,
            "pixels": [[int(p[0]), int(p[1])] for p in chain],
            "smooth_pts": [],
            "is_closed": False,
        })
        edge_id += 1

    # ── Step 5: closed loops ─────────────────────────────────────────────
    # Even a two-pixel component may be a real mark. Singletons and pixels
    # swallowed by keypoint halos are accounted for by final coverage recovery.
    min_loop_pixels = 2

    # Label unclaimed non-kp pixels in image space. The previous implementation
    # materialised every pixel and 8-neighbour link as Python NetworkX objects;
    # a dense 335k-pixel patent component needed nearly 1 GB and minutes just
    # to discover that it was one connected component.
    remaining = set()
    ys, xs = np.where(binary)
    for y, x in zip(ys, xs):
        px = (x, y)
        if px not in kp_map and px not in all_edge_pix:
            remaining.add(px)

    if remaining and unclaimed_mode != "none":
        remaining_mask = np.zeros_like(binary, dtype=np.uint8)
        remaining_array = np.asarray(list(remaining), dtype=np.int32)
        remaining_mask[
            remaining_array[:, 1], remaining_array[:, 0]
        ] = 1
        n_components, component_labels, component_stats, component_centroids = (
            cv2.connectedComponentsWithStats(
                remaining_mask, connectivity=8
            )
        )
        for component_label in range(1, n_components):
            component_size = int(
                component_stats[component_label, cv2.CC_STAT_AREA]
            )
            if component_size < min_loop_pixels:
                continue
            left = int(component_stats[component_label, cv2.CC_STAT_LEFT])
            top = int(component_stats[component_label, cv2.CC_STAT_TOP])
            width = int(component_stats[component_label, cv2.CC_STAT_WIDTH])
            height = int(component_stats[component_label, cv2.CC_STAT_HEIGHT])
            rel_y, rel_x = np.where(
                component_labels[top:top + height, left:left + width]
                == component_label
            )
            component_pixels = set(zip(
                (rel_x + left).tolist(),
                (rel_y + top).tolist(),
            ))
            # A learned keypoint detector can miss an entire dense network.
            # Historically every such residual component was labelled as one
            # closed loop, even when it contained thousands of endpoints and
            # branches.  Record whether this is actually a digital cycle so
            # the quality gate can reject pathological pseudo-loops before
            # Stage 3 attempts to order them.
            component_degrees = []
            raw_degree_sum = 0
            minimum_raw_degree = 8
            for x, y in component_pixels:
                degree = 0
                raw_degree = 0
                for dx, dy in (
                    (-1, -1), (0, -1), (1, -1),
                    (-1, 0),            (1, 0),
                    (-1, 1),  (0, 1),   (1, 1),
                ):
                    if (x + dx, y + dy) not in component_pixels:
                        continue
                    raw_degree += 1
                    # Suppress diagonal corner chords when an orthogonal
                    # connection exists.  Without this, a clean rectangular
                    # one-pixel loop appears branched at every corner.
                    if dx and dy and (
                        (x + dx, y) in component_pixels
                        or (x, y + dy) in component_pixels
                    ):
                        continue
                    degree += 1
                component_degrees.append(degree)
                raw_degree_sum += raw_degree
                minimum_raw_degree = min(minimum_raw_degree, raw_degree)
            is_simple_cycle = bool(
                component_degrees
                and all(degree == 2 for degree in component_degrees)
            )
            if unclaimed_mode == "closed_only":
                if (
                    not component_degrees
                    or minimum_raw_degree < 2
                    or raw_degree_sum // 2 < component_size
                ):
                    continue
            cx, cy = (
                int(component_centroids[component_label, 0]),
                int(component_centroids[component_label, 1]),
            )
            loop_id = len(nodes)
            nodes.append({
                "id": loop_id, "x": cx, "y": cy,
                "type": KP_LOOP_ANCHOR, "confidence": 1.0,
            })
            pixels = [
                [int(point[0]), int(point[1])]
                for point in sorted(component_pixels)
            ]
            edges.append({
                "id": edge_id,
                "source": loop_id,
                "target": loop_id,
                "pixels": pixels,
                "smooth_pts": [],
                "is_closed": True,
                "topology_origin": "unclaimed_component",
                "is_simple_cycle": is_simple_cycle,
            })
            edge_id += 1

    return repair_noncycle_residuals(nodes, edges)


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 2b — GRAPH SIMPLIFICATION (de-fragmentation)
# ═══════════════════════════════════════════════════════════════════════════

def _chain_length(pix) -> float:
    """Euclidean arc-length of a pixel chain."""
    if len(pix) < 2:
        return 0.0
    p = np.asarray(pix, dtype=np.float64)
    return float(np.hypot(*(p[1:] - p[:-1]).T).sum())


def _angle_delta_deg(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _edge_line_features(edge: dict) -> dict | None:
    """
    Measure whether an open graph edge behaves like a short straight hatch line.

    Hachures are not identified by absolute angle: drawings use many hatch
    angles. The useful signal is local repetition: many short, line-like,
    similarly angled edges packed near one another.
    """
    if edge.get("is_closed"):
        return None
    pix = edge.get("pixels") or []
    if len(pix) < 2:
        return None

    pts = np.asarray(pix, dtype=np.float64)
    length = _chain_length(pix)
    if length <= 1e-9:
        return None
    chord = float(np.linalg.norm(pts[-1] - pts[0]))
    center = pts.mean(axis=0)
    centered = pts - center

    if len(pts) == 2:
        direction = pts[-1] - pts[0]
        n = float(np.linalg.norm(direction))
        direction = direction / n if n > 1e-9 else np.array([1.0, 0.0])
    else:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    residual_rms = float(np.sqrt(((centered @ normal) ** 2).mean()))
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)

    return {
        "edge_id": int(edge["id"]),
        "length": float(length),
        "chord": float(chord),
        "straightness": float(chord / length),
        "residual_rms": residual_rms,
        "angle_deg": angle,
        "center": center,
    }


def _drop_unused_nodes(
    nodes: list[dict],
    edges: list[dict],
) -> tuple[list[dict], list[dict]]:
    used = set()
    for edge in edges:
        used.add(edge["source"])
        used.add(edge["target"])
    return [node for node in nodes if node["id"] in used], edges


def _line_angle_from_pix(pix) -> float | None:
    """Orientation in [0,180) of a pixel chain via PCA (robust; ignores stored
    meta, which is missing on ~35 % of removed hatch edges)."""
    p = np.asarray(pix, dtype=np.float64)
    if len(p) < 2:
        return None
    c = p - p.mean(axis=0)
    if len(p) == 2:
        d = p[-1] - p[0]
    else:
        _, _, vt = np.linalg.svd(c, full_matrices=False)
        d = vt[0]
    return float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)


def _hatch_region_params(members: list[dict], cfg: dict) -> tuple[list[float], float]:
    """Estimate hatch pattern angle(s) and spacing for one region.

    Returns (angles_deg, spacing_px). Two angles ⇒ cross-hatch. Angle is the
    line orientation in [0,180); spacing is the median perpendicular gap between
    adjacent parallel lines.
    """
    ang_list, cen_list = [], []
    for m in members:
        a = _line_angle_from_pix(m.get("pixels") or [])
        if a is None:
            continue
        ang_list.append(a)
        cen_list.append(np.asarray(m["pixels"], dtype=np.float64).mean(axis=0))
    if not ang_list:
        return [], 0.0
    angs = np.array(ang_list)
    cen = np.array(cen_list)
    # angle histogram in 12° bins over [0,180); pick dominant, then a distinct
    # second mode if it's substantial (cross-hatch).
    edges_b = np.arange(0.0, 180.0 + 12.0, 12.0)
    hist, _ = np.histogram(angs % 180.0, bins=edges_b)
    order = list(np.argsort(hist)[::-1])
    peaks: list[float] = []
    for bi in order:
        if hist[bi] == 0:
            break
        c = (edges_b[bi] + edges_b[bi + 1]) / 2.0
        if not peaks:
            peaks.append(c)
        elif (all(_angle_delta_deg(c, p) > 25.0 for p in peaks)
              and hist[bi] >= float(cfg.get("hachure_crosshatch_ratio", 0.45)) * hist[order[0]]):
            peaks.append(c)
            break
    angles: list[float] = []
    spacings: list[float] = []
    for c in peaks:
        sel = np.array([_angle_delta_deg(a, c) <= 12.0 for a in angs])
        if sel.sum() == 0:
            continue
        a2 = np.radians(2.0 * angs[sel])             # circular mean (double angle)
        amean = (np.degrees(np.arctan2(np.sin(a2).mean(),
                                       np.cos(a2).mean())) / 2.0) % 180.0
        angles.append(round(float(amean), 2))
        th = np.radians(amean)
        perp = np.array([-np.sin(th), np.cos(th)])   # ⟂ to the lines
        proj = np.sort(cen[sel] @ perp)
        gaps = np.diff(proj)
        gaps = gaps[gaps > 1.0]
        if len(gaps):
            spacings.append(float(np.median(gaps)))
    spacing = round(float(np.median(spacings)), 2) if spacings else 0.0
    return angles, spacing


def _aggregate_hachure_regions(
    removed_hachures: list[dict], image_shape: tuple[int, int], cfg: dict
) -> list[dict]:
    """Group removed hatch edges into filled regions for parametric HATCH export.

    Vectorising hatching line-by-line fragments at every cross-hatch intersection
    and bloats the output (≈45 % of all primitives). Instead we collapse each
    contiguous block of hatch lines into ONE region descriptor — a boundary plus
    a pattern (angle(s) + spacing) — which Stage 4 emits as a native DXF HATCH /
    SVG pattern fill. Contiguity is found by dilating the hatch-pixel mask and
    connected-component labelling (dilation ≈ hatch spacing joins lines in a
    region but keeps distinct regions apart).
    """
    if not removed_hachures:
        return []
    H, W = int(image_shape[0]), int(image_shape[1])
    dilate = int(cfg.get("hachure_region_dilate", 5))
    min_lines = int(cfg.get("hachure_region_min_lines", 6))
    mask = np.zeros((H, W), np.uint8)
    arrs: list[np.ndarray | None] = []
    for e in removed_hachures:
        pix = e.get("pixels") or []
        if pix:
            a = np.asarray(pix, dtype=np.int64)
            arrs.append(a)
            xs = np.clip(a[:, 0], 0, W - 1)
            ys = np.clip(a[:, 1], 0, H - 1)
            mask[ys, xs] = 1
        else:
            arrs.append(None)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1, 2 * dilate + 1))
    n_lab, lab = cv2.connectedComponents(cv2.dilate(mask, k), connectivity=8)
    regions: list[dict] = []
    for rid in range(1, n_lab):
        members, allpix, member_indices = [], [], []
        for edge_index, (e, a) in enumerate(zip(removed_hachures, arrs)):
            if a is None or len(a) == 0:
                continue
            cx = int(np.clip(a[:, 0].mean(), 0, W - 1))
            cy = int(np.clip(a[:, 1].mean(), 0, H - 1))
            if lab[cy, cx] == rid:
                members.append(e)
                allpix.append(a)
                member_indices.append(edge_index)
        if len(members) < min_lines:
            continue
        pix_all = np.vstack(allpix).astype(np.int32)
        try:
            angles, spacing = _hatch_region_params(members, cfg)
        except Exception:                       # never let detection break Stage 2
            continue
        if not angles:
            continue
        hull = cv2.convexHull(pix_all)
        boundary = [[int(p[0][0]), int(p[0][1])] for p in hull]
        if len(boundary) < 3:
            continue
        regions.append({
            "boundary": boundary,
            "angles": angles,
            "spacing": spacing,
            "double": len(angles) > 1,
            "n_lines": len(members),
            "source_hachure_indices": member_indices,
        })
    return regions


def _cleanup_hatch_residue(
    nodes: list[dict], edges: list[dict], regions: list[dict],
    image_shape: tuple[int, int], cfg: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Drop leftover open edges that lie inside a hatch region and match its angle.

    The hatch detector (_remove_hachure_edges) misses some lines (sparser
    diagonal fills), which then survive as fragmented main-graph edges sitting on
    top of the clean HATCH fill. Once regions are known we can safely remove any
    remaining edge that is mostly inside a region AND oriented like that region's
    hatch — the region fill already represents it. Conservative: requires both
    high inside-fraction and an angle match, so a real feature line crossing the
    region (different angle, or only partly inside) is kept. Removed residue is
    returned explicitly so it remains available to the HACHURE side layer.
    """
    if not regions:
        return nodes, edges, []
    H, W = int(image_shape[0]), int(image_shape[1])
    angle_tol = float(cfg.get("hachure_cleanup_angle_tol", 14.0))
    inside_frac = float(cfg.get("hachure_cleanup_inside_frac", 0.80))
    regmask = np.zeros((H, W), np.int32)
    for i, r in enumerate(regions):
        cv2.fillPoly(regmask, [np.asarray(r["boundary"], dtype=np.int32)], i + 1)
    kept: list[dict] = []
    removed: list[dict] = []
    for e in edges:
        pix = e.get("pixels") or []
        if e.get("is_closed") or len(pix) < 2:
            kept.append(e)
            continue
        a = np.asarray(pix, dtype=np.int64)
        xs = np.clip(a[:, 0], 0, W - 1)
        ys = np.clip(a[:, 1], 0, H - 1)
        ids = regmask[ys, xs]
        inside = ids > 0
        if inside.mean() < inside_frac:
            kept.append(e)
            continue
        rid = int(np.bincount(ids[inside]).argmax())
        angles = regions[rid - 1].get("angles") or []
        ang = _line_angle_from_pix(pix)
        angle_delta = (
            min(_angle_delta_deg(ang, ra) for ra in angles)
            if ang is not None and angles
            else None
        )
        if angle_delta is not None and angle_delta <= angle_tol:
            item = dict(e)
            item["is_hachure"] = True
            item["hachure"] = {
                "pass": "region_residue",
                "inside_frac": round(float(inside.mean()), 3),
                "angle_delta": round(float(angle_delta), 3),
                "region_id": rid,
            }
            removed.append(item)
            continue
        kept.append(e)
    nodes, kept = _drop_unused_nodes(nodes, kept)
    return nodes, kept, removed


# ─── Learned hatch-region detector (Phase 2 CNN) ─────────────────────────────

_HATCH_PATCH  = 512
_HATCH_STRIDE = 256
_HATCH_STROKE_LABEL_CONTRACT = "reference-free-hatch-stroke-multilabel-v1"


def load_hatch_model(config: dict):
    """
    Load the Phase 2 HatchUNet hatch-region detector (called once at batch
    start, mirroring ``load_model``).

    Returns None — so Stage 2 transparently falls back to the geometric
    hachure heuristic — when the detector is disabled
    (``stage2.hachure_use_cnn`` is false), the checkpoint is missing, or the
    weights fail to load.
    """
    cfg = config.get("stage2", {})
    if not bool(cfg.get("hachure_use_cnn", False)):
        return None

    ckpt_path = cfg.get("hachure_cnn_model", "models/hatch_unet.pth")
    if not os.path.exists(ckpt_path):
        logger.warning(
            f"Hatch CNN weights not found ({ckpt_path}); "
            "falling back to geometric hachure removal."
        )
        return None

    try:
        import sys
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        import torch
        from tools.hatch_model import HatchUNet

        device = cfg.get("hachure_cnn_device",
                         "cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = HatchUNet(freeze_encoder=False).to(device)
        model.load_state_dict(ck["model_state"])
        model.eval()
        model._hatch_device = torch.device(device)
        logger.info(
            f"Hatch CNN ready ({ckpt_path}, device={device}, "
            f"val_iou_pos={ck.get('val_iou_pos', '?')})."
        )
        return model
    except Exception as exc:
        logger.warning(
            f"Hatch CNN load failed ({exc}); "
            "falling back to geometric hachure removal."
        )
        return None


def load_hatch_stroke_model(config: dict):
    """Load the opt-in two-channel structural/hachure skeleton classifier."""
    cfg = config.get("stage2", {})
    if not bool(cfg.get("hachure_stroke_use_cnn", False)):
        return None

    ckpt_path = cfg.get(
        "hachure_stroke_cnn_model", "models/hatch_stroke_multilabel.pth"
    )
    if not os.path.exists(ckpt_path):
        logger.warning(
            "Hatch-stroke CNN weights not found (%s); classifier disabled.",
            ckpt_path,
        )
        return None

    try:
        import sys
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)
        import torch
        from tools.hatch_model import HatchUNet

        device = cfg.get(
            "hachure_stroke_cnn_device",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        checkpoint = torch.load(
            ckpt_path, map_location=device, weights_only=False
        )
        if checkpoint.get("label_contract") != _HATCH_STROKE_LABEL_CONTRACT:
            raise ValueError("incompatible hatch-stroke label contract")
        model = HatchUNet(
            freeze_encoder=False, out_channels=2, pretrained=False
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        model._hatch_stroke_device = torch.device(device)
        model._hatch_stroke_checkpoint = str(ckpt_path)
        logger.info(
            "Hatch-stroke CNN ready (%s, device=%s, best_epoch=%s).",
            ckpt_path,
            device,
            checkpoint.get("best_epoch", checkpoint.get("epoch", "?")),
        )
        return model
    except Exception as exc:
        logger.warning(
            "Hatch-stroke CNN load failed (%s); classifier disabled.", exc
        )
        return None


def _window_origins(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    origins = list(range(0, length - patch + 1, stride))
    if origins[-1] != length - patch:
        origins.append(length - patch)
    return origins


def _cnn_hatch_stroke_probabilities(
    skeleton: np.ndarray,
    model,
    *,
    patch: int = _HATCH_PATCH,
    stride: int = _HATCH_STRIDE,
) -> np.ndarray:
    """Return structural/hachure probabilities on existing skeleton pixels."""
    import torch

    if patch < 32 or patch % 32 or stride < 1:
        raise ValueError("hatch-stroke patch must be a multiple of 32")
    binary = (np.asarray(skeleton) > 0).astype(np.float32)
    height, width = binary.shape
    accumulated = np.zeros((2, height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    device = (
        getattr(model, "_hatch_stroke_device", None)
        or next(model.parameters()).device
    )

    model.eval()
    with torch.no_grad():
        for top in _window_origins(height, patch, stride):
            for left in _window_origins(width, patch, stride):
                source = binary[top : top + patch, left : left + patch]
                patch_height, patch_width = source.shape
                padded = np.zeros((patch, patch), dtype=np.float32)
                padded[:patch_height, :patch_width] = source
                tensor = torch.from_numpy(padded[None, None]).to(device)
                probabilities = torch.sigmoid(model(tensor))[0].cpu().numpy()
                accumulated[
                    :, top : top + patch_height, left : left + patch_width
                ] += probabilities[:, :patch_height, :patch_width]
                counts[
                    top : top + patch_height, left : left + patch_width
                ] += 1.0

    probabilities = accumulated / np.maximum(counts[None], 1.0)
    probabilities *= binary[None]
    return probabilities


def _cnn_hatch_mask(
    gray: np.ndarray,
    model,
    threshold: float,
    patch: int = _HATCH_PATCH,
    stride: int = _HATCH_STRIDE,
) -> np.ndarray:
    """
    Sliding-window hatch-region inference on a full grayscale image.

    Returns a uint8 {0, 1} mask with the same shape as ``gray`` — 1 where the
    learned detector predicts hatching (prob ≥ threshold).  Mirrors
    ``tools.hatch_infer.infer_tif`` but operates on an in-memory array.
    """
    import torch

    device = getattr(model, "_hatch_device", None) or next(model.parameters()).device
    H, W  = gray.shape
    acc   = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.float32)

    ys = list(range(0, max(1, H - patch), stride)) + [max(0, H - patch)]
    xs = list(range(0, max(1, W - patch), stride)) + [max(0, W - patch)]

    model.eval()
    with torch.no_grad():
        for y0 in dict.fromkeys(ys):
            for x0 in dict.fromkeys(xs):
                y1, x1 = y0 + patch, x0 + patch
                p = gray[y0:y1, x0:x1].astype(np.float32) / 255.0
                ph, pw = p.shape
                if ph < patch or pw < patch:
                    pad = np.zeros((patch, patch), np.float32)
                    pad[:ph, :pw] = p
                    p = pad
                t = torch.from_numpy(p[None, None]).to(device)
                prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
                acc[y0:y0 + ph, x0:x0 + pw]   += prob[:ph, :pw]
                count[y0:y0 + ph, x0:x0 + pw] += 1.0

    prob_map = acc / np.maximum(count, 1e-6)
    return (prob_map >= threshold).astype(np.uint8)


def _aligned_hatch_mask(
    source_image_path: Path,
    hatch_model,
    shape: tuple[int, int],
    cfg: dict,
) -> tuple[np.ndarray, float]:
    """Infer a hatch-region mask aligned to the active Stage-2 skeleton."""
    src_gray = cv2.imread(str(source_image_path), cv2.IMREAD_GRAYSCALE)
    if src_gray is None:
        raise FileNotFoundError(source_image_path)
    threshold = float(cfg.get("hachure_cnn_threshold", 0.70))
    mask = _cnn_hatch_mask(src_gray, hatch_model, threshold)
    height, width = shape
    if mask.shape != shape:
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask, threshold


def _remove_hachures_cnn(
    nodes: list[dict],
    edges: list[dict],
    hatch_mask: np.ndarray,
    cfg: dict,
    *,
    pass_name: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Remove open graph edges that lie inside the learned hatch-region mask.

    Mask overlap is the primary evidence.  The C3 structural-topology path can
    additionally protect edges joining two learned structural seeds and require
    line-like geometry for ambiguous overlap.  Strong mask overlap overrides
    those guards so hatch chords between two real contour nodes are still
    removed.  Removed edges carry ``is_hachure=True`` so the isolation metric
    and region aggregation treat them exactly like geometrically detected
    hachures.
    """
    inside_frac = float(cfg.get("hachure_cnn_inside_frac", 0.60))
    min_len     = float(cfg.get("hachure_cnn_min_length", 4.0))
    H, W = hatch_mask.shape
    main_cnn_pass = (
        pass_name == "cnn"
        or pass_name.startswith("hatch_stroke_multilabel")
    )
    protect_seeded = bool(
        main_cnn_pass
        and cfg.get("hachure_cnn_protect_structural_seed_edges", False)
    )
    require_line_like = bool(
        main_cnn_pass
        and cfg.get("hachure_cnn_main_require_line_like", False)
    )
    node_by_id = {int(node["id"]): node for node in nodes}

    selected_ids: set[int] = set()
    selected_meta: dict[int, dict] = {}
    for edge in edges:
        if edge.get("is_closed"):
            continue
        pix = edge.get("pixels") or []
        if len(pix) < 2:
            continue
        if _chain_length(pix) < min_len:
            continue
        inside = 0
        for px in pix:                       # pixels are [x, y] → index [y, x]
            x, y = int(px[0]), int(px[1])
            if 0 <= y < H and 0 <= x < W and hatch_mask[y, x]:
                inside += 1
        frac = inside / len(pix)
        if frac < inside_frac:
            continue

        feature = None
        line_like = False
        if require_line_like or protect_seeded:
            feature = _edge_line_features(edge)
            minimum_straightness = float(
                cfg.get("hachure_cnn_main_min_straightness", 0.70)
            )
            maximum_residual = float(
                cfg.get("hachure_cnn_main_max_residual_rms", 2.2)
            )
            line_like = bool(
                feature is not None
                and feature["straightness"] >= minimum_straightness
                and feature["residual_rms"] <= maximum_residual
            )
        strong_inside = frac >= float(
            cfg.get("hachure_cnn_strong_inside_frac", 0.80)
        )
        seeded_ends = sum(
            bool(node_by_id.get(int(node_id), {}).get("structural_seed", False))
            for node_id in (edge.get("source"), edge.get("target"))
            if node_id is not None
        )
        if protect_seeded and seeded_ends == 2 and not (
            strong_inside and line_like
        ):
            continue
        if require_line_like and not line_like and not strong_inside:
            continue
        selected_ids.add(int(edge["id"]))
        selected_meta[int(edge["id"])] = {
            "pass":        pass_name,
            "inside_frac": round(frac, 3),
            "n_pixels":    len(pix),
            "strong_inside": bool(strong_inside),
        }

    if not selected_ids:
        return nodes, edges, []

    # Guard: a bad or mis-aligned mask must never nuke most of the drawing.
    # Uses a SEPARATE, higher ceiling than the geometric heuristic's guard
    # (hachure_max_removed_edge_ratio, default 0.75): the CNN mask is a
    # calibrated per-pixel detector (test IoU 0.816, 8/8 held-out negatives
    # clean at this threshold — see hatch-detector-v2-results), not a coarse
    # clustering rule, so a genuinely hatch-dominated drawing (dense
    # cross-sections routinely exceed 75% hachure edges) is expected, not a
    # sign of a bad mask. Evidence: a real cross-section figure with a
    # confidently-detected mask (max prob 0.9997) had candidate ratio 0.835
    # and was silently skipped end to end under the shared 0.75 guard.
    open_edges = [e for e in edges if not e.get("is_closed")]
    max_ratio  = float(cfg.get("hachure_cnn_max_removed_edge_ratio", 0.92))
    if max_ratio > 0 and len(selected_ids) / max(1, len(open_edges)) > max_ratio:
        logger.warning(
            "CNN hachure removal skipped: candidate ratio %.3f exceeds guard %.3f",
            len(selected_ids) / max(1, len(open_edges)),
            max_ratio,
        )
        return nodes, edges, []

    kept: list[dict] = []
    removed: list[dict] = []
    for edge in edges:
        if int(edge["id"]) not in selected_ids:
            kept.append(edge)
            continue
        item = dict(edge)
        item["is_hachure"] = True
        item["hachure"] = selected_meta.get(int(edge["id"]), {"pass": pass_name})
        removed.append(item)

    nodes, kept = _drop_unused_nodes(nodes, kept)
    return nodes, kept, removed


def _remove_hachures_cnn_geometric(
    nodes: list[dict],
    edges: list[dict],
    hatch_mask: np.ndarray,
    cfg: dict,
    *,
    pass_name: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Remove only edges supported by both hatch geometry and the CNN region.

    The hatch network predicts filled regions, so mask overlap alone can erase
    long contours that bound or cross a section. The geometric pass supplies
    the missing stroke-level evidence: short, straight, locally repeated edge
    families. The intersection keeps the region detector as a spatial prior
    without allowing it to classify arbitrary structural ink.
    """
    geometric_cfg = dict(cfg)
    geometric_cfg["hachure_max_removed_edge_ratio"] = float(
        cfg.get("hachure_cnn_geometric_max_candidate_ratio", 0.95)
    )
    geometric_cfg["hachure_max_length"] = float(
        cfg.get(
            "hachure_cnn_geometric_max_length",
            cfg.get("hachure_max_length", 80.0),
        )
    )
    _nodes, _edges, geometric = _remove_hachure_edges(
        nodes,
        edges,
        geometric_cfg,
        pass_name=f"{pass_name}_geometric_candidate",
    )
    if not geometric:
        return nodes, edges, []

    mask_cfg = dict(cfg)
    mask_cfg["hachure_cnn_max_removed_edge_ratio"] = 1.0
    _nodes, _edges, selected = _remove_hachures_cnn(
        nodes,
        geometric,
        hatch_mask,
        mask_cfg,
        pass_name=f"{pass_name}_mask_intersection",
    )
    if not selected:
        return nodes, edges, []

    geometric_meta = {
        int(edge["id"]): dict(edge.get("hachure", {}))
        for edge in geometric
    }
    selected_meta = {
        int(edge["id"]): dict(edge.get("hachure", {}))
        for edge in selected
    }
    selected_ids = set(selected_meta)
    kept: list[dict] = []
    removed: list[dict] = []
    for edge in edges:
        edge_id = int(edge["id"])
        if edge_id not in selected_ids:
            kept.append(edge)
            continue
        item = dict(edge)
        item["is_hachure"] = True
        metadata = geometric_meta.get(edge_id, {})
        metadata.update(selected_meta[edge_id])
        metadata["pass"] = pass_name
        item["hachure"] = metadata
        removed.append(item)

    nodes, kept = _drop_unused_nodes(nodes, kept)
    return nodes, kept, removed


def _remove_hachure_edges(
    nodes: list[dict],
    edges: list[dict],
    cfg: dict,
    *,
    pass_name: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Remove dense parallel short strokes before primitive fitting.

    This is deliberately graph-level rather than Stage 3-level: hachures split
    outlines at skeleton intersections, so the outline must be reconnected while
    topology is still editable. Removed hatch edges are stored in the graph JSON
    as `removed_hachures` and their pixels are ignored by the isolation metric.
    """
    if not bool(cfg.get("remove_hachures", False)):
        return nodes, edges, []

    open_edges = [edge for edge in edges if not edge.get("is_closed")]
    if len(open_edges) < int(cfg.get("hachure_min_graph_edges", 20)):
        return nodes, edges, []

    min_len = float(cfg.get("hachure_min_length", 5.0))
    max_len = float(cfg.get("hachure_max_length", 80.0))
    min_straight = float(cfg.get("hachure_min_straightness", 0.70))
    max_rms = float(cfg.get("hachure_max_residual_rms", 2.2))

    feats: list[dict] = []
    for edge in open_edges:
        feat = _edge_line_features(edge)
        if feat is None:
            continue
        if not (min_len <= feat["length"] <= max_len):
            continue
        if feat["straightness"] < min_straight:
            continue
        if feat["residual_rms"] > max_rms:
            continue
        feats.append(feat)

    if not feats:
        return nodes, edges, []

    n = len(feats)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    angle_tol = float(cfg.get("hachure_angle_tolerance", 12.0))
    cluster_radius = float(cfg.get("hachure_cluster_radius", 95.0))
    for i in range(n):
        ci = feats[i]["center"]
        for j in range(i + 1, n):
            if _angle_delta_deg(feats[i]["angle_deg"], feats[j]["angle_deg"]) > angle_tol:
                continue
            if float(np.linalg.norm(ci - feats[j]["center"])) > cluster_radius:
                continue
            union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, feat in enumerate(feats):
        groups.setdefault(find(i), []).append(feat)

    min_cluster = int(cfg.get("hachure_min_cluster_edges", 4))
    min_total_len = float(cfg.get("hachure_min_cluster_total_length", 35.0))
    selected_ids: set[int] = set()
    selected_meta: dict[int, dict] = {}
    for group in groups.values():
        if len(group) < min_cluster:
            continue
        total_len = sum(float(feat["length"]) for feat in group)
        if total_len < min_total_len:
            continue
        for feat in group:
            selected_ids.add(int(feat["edge_id"]))
            selected_meta[int(feat["edge_id"])] = {
                "pass": pass_name,
                "length": feat["length"],
                "angle_deg": feat["angle_deg"],
                "straightness": feat["straightness"],
                "residual_rms": feat["residual_rms"],
                "cluster_size": len(group),
                "cluster_total_length": float(total_len),
            }

    if not selected_ids:
        return nodes, edges, []

    max_ratio = float(cfg.get("hachure_max_removed_edge_ratio", 0.75))
    if max_ratio > 0 and len(selected_ids) / max(1, len(open_edges)) > max_ratio:
        logger.warning(
            "Hachure removal skipped: candidate ratio %.3f exceeds guard %.3f",
            len(selected_ids) / max(1, len(open_edges)),
            max_ratio,
        )
        return nodes, edges, []

    kept: list[dict] = []
    removed: list[dict] = []
    for edge in edges:
        if int(edge["id"]) not in selected_ids:
            kept.append(edge)
            continue
        item = dict(edge)
        item["is_hachure"] = True
        item["hachure"] = selected_meta.get(int(edge["id"]), {"pass": pass_name})
        removed.append(item)

    nodes, kept = _drop_unused_nodes(nodes, kept)
    return nodes, kept, removed


def _prune_hachure_residual_edges(
    nodes: list[dict],
    edges: list[dict],
    cfg: dict,
    *,
    pass_name: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Remove tiny crumbs left after hatch-line removal.

    This is intentionally much narrower than general spur pruning: it only runs
    when hachures were already found in the sketch, and it defaults to sub-6px
    open edges that are below the pipeline's own micro-edge threshold.
    """
    max_len = float(cfg.get("hachure_residual_prune_max_length", 0.0) or 0.0)
    if max_len <= 0:
        return nodes, edges, []

    kept: list[dict] = []
    removed: list[dict] = []
    for edge in edges:
        if edge.get("is_closed"):
            kept.append(edge)
            continue
        length = _chain_length(edge.get("pixels") or [])
        if length >= max_len:
            kept.append(edge)
            continue
        item = dict(edge)
        item["is_hachure"] = True
        item["hachure"] = {
            "pass": pass_name,
            "length": float(length),
            "reason": "tiny_residual_after_hachure_removal",
        }
        removed.append(item)

    if not removed:
        return nodes, edges, []
    nodes, kept = _drop_unused_nodes(nodes, kept)
    return nodes, kept, removed


def _deduplicate_hachure_edges(
    edges: list[dict], overlap_threshold: float = 0.75
) -> list[dict]:
    """Drop duplicate side-layer strokes emitted by two hatch passes.

    The non-destructive C3 prepass and the main-graph cleanup can describe the
    same hatch with different edge IDs or slightly different endpoint halos.
    Overlap proposes duplicates, but deletion also requires complete source
    containment. A partly overlapping stroke can carry unique endpoints or ink.
    """
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("hachure dedup overlap threshold must be in [0, 1]")
    kept: list[dict] = []
    kept_pixels: list[set[tuple[int, int]]] = []
    for edge in edges:
        pixels = {
            (int(pixel[0]), int(pixel[1]))
            for pixel in edge.get("pixels", [])
        }
        duplicate = False
        if pixels:
            for existing in kept_pixels:
                denominator = min(len(pixels), len(existing))
                if (
                    denominator
                    and len(pixels & existing) / denominator >= overlap_threshold
                    and pixels <= existing
                ):
                    duplicate = True
                    break
        if duplicate:
            continue
        kept.append(edge)
        kept_pixels.append(pixels)
    return kept


_GATE_UNSET = object()


def gate_bound(config, key, default, *, maximum=None):
    """Read a quality-gate bound. `null` disables the gate; a number is literal.

    The older convention disabled a gate on any falsy value, so 0 -- the
    strictest-looking setting a reader could put in the file -- silently
    turned the gate off, and a ratio gate was disabled by setting it above
    1.0, which reads as the strictest setting of all. Both made
    config_deploy.yaml assert the opposite of what it did. Here the only way
    to disable a gate is to say so.
    """
    value = config.get(key, _GATE_UNSET)
    if value is _GATE_UNSET:
        value = default
    if value is None:
        return None
    value = float(value)
    if maximum is not None and value > maximum:
        raise ValueError(
            f"{key}={value:g} exceeds the maximum this quantity can reach ({maximum:g}), "
            f"so the gate could never fire; write `{key}: null` to disable it explicitly"
        )
    return value


def _open_edge_length_stats(edges: list[dict]) -> tuple[list[float], float, float, float]:
    open_lengths = [
        _chain_length([(int(p[0]), int(p[1])) for p in edge["pixels"]])
        for edge in edges
        if not edge.get("is_closed")
    ]
    if not open_lengths:
        return [], 0.0, 0.0, 0.0
    median = float(np.median(open_lengths))
    micro = float(sum(1 for length in open_lengths if length < 6.0) / len(open_lengths))
    short = float(sum(1 for length in open_lengths if length < 15.0) / len(open_lengths))
    return open_lengths, median, micro, short


def _prune_floating_noise(
    nodes: list[dict],
    edges: list[dict],
    cfg: dict,
) -> tuple[list[dict], list[dict], int, set]:
    """Drop isolated tiny open components -- scanned-patent skeleton speckle.

    Distinct from spur pruning (``_simplify_graph``, which trims dead-end
    branches still ATTACHED to the graph): this removes lone 2-node edges whose
    BOTH endpoints have degree 1, i.e. free-floating fragments disconnected from
    everything. On real patent scans these dominate the micro-edge count -- the
    pilot-v3 bolt-circle (EP3499690A1/F0006) reached micro=0.51 with 99/124
    micro-edges floating and 76 of them <=2.5 px, pure speckle scattered across
    the whole sheet -- and wrongly trip the Stage-2 fragmentation gate. Only
    floating components at/under ``floating_noise_max_len`` px are dropped, so a
    genuine small detached feature of real size is kept. Returns
    (nodes, edges, n_removed).
    """
    if not cfg.get("prune_floating_noise", False):
        return nodes, edges, 0, set()
    max_len = float(cfg.get("floating_noise_max_len", 3.0))
    deg: dict = {}
    for e in edges:
        deg[e["source"]] = deg.get(e["source"], 0) + 1
        deg[e["target"]] = deg.get(e["target"], 0) + 1
    drop = set()
    for e in edges:
        if e.get("is_closed"):
            continue
        if deg.get(e["source"], 0) == 1 and deg.get(e["target"], 0) == 1:
            if _chain_length([(int(p[0]), int(p[1])) for p in e["pixels"]]) <= max_len:
                drop.add(e["id"])
    if not drop:
        return nodes, edges, 0, set()
    noise_pixels = {(int(px[0]), int(px[1]))
                    for e in edges if e["id"] in drop
                    for px in e.get("pixels", [])}
    edges = [e for e in edges if e["id"] not in drop]
    nodes, edges = _drop_unused_nodes(nodes, edges)
    return nodes, edges, len(drop), noise_pixels


def _group_dashed_edges(
    nodes: list[dict],
    edges: list[dict],
    cfg: dict,
) -> tuple[list[dict], list[dict], int]:
    """Group dashed straight lines and dashed circles into single logical edges.

    Dashed centre-lines and dashed bolt-circles reach Stage 2 from the skeleton
    as many short, DISCONNECTED segments. Each dash is geometrically correct,
    but the graph counts every dash as its own micro-edge, which inflates
    ``micro_edge_ratio`` and the isolated-component count and can trip the
    Stage-2 fragmentation gate on an otherwise-valid drawing (the bolt-circle
    centre line in the pilot-v3 benchmark, EP3499690A1/F0006, was gated at
    micro=0.51 for exactly this reason). ``_simplify_graph`` cannot help: it
    merges collinear edges that SHARE a node, and dashes share none (the gaps).

    This pass merges runs of collinear, regularly-gapped short segments into one
    open edge, and rings of short segments lying on a common circle into one
    closed edge. Merged edges carry ``is_dashed=True`` and an estimated
    ``dash_pattern`` (or ``circle``) so Stage 3/4 render the gaps AS gaps rather
    than bridging them with solid ink -- the segments are grouped logically, not
    physically joined. Conservative by construction: needs >= min_segments,
    tight collinearity/co-circularity, and regular gaps, so a real feature line
    that merely happens to pass nearby is not swallowed. Returns
    (nodes, edges, n_groups_formed).
    """
    if not cfg.get("dashed_grouping", False):
        return nodes, edges, 0

    dmax        = float(cfg.get("dash_max_length", 50.0))
    dmin        = float(cfg.get("dash_min_length", 1.5))
    ang_tol     = float(cfg.get("dash_angle_tol_deg", 12.0))
    lat_tol     = float(cfg.get("dash_lateral_tol_px", 3.5))
    gap_factor  = float(cfg.get("dash_max_gap_factor", 3.5))
    gap_abs     = float(cfg.get("dash_max_gap_px", 45.0))
    min_dashes  = int(cfg.get("dash_min_segments", 3))
    circ_min    = int(cfg.get("dash_circle_min_segments", 6))
    circ_rms    = float(cfg.get("dash_circle_rms_frac", 0.06))
    circ_resid  = float(cfg.get("dash_circle_max_resid_px", 4.0))
    circ_span   = float(cfg.get("dash_circle_min_span_deg", 200.0))

    cand: list[dict] = []
    other: list[dict] = []
    for e in edges:
        pix = e.get("pixels") or []
        if e.get("is_closed") or len(pix) < 2:
            other.append(e); continue
        L = _chain_length([(int(p[0]), int(p[1])) for p in pix])
        if dmin <= L <= dmax:
            p0 = np.asarray(pix[0], float); p1 = np.asarray(pix[-1], float)
            d = p1 - p0
            cand.append({"e": e, "p0": p0, "p1": p1, "mid": 0.5 * (p0 + p1),
                         "ang": float(np.degrees(np.arctan2(d[1], d[0])) % 180.0),
                         "len": L})
        else:
            other.append(e)

    if len(cand) < min_dashes:
        return nodes, edges, 0

    used = [False] * len(cand)
    new_edges: list[dict] = []
    next_id = max((e["id"] for e in edges), default=-1) + 1
    n_groups = 0

    def _perp(mid, base, ang):
        th = np.radians(ang); dv = np.array([np.cos(th), np.sin(th)])
        v = mid - base
        return abs(-v[0] * dv[1] + v[1] * dv[0])

    # ── straight dashed lines ────────────────────────────────────────────────
    for i in range(len(cand)):
        if used[i]:
            continue
        ci = cand[i]
        th = np.radians(ci["ang"]); dv = np.array([np.cos(th), np.sin(th)])
        group = [i]
        for j in range(len(cand)):
            if j == i or used[j]:
                continue
            cj = cand[j]
            da = abs(ci["ang"] - cj["ang"]); da = min(da, 180.0 - da)
            if da <= ang_tol and _perp(cj["mid"], ci["mid"], ci["ang"]) <= lat_tol:
                group.append(j)
        if len(group) < min_dashes:
            continue

        def _span(k):
            a = float(cand[k]["p0"] @ dv); b = float(cand[k]["p1"] @ dv)
            return (a, b) if a <= b else (b, a)

        members = sorted(group, key=lambda k: float(cand[k]["mid"] @ dv))
        run = [members[0]]; best = []
        for a, b in zip(members, members[1:]):
            gap = _span(b)[0] - _span(a)[1]
            local = max(cand[a]["len"], cand[b]["len"])
            if 0.0 <= gap <= min(gap_abs, gap_factor * local):
                run.append(b)
            else:
                if len(run) > len(best):
                    best = run
                run = [b]
        if len(run) > len(best):
            best = run
        if len(best) < min_dashes:
            continue

        for k in best:
            used[k] = True
        ordered = sorted(best, key=lambda k: float(cand[k]["mid"] @ dv))
        pix_concat = [[int(p[0]), int(p[1])]
                      for k in ordered for p in cand[k]["e"]["pixels"]]
        gaps = [max(0.0, _span(b)[0] - _span(a)[1])
                for a, b in zip(ordered, ordered[1:])]
        new_edges.append({
            "id": next_id,
            "source": cand[ordered[0]]["e"]["source"],
            "target": cand[ordered[-1]]["e"]["target"],
            "pixels": pix_concat, "smooth_pts": [], "is_closed": False,
            "is_dashed": True,
            "dash_pattern": [round(float(np.mean([cand[k]["len"] for k in ordered])), 1),
                             round(float(np.mean(gaps)), 1) if gaps else 0.0],
        })
        next_id += 1; n_groups += 1

    # ── dashed circles (RANSAC: robust to stray short segments) ───────────────
    def _kasa(pts):
        x, y = pts[:, 0], pts[:, 1]
        A = np.c_[2 * x, 2 * y, np.ones(len(x))]
        sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
        cx, cy, c = sol
        return float(cx), float(cy), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))

    def _circ3(p, q, s):
        ax, ay = p; bx, by = q; cx_, cy_ = s
        d = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        if abs(d) < 1e-6:
            return None
        a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx_ * cx_ + cy_ * cy_
        ux = (a2 * (by - cy_) + b2 * (cy_ - ay) + c2 * (ay - by)) / d
        uy = (a2 * (cx_ - bx) + b2 * (ax - cx_) + c2 * (bx - ax)) / d
        return ux, uy, np.hypot(ax - ux, ay - uy)

    def _span_deg(idxs, cx, cy):
        angs = np.sort(np.array([
            np.degrees(np.arctan2(cand[i]["mid"][1] - cy,
                                  cand[i]["mid"][0] - cx)) % 360.0 for i in idxs]))
        return 360.0 - float(np.diff(np.concatenate([angs, [angs[0] + 360]])).max())

    rng = np.random.default_rng(0)
    for _ in range(4):  # up to 4 separate dashed circles
        rem = [i for i in range(len(cand)) if not used[i]]
        if len(rem) < circ_min:
            break
        mids = {i: cand[i]["mid"] for i in rem}
        allm = np.array([mids[i] for i in rem])
        # a circle can't have radius >> the span of its own points; this rejects
        # the degenerate huge-radius fits that near-collinear samples produce.
        max_r = float(max(np.ptp(allm[:, 0]), np.ptp(allm[:, 1]))) or 1.0

        def _tol(r):  # absolute residual cap so big-r fits can't swallow strays
            return min(circ_rms * r, circ_resid)

        best: list[int] = []
        best_model = None
        for _it in range(200):
            trio = rng.choice(len(rem), size=3, replace=False)
            m = _circ3(*(mids[rem[t]] for t in trio))
            if m is None or not np.isfinite(m).all() or not (5.0 <= m[2] <= max_r):
                continue
            cx, cy, r = m
            tol = _tol(r)
            inl = [i for i in rem
                   if abs(np.hypot(mids[i][0] - cx, mids[i][1] - cy) - r) <= tol]
            if len(inl) > len(best):
                best, best_model = inl, m
        if len(best) < circ_min or _span_deg(best, best_model[0], best_model[1]) < circ_span:
            break
        # refine on the consensus set, re-select inliers
        pts = np.array([mids[i] for i in best])
        cx, cy, r = _kasa(pts)
        if not (5.0 <= r <= max_r):
            cx, cy, r = best_model
        inliers = [i for i in best
                   if abs(np.hypot(mids[i][0] - cx, mids[i][1] - cy) - r) <= _tol(r)]
        if len(inliers) < circ_min or _span_deg(inliers, cx, cy) < circ_span:
            break
        order = sorted(inliers, key=lambda i: np.degrees(
            np.arctan2(cand[i]["mid"][1] - cy, cand[i]["mid"][0] - cx)) % 360.0)
        for i in inliers:
            used[i] = True
        pix_concat = [[int(p[0]), int(p[1])]
                      for k in order for p in cand[k]["e"]["pixels"]]
        src = cand[order[0]]["e"]["source"]
        new_edges.append({
            "id": next_id, "source": src, "target": src,
            "pixels": pix_concat, "smooth_pts": [], "is_closed": True,
            "is_dashed": True,
            "circle": [round(float(cx), 1), round(float(cy), 1), round(float(r), 1)],
        })
        next_id += 1; n_groups += 1

    if n_groups == 0:
        return nodes, edges, 0

    kept_cand = [cand[i]["e"] for i in range(len(cand)) if not used[i]]
    edges = other + kept_cand + new_edges
    nodes, edges = _drop_unused_nodes(nodes, edges)
    return nodes, edges, n_groups


def _should_run_hachure_cleanup(edges: list[dict], cfg: dict) -> bool:
    open_lengths, _median, micro, short = _open_edge_length_stats(edges)
    if len(open_lengths) < int(cfg.get("hachure_trigger_min_open_edges", 40)):
        return False
    micro_trigger = float(cfg.get("hachure_trigger_micro_edge_ratio", 0.20))
    short_trigger = float(cfg.get("hachure_trigger_short_edge_ratio", 0.55))
    return micro >= micro_trigger or short >= short_trigger


def _leave_direction(pix, at_start: bool, baseline: float = 8.0):
    """
    Unit direction in which a pixel chain leaves one of its ends.

    `at_start=True`  → direction leaving pix[0]   (into the chain)
    `at_start=False` → direction leaving pix[-1]  (into the chain)

    Sampled over up to `baseline` px of arc-length so a single staircase
    pixel does not dominate the estimate; for chains shorter than the
    baseline this is just the chord direction.
    """
    seq = pix if at_start else pix[::-1]
    if len(seq) < 2:
        return None
    p0 = np.asarray(seq[0], dtype=np.float64)
    acc = 0.0
    far = seq[-1]
    for k in range(1, len(seq)):
        far = seq[k]
        acc += float(np.hypot(seq[k][0] - seq[k - 1][0],
                              seq[k][1] - seq[k - 1][1]))
        if acc >= baseline:
            break
    v = np.asarray(far, dtype=np.float64) - p0
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _merge_close_junctions(
    E: list, node_by_id: dict, radius: float
) -> bool:
    """
    Collapse clusters of junction nodes that sit within `radius` px of each
    other, connected by a short edge. On dense patent scans the CN map fires
    a forest of junctions 2–4 px apart (a "hairball" of tiny inter-junction
    stubs); merging them into one node removes the stubs and lets the
    surviving strokes reconnect. Union-find over a single sweep. Returns True
    if anything merged.
    """
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    merged_any = False
    for e in E:
        if e is None or e["a"] == e["b"]:
            continue
        if e.get("residual_parents"):
            continue
        na, nb = node_by_id.get(e["a"]), node_by_id.get(e["b"])
        if na is None or nb is None:
            continue
        if na["type"] != KP_JUNCTION or nb["type"] != KP_JUNCTION:
            continue
        if _chain_length(e["pix"]) < radius:
            union(e["a"], e["b"])
            merged_any = True

    if not merged_any:
        return False

    # representative = cluster root; recompute its centroid from members
    members = {}
    for nid in list(node_by_id.keys()):
        if nid in parent:
            members.setdefault(find(nid), []).append(nid)
    for root, mids in members.items():
        if len(mids) <= 1:
            continue
        xs = [node_by_id[m]["x"] for m in mids]
        ys = [node_by_id[m]["y"] for m in mids]
        node_by_id[root]["x"] = int(round(sum(xs) / len(xs)))
        node_by_id[root]["y"] = int(round(sum(ys) / len(ys)))
        for m in mids:
            if m != root:
                node_by_id.pop(m, None)

    # relabel edge endpoints to cluster roots; drop edges that collapse to a
    # short self-loop (the stubs we merged across).
    for i, e in enumerate(E):
        if e is None:
            continue
        a = find(e["a"]) if e["a"] in parent else e["a"]
        b = find(e["b"]) if e["b"] in parent else e["b"]
        if a == b and not e.get("residual_parents") and _chain_length(e["pix"]) < max(radius * 2.0, 6.0):
            E[i] = None
            continue
        e["a"], e["b"] = a, b
    return True


def _simplify_graph(
    nodes: list[dict],
    edges: list[dict],
    spur_min_len: float = 6.0,
    collinear_max_angle: float = 28.0,
    collinear_tangent_baseline: float = 8.0,
    junction_merge_radius: float = 4.0,
    max_iter: int = 40,
) -> tuple[list[dict], list[dict]]:
    """
    De-fragment the stroke graph produced by `_extract_topology`.

    Three operations, iterated to a fixed point:
      (a) Spur pruning — delete short dead-end edges that dangle off a
          junction (one end has graph-degree 1, the other is a junction,
          chain length < spur_min_len). These are Zhang-Suen barbs and
          scan-noise whiskers, the dominant fragment source on patent TIFs.
      (b) Degree-2 dissolution — a node where exactly two open edges meet is
          a phantom junction (genuine corners are CN=2, never CN>=3 junctions),
          so the two edges are merged into one continuous chain and the node
          removed. This stitches long strokes that the CN map split at
          staircase artefacts.
      (c) Collinear through-merge — at a real junction (degree >= 3), pairs of
          incident edges that continue nearly straight through the node
          (turn within collinear_max_angle of 180°) are merged, so a line
          passing through a T-junction/crossing stays a single primitive.

    Closed-loop edges (and their loop_anchor nodes) are passed through
    untouched. Returns renumbered (nodes, edges).
    """
    from collections import defaultdict

    if collinear_tangent_baseline <= 0:
        raise ValueError("collinear tangent baseline must be positive")

    node_by_id = {n["id"]: dict(n) for n in nodes}

    closed_edges = [e for e in edges if e.get("is_closed")]
    # Working representation for open edges: {a, b, pix}
    E: list[dict | None] = []
    for e in edges:
        if e.get("is_closed"):
            continue
        pix = [(int(p[0]), int(p[1])) for p in e["pixels"]]
        E.append({"a": e["source"], "b": e["target"], "pix": pix,
                  "residual_parents": set(e.get("residual_parent_edge_ids", []))})

    cos_thresh = np.cos(np.radians(collinear_max_angle))

    def build_adj():
        adj = defaultdict(list)
        for i, e in enumerate(E):
            if e is None:
                continue
            adj[e["a"]].append(i)
            adj[e["b"]].append(i)
        return adj

    def other(e, node):
        return e["b"] if e["a"] == node else e["a"]

    def merge(i, j, node):
        """Merge alive edges i, j that share `node`; node becomes interior."""
        ei, ej = E[i], E[j]
        # orient ei to END at node
        if ei["b"] == node:
            pi, a_node = ei["pix"], ei["a"]
        else:
            pi, a_node = ei["pix"][::-1], ei["b"]
        # orient ej to START at node
        if ej["a"] == node:
            pj, b_node = ej["pix"], ej["b"]
        else:
            pj, b_node = ej["pix"][::-1], ej["a"]
        if pi and pj and pi[-1] == pj[0]:
            pj = pj[1:]
        E[i] = {"a": a_node, "b": b_node, "pix": pi + pj,
                "residual_parents": ei.get("residual_parents", set()) | ej.get("residual_parents", set())}
        E[j] = None

    for _ in range(max_iter):
        changed = False

        # ── (a0) collapse hairball: merge junctions within radius ───────────
        if junction_merge_radius > 0 and _merge_close_junctions(
                E, node_by_id, junction_merge_radius):
            continue

        adj = build_adj()
        deg = {nid: len(idxs) for nid, idxs in adj.items()}

        # ── (a) spur pruning ────────────────────────────────────────────────
        for i, e in enumerate(E):
            if e is None:
                continue
            a, b = e["a"], e["b"]
            if e.get("residual_parents"):
                continue
            da, db = deg.get(a, 0), deg.get(b, 0)
            # dead-end = degree-1 end; keep it only if it is a free stroke
            # (both ends degree 1) or long enough to be real.
            tip = None
            root = None
            if da == 1 and db >= 3:
                tip, root = a, b
            elif db == 1 and da >= 3:
                tip, root = b, a
            if tip is None:
                continue
            if _chain_length(e["pix"]) < spur_min_len:
                E[i] = None
                deg[tip] = 0
                deg[root] = deg.get(root, 1) - 1
                changed = True
        if changed:
            continue   # recompute adjacency before dissolving

        # ── (b) degree-2 dissolution ────────────────────────────────────────
        # Maintain a LIVE adjacency: each merge in this pass rewrites edge
        # endpoints, so a once-per-pass static adjacency would let us pop a node
        # that an earlier merge just made an endpoint of — leaving that edge
        # pointing at a deleted node (the dangling-reference bug). Updating
        # incidence after every merge keeps the "exactly two incident edges"
        # test honest while still dissolving whole chains in one pass.
        adj = build_adj()
        live = defaultdict(set)
        for n_, idxs in adj.items():
            for k in idxs:
                if E[k] is not None:
                    live[n_].add(k)
        for nid in list(live.keys()):
            inc = [k for k in live[nid] if E[k] is not None]
            if len(inc) != 2:
                continue
            # Learned/explicit corners are intentional degree-2 split points.
            # Dissolving them makes the corner detector ineffective and hands
            # Stage 3 a compound polyline instead of two fit-ready primitives.
            node = node_by_id.get(nid)
            if node is not None and node.get("type") == KP_CORNER:
                continue
            i, j = inc
            if i == j:
                continue   # self-loop edge through this node — leave it
            fa, fb = other(E[i], nid), other(E[j], nid)
            if fa == nid or fb == nid:
                continue
            merge(i, j, nid)            # E[i] := fa—fb ; E[j] := None
            node_by_id.pop(nid, None)
            live.pop(nid, None)
            live[fa].discard(j); live[fb].discard(j)
            live[fa].add(i);     live[fb].add(i)
            changed = True
        if changed:
            continue

        # ── (c) collinear through-merge at real junctions ───────────────────
        adj = build_adj()
        for nid, idxs in adj.items():
            alive = [k for k in idxs if E[k] is not None]
            if len(alive) < 3:
                continue
            # direction each incident edge leaves the node
            dirs = {}
            for k in alive:
                e = E[k]
                d = _leave_direction(
                    e["pix"],
                    at_start=(e["a"] == nid),
                    baseline=collinear_tangent_baseline,
                )
                if d is not None:
                    dirs[k] = d
            # candidate straight-through pairs (leaving dirs ~opposite)
            cands = []
            ks = list(dirs.keys())
            for a_i in range(len(ks)):
                for b_i in range(a_i + 1, len(ks)):
                    ka, kb = ks[a_i], ks[b_i]
                    # straight-through ⇒ leaving directions point opposite ways
                    straightness = -float(np.dot(dirs[ka], dirs[kb]))
                    if straightness >= cos_thresh:
                        # avoid creating a self-loop (both far ends same node)
                        if other(E[ka], nid) == other(E[kb], nid):
                            continue
                        cands.append((straightness, ka, kb))
            cands.sort(reverse=True)
            used = set()
            for _s, ka, kb in cands:
                if ka in used or kb in used:
                    continue
                if E[ka] is None or E[kb] is None:
                    continue
                merge(ka, kb, nid)
                used.add(ka)
                used.add(kb)
                changed = True
            # node stays (it still has the un-merged incident edges)
        if not changed:
            break

    # ── rebuild node/edge lists ─────────────────────────────────────────────
    alive_edges = [e for e in E if e is not None]
    used_nodes = set()
    for e in alive_edges:
        used_nodes.add(e["a"])
        used_nodes.add(e["b"])
    for e in closed_edges:
        used_nodes.add(e["source"])
        used_nodes.add(e["target"])

    out_nodes = [node_by_id[nid] for nid in node_by_id if nid in used_nodes]

    # Invariant guard: every edge endpoint must exist as a node. The live-
    # adjacency dissolution above keeps this true, but synthesize any missing
    # endpoint from the edge's pixel coords so the contract holds unconditionally
    # (a dangling reference breaks every downstream topology consumer).
    have = {n["id"] for n in out_nodes}
    for e in alive_edges:
        for nid, px in ((e["a"], e["pix"][0] if e["pix"] else None),
                        (e["b"], e["pix"][-1] if e["pix"] else None)):
            if nid not in have:
                x, y = (int(px[0]), int(px[1])) if px is not None else (0, 0)
                out_nodes.append({"id": nid, "x": x, "y": y,
                                  "type": KP_ENDPOINT, "confidence": 1.0})
                have.add(nid)

    out_edges = []
    eid = 0
    for e in alive_edges:
        is_closed = e["a"] == e["b"]
        out_edges.append({
            "id": eid, "source": e["a"], "target": e["b"],
            "pixels": [[int(p[0]), int(p[1])] for p in e["pix"]],
            "smooth_pts": [], "is_closed": is_closed,
        })
        if e.get("residual_parents"):
            out_edges[-1].update(topology_origin="recovered_residual",
                                 residual_parent_edge_ids=sorted(e["residual_parents"]))
        eid += 1
    for e in closed_edges:
        ce = dict(e)
        ce["id"] = eid
        out_edges.append(ce)
        eid += 1

    return out_nodes, out_edges


def _skeleton_without_hachure_edges(
    skeleton: np.ndarray,
    kept_edges: list[dict],
    removed_edges: list[dict],
) -> tuple[np.ndarray, int]:
    """Remove hatch-only pixels while retaining every structural shared pixel."""
    kept_pixels = {
        (int(pixel[0]), int(pixel[1]))
        for edge in kept_edges
        for pixel in edge.get("pixels", [])
    }
    removed_pixels = {
        (int(pixel[0]), int(pixel[1]))
        for edge in removed_edges
        for pixel in edge.get("pixels", [])
    }
    exclusive = removed_pixels - kept_pixels
    cleaned = np.where(skeleton > 0, 255, 0).astype(np.uint8)
    height, width = cleaned.shape
    for x, y in exclusive:
        if 0 <= x < width and 0 <= y < height:
            cleaned[y, x] = 0
    return cleaned, len(exclusive)


def _endpoint_tangent(
    binary: np.ndarray,
    endpoint: tuple[int, int],
    support_length: float,
) -> np.ndarray | None:
    """Estimate the direction from an endpoint into its surviving stroke."""
    height, width = binary.shape
    start = (int(endpoint[0]), int(endpoint[1]))
    if not (0 <= start[0] < width and 0 <= start[1] < height):
        return None
    if not binary[start[1], start[0]]:
        return None

    distances = {start: 0.0}
    queue = [(0.0, start[1], start[0])]
    farthest = start
    while queue:
        distance, y, x = heapq.heappop(queue)
        point = (x, y)
        if distance != distances.get(point):
            continue
        if distance > distances.get(farthest, 0.0):
            farthest = point
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if not (0 <= nx_ < width and 0 <= ny_ < height):
                    continue
                if not binary[ny_, nx_]:
                    continue
                step = float(np.hypot(dx, dy))
                candidate = distance + step
                if candidate > support_length:
                    continue
                neighbour = (nx_, ny_)
                if candidate + 1e-9 >= distances.get(neighbour, float("inf")):
                    continue
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, ny_, nx_))

    vector = np.asarray(farthest, dtype=np.float64) - np.asarray(
        start, dtype=np.float64
    )
    norm = float(np.linalg.norm(vector))
    if norm < min(2.0, support_length * 0.5):
        return None
    return vector / norm


def _removed_component_path(
    labels: np.ndarray,
    component_id: int,
    start: tuple[int, int],
    target: tuple[int, int],
    maximum_length: float,
) -> list[tuple[int, int]] | None:
    """Shortest 8-connected path through one removed-pixel component."""
    height, width = labels.shape
    queue = [(0.0, start[1], start[0])]
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    distances = {start: 0.0}
    while queue:
        distance, y, x = heapq.heappop(queue)
        point = (x, y)
        if distance != distances.get(point):
            continue
        if point == target:
            break
        if distance >= maximum_length:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                neighbour = (x + dx, y + dy)
                nx_, ny_ = neighbour
                if not (0 <= nx_ < width and 0 <= ny_ < height):
                    continue
                if neighbour in parent:
                    previous = distances.get(neighbour, float("inf"))
                else:
                    previous = float("inf")
                if neighbour != target and labels[ny_, nx_] != component_id:
                    continue
                candidate = distance + float(np.hypot(dx, dy))
                if candidate > maximum_length or candidate + 1e-9 >= previous:
                    continue
                parent[neighbour] = point
                distances[neighbour] = candidate
                heapq.heappush(queue, (candidate, ny_, nx_))

    if target not in parent:
        return None
    path = []
    cursor: tuple[int, int] | None = target
    while cursor is not None:
        path.append(cursor)
        cursor = parent[cursor]
    path.reverse()
    return path


def _repair_hachure_crossing_gaps(
    original_skeleton: np.ndarray,
    cleaned_skeleton: np.ndarray,
    removed_hachures: list[dict],
    cfg: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Reconnect short structural gaps created while subtracting hatch ink.

    Candidate endpoints must touch the same removed-pixel component, continue
    nearly straight through it, and be connected by pixels present in the
    original skeleton. Paths aligned with a nearby removed hatch edge are
    rejected. This restores contour crossings without redrawing the hatch
    family or inventing geometry across blank raster space.
    """
    if original_skeleton.shape != cleaned_skeleton.shape:
        raise ValueError("original and cleaned skeleton shapes must match")
    repaired = np.where(cleaned_skeleton > 0, 255, 0).astype(np.uint8)
    original = original_skeleton > 0
    cleaned = repaired > 0
    removed = original & ~cleaned
    if not np.any(removed):
        return repaired, []

    max_path_length = float(cfg.get("hachure_gap_repair_max_path_length", 14.0))
    max_angle = float(cfg.get("hachure_gap_repair_max_angle", 24.0))
    tangent_support = float(cfg.get("hachure_gap_repair_tangent_support", 10.0))
    hatch_angle_margin = float(
        cfg.get("hachure_gap_repair_hatch_angle_margin", 16.0)
    )
    hatch_context_radius = float(
        cfg.get("hachure_gap_repair_hatch_context_radius", 64.0)
    )
    if max_path_length <= 0 or tangent_support <= 0:
        return repaired, []

    _count, labels = cv2.connectedComponents(
        removed.astype(np.uint8), connectivity=8
    )
    endpoints: list[dict] = []
    for cluster in _cn_keypoint_clusters(repaired):
        if cluster.get("type") != KP_ENDPOINT:
            continue
        point = min(
            ((int(x), int(y)) for x, y in cluster.get("pixels", [])),
            key=lambda p: (p[1], p[0]),
            default=(int(cluster["x"]), int(cluster["y"])),
        )
        tangent = _endpoint_tangent(cleaned, point, tangent_support)
        if tangent is None:
            continue
        component_ids = set()
        x, y = point
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nx_, ny_ = x + dx, y + dy
                if 0 <= ny_ < labels.shape[0] and 0 <= nx_ < labels.shape[1]:
                    component_id = int(labels[ny_, nx_])
                    if component_id > 0:
                        component_ids.add(component_id)
        if component_ids:
            endpoints.append({
                "point": point,
                "tangent": tangent,
                "components": component_ids,
            })

    hatch_features = []
    for edge in removed_hachures:
        feature = _edge_line_features(edge)
        if feature is not None:
            hatch_features.append(feature)

    cosine_threshold = float(np.cos(np.radians(max_angle)))
    candidates: list[dict] = []
    for first_index in range(len(endpoints)):
        first = endpoints[first_index]
        for second_index in range(first_index + 1, len(endpoints)):
            second = endpoints[second_index]
            shared_components = first["components"] & second["components"]
            if not shared_components:
                continue
            delta = np.asarray(second["point"], dtype=np.float64) - np.asarray(
                first["point"], dtype=np.float64
            )
            chord = float(np.linalg.norm(delta))
            if chord <= 1.0 or chord > max_path_length:
                continue
            direction = delta / chord
            alignment = min(
                -float(np.dot(first["tangent"], direction)),
                float(np.dot(second["tangent"], direction)),
                -float(np.dot(first["tangent"], second["tangent"])),
            )
            if alignment < cosine_threshold:
                continue

            path = None
            component_id = None
            for candidate_component in sorted(shared_components):
                candidate_path = _removed_component_path(
                    labels,
                    candidate_component,
                    first["point"],
                    second["point"],
                    max_path_length,
                )
                if candidate_path is None:
                    continue
                if _chain_length(candidate_path) > max_path_length:
                    continue
                if path is None or _chain_length(candidate_path) < _chain_length(path):
                    path = candidate_path
                    component_id = candidate_component
            if path is None:
                continue

            candidate_angle = float(
                np.degrees(np.arctan2(direction[1], direction[0])) % 180.0
            )
            midpoint = 0.5 * (
                np.asarray(first["point"], dtype=np.float64)
                + np.asarray(second["point"], dtype=np.float64)
            )
            local_hatch_angles = [
                float(feature["angle_deg"])
                for feature in hatch_features
                if float(np.linalg.norm(feature["center"] - midpoint))
                <= hatch_context_radius
            ]
            nearest_hatch_delta = min(
                (_angle_delta_deg(candidate_angle, angle)
                 for angle in local_hatch_angles),
                default=90.0,
            )
            if nearest_hatch_delta <= hatch_angle_margin:
                continue

            restored_pixels = [
                point for point in path if removed[point[1], point[0]]
            ]
            if not restored_pixels:
                continue
            candidates.append({
                "first": first_index,
                "second": second_index,
                "component_id": int(component_id),
                "path": path,
                "restored_pixels": restored_pixels,
                "alignment": alignment,
                "candidate_angle": candidate_angle,
                "nearest_hatch_delta": nearest_hatch_delta,
                "path_length": _chain_length(path),
            })

    candidates.sort(
        key=lambda item: (
            -item["alignment"],
            item["path_length"],
            item["first"],
            item["second"],
        )
    )
    used_endpoints: set[int] = set()
    repairs: list[dict] = []
    for candidate in candidates:
        if (
            candidate["first"] in used_endpoints
            or candidate["second"] in used_endpoints
        ):
            continue
        for x, y in candidate["restored_pixels"]:
            repaired[y, x] = 255
        used_endpoints.update((candidate["first"], candidate["second"]))
        repairs.append({
            "start": list(endpoints[candidate["first"]]["point"]),
            "end": list(endpoints[candidate["second"]]["point"]),
            "path_length": float(candidate["path_length"]),
            "restored_pixels": len(candidate["restored_pixels"]),
            "alignment": float(candidate["alignment"]),
            "hatch_angle_delta": float(candidate["nearest_hatch_delta"]),
        })
    return repaired, repairs


def _edge_support_mask(
    edges: list[dict],
    shape: tuple[int, int],
    dilation: int = 0,
) -> np.ndarray:
    """Rasterize edge pixels into a binary support mask."""
    if dilation < 0:
        raise ValueError("edge support dilation must be non-negative")
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for edge in edges:
        for pixel in edge.get("pixels", []):
            x, y = int(pixel[0]), int(pixel[1])
            if 0 <= x < width and 0 <= y < height:
                mask[y, x] = 1
    if dilation > 0 and np.any(mask):
        size = dilation * 2 + 1
        mask = cv2.dilate(mask, np.ones((size, size), dtype=np.uint8))
    return mask


def _separate_hatch_additive_masks(
    region_mask: np.ndarray | None,
    stroke_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep established region evidence separate from model-only evidence."""
    stroke = np.asarray(stroke_mask) > 0
    if region_mask is None:
        region = np.zeros(stroke.shape, dtype=np.uint8)
    else:
        region_array = np.asarray(region_mask)
        if region_array.shape != stroke.shape:
            raise ValueError(
                "region and hatch-stroke masks must have identical shapes"
            )
        region = (region_array > 0).astype(np.uint8)
    additive = (stroke & (region == 0)).astype(np.uint8)
    return region, additive


def _cluster_scalar_positions(
    values: list[float], tolerance: float
) -> list[float]:
    """Merge nearby scalar observations into deterministic cluster centres."""
    if not values:
        return []
    groups: list[list[float]] = []
    for value in sorted(float(item) for item in values):
        if not groups or value - groups[-1][-1] > tolerance:
            groups.append([value])
        else:
            groups[-1].append(value)
    return [float(np.mean(group)) for group in groups]


def _hough_hachure_stroke_mask(
    skeleton: np.ndarray,
    hatch_region_mask: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, list[dict]]:
    """Find periodic line families inside a learned filled hatch region.

    Hough fragments on one structural contour collapse to one or two normal
    offsets, whereas a hatch family yields many distinct, regularly separated
    offsets. Selecting by physical-line count avoids treating the most common
    fragment angle as the hatch direction on dense mechanical drawings.
    """
    if skeleton.shape != hatch_region_mask.shape:
        raise ValueError("skeleton and hatch region mask shapes must match")
    height, width = skeleton.shape
    output = np.zeros((height, width), dtype=np.uint8)
    metadata: list[dict] = []
    if not np.any(skeleton) or not np.any(hatch_region_mask):
        return output, metadata

    minimum_region_area = int(cfg.get("hachure_hough_min_region_area", 500))
    threshold = int(cfg.get("hachure_hough_threshold", 15))
    minimum_length = float(cfg.get("hachure_hough_min_line_length", 8.0))
    maximum_gap = float(cfg.get("hachure_hough_max_line_gap", 4.0))
    angle_step = float(cfg.get("hachure_hough_angle_step", 5.0))
    angle_tolerance = float(cfg.get("hachure_hough_angle_tolerance", 7.5))
    rho_tolerance = float(cfg.get("hachure_hough_rho_tolerance", 4.0))
    minimum_family_lines = int(cfg.get("hachure_hough_min_family_lines", 10))
    maximum_families = int(cfg.get("hachure_hough_max_families", 2))
    second_family_ratio = float(cfg.get("hachure_hough_second_family_ratio", 0.45))
    minimum_angle_separation = float(
        cfg.get("hachure_hough_min_angle_separation", 25.0)
    )
    line_thickness = int(cfg.get("hachure_hough_mask_thickness", 3))
    interior_margin = int(cfg.get("hachure_hough_interior_margin", 2))
    if angle_step <= 0 or angle_tolerance <= 0 or rho_tolerance < 0:
        raise ValueError("invalid hachure Hough angular/rho configuration")
    if maximum_families <= 0 or line_thickness <= 0:
        return output, metadata

    regions = hatch_region_mask.astype(np.uint8)
    if interior_margin > 0:
        size = interior_margin * 2 + 1
        eroded = cv2.erode(regions, np.ones((size, size), np.uint8))
        if np.any(eroded):
            regions = eroded
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        regions, connectivity=8
    )
    skeleton_binary = skeleton > 0
    angle_centres = np.arange(0.0, 180.0, angle_step)

    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < minimum_region_area:
            continue
        component = labels == component_id
        component_ink = np.where(
            skeleton_binary & component, 255, 0
        ).astype(np.uint8)
        lines = cv2.HoughLinesP(
            component_ink,
            1.0,
            np.pi / 360.0,
            threshold=threshold,
            minLineLength=minimum_length,
            maxLineGap=maximum_gap,
        )
        if lines is None:
            continue

        records: list[dict] = []
        for x1, y1, x2, y2 in lines[:, 0]:
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < minimum_length:
                continue
            angle = float(
                np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
            )
            records.append({
                "points": (int(x1), int(y1), int(x2), int(y2)),
                "angle": angle,
                "length": length,
                "center": (0.5 * (x1 + x2), 0.5 * (y1 + y2)),
            })
        if not records:
            continue

        candidates: list[dict] = []
        for centre in angle_centres:
            member_indices = [
                index for index, record in enumerate(records)
                if _angle_delta_deg(record["angle"], float(centre))
                <= angle_tolerance
            ]
            if not member_indices:
                continue
            theta = np.radians(float(centre))
            normal = np.asarray([-np.sin(theta), np.cos(theta)])
            rhos = [
                float(np.dot(np.asarray(records[index]["center"]), normal))
                for index in member_indices
            ]
            line_positions = _cluster_scalar_positions(rhos, rho_tolerance)
            if len(line_positions) < minimum_family_lines:
                continue
            gaps = np.diff(line_positions)
            usable_gaps = gaps[
                (gaps > rho_tolerance)
                & (gaps <= float(cfg.get("hachure_hough_max_spacing", 50.0)))
            ]
            spacing = float(np.median(usable_gaps)) if len(usable_gaps) else 0.0
            spacing_cv = (
                float(np.std(usable_gaps) / np.mean(usable_gaps))
                if len(usable_gaps) > 1 and float(np.mean(usable_gaps)) > 0
                else 0.0
            )
            candidates.append({
                "angle": float(centre),
                "member_indices": member_indices,
                "line_count": len(line_positions),
                "segment_count": len(member_indices),
                "total_length": float(sum(
                    records[index]["length"] for index in member_indices
                )),
                "spacing": spacing,
                "spacing_cv": spacing_cv,
            })

        candidates.sort(
            key=lambda item: (
                -item["line_count"],
                -item["total_length"],
                item["angle"],
            )
        )
        selected: list[dict] = []
        for candidate in candidates:
            if any(
                _angle_delta_deg(candidate["angle"], item["angle"])
                < minimum_angle_separation
                for item in selected
            ):
                continue
            if selected and (
                candidate["line_count"]
                < selected[0]["line_count"] * second_family_ratio
            ):
                continue
            selected.append(candidate)
            if len(selected) >= maximum_families:
                break

        for family_index, family in enumerate(selected):
            family_mask = np.zeros_like(output)
            for record_index in family["member_indices"]:
                x1, y1, x2, y2 = records[record_index]["points"]
                cv2.line(
                    family_mask,
                    (x1, y1),
                    (x2, y2),
                    1,
                    thickness=line_thickness,
                    lineType=cv2.LINE_8,
                )
            family_mask &= component.astype(np.uint8)
            output |= family_mask
            metadata.append({
                "component_id": int(component_id),
                "family_index": int(family_index),
                "bbox": [
                    int(stats[component_id, cv2.CC_STAT_LEFT]),
                    int(stats[component_id, cv2.CC_STAT_TOP]),
                    int(stats[component_id, cv2.CC_STAT_WIDTH]),
                    int(stats[component_id, cv2.CC_STAT_HEIGHT]),
                ],
                "angle_deg": round(float(family["angle"]), 3),
                "line_count": int(family["line_count"]),
                "segment_count": int(family["segment_count"]),
                "total_length": round(float(family["total_length"]), 3),
                "spacing": round(float(family["spacing"]), 3),
                "spacing_cv": round(float(family["spacing_cv"]), 3),
            })

    # Keep only support near actual skeleton ink. This prevents line gaps from
    # turning a Hough segment into a broad removal corridor.
    support = cv2.dilate(
        skeleton_binary.astype(np.uint8), np.ones((3, 3), np.uint8)
    )
    output &= support
    return output, metadata


def _run_hachure_topology_prepass(
    skeleton: np.ndarray,
    hatch_mask: np.ndarray,
    cfg: dict,
    max_search_radius: int,
) -> tuple[np.ndarray, list[dict], int]:
    """Extract repeated hatch strokes inside the CNN hatch-region prior."""
    nodes, edges = _extract_topology(
        skeleton,
        _cn_keypoint_clusters(skeleton),
        max_search_radius,
        directional_walk=bool(cfg.get("topology_directional_walk", False)),
        directional_walk_baseline=float(
            cfg.get("topology_directional_walk_baseline", 6.0)
        ),
    )
    if cfg.get("simplify_graph", True):
        nodes, edges = _simplify_graph(
            nodes,
            edges,
            spur_min_len=cfg.get("spur_min_length", 6.0),
            collinear_max_angle=cfg.get("merge_collinear_max_angle", 28.0),
            collinear_tangent_baseline=cfg.get(
                "hachure_topology_prepass_tangent_baseline", 8.0
            ),
            junction_merge_radius=cfg.get("junction_merge_radius", 0.0),
        )
    geometric_cfg = dict(cfg)
    geometric_cfg["hachure_max_removed_edge_ratio"] = float(
        cfg.get("hachure_topology_prepass_max_edge_ratio", 0.95)
    )
    _nodes, _kept, geometric = _remove_hachure_edges(
        nodes,
        edges,
        geometric_cfg,
        pass_name="topology_prepass_geometric",
    )
    if not geometric:
        return skeleton, [], 0

    interior_margin = int(
        cfg.get("hachure_topology_prepass_interior_margin", 0)
    )
    if interior_margin < 0:
        raise ValueError("hachure prepass interior margin must be non-negative")
    interior_mask = hatch_mask
    if interior_margin > 0:
        size = interior_margin * 2 + 1
        interior_mask = cv2.erode(
            hatch_mask.astype(np.uint8),
            np.ones((size, size), dtype=np.uint8),
        )
        if not np.any(interior_mask):
            return skeleton, [], 0

    mask_cfg = dict(cfg)
    mask_cfg["hachure_cnn_max_removed_edge_ratio"] = 1.0
    _nodes, _kept, removed = _remove_hachures_cnn(
        nodes,
        geometric,
        interior_mask,
        mask_cfg,
        pass_name="cnn_topology_prepass",
    )
    if not removed:
        return skeleton, [], 0

    geometric_meta = {
        int(edge["id"]): dict(edge.get("hachure", {})) for edge in geometric
    }
    selected_ids = {int(edge["id"]) for edge in removed}
    for edge in removed:
        metadata = geometric_meta.get(int(edge["id"]), {})
        metadata.update(edge.get("hachure", {}))
        metadata["pass"] = "cnn_topology_prepass_geometric"
        edge["hachure"] = metadata
    kept = [edge for edge in edges if int(edge["id"]) not in selected_ids]

    cleaned, removed_pixels = _skeleton_without_hachure_edges(
        skeleton, kept, removed
    )
    foreground = int(np.count_nonzero(skeleton))
    maximum_ratio = float(
        cfg.get("hachure_topology_prepass_max_removed_pixel_ratio", 0.95)
    )
    removed_ratio = removed_pixels / max(foreground, 1)
    if not np.any(cleaned) or (
        maximum_ratio > 0 and removed_ratio > maximum_ratio
    ):
        logger.warning(
            "Hachure topology prepass skipped: removed-pixel ratio %.3f "
            "exceeds guard %.3f",
            removed_ratio,
            maximum_ratio,
        )
        return skeleton, [], 0
    return cleaned, removed, removed_pixels


def _trace_hatch_stroke_side_layer(
    mask: np.ndarray,
    cfg: dict,
    max_search_radius: int,
) -> list[dict]:
    """Trace a predicted hatch skeleton without changing the main graph."""
    binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
    if not np.any(binary):
        return []
    nodes, edges = _extract_topology(
        binary,
        _cn_keypoint_clusters(binary),
        max_search_radius,
        directional_walk=bool(cfg.get("topology_directional_walk", False)),
        directional_walk_baseline=float(
            cfg.get("topology_directional_walk_baseline", 6.0)
        ),
    )
    if cfg.get("simplify_graph", True):
        _nodes, edges = _simplify_graph(
            nodes,
            edges,
            spur_min_len=0.0,
            collinear_max_angle=cfg.get("merge_collinear_max_angle", 28.0),
            junction_merge_radius=0.0,
        )
    for edge in edges:
        edge["is_hachure"] = True
        edge["hachure"] = {
            "pass": "stroke_multilabel_prepass",
            "source": "hatch_stroke_cnn",
        }
    return edges


def _hatch_stroke_masks_from_probabilities(
    skeleton: np.ndarray,
    probabilities: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Threshold multilabel predictions on existing skeleton ink only."""
    binary = np.asarray(skeleton) > 0
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.shape != (2, *binary.shape):
        raise ValueError(
            "hatch-stroke probabilities must have shape (2, H, W)"
        )
    hatch_threshold = float(
        cfg.get("hachure_stroke_hatch_high_threshold", 0.8)
    )
    structural_threshold = float(
        cfg.get("hachure_stroke_structural_low_threshold", 0.2)
    )
    if not 0.0 <= hatch_threshold <= 1.0:
        raise ValueError("hatch-stroke hatch threshold must be in [0, 1]")
    if not 0.0 <= structural_threshold <= 1.0:
        raise ValueError("hatch-stroke structural threshold must be in [0, 1]")

    predicted_hatch = binary & (probabilities[1] >= hatch_threshold)
    safe_hatch_only = predicted_hatch & (
        probabilities[0] < structural_threshold
    )
    foreground_pixels = int(np.count_nonzero(binary))
    safe_candidate_pixels = int(np.count_nonzero(safe_hatch_only))
    safe_candidate_ratio = safe_candidate_pixels / max(foreground_pixels, 1)
    maximum_ratio = float(
        cfg.get("hachure_stroke_max_removed_pixel_ratio", 0.75)
    )
    stats = {
        "applied": False,
        "foreground_pixels": foreground_pixels,
        "predicted_hatch_pixels": int(np.count_nonzero(predicted_hatch)),
        "safe_candidate_pixels": safe_candidate_pixels,
        "safe_candidate_ratio": safe_candidate_ratio,
        "safe_removed_pixels": 0,
        "safe_removed_ratio": 0.0,
        "hatch_high_threshold": hatch_threshold,
        "structural_low_threshold": structural_threshold,
        "maximum_removed_pixel_ratio": maximum_ratio,
        "side_layer_pixels": 0,
        "side_layer_edges": 0,
    }
    return binary, predicted_hatch, safe_hatch_only, stats


def _hatch_stroke_prepass_from_probabilities(
    skeleton: np.ndarray,
    probabilities: np.ndarray,
    cfg: dict,
    max_search_radius: int,
) -> tuple[np.ndarray, list[dict], dict]:
    """Subtract only confident hatch-only pixels and retain a hatch side layer."""
    binary, predicted_hatch, safe_hatch_only, stats = (
        _hatch_stroke_masks_from_probabilities(
            skeleton, probabilities, cfg
        )
    )
    removed_pixels = int(stats["safe_candidate_pixels"])
    removed_ratio = float(stats["safe_candidate_ratio"])
    maximum_ratio = float(stats["maximum_removed_pixel_ratio"])
    stats.update({
        "mode": "pre_topology",
        "safe_removed_pixels": removed_pixels,
        "safe_removed_ratio": removed_ratio,
    })
    if removed_pixels == 0:
        return np.asarray(skeleton).copy(), [], stats
    if maximum_ratio > 0 and removed_ratio > maximum_ratio:
        stats["guard"] = "maximum_removed_pixel_ratio"
        return np.asarray(skeleton).copy(), [], stats

    # Recover multilabel crossing pixels only near confidently hatch-only ink.
    # Component-wide propagation is unsafe on patent drawings: one crossing can
    # connect a hatch family to a large structural network and duplicate that
    # entire network in the side layer.
    recovery_radius = int(
        cfg.get("hachure_stroke_overlap_recovery_radius", 1)
    )
    if recovery_radius < 0:
        raise ValueError("hatch-stroke overlap recovery radius must be non-negative")
    if recovery_radius:
        size = recovery_radius * 2 + 1
        local_support = cv2.dilate(
            safe_hatch_only.astype(np.uint8),
            np.ones((size, size), dtype=np.uint8),
        ) > 0
    else:
        local_support = safe_hatch_only
    side_mask = predicted_hatch & local_support
    side_edges = _trace_hatch_stroke_side_layer(
        side_mask, cfg, max_search_radius
    )
    if not side_edges:
        stats["guard"] = "empty_side_layer"
        return np.asarray(skeleton).copy(), [], stats

    cleaned = binary & ~safe_hatch_only
    if not np.any(cleaned):
        stats["guard"] = "empty_structural_skeleton"
        return np.asarray(skeleton).copy(), [], stats
    stats.update({
        "applied": True,
        "side_layer_pixels": int(np.count_nonzero(side_mask)),
        "side_layer_edges": len(side_edges),
        "overlap_recovery_radius": recovery_radius,
    })
    for edge in side_edges:
        edge["hachure"].update({
            "hatch_high_threshold": stats["hatch_high_threshold"],
            "structural_low_threshold": stats["structural_low_threshold"],
        })
    return cleaned.astype(np.uint8) * 255, side_edges, stats


def _hatch_stroke_postmask_from_probabilities(
    skeleton: np.ndarray,
    probabilities: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray | None, dict]:
    """Build a structural-vetoed hatch mask without changing the skeleton."""
    _binary, _predicted_hatch, safe_hatch_only, stats = (
        _hatch_stroke_masks_from_probabilities(
            skeleton, probabilities, cfg
        )
    )
    candidate_pixels = int(stats["safe_candidate_pixels"])
    candidate_ratio = float(stats["safe_candidate_ratio"])
    maximum_ratio = float(stats["maximum_removed_pixel_ratio"])
    stats["mode"] = "post_topology"
    if candidate_pixels == 0:
        stats["guard"] = "empty_safe_hatch_mask"
        return None, stats
    if maximum_ratio > 0 and candidate_ratio > maximum_ratio:
        stats["guard"] = "maximum_removed_pixel_ratio"
        return None, stats

    stats.update({
        "applied": True,
        "edge_mask_pixels": candidate_pixels,
        "pixel_subtraction_applied": False,
    })
    return safe_hatch_only.astype(np.uint8), stats


def _run_hatch_stroke_prepass(
    skeleton: np.ndarray,
    model,
    cfg: dict,
    max_search_radius: int,
) -> tuple[np.ndarray, list[dict], dict]:
    probabilities = _cnn_hatch_stroke_probabilities(
        skeleton,
        model,
        patch=int(cfg.get("hachure_stroke_patch", _HATCH_PATCH)),
        stride=int(cfg.get("hachure_stroke_stride", _HATCH_STRIDE)),
    )
    return _hatch_stroke_prepass_from_probabilities(
        skeleton, probabilities, cfg, max_search_radius
    )


def _run_hatch_stroke_postmask(
    skeleton: np.ndarray,
    model,
    cfg: dict,
) -> tuple[np.ndarray | None, dict]:
    probabilities = _cnn_hatch_stroke_probabilities(
        skeleton,
        model,
        patch=int(cfg.get("hachure_stroke_patch", _HATCH_PATCH)),
        stride=int(cfg.get("hachure_stroke_stride", _HATCH_STRIDE)),
    )
    return _hatch_stroke_postmask_from_probabilities(
        skeleton, probabilities, cfg
    )


def _snap_to_skeleton(
    binary: np.ndarray, x: int, y: int, radius: int = 6
) -> Optional[tuple[int, int]]:
    """Find the nearest foreground pixel to (x, y) within radius."""
    H, W = binary.shape
    best_dist = float("inf")
    best_px   = None
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx_ < W and binary[ny, nx_]:
                d = dx*dx + dy*dy
                if d < best_dist:
                    best_dist = d
                    best_px   = (nx_, ny)
    return best_px


def _path_is_direct(
    path: list[tuple[int, int]], max_detour_ratio: float
) -> bool:
    """
    Accept a Dijkstra path only if its total pixel-length is not more than
    max_detour_ratio times the straight-line distance between endpoints.
    Rejects paths that snake through unrelated stroke regions.
    """
    if len(path) < 2:
        return True
    dx = path[-1][0] - path[0][0]
    dy = path[-1][1] - path[0][1]
    straight = max(np.hypot(dx, dy), 1.0)
    path_len = sum(
        np.hypot(path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
        for i in range(len(path)-1)
    )
    return (path_len / straight) <= max_detour_ratio


# ═══════════════════════════════════════════════════════════════════════════
# LAYER 3 — CURVE SMOOTHING
# ═══════════════════════════════════════════════════════════════════════════

def _smooth_edges(
    edges: list[dict],
    rdp_epsilon: float = 1.5,
    spline_smoothing: float = 2.0,
    min_smooth_pts: int = 4,
    spline_overshoot_limit: float = 5.0,
) -> list[dict]:
    """
    For each edge, apply:
      1. RDP simplification (removes collinear redundancy)
      2. B-spline fitting (scipy splprep) for sub-pixel smooth coordinates

    Results stored in edge["smooth_pts"] as [[x, y], ...].

    The spline is validated against the raw pixel bounding box: if any
    smooth point lies more than spline_overshoot_limit px outside the
    raw-pixel bbox the spline is discarded and RDP-simplified points are
    used instead. This prevents scipy splprep end-effect oscillations from
    introducing false curvature on long, nearly-straight edges.
    """
    for edge in edges:
        pixels = edge["pixels"]
        # Closed-loop pixels come from a connected-component set and are sorted
        # lexicographically, not in contour order. RDP/splprep on that ordering
        # is both meaningless and quadratic on larger circles. Closed-loop
        # fitters deliberately consume ``pixels`` directly, so leave the
        # smoothing side channel empty.
        if edge.get("is_closed"):
            edge["smooth_pts"] = []
            continue
        if len(pixels) < 2:
            edge["smooth_pts"] = pixels
            continue

        pts = np.array(pixels, dtype=np.float64)  # (N, 2)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()

        # Step 1: RDP simplification
        simplified = rdp(pts, epsilon=rdp_epsilon)
        if len(simplified) < 2:
            edge["smooth_pts"] = pixels
            continue

        rdp_pts = [[float(p[0]), float(p[1])] for p in simplified]

        # Step 2: B-spline fitting (needs at least min_smooth_pts points)
        if len(simplified) < min_smooth_pts:
            edge["smooth_pts"] = rdp_pts
            continue

        try:
            x = simplified[:, 0]
            y = simplified[:, 1]

            # Degree: cubic (k=3) unless too few points
            k = min(3, len(simplified) - 1)

            # splprep fits a parametric spline through (x, y)
            tck, u = splprep([x, y], s=spline_smoothing, k=k, quiet=True)

            # Evaluate at uniform parameter steps
            n_eval = max(len(pixels) // 2, 10)
            u_new  = np.linspace(0, 1, n_eval)
            x_new, y_new = splev(u_new, tck)

            # Guard: reject the spline if it overshoots the raw pixel bbox.
            # scipy splprep can oscillate badly on long near-straight edges
            # (end-effect / Runge phenomenon), introducing tens of pixels of
            # false curvature. Falling back to RDP points is always safe.
            lim = spline_overshoot_limit
            if (x_new.min() < x_min - lim or x_new.max() > x_max + lim or
                    y_new.min() < y_min - lim or y_new.max() > y_max + lim):
                logger.debug(
                    f"Spline overshoot on edge {edge['id']} "
                    f"(bbox x=[{x_min:.0f},{x_max:.0f}] y=[{y_min:.0f},{y_max:.0f}] "
                    f"spline x=[{x_new.min():.0f},{x_new.max():.0f}] "
                    f"y=[{y_new.min():.0f},{y_new.max():.0f}]) — using RDP fallback"
                )
                edge["smooth_pts"] = rdp_pts
                continue

            # Guard 2: reject the spline if it wanders off the actual skeleton.
            # The bbox guard above misses *interior* oscillations (the Runge
            # "leaf"): the spline stays inside the edge bbox yet deviates tens of
            # px from the raw pixel chain, and the polyline fallback then traces
            # that garbage. Measure max distance from each spline sample to the
            # nearest raw skeleton pixel; if it exceeds the overshoot limit the
            # spline failed — fall back to RDP points, which stay on-skeleton.
            max_dev = float(cKDTree(pts).query(np.column_stack([x_new, y_new]))[0].max())
            if max_dev > spline_overshoot_limit:
                logger.debug(
                    f"Spline off-skeleton on edge {edge['id']} "
                    f"(max_dev={max_dev:.1f}px > {spline_overshoot_limit}) — "
                    f"using RDP fallback"
                )
                edge["smooth_pts"] = rdp_pts
                continue

            edge["smooth_pts"] = [
                [float(xi), float(yi)] for xi, yi in zip(x_new, y_new)
            ]
        except Exception as exc:
            logger.debug(f"Spline fitting failed for edge {edge['id']}: {exc}")
            edge["smooth_pts"] = rdp_pts

    return edges


# ═══════════════════════════════════════════════════════════════════════════
# CONFIDENCE SIGNAL
# ═══════════════════════════════════════════════════════════════════════════

def _compute_isolation_ratio(
    skeleton: np.ndarray,
    edges: list[dict],
    ignored_pixels: set[tuple[int, int]] | None = None,
) -> float:
    """
    Fraction of foreground pixels not captured by any edge.
    isolation_ratio → 0 : perfect coverage
    isolation_ratio → 1 : almost nothing was captured
    """
    binary = (skeleton > 0)
    ignored_pixels = ignored_pixels or set()
    total = int(binary.sum())
    if ignored_pixels:
        total -= sum(
            1 for x, y in ignored_pixels
            if 0 <= y < skeleton.shape[0] and 0 <= x < skeleton.shape[1] and binary[y, x]
        )
    if total == 0:
        return 0.0

    covered = set()
    for edge in edges:
        for px in edge["pixels"]:
            covered.add((px[0], px[1]))

    uncovered = 0
    ys, xs = np.where(binary)
    for y, x in zip(ys, xs):
        if (int(x), int(y)) in ignored_pixels:
            continue
        if (x, y) not in covered:
            uncovered += 1

    return uncovered / total


# ═══════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════

def load_model(config: dict) -> Optional[PuhachovKeypointDetector]:
    """
    Attempt to load the Puhachov keypoint CNN.
    Returns None if weights are not available — classical fallback will be used.
    Called once at batch start.
    """
    # Limit PyTorch intra-op threads to 1 per process so that when multiple
    # worker processes run in parallel they don't compete for the same CPU cores.
    # Parallelism is provided at the process level by ProcessPoolExecutor.
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    weights_path = config.get("puhachov", {}).get("weights", "")
    device       = config.get("puhachov", {}).get("device", "cuda")

    try:
        model = PuhachovKeypointDetector(weights_path=weights_path, device=device)
        logger.info("Puhachov keypoint CNN ready.")
        return model
    except ModelNotAvailableError as exc:
        logger.warning(
            f"Puhachov CNN not available: {exc}\n"
            "  → Classical crossing-number detection will be used."
        )
        return None


# ─── Noise-loop circularity guard ─────────────────────────────────────────

def _is_circular_loop(pixels) -> bool:
    """
    Return True if the pixel sequence approximates a circle.

    Used to distinguish genuine small circles (e.g. construction points in
    clean CAD rasterizations) from irregular noise blobs in scanned patent
    TIFs — box corners, junction-clustering artefacts, text-glyph loops
    ('o', '0', 'D') and stipple/hatch texture dots. A real skeleton circle has
    pixels at a nearly uniform radius from the centroid AND traces close to a
    full revolution; a noise blob is irregular and/or only partially traced.

    Two independent criteria must both pass:
      1. RMS of radial deviations < threshold × mean radius, where the
         threshold is relaxed for small circles (r_mean < 12 px) because
         Zhang-Suen skeletonization produces staircase artefacts that inflate
         the RMS on tiny rings, even when geometrically a perfect circle.
      2. Path-completeness: the walked pixel-chain length must be within
         [0.7, 1.6] of the expected circumference (2*pi*r_mean). A spurious
         loop formed by topology mis-joining two nearby corner keypoints
         zig-zags back and forth in a small area — its walked path is 2-4x
         the circumference of its own bounding circle, which this range
         rejects; a real traced circle's path is close to 1x (skeleton
         staircase, at most mildly longer than the ideal circumference).

    Evidence (2026-07-22 patent pilot): on a pure flowchart figure with ZERO
    real circles, every one of 34-219 tiny closed loops per Stage-2 config
    passed the old single-criterion (RMS-only) check and became a hallucinated
    'circle' primitive in Stage 3. Adding the completeness criterion rejected
    14/15 sampled ghost loops in that figure while a threshold-only tightening
    could not do so without also risking genuine small circles.
    """
    pts = np.array(pixels, dtype=np.float64)
    if len(pts) < 6:
        return False
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    radii = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
    r_mean = radii.mean()
    if r_mean < 1.0:
        return False
    rms = float(np.sqrt(((radii - r_mean) ** 2).mean()))
    # Small circles (r < 12 px) have disproportionate staircase error; allow
    # up to 55 % relative RMS.  Larger circles keep the stricter 30 % limit.
    threshold = 0.55 if r_mean < 12.0 else 0.30
    if rms >= threshold * r_mean:
        return False

    path_len = float(np.hypot(*np.diff(pts, axis=0).T).sum())
    circumference = 2.0 * np.pi * r_mean
    if circumference < 1e-6:
        return False
    completeness = path_len / circumference
    return 0.7 <= completeness <= 1.6


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC STAGE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run(
    skeleton_path: Path,
    output_dir: Path,
    sketch_id: str,
    config: dict,
    model: Optional[PuhachovKeypointDetector] = None,
    *,
    source_image_path: Optional[Path] = None,
    hatch_model=None,
    hatch_stroke_model=None,
) -> Stage2Result:
    """
    Run Stage 2 on a single skeleton image.

    Parameters
    ----------
    skeleton_path : Path
        1px binary skeleton PNG from Stage 1.
    output_dir : Path
        Root output directory. Writes to output_dir/graphs/.
    sketch_id : str
        Unique identifier for this sketch.
    config : dict
        Parsed config.yaml content.
    model : PuhachovKeypointDetector | None
        Pre-loaded CNN instance; None → classical fallback.
    source_image_path : Path | None
        The grayscale image Stage 1 consumed (same coordinate frame as the
        skeleton).  Required for the learned hatch detector; None → geometric
        hachure removal.
    hatch_model : HatchUNet | None
        Pre-loaded Phase 2 hatch-region detector (see ``load_hatch_model``).
        When present alongside ``source_image_path``, its mask is the primary
        hachure signal; otherwise the geometric heuristic runs.
    hatch_stroke_model : HatchUNet | None
        Opt-in two-channel skeleton classifier. Confident hatch-only pixels are
        removed before topology tracing and retained in the hachure side layer.

    Returns
    -------
    Stage2Result
    """
    t_start = time.perf_counter()
    strict_models = bool(config.get("pipeline", {}).get("deployment", {}).get("strict_models", False))
    if strict_models:
        if model is None:
            raise RuntimeError("Deployment Puhachov model is required; fallback is disabled")
        if config.get("stage2", {}).get("hachure_use_cnn", False) and (
            hatch_model is None or source_image_path is None
        ):
            raise RuntimeError("Deployment hatch CNN requires its model and source image")

    graphs_dir = output_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    graph_path = graphs_dir / f"{sketch_id}_graph.json"

    # ── Load skeleton ─────────────────────────────────────────────────────
    skeleton = cv2.imread(str(skeleton_path), cv2.IMREAD_GRAYSCALE)
    if skeleton is None:
        raise FileNotFoundError(f"Cannot read skeleton: {skeleton_path}")

    H, W = skeleton.shape
    orig_H, orig_W = H, W
    stage2_scale = 1.0
    cfg_kp = config.get("stage2", {})

    # ── Resolution cap ────────────────────────────────────────────────────
    # Downsample the skeleton if it exceeds max_input_resolution.
    # The Puhachov CNN was trained on ~512 px images; feeding it 2000–2700 px
    # patent TIFs causes massive over-segmentation (10–300 k edges vs expected
    # 200–2 k).  Capping at 1000 px keeps the CNN in its effective range while
    # preserving enough geometric detail for the graph and RANSAC stages.
    # 1-px skeleton lines are dilated before downsampling so they survive
    # the resize without topological breaks.
    #
    # Tiled inference (puhachov.tiled) is the alternative: the CNN runs on
    # 512 px tiles at *native* resolution, so the cap is skipped entirely — the
    # detector stays in its trained receptive field without discarding detail.
    tiled   = bool(cfg_puh_pre.get("tiled", False)) if (cfg_puh_pre := config.get("puhachov", {})) else False
    max_res = cfg_kp.get("max_input_resolution", 0)
    if max_res and max(H, W) > max_res and not tiled:
        stage2_scale = max_res / max(H, W)
        new_W   = max(1, round(W * stage2_scale))
        new_H   = max(1, round(H * stage2_scale))
        # Dilate before resize so 1-px lines survive the interpolation step;
        # then re-skeletonize to restore 1-px width before graph building.
        ksize   = max(3, int(1.0 / stage2_scale) * 2 + 1)
        skeleton = cv2.dilate(skeleton, np.ones((ksize, ksize), np.uint8))
        skeleton = cv2.resize(skeleton, (new_W, new_H), interpolation=cv2.INTER_AREA)
        skeleton = (skeleton > 30).astype(np.uint8)
        skeleton = _skeletonize(skeleton).astype(np.uint8) * 255
        H, W    = skeleton.shape
        logger.info(f"[{sketch_id}] Capped skeleton to {W}×{H}px "
                    f"(scale={stage2_scale:.2f}, fg_px={int((skeleton>0).sum())})")

    logger.info(f"[{sketch_id}] Stage 2 — skeleton {W}×{H}px, "
                f"{int((skeleton > 0).sum())} foreground px")
    coverage_ledger = _CoverageLedger(skeleton)

    max_radius = cfg_kp.get("max_search_radius", 60)
    hatch_mask = None
    hatch_stroke_mask = None
    hatch_stroke_additive_mask = None
    prepass_subtracted = False
    hachure_gap_repairs: list[dict] = []
    hachure_hough_families: list[dict] = []
    removed_hachures: list[dict] = []
    preliminary_hachures: list[dict] = []
    hatch_stroke_mode = "disabled"
    hatch_stroke_stats: dict = {"applied": False}
    if (
        hatch_stroke_model is not None
        and cfg_kp.get("remove_hachures", False)
    ):
        try:
            hatch_stroke_mode = str(
                cfg_kp.get("hachure_stroke_mode", "pre_topology")
            ).strip().lower()
            if hatch_stroke_mode not in {
                "pre_topology",
                "post_topology",
                "post_topology_additive",
            }:
                raise ValueError(
                    "hachure_stroke_mode must be pre_topology, post_topology, "
                    "or post_topology_additive"
                )
            original_foreground = int(np.count_nonzero(skeleton))
            if hatch_stroke_mode == "pre_topology":
                cleaned, preliminary_hachures, hatch_stroke_stats = (
                    _run_hatch_stroke_prepass(
                        skeleton, hatch_stroke_model, cfg_kp, max_radius
                    )
                )
                if hatch_stroke_stats.get("applied"):
                    skeleton = cleaned
                    removed_hachures.extend(preliminary_hachures)
                    hatch_stroke_mask = _edge_support_mask(
                        preliminary_hachures, (H, W)
                    )
                    hatch_mask = hatch_stroke_mask
                    prepass_subtracted = True
                    logger.info(
                        f"[{sketch_id}] Hatch-stroke prepass: "
                        f"{len(preliminary_hachures)} side-layer edges, "
                        f"{hatch_stroke_stats['safe_removed_pixels']}/"
                        f"{original_foreground} confident hatch-only pixels "
                        "removed before topology"
                    )
            else:
                hatch_stroke_mask, hatch_stroke_stats = (
                    _run_hatch_stroke_postmask(
                        skeleton, hatch_stroke_model, cfg_kp
                    )
                )
                if hatch_stroke_stats.get("applied"):
                    if hatch_stroke_mode == "post_topology":
                        hatch_mask = hatch_stroke_mask
                    logger.info(
                        f"[{sketch_id}] Hatch-stroke {hatch_stroke_mode} mask: "
                        f"{hatch_stroke_stats['safe_candidate_pixels']}/"
                        f"{original_foreground} structurally-vetoed hatch "
                        "pixels; skeleton retained unchanged"
                    )
            if (
                not hatch_stroke_stats.get("applied")
                and hatch_stroke_stats.get("guard")
            ):
                logger.warning(
                    f"[{sketch_id}] Hatch-stroke {hatch_stroke_mode} skipped "
                    "by "
                    f"{hatch_stroke_stats['guard']} guard"
                )
        except Exception as exc:
            hatch_stroke_stats = {
                "applied": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            logger.warning(
                f"[{sketch_id}] Hatch-stroke inference failed ({exc}); "
                "continuing with the existing hachure path."
            )
    hatch_requested = (
        hatch_model is not None
        and source_image_path is not None
        and cfg_kp.get("remove_hachures", False)
    )
    if (
        hatch_requested
        and not prepass_subtracted
        and cfg_kp.get("hachure_hough_bridge_keypoints", False)
        and not cfg_kp.get("hachure_topology_prepass", False)
    ):
        try:
            hatch_mask, threshold = _aligned_hatch_mask(
                source_image_path, hatch_model, (H, W), cfg_kp
            )
            hatch_stroke_mask, hachure_hough_families = (
                _hough_hachure_stroke_mask(
                    skeleton,
                    hatch_mask,
                    cfg_kp,
                )
            )
            logger.info(
                f"[{sketch_id}] Early hatch Hough mask: "
                f"{len(hachure_hough_families)} periodic families, "
                f"{int(np.count_nonzero(hatch_stroke_mask))} support pixels"
            )
        except Exception as exc:
            logger.warning(
                f"[{sketch_id}] Early hatch Hough inference failed ({exc}); "
                "continuing without temporary hatch routing keypoints."
            )
            hatch_mask = None
            hatch_stroke_mask = None
            hachure_hough_families = []
    if (
        hatch_requested
        and not prepass_subtracted
        and cfg_kp.get("hachure_topology_prepass", False)
    ):
        try:
            hatch_mask, threshold = _aligned_hatch_mask(
                source_image_path, hatch_model, (H, W), cfg_kp
            )
            logger.info(
                f"[{sketch_id}] Hatch CNN mask: "
                f"{int(hatch_mask.sum())}/{H*W} px "
                f"({hatch_mask.mean()*100:.1f}%) @ thr={threshold}"
            )
            original_foreground = int(np.count_nonzero(skeleton))
            prepass_skeleton, preliminary_hachures, removed_pixels = (
                _run_hachure_topology_prepass(
                    skeleton, hatch_mask, cfg_kp, max_radius
                )
            )
            removed_hachures.extend(preliminary_hachures)
            if preliminary_hachures:
                hatch_stroke_mask = _edge_support_mask(
                    preliminary_hachures, (H, W)
                )
                subtract_pixels = bool(
                    cfg_kp.get("hachure_topology_prepass_subtract", True)
                )
                if subtract_pixels:
                    if cfg_kp.get("hachure_gap_repair", False):
                        prepass_skeleton, hachure_gap_repairs = (
                            _repair_hachure_crossing_gaps(
                                skeleton,
                                prepass_skeleton,
                                preliminary_hachures,
                                cfg_kp,
                            )
                        )
                        restored_pixels = sum(
                            int(repair["restored_pixels"])
                            for repair in hachure_gap_repairs
                        )
                        removed_pixels = max(0, removed_pixels - restored_pixels)
                        if hachure_gap_repairs:
                            logger.info(
                                f"[{sketch_id}] Hatch gap repair: "
                                f"{len(hachure_gap_repairs)} contour gaps, "
                                f"{restored_pixels} original pixels restored"
                            )
                    skeleton = prepass_skeleton
                    prepass_subtracted = True
                logger.info(
                    f"[{sketch_id}] Hatch topology prepass: "
                    f"{len(preliminary_hachures)} side-layer edges, "
                    f"{removed_pixels}/{original_foreground} exclusive pixels "
                    f"{'removed' if subtract_pixels else 'retained'} before "
                    "learned topology"
                )
        except Exception as exc:
            logger.warning(
                f"[{sketch_id}] Hatch topology prepass failed ({exc}); "
                "using the post-topology hachure path."
            )
            hatch_mask = None
            removed_hachures = []

    # ── Layer 1: Keypoint detection ───────────────────────────────────────
    # Keypoint detector knobs live in the `puhachov` config block; fall back to
    # `stage2` then a default for backward compatibility.
    cfg_puh     = config.get("puhachov", {})
    conf_thresh = cfg_puh.get("keypoint_threshold", cfg_kp.get("keypoint_threshold", 0.50))
    nms_radius  = cfg_puh.get("nms_radius",         cfg_kp.get("nms_radius", 5))

    # Scale NMS radius up for images larger than the model's training resolution.
    ref_res = cfg_kp.get("nms_reference_resolution", 512)
    if ref_res and max(H, W) > ref_res:
        nms_radius = max(nms_radius, int(round(nms_radius * max(H, W) / ref_res)))
        logger.debug(f"[{sketch_id}] Adaptive NMS radius: {nms_radius} "
                     f"(image {max(H,W)}px vs ref {ref_res}px)")

    # Keypoint clusters seed topology extraction. The CN path produces the same
    # clusters _extract_topology used to recompute internally (byte-identical);
    # a learned detector returns bare points that are snapped onto the skeleton.
    fusion        = bool(cfg_puh.get("fusion", False))
    fusion_dist   = cfg_puh.get("fusion_min_corner_dist", 5.0)
    tile_size     = int(cfg_puh.get("tile_size", 512))
    tile_stride   = int(cfg_puh.get("tile_stride", tile_size // 2))
    if model is not None:
        try:
            if tiled:
                keypoints = model.detect_tiled(
                    skeleton, conf_thresh, nms_radius, tile_size, tile_stride)
            else:
                keypoints = model.detect(skeleton, conf_thresh, nms_radius)
            if fusion:
                # CN endpoints/junctions + CNN corners only
                kp_clusters = _fuse_cn_cnn_clusters(skeleton, keypoints, fusion_dist)
                kp_source   = "fusion"
            else:
                kp_clusters = _clusters_from_points(keypoints, skeleton)
                kp_source   = "cnn"
            if tiled:
                kp_source += "_tiled"
            logger.info(f"[{sketch_id}] CNN detected {len(keypoints)} keypoints "
                        f"→ {len(kp_clusters)} clusters ({kp_source}"
                        f"{f', {tile_size}px tiles' if tiled else ''})")
        except Exception as exc:
            if strict_models:
                raise RuntimeError("Deployment Puhachov inference failed; fallback is disabled") from exc
            logger.warning(f"[{sketch_id}] CNN keypoint detection failed "
                           f"({exc}), using classical fallback")
            kp_clusters = _cn_keypoint_clusters(skeleton)
            kp_source   = "classical_fallback"
    else:
        kp_clusters = _cn_keypoint_clusters(skeleton)
        kp_source   = "classical"
        logger.info(f"[{sketch_id}] Classical CN: {len(kp_clusters)} keypoints")
    if prepass_subtracted:
        kp_source += "_hatchstroke_subtracted"
    elif hatch_stroke_stats.get("applied"):
        kp_source += "_hatchstroke_edge_mask"

    unclaimed_mode = "all"
    hachure_routing_candidate_junctions = 0
    hachure_routing_junctions = 0
    hachure_routing_endpoints = 0
    structural_topology = bool(
        cfg_kp.get("hachure_structural_topology", False)
        and model is not None
        and kp_source.startswith("cnn")
    )
    if structural_topology:
        for cluster in kp_clusters:
            cluster["structural_seed"] = True
            cluster["topology_origin"] = "cnn_structural"
        structural_count = len(kp_clusters)
        if hatch_mask is not None:
            if prepass_subtracted:
                kp_source += "_hatch_subtracted"
                logger.info(
                    f"[{sketch_id}] Hatch structural topology: "
                    f"{structural_count} learned structural seeds on the "
                    "hatch-subtracted skeleton"
                )
            else:
                bridge_mask = (
                    hatch_stroke_mask
                    if hatch_stroke_mask is not None
                    else np.zeros_like(hatch_mask)
                )
                bridge_dilation = int(
                    cfg_kp.get("hachure_bridge_mask_dilation", 0)
                )
                if bridge_dilation > 0:
                    size = bridge_dilation * 2 + 1
                    bridge_mask = cv2.dilate(
                        bridge_mask,
                        np.ones((size, size), dtype=np.uint8),
                    )
                recovered_endpoints: list[dict] = []
                if cfg_kp.get(
                    "hachure_structural_recover_cn_endpoints", False
                ):
                    recovered_endpoints = _structural_cn_endpoint_clusters(
                        skeleton,
                        bridge_mask,
                        kp_clusters,
                        dedup_radius=float(
                            cfg_kp.get(
                                "hachure_endpoint_recovery_dedup_radius",
                                nms_radius,
                            )
                        ),
                    )
                    kp_clusters.extend(recovered_endpoints)
                bridges = _hatch_bridge_junction_clusters(
                    skeleton,
                    bridge_mask,
                    kp_clusters,
                    dedup_radius=float(
                        cfg_kp.get("hachure_bridge_dedup_radius", nms_radius)
                    ),
                )
                kp_clusters.extend(bridges)
                unclaimed_mode = "closed_only"
                if recovered_endpoints:
                    kp_source += "_cn_endpoints"
                kp_source += "_hatch_bridged"
                logger.info(
                    f"[{sketch_id}] Hatch structural topology: "
                    f"{structural_count} learned structural seeds + "
                    f"{len(recovered_endpoints)} recovered CN endpoints + "
                    f"{len(bridges)} temporary CN junctions"
                )
        else:
            logger.warning(
                f"[{sketch_id}] Hatch structural topology requested without "
                "an aligned hatch mask; using the regular learned tracer."
            )

    # The learned keypoint model intentionally suppresses many raster-only
    # hatch crossings. Without an explicit node there, the pixel walker can
    # turn from a contour onto a hatch stroke and create one mixed edge. Add
    # only CN nodes supported by a verified thin Hough stroke mask: the tracer
    # stops locally, simplification pairs straight-through arms, and cleanup
    # can then classify the separated hatch edge without deleting the contour.
    if (
        cfg_kp.get("hachure_hough_bridge_keypoints", False)
        and hatch_stroke_mask is not None
        and np.any(hatch_stroke_mask)
        and not prepass_subtracted
    ):
        bridge_dedup_radius = float(
            cfg_kp.get("hachure_hough_bridge_dedup_radius", nms_radius)
        )
        bridge_candidates = _hatch_bridge_junction_clusters(
            skeleton,
            hatch_stroke_mask,
            kp_clusters,
            dedup_radius=bridge_dedup_radius,
        )
        if cfg_kp.get("hachure_hough_bridge_mixed_only", True):
            bridges = _mixed_hatch_junction_clusters(
                skeleton,
                hatch_stroke_mask,
                bridge_candidates,
                cfg_kp,
                hachure_hough_families,
            )
        else:
            bridges = bridge_candidates
        kp_clusters.extend(bridges)

        hatch_endpoints: list[dict] = []
        if cfg_kp.get("hachure_hough_recover_cn_endpoints", True):
            endpoint_mask = hatch_stroke_mask
            endpoint_dilation = int(
                cfg_kp.get("hachure_hough_endpoint_mask_dilation", 2)
            )
            if endpoint_dilation < 0:
                raise ValueError(
                    "hachure Hough endpoint mask dilation must be non-negative"
                )
            if endpoint_dilation > 0:
                size = endpoint_dilation * 2 + 1
                endpoint_mask = cv2.dilate(
                    endpoint_mask,
                    np.ones((size, size), dtype=np.uint8),
                )
            hatch_endpoints = _hatch_cn_endpoint_clusters(
                skeleton,
                endpoint_mask,
                kp_clusters,
                dedup_radius=float(
                    cfg_kp.get(
                        "hachure_hough_endpoint_dedup_radius",
                        nms_radius,
                    )
                ),
            )
            kp_clusters.extend(hatch_endpoints)

        hachure_routing_candidate_junctions = len(bridge_candidates)
        hachure_routing_junctions = len(bridges)
        hachure_routing_endpoints = len(hatch_endpoints)
        # Keep all unclaimed components. Unlike the older structural-bridge
        # experiment, Hough endpoints make open hatch strokes traceable, so
        # dropping open unclaimed ink would only lower structural recall.
        unclaimed_mode = "all"
        kp_source += "_hough_cn_routed"
        logger.info(
            f"[{sketch_id}] Hough-guided topology routing: "
            f"{len(bridge_candidates)} candidate -> {len(bridges)} mixed "
            "temporary CN junctions + "
            f"{len(hatch_endpoints)} temporary CN endpoints"
        )

    # ── Layer 2: Topology extraction ──────────────────────────────────────
    coverage_ledger.record("topology_prepasses", [], removed_hachures, active_skeleton=skeleton)
    nodes, edges = _extract_topology(
        skeleton,
        kp_clusters,
        max_radius,
        unclaimed_mode=unclaimed_mode,
        directional_walk=bool(
            cfg_kp.get("topology_directional_walk", False)
        ),
        directional_walk_baseline=float(
            cfg_kp.get("topology_directional_walk_baseline", 6.0)
        ),
    )
    logger.info(f"[{sketch_id}] Graph: {len(nodes)} nodes, {len(edges)} edges")
    coverage_ledger.record("topology_extraction", edges, removed_hachures)

    # ── Learned hatch-region mask (Phase 2 CNN) ───────────────────────────
    # Run the detector on the same image Stage 1 consumed (identical coordinate
    # frame as the skeleton), then downscale the mask by stage2_scale so it
    # aligns with the possibly resolution-capped skeleton the graph lives in.
    if hatch_mask is None and hatch_requested and not prepass_subtracted:
        try:
            hatch_mask, threshold = _aligned_hatch_mask(
                source_image_path, hatch_model, (H, W), cfg_kp
            )
            logger.info(
                f"[{sketch_id}] Hatch CNN mask: {int(hatch_mask.sum())}/{H*W} px "
                f"({hatch_mask.mean()*100:.1f}%) @ thr={threshold}"
            )
        except Exception as exc:
            if strict_models:
                raise RuntimeError("Deployment hatch CNN inference failed; fallback is disabled") from exc
            logger.warning(
                f"[{sketch_id}] Hatch CNN inference failed ({exc}); "
                "using geometric hachure removal."
            )
            hatch_mask = None
    cleanup_hatch_mask = hatch_mask
    if (
        hatch_stroke_mode == "post_topology_additive"
        and hatch_stroke_stats.get("applied")
        and hatch_stroke_mask is not None
    ):
        region_mask, hatch_stroke_additive_mask = (
            _separate_hatch_additive_masks(hatch_mask, hatch_stroke_mask)
        )
        # Preserve the production region pass exactly. Model-only pixels are
        # evaluated later against the remaining graph with independent
        # geometric guards; unioning both masks here made source attribution
        # impossible and let long structural chains bypass hatch limits.
        cleanup_hatch_mask = region_mask
        hatch_stroke_stats.update({
            "region_mask_pixels": int(np.count_nonzero(region_mask)),
            "additive_mask_pixels": int(
                np.count_nonzero(hatch_stroke_additive_mask)
            ),
            "combined_mask_pixels": int(
                np.count_nonzero(
                    np.maximum(region_mask, hatch_stroke_mask)
                )
            ),
        })
    if prepass_subtracted and hatch_mask is not None:
        cleanup_hatch_mask = np.zeros_like(hatch_mask)
    elif structural_topology:
        if hatch_mask is None:
            cleanup_hatch_mask = None
        elif hatch_stroke_mask is None:
            cleanup_hatch_mask = np.zeros_like(hatch_mask)
        else:
            cleanup_hatch_mask = _edge_support_mask(
                preliminary_hachures,
                (H, W),
                dilation=int(
                    cfg_kp.get("hachure_cnn_stroke_mask_dilation", 2)
                ),
            )
    if (
        cleanup_hatch_mask is not None
        and cfg_kp.get("hachure_hough_stroke_mask", False)
    ):
        if (
            cfg_kp.get("hachure_hough_bridge_keypoints", False)
            and hatch_stroke_mask is not None
        ):
            cleanup_hatch_mask = hatch_stroke_mask
            logger.info(
                f"[{sketch_id}] Reusing early hatch Hough mask for cleanup: "
                f"{len(hachure_hough_families)} periodic families, "
                f"{int(np.count_nonzero(cleanup_hatch_mask))} support pixels"
            )
        else:
            cleanup_hatch_mask, hachure_hough_families = (
                _hough_hachure_stroke_mask(
                    skeleton,
                    cleanup_hatch_mask,
                    cfg_kp,
                )
            )
            logger.info(
                f"[{sketch_id}] Hatch Hough mask: "
                f"{len(hachure_hough_families)} periodic families, "
                f"{int(np.count_nonzero(cleanup_hatch_mask))} support pixels"
            )
    use_cnn_hatch = (
        cleanup_hatch_mask is not None and np.any(cleanup_hatch_mask)
    )

    if (cfg_kp.get("remove_hachures", False) and cfg_kp.get("hachure_pre_pass", False)
            and not use_cnn_hatch):
        n0, e0 = len(nodes), len(edges)
        nodes, edges, removed = _remove_hachure_edges(
            nodes, edges, cfg_kp, pass_name="pre_simplify"
        )
        removed_hachures.extend(removed)
        if removed:
            logger.info(
                f"[{sketch_id}] Hachures pre-simplify: "
                f"{n0}→{len(nodes)} nodes, {e0}→{len(edges)} edges "
                f"({len(removed)} removed)"
            )

    # ── Layer 2b: Graph simplification (de-fragmentation) ─────────────────
    coverage_ledger.record("hatch_pre_simplify", edges, removed_hachures)
    # Prune skeleton spurs, dissolve phantom degree-2 junctions, and merge
    # collinear edges that pass straight through real junctions. Without this
    # a single logical stroke fragments into many primitives: dense patent
    # scans produce thousands of 2–3 px junction stubs, and clean CAD drawings
    # split a straight edge at every T-junction it crosses.
    if cfg_kp.get("simplify_graph", True):
        n0, e0 = len(nodes), len(edges)
        nodes, edges = _simplify_graph(
            nodes, edges,
            spur_min_len          = cfg_kp.get("spur_min_length", 6.0),
            collinear_max_angle   = cfg_kp.get("merge_collinear_max_angle", 28.0),
            # Junction-cluster merging welds parallel strokes that run close
            # together (concentric circles, thin-ring/washer outlines, double
            # walls), destroying them — it is OFF by default. Enable with a
            # small radius only on corpora known to be free of close parallels.
            junction_merge_radius = cfg_kp.get("junction_merge_radius", 0.0),
        )
        logger.info(f"[{sketch_id}] Simplified: {n0}→{len(nodes)} nodes, "
                    f"{e0}→{len(edges)} edges")

    # ── Hachure removal: learned mask (primary) or geometric (fallback) ────
    coverage_ledger.record("graph_simplification", edges, removed_hachures)
    # The CNN mask is the primary signal when the Phase 2 detector is loaded;
    # the geometric clustering heuristic runs only when no mask is available.
    if use_cnn_hatch:
        n0, e0 = len(nodes), len(edges)
        cleanup_pass_name = (
            "hatch_stroke_multilabel"
            if hatch_stroke_mode == "post_topology"
            and hatch_stroke_stats.get("applied")
            else "cnn"
        )
        if cfg_kp.get("hachure_cnn_intersect_geometric", False):
            nodes, edges, removed = _remove_hachures_cnn_geometric(
                nodes,
                edges,
                cleanup_hatch_mask,
                cfg_kp,
                pass_name=(
                    f"{cleanup_pass_name}_geometric"
                    if cleanup_pass_name != "cnn"
                    else "cnn_geometric"
                ),
            )
        else:
            nodes, edges, removed = _remove_hachures_cnn(
                nodes,
                edges,
                cleanup_hatch_mask,
                cfg_kp,
                pass_name=cleanup_pass_name,
            )
        if cleanup_pass_name.startswith("hatch_stroke_multilabel"):
            for edge in removed:
                edge.setdefault("hachure", {})["source"] = "hatch_stroke_cnn"
            hatch_stroke_stats.update({
                "classified_hachure_edges": len(removed),
                "side_layer_edges": len(removed),
                "side_layer_pixels": sum(
                    len(edge.get("pixels") or []) for edge in removed
                ),
            })
        removed_hachures.extend(removed)
        did_remove_hachures = bool(removed_hachures)
        if removed:
            logger.info(
                f"[{sketch_id}] Hachures (CNN): "
                f"{n0}→{len(nodes)} nodes, {e0}→{len(edges)} edges "
                f"({len(removed)} removed)"
            )
    else:
        did_remove_hachures = False
        run_hachure_post = (
            cfg_kp.get("remove_hachures", False)
            and cfg_kp.get("hachure_second_pass", True)
            and not prepass_subtracted
            and _should_run_hachure_cleanup(edges, cfg_kp)
        )
        if run_hachure_post:
            n0, e0 = len(nodes), len(edges)
            nodes, edges, removed = _remove_hachure_edges(
                nodes, edges, cfg_kp, pass_name="post_simplify"
            )
            removed_hachures.extend(removed)
            if removed:
                did_remove_hachures = True
                logger.info(
                    f"[{sketch_id}] Hachures post-simplify: "
                    f"{n0}→{len(nodes)} nodes, {e0}→{len(edges)} edges "
                    f"({len(removed)} removed)"
                )

    if (
        hatch_stroke_mode == "post_topology_additive"
        and hatch_stroke_stats.get("applied")
        and hatch_stroke_additive_mask is not None
        and np.any(hatch_stroke_additive_mask)
    ):
        n0, e0 = len(nodes), len(edges)
        additive_cfg = dict(cfg_kp)
        additive_cfg["hachure_cnn_geometric_max_length"] = float(
            cfg_kp.get(
                "hachure_stroke_additive_max_length",
                cfg_kp.get("hachure_max_length", 80.0),
            )
        )
        additive_cfg["hachure_cnn_geometric_max_candidate_ratio"] = float(
            cfg_kp.get("hachure_stroke_additive_max_candidate_ratio", 0.75)
        )
        if cfg_kp.get("hachure_stroke_additive_require_geometric", True):
            nodes, edges, additive_removed = _remove_hachures_cnn_geometric(
                nodes,
                edges,
                hatch_stroke_additive_mask,
                additive_cfg,
                pass_name="hatch_stroke_multilabel_additive",
            )
        else:
            nodes, edges, additive_removed = _remove_hachures_cnn(
                nodes,
                edges,
                hatch_stroke_additive_mask,
                additive_cfg,
                pass_name="hatch_stroke_multilabel_additive",
            )
        for edge in additive_removed:
            edge.setdefault("hachure", {})["source"] = "hatch_stroke_cnn"
        removed_hachures.extend(additive_removed)
        did_remove_hachures = bool(removed_hachures)
        hatch_stroke_stats.update({
            "additive_classified_hachure_edges": len(additive_removed),
            "additive_side_layer_pixels": sum(
                len(edge.get("pixels") or []) for edge in additive_removed
            ),
            "additive_require_geometric": bool(
                cfg_kp.get(
                    "hachure_stroke_additive_require_geometric", True
                )
            ),
            "additive_max_length": float(
                additive_cfg["hachure_cnn_geometric_max_length"]
            ),
        })
        if additive_removed:
            logger.info(
                f"[{sketch_id}] Hachures (stroke additive): "
                f"{n0}→{len(nodes)} nodes, {e0}→{len(edges)} edges "
                f"({len(additive_removed)} removed)"
            )

    # Shared post-removal cleanup (residual crumbs + re-simplify) — identical
    # for both the learned and geometric paths.
    coverage_ledger.record("hatch_separation", edges, removed_hachures)
    if did_remove_hachures:
        min_removed_for_prune = int(
            cfg_kp.get("hachure_residual_prune_min_removed", 1)
        )
        if len(removed_hachures) >= min_removed_for_prune:
            n2, e2 = len(nodes), len(edges)
            nodes, edges, residuals = _prune_hachure_residual_edges(
                nodes,
                edges,
                cfg_kp,
                pass_name="post_hachure_residual_prune",
            )
            removed_hachures.extend(residuals)
            if residuals:
                logger.info(
                    f"[{sketch_id}] Hachure residuals: "
                    f"{n2}→{len(nodes)} nodes, {e2}→{len(edges)} edges "
                    f"({len(residuals)} removed)"
                )
        coverage_ledger.record("hatch_residual_pruning", edges, removed_hachures)
        if cfg_kp.get("simplify_graph", True):
            n1, e1 = len(nodes), len(edges)
            nodes, edges = _simplify_graph(
                nodes, edges,
                spur_min_len          = cfg_kp.get("spur_min_length", 6.0),
                collinear_max_angle   = cfg_kp.get("merge_collinear_max_angle", 28.0),
                junction_merge_radius = cfg_kp.get("junction_merge_radius", 0.0),
            )
            logger.info(
                f"[{sketch_id}] Re-simplified after hachures: "
                f"{n1}→{len(nodes)} nodes, {e1}→{len(edges)} edges"
            )

    coverage_ledger.record("post_hatch_simplification", edges, removed_hachures)
    if removed_hachures:
        before_dedup = len(removed_hachures)
        removed_hachures = _deduplicate_hachure_edges(
            removed_hachures,
            overlap_threshold=float(
                cfg_kp.get("hachure_side_dedup_overlap", 0.75)
            ),
        )
        if len(removed_hachures) != before_dedup:
            logger.info(
                f"[{sketch_id}] Hachure side-layer dedup: "
                f"{before_dedup}→{len(removed_hachures)} edges"
            )

    coverage_ledger.record("hatch_deduplication", edges, removed_hachures)
    # Aggregate hatch lines before final metrics, then classify any matching
    # residue and re-simplify. The old late cleanup silently deleted residue
    # after metrics and left its structural neighbours fragmented.
    early_region_cleanup = bool(
        cfg_kp.get("hachure_region_cleanup_before_metrics", False)
    )
    hachure_regions: list[dict] = []
    if (
        early_region_cleanup
        and cfg_kp.get("hachure_mode", "region") == "region"
        and removed_hachures
    ):
        try:
            hachure_regions = _aggregate_hachure_regions(
                removed_hachures, (H, W), cfg_kp
            )
            if hachure_regions:
                n_before = len(edges)
                nodes, edges, region_residuals = _cleanup_hatch_residue(
                    nodes, edges, hachure_regions, (H, W), cfg_kp
                )
                removed_hachures.extend(region_residuals)
                if region_residuals and cfg_kp.get("simplify_graph", True):
                    nodes, edges = _simplify_graph(
                        nodes,
                        edges,
                        # Residue removal can expose genuine structural ends.
                        # Reconnect only; do not prune those newly exposed arms.
                        spur_min_len=0.0,
                        collinear_max_angle=cfg_kp.get(
                            "merge_collinear_max_angle", 28.0
                        ),
                        junction_merge_radius=cfg_kp.get(
                            "junction_merge_radius", 0.0
                        ),
                    )
                logger.info(
                    f"[{sketch_id}] Hachure regions: {len(hachure_regions)} "
                    f"from {len(removed_hachures) - len(region_residuals)} "
                    "lines "
                    f"({sum(r['n_lines'] for r in hachure_regions)} grouped); "
                    f"residue cleanup {n_before}→{len(edges)} edges "
                    f"({len(region_residuals)} preserved in side layer)"
                )
        except Exception as exc:
            logger.warning(f"[{sketch_id}] Hachure region aggregation failed: {exc}")
            hachure_regions = []

    # ── Layer 3: Curve smoothing ──────────────────────────────────────────
    coverage_ledger.record("early_region_cleanup", edges, removed_hachures)
    rdp_eps        = cfg_kp.get("rdp_epsilon",          1.5)
    spline_s       = cfg_kp.get("spline_smoothing",     2.0)
    overshoot_lim  = cfg_kp.get("spline_overshoot_limit", 5.0)
    edges = _smooth_edges(edges, rdp_eps, spline_s,
                          spline_overshoot_limit=overshoot_lim)

    # ── Filter noise closed loops ─────────────────────────────────────────
    # Size/non-circularity alone cannot establish noise. Retain these source
    # shapes; the count remains useful for auditing the former deletion rule.
    min_loop_px = cfg_kp.get("min_closed_loop_pixels", 80)
    noise_loops = [e for e in edges
                   if e.get("is_closed") and len(e["pixels"]) < min_loop_px
                   and not e.get("residual_parent_edge_ids")
                   and not _is_circular_loop(e["pixels"])]
    coverage_ledger.record("smoothing_and_small_loop_preservation", edges, removed_hachures)

    # ── Prune free-floating skeleton speckle (opt-in) ─────────────────────
    # Removes lone tiny disconnected fragments (scan noise) that dominate the
    # micro-edge count on real patent scans. Runs before dashed grouping and
    # the metrics so it also feeds cleaner input to the dash RANSAC.
    pruned_noise_pixels: set = set()
    if cfg_kp.get("prune_floating_noise", False):
        n_before = len(edges)
        nodes, edges, n_noise, pruned_noise_pixels = _prune_floating_noise(
            nodes, edges, cfg_kp)
        if n_noise:
            logger.info(
                f"[{sketch_id}] Floating-noise prune: removed {n_noise} speckle "
                f"fragment(s), {n_before}→{len(edges)} edges"
            )

    coverage_ledger.record("floating_noise_pruning", edges, removed_hachures)
    # ── Group dashed centre-lines / bolt-circles (opt-in) ─────────────────
    # Runs BEFORE the fragmentation metrics so grouped dashes stop inflating
    # micro_edge_ratio / isolation on valid dashed drawings.
    if cfg_kp.get("dashed_grouping", False):
        n_before = len(edges)
        nodes, edges, n_dash_groups = _group_dashed_edges(nodes, edges, cfg_kp)
        if n_dash_groups:
            logger.info(
                f"[{sketch_id}] Dashed grouping: {n_dash_groups} group(s), "
                f"{n_before}→{len(edges)} edges"
            )

    coverage_ledger.record("dashed_grouping", edges, removed_hachures)

    # Finish legacy ordering before final recovery and metrics. Preserve the
    # residue side channel here too; no cleanup may delete ink after accounting.
    if (not early_region_cleanup and cfg_kp.get("hachure_mode", "region") == "region"
            and removed_hachures):
        try:
            hachure_regions = _aggregate_hachure_regions(removed_hachures, (H, W), cfg_kp)
            if hachure_regions:
                nodes, edges, late_residuals = _cleanup_hatch_residue(
                    nodes, edges, hachure_regions, (H, W), cfg_kp)
                removed_hachures.extend(late_residuals)
        except Exception as exc:
            logger.warning(f"[{sketch_id}] Hachure region aggregation failed: {exc}")
            hachure_regions = []
    coverage_ledger.record("late_region_cleanup", edges, removed_hachures)
    nodes, edges, coverage_recovery = recover_source_coverage(
        coverage_ledger.source, nodes, edges, removed_hachures)
    coverage_ledger.record("final_source_recovery", edges, removed_hachures)
    nodes, edges, integration = integrate_recovered_connections(
        coverage_ledger.source, nodes, edges, removed_hachures)
    coverage_recovery["integration"] = integration
    coverage_ledger.record("recovered_connection_integration", edges, removed_hachures)
    coverage_report = coverage_ledger.report(
        coverage_recovery, scale=stage2_scale, original_shape=(orig_H, orig_W))
    coverage_report["small_closed_shapes_preserved"] = len(noise_loops)
    logger.info(f"[{sketch_id}] Source coverage: {coverage_recovery['recovered_pixels']} pixels "
                f"recovered as {coverage_recovery['new_edges']} traces; "
                f"{coverage_recovery['unresolved_pixels']} pixels remain explicit review items")

    # ── Confidence signal ─────────────────────────────────────────────────
    ignored_hachure_pixels = {
        (int(px[0]), int(px[1]))
        for edge in removed_hachures
        for px in edge.get("pixels", [])
    }
    # Only actual preserved hatch geometry leaves the structural denominator.
    # Unresolved source marks are not silently declared noise.
    ignored_pixels = ignored_hachure_pixels
    iso_ratio = _compute_isolation_ratio(
        coverage_ledger.source,
        edges,
        ignored_pixels=ignored_pixels,
    )
    threshold = cfg_kp.get("isolation_threshold",
                            config.get("stage2", {}).get("isolation_threshold", 0.05))
    open_lengths, median_edge_length, micro_edge_ratio, short_edge_ratio = (
        _open_edge_length_stats(edges)
    )
    n_open = len(open_lengths)
    n_closed = sum(1 for e in edges if e.get("is_closed"))

    frag_cfg = cfg_kp.get("fragmentation", {})
    max_micro_ratio = gate_bound(frag_cfg, "max_micro_edge_ratio", 0.50, maximum=1.0)
    max_short_ratio = gate_bound(frag_cfg, "max_short_edge_ratio", 0.80, maximum=1.0)
    max_edges = gate_bound(frag_cfg, "max_edges", 5000)
    noncycle_unclaimed_sizes = [
        len(edge.get("pixels") or [])
        for edge in edges
        if edge.get("topology_origin") == "unclaimed_component"
        and not edge.get("is_simple_cycle", False)
    ]
    max_noncycle_unclaimed_pixels = max(
        noncycle_unclaimed_sizes, default=0
    )
    max_allowed_noncycle_unclaimed_pixels = gate_bound(
        frag_cfg, "max_unclaimed_noncycle_pixels", 100000
    )
    flagged = (
        iso_ratio > threshold
        or any(c["reason"] in {"component_budget", "edge_budget"}
               for c in coverage_recovery["components"])
        or (max_edges is not None and len(edges) > max_edges)
        or (
            max_allowed_noncycle_unclaimed_pixels is not None
            and max_noncycle_unclaimed_pixels
            > max_allowed_noncycle_unclaimed_pixels
        )
        or (max_micro_ratio is not None
            and n_open >= 50 and micro_edge_ratio > max_micro_ratio)
        or (max_short_ratio is not None
            and n_open >= 50 and short_edge_ratio > max_short_ratio)
    )

    # ── Serialise graph ───────────────────────────────────────────────────
    def _to_python(obj):
        """Recursively convert numpy scalars to native Python types."""
        if isinstance(obj, dict):
            return {k: _to_python(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_python(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    graph_doc = _to_python({
        "sketch_id":   sketch_id,
        "image_shape": [H, W],
        "original_image_shape": [orig_H, orig_W],
        "stage2_scale": stage2_scale,
        "keypoint_source": kp_source,
        "coverage": coverage_report,
        "hachure_source": ("cnn" if hatch_mask is not None else
                           "geometric" if cfg_kp.get("remove_hachures", False) else "disabled"),
        "metrics": {
            "n_closed_edges": n_closed,
            "n_hachure_edges_removed": len(removed_hachures),
            "n_hachure_pixels_ignored": len(ignored_hachure_pixels),
            "n_hachure_regions": len(hachure_regions),
            "n_hachure_hough_families": len(hachure_hough_families),
            "n_hachure_routing_candidate_junctions": (
                hachure_routing_candidate_junctions
            ),
            "n_hachure_routing_junctions": hachure_routing_junctions,
            "n_hachure_routing_endpoints": hachure_routing_endpoints,
            "n_hachure_gaps_repaired": len(hachure_gap_repairs),
            "n_hachure_gap_pixels_restored": sum(
                int(repair["restored_pixels"])
                for repair in hachure_gap_repairs
            ),
            "n_hachure_stroke_pixels_removed": int(
                hatch_stroke_stats.get("safe_removed_pixels", 0)
            ),
            "n_unclaimed_noncycle_components": len(
                noncycle_unclaimed_sizes
            ),
            "max_unclaimed_noncycle_pixels": (
                max_noncycle_unclaimed_pixels
            ),
            "median_edge_length": median_edge_length,
            "micro_edge_ratio": micro_edge_ratio,
            "short_edge_ratio": short_edge_ratio,
        },
        "nodes": nodes,
        "edges": edges,
        "removed_hachures": removed_hachures,
        "hachure_regions": hachure_regions,
        "hachure_hough_families": hachure_hough_families,
        "hachure_gap_repairs": hachure_gap_repairs,
        "hachure_stroke": hatch_stroke_stats,
    })
    with open(graph_path, "w") as f:
        json.dump(graph_doc, f, indent=2)

    elapsed = time.perf_counter() - t_start

    if flagged:
        logger.warning(
            f"[{sketch_id}] FLAGGED — isolation ratio {iso_ratio:.3f} "
            f"> threshold {threshold:.2f}; largest non-cycle residual "
            f"component={max_noncycle_unclaimed_pixels} px"
        )
    else:
        logger.info(
            f"[{sketch_id}] Stage 2 done in {elapsed:.2f}s — "
            f"isolation={iso_ratio:.3f} kp_source={kp_source}"
        )

    return Stage2Result(
        sketch_id         = sketch_id,
        graph_path        = graph_path,
        isolation_ratio   = iso_ratio,
        flagged           = flagged,
        processing_time_s = elapsed,
        keypoint_source   = kp_source,
        n_nodes           = len(nodes),
        n_edges           = len(edges),
        n_closed_edges    = n_closed,
        n_hachure_edges_removed = len(removed_hachures),
        median_edge_length = median_edge_length,
        micro_edge_ratio  = micro_edge_ratio,
        short_edge_ratio  = short_edge_ratio,
        n_unclaimed_noncycle_components = len(
            noncycle_unclaimed_sizes
        ),
        max_unclaimed_noncycle_pixels = max_noncycle_unclaimed_pixels,
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI FOR STANDALONE TESTING
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import yaml

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Stage 2 — Stroke Extraction: build stroke graph from skeleton."
    )
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    parser.add_argument("input",    type=Path, help="Input 1px binary skeleton PNG")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output",
                        help="Output root directory (default: <project>/output)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml",
                        help="Pipeline config file (default: <project>/config.yaml)")
    parser.add_argument("--id",     type=str,  default=None,
                        help="Sketch ID (default: input filename stem)")
    args = parser.parse_args()

    cfg = {}
    if args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}

    sketch_id = args.id or args.input.stem
    mdl = load_model(cfg)

    result = run(
        skeleton_path = args.input,
        output_dir    = args.output,
        sketch_id     = sketch_id,
        config        = cfg,
        model         = mdl,
    )

    print(f"\n{'─'*56}")
    print(f"  Sketch ID        : {result.sketch_id}")
    print(f"  Keypoint source  : {result.keypoint_source}")
    print(f"  Nodes            : {result.n_nodes}")
    print(f"  Edges            : {result.n_edges}")
    print(f"  Isolation ratio  : {result.isolation_ratio:.3f}")
    print(f"  Flagged          : {'YES ⚠' if result.flagged else 'no'}")
    print(f"  Graph JSON       : {result.graph_path}")
    print(f"  Processing time  : {result.processing_time_s:.2f}s")
    print(f"{'─'*56}")
