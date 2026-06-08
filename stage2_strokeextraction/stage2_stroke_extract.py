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
import logging
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


def _extract_topology(
    skeleton: np.ndarray,
    keypoints: list[dict] = (),   # unused — topology is built from CN internally
    max_search_radius: int = 60,  # unused — walk terminates at extended kp regions
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
      5. Unclaimed CCs >= min_loop_pixels -> closed loops.

    Returns (nodes, edges) as plain dicts for JSON serialisation.
    The `keypoints` parameter is accepted for API compatibility but is not
    used — keypoints are derived from the CN map to keep layers independent.
    """
    binary = (skeleton > 0).astype(np.uint8)
    H, W   = binary.shape

    # ── Step 1: vectorised CN map ────────────────────────────────────────
    cn_map = _cn_map_vectorized(binary)

    # ── Step 2: cluster keypoints ────────────────────────────────────────
    kp_info   = []   # list of {id, x, y, type, confidence}
    kp_map    = {}   # pixel (x,y) -> kp_id  (core + extended 8-neighbourhood)
    kp_pixels = {}   # kp_id -> set of pixels in extended region (O(1) reverse lookup)

    _k3 = np.ones((3, 3), np.uint8)   # 3×3 dilation kernel (shared)

    def _add_cluster(mask, kp_type):
        """
        Register core pixels of each CN cluster; extension happens in bulk below.

        Critical: never call np.where(labels == c) inside a loop — that scans
        the full image once per cluster (O(n_clusters × H × W)).  Instead, read
        all labeled pixels once, sort by label, and slice into per-cluster groups.
        """
        n, labels = cv2.connectedComponents(mask)
        if n <= 1:
            return
        # Single pass: all foreground pixels and their cluster labels
        ys_all, xs_all = np.where(labels > 0)
        if len(ys_all) == 0:
            return
        lv = labels[ys_all, xs_all]
        # Sort so pixels of the same cluster are contiguous
        order  = np.argsort(lv, kind="stable")
        xs_s   = xs_all[order]; ys_s = ys_all[order]; lv_s = lv[order]
        splits = np.where(np.diff(lv_s))[0] + 1
        starts = np.concatenate([[0], splits])
        ends   = np.concatenate([splits, [len(lv_s)]])
        for s, e in zip(starts.tolist(), ends.tolist()):
            xs_g = xs_s[s:e]; ys_g = ys_s[s:e]
            kid  = len(kp_info)
            kp_info.append({
                "id": kid,
                "x": int(np.mean(xs_g)), "y": int(np.mean(ys_g)),
                "type": kp_type, "confidence": 1.0,
            })
            kp_pixels[kid] = set()
            for x, y in zip(xs_g.tolist(), ys_g.tolist()):
                kp_map[(x, y)] = kid
                kp_pixels[kid].add((x, y))

    _add_cluster((cn_map == 1).astype(np.uint8), KP_ENDPOINT)
    _add_cluster((cn_map >= 3).astype(np.uint8), KP_JUNCTION)

    # Extend each cluster by 1px using a single bulk dilation of a float label image.
    # This avoids calling cv2.dilate once per cluster (O(n_clusters × H × W)) and
    # replaces it with a single O(H × W) operation.
    # Label encoding: label_img[y,x] = kid+1 so that 0 = background.
    # cv2.dilate with float uses max-pool: each extended pixel gets the highest
    # adjacent kid+1.  Ties are broken deterministically (higher ID wins); this is
    # fine because any adjacent cluster will stop the walk correctly.
    if kp_info:
        label_img = np.zeros((H, W), dtype=np.float32)
        for (x, y), kid in kp_map.items():
            label_img[y, x] = float(kid + 1)
        dilated = cv2.dilate(label_img, _k3)
        ext_ys, ext_xs = np.where((dilated > 0) & (binary > 0) & (label_img == 0))
        for y, x in zip(ext_ys.tolist(), ext_xs.tolist()):
            kid = int(dilated[y, x]) - 1
            if (x, y) not in kp_map:
                kp_map[(x, y)] = kid
                kp_pixels[kid].add((x, y))

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
                        nxt = nbs[0]
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
    # Keep this low (8) so tiny genuine circles (≥4 px radius) are captured
    # as closed-loop edges.  The noise filter in run() then discards non-circular
    # small loops using _is_circular_loop, so patent-TIF ink blobs are still
    # suppressed.
    min_loop_pixels = 8

    # Build adjacency among unclaimed non-kp pixels
    remaining = set()
    ys, xs = np.where(binary)
    for y, x in zip(ys, xs):
        px = (x, y)
        if px not in kp_map and px not in all_edge_pix:
            remaining.add(px)

    if remaining:
        import networkx as nx
        G_rem = nx.Graph()
        for px in remaining:
            for nb in _neighbours8(*px):
                if nb in remaining:
                    G_rem.add_edge(px, nb)
        for cc in nx.connected_components(G_rem):
            if len(cc) < min_loop_pixels:
                continue
            xs2 = [p[0] for p in cc]
            ys2 = [p[1] for p in cc]
            cx, cy = int(np.mean(xs2)), int(np.mean(ys2))
            loop_id = len(nodes)
            nodes.append({
                "id": loop_id, "x": cx, "y": cy,
                "type": KP_LOOP_ANCHOR, "confidence": 1.0,
            })
            pixels = [[int(p[0]), int(p[1])] for p in sorted(cc)]
            edges.append({
                "id": edge_id,
                "source": loop_id,
                "target": loop_id,
                "pixels": pixels,
                "smooth_pts": [],
                "is_closed": True,
            })
            edge_id += 1

    return nodes, edges


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
        if a == b and _chain_length(e["pix"]) < max(radius * 2.0, 6.0):
            E[i] = None
            continue
        e["a"], e["b"] = a, b
    return True


def _simplify_graph(
    nodes: list[dict],
    edges: list[dict],
    spur_min_len: float = 6.0,
    collinear_max_angle: float = 28.0,
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

    node_by_id = {n["id"]: dict(n) for n in nodes}

    closed_edges = [e for e in edges if e.get("is_closed")]
    # Working representation for open edges: {a, b, pix}
    E: list[dict | None] = []
    for e in edges:
        if e.get("is_closed"):
            continue
        pix = [(int(p[0]), int(p[1])) for p in e["pixels"]]
        E.append({"a": e["source"], "b": e["target"], "pix": pix})

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
        E[i] = {"a": a_node, "b": b_node, "pix": pi + pj}
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
        adj = build_adj()
        for nid, idxs in adj.items():
            alive = [k for k in idxs if E[k] is not None]
            if len(alive) != 2:
                continue
            i, j = alive
            if i == j:
                continue   # self-loop edge through this node — leave it
            if other(E[i], nid) == nid or other(E[j], nid) == nid:
                continue
            merge(i, j, nid)
            node_by_id.pop(nid, None)
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
                d = _leave_direction(e["pix"], at_start=(e["a"] == nid))
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

    out_edges = []
    eid = 0
    for e in alive_edges:
        out_edges.append({
            "id": eid, "source": e["a"], "target": e["b"],
            "pixels": [[int(p[0]), int(p[1])] for p in e["pix"]],
            "smooth_pts": [], "is_closed": False,
        })
        eid += 1
    for e in closed_edges:
        ce = dict(e)
        ce["id"] = eid
        out_edges.append(ce)
        eid += 1

    return out_nodes, out_edges


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
    TIFs.  A real skeleton circle has pixels at a nearly uniform radius from
    the centroid; a noise blob is irregular.

    Criterion: RMS of radial deviations < threshold × mean radius, where the
    threshold is relaxed for small circles (r_mean < 12 px) because Zhang-Suen
    skeletonization produces staircase artefacts that inflate the RMS on tiny
    rings, even when they are geometrically perfect circles.
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
    return rms < threshold * r_mean


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC STAGE FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def run(
    skeleton_path: Path,
    output_dir: Path,
    sketch_id: str,
    config: dict,
    model: Optional[PuhachovKeypointDetector] = None,
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

    Returns
    -------
    Stage2Result
    """
    t_start = time.perf_counter()

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
    max_res = cfg_kp.get("max_input_resolution", 0)
    if max_res and max(H, W) > max_res:
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

    # ── Layer 1: Keypoint detection ───────────────────────────────────────
    conf_thresh = cfg_kp.get("keypoint_threshold", 0.50)
    nms_radius  = cfg_kp.get("nms_radius",         5)

    # Scale NMS radius up for images larger than the model's training resolution.
    ref_res = cfg_kp.get("nms_reference_resolution", 512)
    if ref_res and max(H, W) > ref_res:
        nms_radius = max(nms_radius, int(round(nms_radius * max(H, W) / ref_res)))
        logger.debug(f"[{sketch_id}] Adaptive NMS radius: {nms_radius} "
                     f"(image {max(H,W)}px vs ref {ref_res}px)")

    if model is not None:
        try:
            keypoints = model.detect(skeleton, conf_thresh, nms_radius)
            kp_source = "cnn"
            logger.info(f"[{sketch_id}] CNN detected {len(keypoints)} keypoints")
        except Exception as exc:
            logger.warning(f"[{sketch_id}] CNN keypoint detection failed "
                           f"({exc}), using classical fallback")
            keypoints = _classical_keypoints(skeleton)
            kp_source = "classical_fallback"
    else:
        keypoints = _classical_keypoints(skeleton)
        kp_source = "classical"
        logger.info(f"[{sketch_id}] Classical CN: {len(keypoints)} keypoints")

    # ── Layer 2: Topology extraction ──────────────────────────────────────
    max_radius = cfg_kp.get("max_search_radius", 60)
    nodes, edges = _extract_topology(skeleton, keypoints, max_radius)
    logger.info(f"[{sketch_id}] Graph: {len(nodes)} nodes, {len(edges)} edges")

    removed_hachures: list[dict] = []
    if cfg_kp.get("remove_hachures", False) and cfg_kp.get("hachure_pre_pass", False):
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

    run_hachure_post = (
        cfg_kp.get("remove_hachures", False)
        and cfg_kp.get("hachure_second_pass", True)
        and _should_run_hachure_cleanup(edges, cfg_kp)
    )
    if run_hachure_post:
        n0, e0 = len(nodes), len(edges)
        nodes, edges, removed = _remove_hachure_edges(
            nodes, edges, cfg_kp, pass_name="post_simplify"
        )
        removed_hachures.extend(removed)
        if removed:
            logger.info(
                f"[{sketch_id}] Hachures post-simplify: "
                f"{n0}→{len(nodes)} nodes, {e0}→{len(edges)} edges "
                f"({len(removed)} removed)"
            )
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

    # ── Layer 3: Curve smoothing ──────────────────────────────────────────
    rdp_eps        = cfg_kp.get("rdp_epsilon",          1.5)
    spline_s       = cfg_kp.get("spline_smoothing",     2.0)
    overshoot_lim  = cfg_kp.get("spline_overshoot_limit", 5.0)
    edges = _smooth_edges(edges, rdp_eps, spline_s,
                          spline_overshoot_limit=overshoot_lim)

    # ── Filter noise closed loops ─────────────────────────────────────────
    # Tiny closed loops that are NOT geometrically circular are skeleton noise
    # (ink blobs, dust from scanned patent TIFs).  Genuine small circles
    # (e.g. construction points in clean CAD rasterizations) have uniform
    # radial distance from their centroid and are preserved via the
    # _is_circular_loop circularity guard.
    min_loop_px = cfg_kp.get("min_closed_loop_pixels", 80)
    noise_loops = [e for e in edges
                   if e.get("is_closed") and len(e["pixels"]) < min_loop_px
                   and not _is_circular_loop(e["pixels"])]
    if noise_loops:
        noise_ids = {e["id"] for e in noise_loops}
        edges = [e for e in edges if e["id"] not in noise_ids]
        logger.debug(f"[{sketch_id}] Removed {len(noise_loops)} noise closed loop(s) "
                     f"(< {min_loop_px} px, non-circular): edge ids {sorted(noise_ids)}")

    # ── Confidence signal ─────────────────────────────────────────────────
    ignored_hachure_pixels = {
        (int(px[0]), int(px[1]))
        for edge in removed_hachures
        for px in edge.get("pixels", [])
    }
    iso_ratio = _compute_isolation_ratio(
        skeleton,
        edges,
        ignored_pixels=ignored_hachure_pixels,
    )
    threshold = cfg_kp.get("isolation_threshold",
                            config.get("stage2", {}).get("isolation_threshold", 0.05))
    open_lengths, median_edge_length, micro_edge_ratio, short_edge_ratio = (
        _open_edge_length_stats(edges)
    )
    n_open = len(open_lengths)
    n_closed = sum(1 for e in edges if e.get("is_closed"))

    frag_cfg = cfg_kp.get("fragmentation", {})
    max_micro_ratio = frag_cfg.get("max_micro_edge_ratio", 0.50)
    max_short_ratio = frag_cfg.get("max_short_edge_ratio", 0.80)
    max_edges = frag_cfg.get("max_edges", 5000)
    flagged = (
        iso_ratio > threshold
        or len(edges) > max_edges
        or (n_open >= 50 and micro_edge_ratio > max_micro_ratio)
        or (n_open >= 50 and short_edge_ratio > max_short_ratio)
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
        "metrics": {
            "n_closed_edges": n_closed,
            "n_hachure_edges_removed": len(removed_hachures),
            "n_hachure_pixels_ignored": len(ignored_hachure_pixels),
            "median_edge_length": median_edge_length,
            "micro_edge_ratio": micro_edge_ratio,
            "short_edge_ratio": short_edge_ratio,
        },
        "nodes": nodes,
        "edges": edges,
        "removed_hachures": removed_hachures,
    })
    with open(graph_path, "w") as f:
        json.dump(graph_doc, f, indent=2)

    elapsed = time.perf_counter() - t_start

    if flagged:
        logger.warning(
            f"[{sketch_id}] FLAGGED — isolation ratio {iso_ratio:.3f} "
            f"> threshold {threshold:.2f}"
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
