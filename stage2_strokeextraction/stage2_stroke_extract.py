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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
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
            model.load_state_dict(state_dict, strict=False)
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

def _classical_keypoints(skeleton: np.ndarray) -> list[dict]:
    """
    Classify foreground pixels by crossing number (CN):
      CN = 1  → endpoint
      CN >= 3 → junction
    Returns list of {x, y, type, confidence}.

    Also performs junction cluster merging: connected components of
    junction pixels are collapsed to a single centroid node.
    """
    binary = (skeleton > 0).astype(np.uint8)
    H, W   = binary.shape

    endpoint_mask  = np.zeros((H, W), dtype=np.uint8)
    junction_mask  = np.zeros((H, W), dtype=np.uint8)

    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if not binary[y, x]:
                continue
            neighbourhood = [
                binary[y-1, x],   binary[y-1, x+1],
                binary[y,   x+1], binary[y+1, x+1],
                binary[y+1, x],   binary[y+1, x-1],
                binary[y,   x-1], binary[y-1, x-1],
            ]
            # Crossing number: count 0→1 transitions (circular)
            cn = sum(
                abs(int(neighbourhood[i]) - int(neighbourhood[(i+1) % 8]))
                for i in range(8)
            ) // 2
            if cn == 1:
                endpoint_mask[y, x] = 255
            elif cn >= 3:
                junction_mask[y, x] = 255

    keypoints = []

    # Endpoints: each connected component → one keypoint at centroid
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(endpoint_mask)
    for i in range(1, n):
        cx, cy = centroids[i]
        keypoints.append({
            "x": int(round(cx)), "y": int(round(cy)),
            "type": KP_ENDPOINT, "confidence": 1.0,
        })

    # Junctions: merge cluster → centroid
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
    keypoints: list[dict],
    max_search_radius: int = 60,
) -> tuple[list[dict], list[dict]]:
    """
    Connect keypoints into a stroke graph using CN-cluster skeleton tracing.

    Algorithm (no Dijkstra, no Gurobi):
      1. Compute crossing number (CN) for every foreground pixel.
      2. Cluster CN=1 pixels -> endpoints; cluster CN>=3 pixels -> junctions.
      3. Build an extended kp_map: every keypoint pixel PLUS its 8-connected
         foreground neighbours map to that keypoint's ID. This 1px extension
         absorbs the staircase artefacts of Zhang-Suen thinning so walks
         stop correctly even when entering a junction from a diagonal pixel.
      4. From each keypoint cluster, walk outward along the skeleton. Stop
         when the walk enters another keypoint's extended region. One walk
         per unique direction -> one edge per unique (src, dst) pair.
      5. CCs with no keypoints and >= min_loop_pixels -> closed loops.

    Returns (nodes, edges) as plain dicts for JSON serialisation.
    """
    binary = (skeleton > 0).astype(np.uint8)
    H, W   = binary.shape

    # ── Step 1: compute crossing number map ─────────────────────────────
    cn_map = np.zeros((H, W), dtype=np.int32)
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            if not binary[y, x]:
                continue
            ring = [
                binary[y-1, x],   binary[y-1, x+1], binary[y,   x+1],
                binary[y+1, x+1], binary[y+1, x],   binary[y+1, x-1],
                binary[y,   x-1], binary[y-1, x-1],
            ]
            cn_map[y, x] = sum(
                abs(int(ring[i]) - int(ring[(i+1) % 8])) for i in range(8)
            ) // 2

    # ── Step 2: cluster keypoints ────────────────────────────────────────
    kp_info = []   # list of {id, x, y, type, confidence}
    kp_map  = {}   # pixel (x,y) -> kp_id  (extended: includes 8-neighbours)

    def _add_cluster(mask, kp_type):
        n, labels = cv2.connectedComponents(mask.astype(np.uint8))
        for c in range(1, n):
            ys, xs = np.where(labels == c)
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            kid = len(kp_info)
            kp_info.append({
                "id": kid, "x": cx, "y": cy,
                "type": kp_type, "confidence": 1.0,
            })
            # Core pixels
            for y, x in zip(ys, xs):
                kp_map[(x, y)] = kid
            # Extended: 8-connected foreground neighbours
            for y, x in zip(ys, xs):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx_, ny_ = x + dx, y + dy
                        if (0 <= ny_ < H and 0 <= nx_ < W
                                and binary[ny_, nx_]
                                and (nx_, ny_) not in kp_map):
                            kp_map[(nx_, ny_)] = kid

    _add_cluster((cn_map == 1), KP_ENDPOINT)
    _add_cluster((cn_map >= 3), KP_JUNCTION)

    # ── Step 3: walk from each keypoint cluster outward ─────────────────
    def _neighbours8(x, y):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= ny_ < H and 0 <= nx_ < W and binary[ny_, nx_]:
                    yield (nx_, ny_)

    edges_raw  = {}   # frozenset({sid, did}) -> pixel chain
    all_edge_pix = set()

    for src_kp in kp_info:
        sid     = src_kp["id"]
        src_pxs = {p for p, k in kp_map.items() if k == sid}

        for sp in list(src_pxs):
            for entry in _neighbours8(*sp):
                if kp_map.get(entry) == sid:
                    continue   # same cluster, skip
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
                    key = frozenset([sid, found])
                    if key not in edges_raw:
                        edges_raw[key] = [
                            [int(p[0]), int(p[1])] for p in chain
                        ]
                        for p in chain:
                            all_edge_pix.add((p[0], p[1]))

    # ── Step 4: build final node/edge lists ──────────────────────────────
    nodes   = list(kp_info)
    edges   = []
    edge_id = 0

    for key, pixels in edges_raw.items():
        a, b = tuple(key)
        edges.append({
            "id": edge_id,
            "source": a,
            "target": b,
            "pixels": pixels,
            "smooth_pts": [],
            "is_closed": False,
        })
        edge_id += 1

    # ── Step 5: closed loops ─────────────────────────────────────────────
    min_loop_pixels = 40

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
) -> float:
    """
    Fraction of foreground pixels not captured by any edge.
    isolation_ratio → 0 : perfect coverage
    isolation_ratio → 1 : almost nothing was captured
    """
    binary = (skeleton > 0)
    total  = int(binary.sum())
    if total == 0:
        return 0.0

    covered = set()
    for edge in edges:
        for px in edge["pixels"]:
            covered.add((px[0], px[1]))

    uncovered = 0
    ys, xs = np.where(binary)
    for y, x in zip(ys, xs):
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

    Criterion: RMS of radial deviations < 30 % of the mean radius.
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
    return rms < 0.30 * r_mean


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
    logger.info(f"[{sketch_id}] Stage 2 — skeleton {W}×{H}px, "
                f"{int((skeleton > 0).sum())} foreground px")

    # ── Layer 1: Keypoint detection ───────────────────────────────────────
    cfg_kp      = config.get("stage2", {})
    conf_thresh = cfg_kp.get("keypoint_threshold", 0.50)
    nms_radius  = cfg_kp.get("nms_radius",         5)

    # Scale NMS radius up for images larger than the model's training resolution
    # so that spurious duplicate junctions are suppressed on high-res patent TIFs
    # without any loss of geometric accuracy (full-resolution CNN still runs).
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
    iso_ratio = _compute_isolation_ratio(skeleton, edges)
    threshold = cfg_kp.get("isolation_threshold",
                            config.get("stage2", {}).get("isolation_threshold", 0.05))
    flagged = iso_ratio > threshold

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
        "nodes": nodes,
        "edges": edges,
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
