# Roadmap revision 3 — the defects the first release left in

Status: proposal, 2026-09-29. Follows `DEPLOYABILITY_ROADMAP_2026-09-21.md`,
whose phases closed with `docs/RELEASE_2026-09-28.md` (1,269 drawings). This
revision covers the three remarks the owner made on auditing that release.

Every number here was measured on the release or its audit; the file is named
beside it. Anything not measured is marked **hypothesis**.

---

## 0. Decisions taken on 2026-09-29

| # | decision | what it settles |
| --- | --- | --- |
| 1 | **A solid filled region becomes a filled polygon** — SVG path with fill, DXF SOLID hatch. | A new primitive type. The export schema, the geometry validator and the acceptance record all learn it. |
| 2 | **Fine shading strokes are removed like hatching.** | The detector is retrained on the style; the strokes leave the geometry and the region is recorded. |
| 3 | **A curved leader is reinjected as its traced path**, a polyline. | The removal mask and the DXF leader follow the same curve, which is what removes the ghost. |
| 4 | **The owner will label ~40 figures by hatch style** with `tools/hatch_mask_label.py`. | The detector retrain has data, and per-style recall is measured for the first time. |

---

## 1. What the audit remarks turned out to be

The owner flagged three things: thick strokes (including black areas and solid
hatching), out-of-distribution hatching, and ghost leader lines. Read against
the figures, they are **two mechanisms and one retrain**.

### "Thick strokes" is three defects

The 12 flagged figures, source beside reconstruction
(`docs/audits/2026-09-29/remaining_defects_evidence.json`):

| what it actually is | figures | share of release (proxy) | what the skeleton does to it |
| --- | --- | --- | --- |
| **solid filled region** — section fill, filled part | 4 | ~2.0% (26 figures with a blob ≥ 5,000 px) | a branching medial axis: a scribble |
| **fine shading strokes** on a perspective render | 3 | ~2.9% (37 figures with ≥ 40 parallel thin strokes) | vectorized as a dense mass |
| **thick outline proper** — bold frame, cable | 3 | — | a thin line or a double edge; tolerable |
| mixed / mainly ghost leaders | 2 | — | — |

The two "OOD hatching" figures are **wide diagonal bands** — bold hatching. A
solid section fill is a band at 100% duty. So remark 1 and remark 2 are one
family: **ink that means an area, reduced to a centreline that means a line.**

### The hatch machinery fails on that family twice

| population | HatchUNet predicted-hatch share of ink |
| --- | --- |
| ordinary figures with thin hatching (control) | median **28.5%** |
| bold-band figures | median **5.4%** |
| shading figures | median **9.3%** |
| the two audit hatch figures | 34.4% |

On most bold and shading figures the detector is **blind**. On the two audit
figures it **fired** — Stage 2 removed 27,708 hachure edges on EP2889159B1 —
and the bands *still* exported as outlines, because the remover is built for
thin line strokes and a wide band's boundary is not one. Detection and
representation both need work, and the representation gap is the deeper one:
**there is no filled-region primitive anywhere in Stage 3 or Stage 4.** The
only `fill` in the SVG writer is text.

### Ghost leaders are a discarded path

Measured over 2,611 removed leaders in the release:

| leaders | share | ink left behind after removal |
| --- | --- | --- |
| straight (chord lies ≥ 90% on ink) | 53.8% | **0%** |
| curved (chord lies < 60% on ink) | 34.9% | **75%** |

`_trace_ocr_leaders` walks the skeleton along the curve and then **keeps only
the two endpoints**. `_build_removal_mask` draws a straight `cv2.line` between
them; Stage 4 reinjects `add_line(p1, p2)`. The curve escapes the chord, its
ink survives into Stage 2 and is vectorized, and a straight leader is drawn
beside it. Leaders curved beyond `ocr_leader_straightness: 0.80` are not
traced at all — no ghost, but the leader ink stays as geometry. **34.8% of
identified numerals (5,268 of 15,144) have no traced leader**, a mix of
numerals that genuinely have none and leaders too curved to pass the cut.

---

## 2. Exit criteria

| # | criterion | now | target |
| --- | --- | --- | --- |
| R1 | ink left behind on curved leaders | 75% | **≤ 10%** |
| R2 | ghost strokes beside a leader, on the 100 audit figures | present on ≥ 3 figures | **0** |
| R3 | figures with a filled blob ≥ 5,000 px exported as a filled polygon | 0 of 26 | **≥ 22 of 26**, none as a scribble |
| R4 | hatch recall **by style**, real figures | never measured | measured; bold-band and shading recall ≥ 0.70 (**hypothesis** until R4a) |
| R5 | original-style hatch test IoU | 0.816 | **≥ 0.816** after retrain, no regression |
| R6 | audit "no clean render" | 12 of 100 | **≤ 3 of 100** on the second release's audit |
| R7 | second release accepted count | 1,269 | **≥ 1,269**, with every release-1 usable figure still usable |

R6 and R7 are release gates; the others are phase gates.

---

## 3. Phases

Re-lettered L–O to follow revision 2's G–K.

### Phase L — Path-faithful leaders

Goal: R1, R2. Independent of everything else; the cheapest and the most
visible, since a third of all leaders are curved.

#### L1. Keep the path

`_trace_ocr_leaders` already has the skeleton path in `best[0]`. Store it —
`"path": [[x, y], …]` in figure coordinates, simplified with
Douglas–Peucker at ~1 px — beside `p1`/`p2`, which stay for compatibility.
Carry `path` through `_build_reference_doc` and
`annotations_from_reference_json`.

#### L2. Mask along the path

In `_build_removal_mask`, draw `cv2.polylines` over `path` with
`leader_mask_thickness` when a path exists; fall back to the chord when it
does not. Keep `leader_tip_trim` on the final segment only.

**Accept when.** Ink-left-behind on curved leaders, re-measured with the
script in the evidence file, falls from 75% to ≤ 10% (R1).

#### L3. Reinject the path

Stage 4 draws `add_lwpolyline(path)` on the `LEADER` layer for DXF and a
`<polyline>` for SVG when a path exists; `add_line` otherwise. The export
audit's `leaders_written` counts `LWPOLYLINE` as well as `LINE`.
`acceptance._check_references` compares `path` between reference and
annotation the way it compares `p1`/`p2`.

#### L4. Trace what the straightness cut rejects

Lower `ocr_leader_straightness` from 0.80 in steps and measure two things on
the 100 audit figures: leaders newly traced, and any new false leaders
(traces that wander onto a feature line). The turn-tolerance in the tracer is
what guards against wandering; if it holds, the 34.8% leaderless share should
drop. If false leaders appear, stop at the last safe value. **Hypothesis**
until measured.

**Accept when.** R2 holds on the 100 audit figures; policy version bumps.

**Cost.** 2 days. **Risk.** Low; every change is additive and falls back to
the chord.

### Phase M — A filled-region primitive  ·  *the largest change*

Goal: R3, and the representation half of R6. Touches Stages 2, 3, 4, the
validator and the acceptance record, so it carries version bumps throughout.

#### M1. A region detector, measured before it ships

Detect ink that means an area: connected components where the distance
transform exceeds ~5 px over a substantial fraction of the component, or
whose skeleton-length-to-area ratio is far below a stroke's. The sizing
proxy is the starting point, not the detector: it caught 4 of the 12
audit figures at a threshold that also flagged 3 of 88 good ones.

**Accept when.** Precision measured the way every discriminator this month
was — against a labelled negative class: on the 88 audit figures the owner
called clean, false regions on ≤ 2; on the 26 figures with a blob ≥ 5,000 px,
the region found on ≥ 22. If the detector cannot reach this, Phase M stops
here and the regions are routed to review rather than mis-represented.

#### M2. Route regions before skeletonization

In Stage 2, where hachure removal already subtracts ink before the
topology pass: subtract region ink the same way, and hand each region's
contour to Stage 3 as a `filled_polygon` primitive (contour simplified,
holes kept as inner rings). The coverage ledger records region ink as
`represented` by the region, so ink routing still accounts for every pixel.

#### M3. Export

SVG: `<path d=… fill="black">`. DXF: a `HATCH` entity with the `SOLID`
pattern on a `FILL` layer — the native form, and the DXF `hatch` type already
exists in Stage 4. The export audit gains the type; `label_required` is
false for it.

#### M4. Validate the boundary, not the interior

`geometry_validation.sample_primitive` samples a `filled_polygon` as its
boundary. The source-support check then asks whether the boundary lies on
the ink's *edge* rather than its skeleton — a filled region's skeleton is
exactly the thing this phase stops trusting. Validator version → 6.

#### M5. Bold bands are regions too

When the hatch detector's mask covers a component whose stroke width is
≥ 4 px (the band case), route it through M2 as a region with
`"pattern": "hatch"` and the band angle recorded, instead of through the
thin-line remover. This is the removal-side fix for the two audit hatch
figures, and it needs no retrain.

**Accept when.** R3. On the 12 + 2 flagged audit figures, reconstruction
re-rendered and judged by the owner: no scribbles.

**Cost.** 4–5 days. **Risk.** Medium. The schema change is the risk, not the
detector — every consumer of the primitives list must learn the new type or
fail closed on it. The version bumps make unhandled records unverifiable
rather than silently wrong.

### Phase N — Hatch styles: detection

Goal: R4, R5, and the detection half of R6. N1 is the owner's; the rest
waits on it.

#### N1. Label ~40 figures by style  ·  *owner, ~1–2 hours*

A candidate list drawn from the sizing proxies so the hour is spent where
the styles occur: the 39 bold-band figures, the 37 shading figures, the 26
filled-blob figures, plus 20 ordinary hatched figures as controls. The owner
draws the true hatch areas with `tools/hatch_mask_label.py --out
output/PatentData/hatch_gt_styles`, tagging each region's style. The tool
draws freehand and so captures what the detector misses; `hatch_label.py`
only judges detector proposals and would not.

#### N2. Per-style recall of the shipped detector

Run `hatch_unet.pth` over N1's masks; report IoU and recall **by style**. This
is the measurement that has never been made — Phase C's 98.2% was on
synthetic sheets drawn at one thin stroke width. It sets R4's target from
data rather than from the 0.70 guess above.

#### N3. Retrain on the union

`tools/hatch_train_cnn.py --gt <268 originals + N1>`, starting from the
shipped recipe (lr 3e-4, pos_weight 3.0). Select by per-style validation
recall **subject to** original-style test IoU ≥ 0.816.

**Accept when.** R4 and R5 both hold. If R5 fails — the new styles cost the
old ones — the retrain is rejected and bold bands rely on M5 alone.

**Cost.** 2 days plus the owner's labelling plus GPU time on card 1.

### Phase O — Second release

- **O1.** Freeze first: all code changes in, tests green, identity recorded.
  Then rerun the 1,465 labelled figures under the new identity.
- **O2.** Audit: 100 figures at random as before, **plus** the 14 flagged
  figures re-judged side by side with their release-1 output, **plus** the
  numeral contact sheets that release 1 shipped without
  (`tools/build_label_audit.py`, ~20 minutes). The numeral audit is the one
  release-1 gate still open.
- **O3.** `docs/RELEASE_<date>.md` with R6 and R7, and a diff against
  release 1 on every criterion.

**Cost.** 1 day plus compute plus ~1.5 owner-hours.

---

## 4. Dependency graph

```
L1 keep path ─► L2 mask ─► L3 reinject ─► L4 straightness ──────────────┐
                                                                         │
M1 detector ─► M2 route ─► M3 export ─► M4 validate ─► M5 bands ─────────┤
                                                                         ├─► O1 freeze+run ─► O2 audit (owner) ─► O3 release
N1 labels (owner) ─► N2 measure ─► N3 retrain ───────────────────────────┤
                                                                         │
                          (N3 needs M2's routing to remove what it detects)
```

L, M and N1 can all start now. N2 waits on N1 only. N3 waits on N1 and M2.
Suggested order of attention: **L first** — two days, a third of all
leaders, no schema risk — then M, with N1 running in parallel on the owner's
side.

Rough effort: L 2 days · M 4–5 · N 2 plus labelling and training · O 1 plus
compute. Three working weeks. Owner time: ~1–2 h labelling, ~1.5 h auditing.

---

## 5. What this revision does not fix

- **Thick outlines proper.** The three figures whose bold frame or cable
  skeletonized to a thin line or a double edge. Tolerable; recording stroke
  width per edge would fix it, and is not scheduled.
- **Numerals that genuinely have no leader.** L4 recovers the too-curved
  ones; it cannot invent a leader that was never drawn.
- **The isolation gate.** 48 of 49 Stage-2 rejections in the pre-release run
  tripped it on healthy graphs — the `max_edges` defect in the gate that
  replaced it. Separate work; the excluded figures are recoverable.
- **Reference text remains unverified** until O2's numeral sheets are done.
- **Every effort estimate is a guess.** Revision 2's were right about the
  release pipeline and wrong about the classifier. Phase M is the one most
  likely to run long.

---

## 6. Evidence

| finding | file |
| --- | --- |
| taxonomy, sizing, detector blindness, removal failure, ghost mechanism | `docs/audits/2026-09-29/remaining_defects_evidence.json` |
| per-figure sizing proxies over the release | `docs/audits/2026-09-29/defect_sizing.json` |
| the owner's audit | `releases/2026-09-28/k2_geometry_audit.csv` |
| release 1 | `docs/RELEASE_2026-09-28.md` |
| existing hatch ground truth | `output/PatentData/hatch_gt/` (268 figures) |
