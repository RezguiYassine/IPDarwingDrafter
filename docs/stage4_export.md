# Stage 4 — Vector Export

> AP3 Vectorization Pipeline · IP DrawingDrafter · HAW Landshut
> Module: `stage4_export.py`

---

## 1. Position in the Pipeline

Stage 4 is the **terminal stage** of the AP3 vectorization pipeline. It converts the geometric primitives produced by Stage 3 (RANSAC fitting) into deliverable vector files — SVG for review, DXF for downstream CAD software.

```
Stage 3 (RANSAC)  →  primitives.json  →  Stage 4 (Export)  →  .svg / .dxf
                       (lines/arcs/circles)                    ↓
                                                            AP6 (GST) /
                                                          User download
```

**Representation upgrade framing.** Each pipeline stage takes data in one form and produces a richer one:

| Stage | In                    | Out                                 |
|-------|-----------------------|-------------------------------------|
| 1     | Raw raster            | Cleaned skeleton                    |
| 2     | Skeleton              | Stroke graph                        |
| 3     | Stroke graph          | Geometric primitives (parameterised)|
| **4** | **Geometric primitives** | **Vector files (SVG, DXF)**      |

Stage 4 is the **only stage that produces files for human/CAD consumption** — every other stage produces JSON intermediates. This means it owns all format-specific concerns (coordinate conventions, layer organisation, linetypes, units) so no upstream stage has to know about CAD.

---

## 2. Input Contract

Stage 4 reads a single JSON file produced by Stage 3:

```json
{
  "sketch_id":  "...",
  "image_size": [W, H],
  "primitives": [
    {"type": "line",   "p1": [x,y], "p2": [x,y], "confidence": 0..1, "style": "..."},
    {"type": "circle", "center": [x,y], "radius": r, "confidence": 0..1, "style": "..."},
    {"type": "arc",    "center": [x,y], "radius": r,
     "start_angle": deg, "end_angle": deg, "confidence": 0..1, "style": "..."},
    {"type": "path", "segments": [{"type": "line|arc|bezier", "...": "..."}]},
    {"type": "hatch", "boundary": [[x,y], "..."], "angles": [45], "spacing": 8}
  ],
  "annotations": [               // optional, supplied by AP6 when available
    {"id": "1", "text": "Welle",
     "position":  [x, y],        // text anchor, image space
     "leader_to": [x, y]}        // tip of leader arrow, image space
  ]
}
```

**Field semantics:**

- `style` is **optional** on every primitive. Valid values: `visible | hidden | center | construction`. Missing or unknown values fall back to `visible`. Stage 3 typically does not know styles — they will be assigned by AP3.3 (element relations) or AP6.
- `annotations` is **optional**. Stage 4 does not generate Bezugszeichen — that is AP6.4's responsibility per the Vorhabensbeschreibung. If absent, patent-mode DXF still produces a valid layered file, just without numerals.
- Coordinates are **raster pixel indices in image space, Y-down**. The integer
  pair `[x, y]` identifies a pixel whose centre is at continuous vector
  coordinate `[x + 0.5, y + 0.5]`; Stage 4 owns that conversion.

---

## 3. Output Modes

Two orthogonal choices — **format** and (for DXF) **compliance level**:

| `--format` | `--dxf-mode` | Output                                                          |
|------------|--------------|-----------------------------------------------------------------|
| `svg`      | (n/a)        | Single `.svg` file, view-friendly, ISO-styled stroke widths     |
| `dxf`      | `basic`      | Single-layer DXF, AutoCAD-readable, no styling (quick QA)       |
| `dxf`      | `patent`     | ISO 128 layered DXF + center crosses + Bezugszeichen (if given) |
| `both`     | either       | SVG + DXF emitted together                                      |

**Why two DXF modes:** the basic mode is for fast iteration during AP3 development — open in any viewer, confirm geometry. Patent mode is the production deliverable for the Verwertung phase (PatentDrafter Drawing product).

---

## 4. ISO 128 Layer Specification (Patent Mode)

Patent mode creates six layers, all with AutoCAD-standard names and linetypes. Lineweights are in 1/100 mm (DXF convention).

| Layer          | Linetype     | Lineweight | Purpose                              |
|----------------|--------------|-----------:|--------------------------------------|
| `VISIBLE`      | CONTINUOUS   | 0.50 mm    | Visible edges                        |
| `HIDDEN`       | DASHED       | 0.25 mm    | Hidden edges                         |
| `CENTER`       | CENTER       | 0.25 mm    | Centerlines + auto center-crosses    |
| `CONSTRUCTION` | CONTINUOUS   | 0.13 mm    | Construction / projection lines      |
| `TEXT`         | CONTINUOUS   | 0.25 mm    | Bezugszeichen (MTEXT)                |
| `LEADER`       | CONTINUOUS   | 0.18 mm    | Leader lines pointing from text to feature |

**Auto-additions in patent mode:**

- Every circle gets a **center cross** on the `CENTER` layer (extends 15% past the circle radius), per ISO 128 convention for circular features.
- Bezugszeichen are emitted as `MTEXT` on `TEXT`, with optional leader `LINE` on `LEADER` per EPO Rule 46 ("uniform size, clearly identifies the referenced feature").

The `ISO128_LAYERS` dict at the top of the module is the **single source of truth** for layer specs. Adding/changing a layer = edit the dict.

---

## 5. Coordinate System Handling

This is the most subtle part of Stage 4 and the part most likely to confuse downstream developers.

| System | Origin     | Y direction | Used for                        |
|--------|------------|-------------|---------------------------------|
| Input  | Top-left   | **Y-down**  | Raster-index frame used by Stages 1–3 |
| SVG    | Top-left   | **Y-down**  | Pixel centres: `(x + 0.5, y + 0.5)` |
| DXF    | Bottom-left| **Y-up**    | Pixel centres plus Y flip |

The half-pixel conversion is defined by `PIXEL_CENTER_OFFSET`. SVG applies it
once to a group containing all geometry and re-injected Stage 0 annotations.
DXF applies the equivalent conversion in `_flip_y_point()`:

```text
SVG: (x, y) -> (x + 0.5, y + 0.5)
DXF: (x, y) -> (x + 0.5, H - (y + 0.5))
```

Arc direction conversion lives in `_flip_y_arc_angles()`. As a result:

- SVG previews match the source sketch 1:1 — useful for QC.
- DXF imports into AutoCAD/SolidWorks/KiCad with the correct visual orientation.
- Arcs require a special case: mirroring across the X-axis flips CCW direction, so we mirror the angles around 0° **and** swap start/end. ezdxf draws arcs CCW from start to end, so this preserves the visual arc.

Compound paths have a second orientation problem: an independently fitted arc
describes the occupied angular interval but not the source chain direction.
`_orient_path_segments()` therefore uses a deterministic two-state dynamic
program to choose the forward/reverse direction of every line, arc, and Bezier
segment while minimizing total connector length. SVG and DXF consume the same
oriented sequence, preventing chord-sized jumps when an arc is the first path
atom and preserving continuity across mixed segment types.

---

## 6. Script Architecture (`stage4_export.py`)

The module follows the same shape as `stage1_preprocess.py` for consistency across the pipeline.

```
┌─────────────────────────────────────────────────────────────┐
│  Output contract                                            │
│  ─ Stage4Result (dataclass)                                 │
├─────────────────────────────────────────────────────────────┤
│  Configuration constants                                    │
│  ─ ISO128_LAYERS  ← single source of truth for layer specs  │
│  ─ DEFAULT_STYLE                                            │
├─────────────────────────────────────────────────────────────┤
│  Coordinate helpers                                         │
│  ─ _flip_y_point()        image → CAD                       │
│  ─ _flip_y_arc_angles()   handle CCW direction flip         │
├─────────────────────────────────────────────────────────────┤
│  Input handling                                             │
│  ─ _load_input()          JSON load + validation            │
│  ─ _primitive_style()     style extraction with fallback    │
├─────────────────────────────────────────────────────────────┤
│  Format-specific exporters (independent, never call each other)│
│  ─ _export_svg()                                            │
│  ─ _export_dxf_basic()                                      │
│  ─ _export_dxf_patent()                                     │
│      └─ _ensure_layers()                                    │
│      └─ _add_bezugszeichen()                                │
├─────────────────────────────────────────────────────────────┤
│  Public stage entry point                                   │
│  ─ run(input_json, output_dir, sketch_id, formats, dxf_mode)│
├─────────────────────────────────────────────────────────────┤
│  CLI for standalone testing                                 │
│  ─ argparse + run() invocation                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.1 Why three exporters instead of one parameterised exporter

The three export functions (`_export_svg`, `_export_dxf_basic`, `_export_dxf_patent`) intentionally **do not share code**. Each writes a different file format with different conventions, and the patent DXF in particular has format-specific concerns (layers, center crosses, MTEXT) that don't apply to SVG. Trying to factor them through a common interface produces a leaky abstraction.

What *is* shared:
- Coordinate helpers (called identically by both DXF exporters)
- `_primitive_style()` (called by SVG and DXF-patent)
- The `ISO128_LAYERS` dict (used to derive both DXF layer specs and SVG stroke widths)

### 6.2 Error handling

Each primitive is exported inside a `try`/`except`. If a single primitive fails (e.g., malformed JSON, unknown type), the export logs a warning and continues. The stage flags itself if `n_out < n_in` — same QC pattern as Stage 1's `flagged` field.

This means **a corrupt primitive does not block the rest of the sketch** — Stage 4 will produce a partial file and flag it for review.

### 6.3 Stage-1 consistency

Patterns deliberately mirrored from `stage1_preprocess.py`:

- Public `run()` returning a typed `@dataclass` result
- `flagged: bool` on the result for QC pipeline integration
- Per-sketch logging with `[sketch_id]` prefix
- CLI entry point under `if __name__ == "__main__"`
- Output directory created on demand (`mkdir(parents=True, exist_ok=True)`)

---

## 7. Extension Points

Three places future stages will plug in:

| Hook                   | Filled by | What it does                                             |
|------------------------|-----------|----------------------------------------------------------|
| `style` field on primitives | AP3.3 / AP6 | Maps primitives onto ISO 128 layers (visible/hidden/center/construction) |
| `annotations` block         | AP6.4      | Bezugszeichen text + leader endpoints                  |
| New layer in `ISO128_LAYERS` | Any        | Add a new line class — appears automatically in patent DXF |

**What Stage 4 will *not* do** (deliberate scope boundary):

- Style assignment (which line is a centerline?) — needs context Stage 3 doesn't have.
- Bezugszeichen *generation* (which features get numbered, in what order?) — that's AP6.4.
- Dimensioning — explicitly forbidden by patent rules (no Vermassung).
- Colour — patent rules forbid colour in most jurisdictions.
- Title blocks, borders, sheet frames — Verwertung-phase product concern, not pipeline.

---

## 8. CLI Reference

```bash
# SVG only (fast QA)
python3 stage4_export.py primitives.json --format svg

# Basic DXF (single-layer, AutoCAD-readable)
python3 stage4_export.py primitives.json --format dxf --dxf-mode basic

# Patent-ready DXF (ISO 128 layered + Bezugszeichen)
python3 stage4_export.py primitives.json --format dxf --dxf-mode patent

# Both formats at once
python3 stage4_export.py primitives.json --format both --dxf-mode patent

# Override sketch ID and output directory
python3 stage4_export.py primitives.json \
    --format both --dxf-mode patent \
    --output ./results --id gear_demo_001
```

Outputs land in `<output>/vectors/<sketch_id>.{svg,dxf}`.

---

## 9. Verification Status

Verified end-to-end on a synthetic test sketch (`example_stage3_input.json`) covering:

- All three primitive types (line, circle, arc)
- All four styles (visible, hidden, center, construction)
- Three Bezugszeichen with leader lines
- Both DXF compliance modes

Confirmed:

- 9/9 primitives written in both modes
- Patent DXF produces all 6 ISO 128 layers with correct linetypes
- Pixel-centre conversion is shared by SVG and DXF (input `[4,16]` in a
  32px image exports to SVG `[4.5,16.5]` and DXF `[4.5,15.5]`).
- Arc angles mirrored such that visual orientation matches the SVG preview
- MTEXT entries land on `TEXT` layer with leaders on `LEADER`
- Circles automatically gain center crosses on the `CENTER` layer

The Drawing2CAD 250-sample/all-view sweep selected `+0.5 px` over `+0.25`,
`+0.75`, and `+1.0`: candidate mean Chamfer fell from `0.9237` to `0.3423`
and median Chamfer from `0.9123` to `0.2391`. The old exporter placed common
integer-coordinate strokes one raster pixel up and left after SVG rendering.

Exporter-only changes can be evaluated without repeating Stages 1–3:

```bash
python -m tools.rescore_d2c_stage4 \
  --run-dir output/Drawing2CAD/<completed-run> --workers 12
```

This writes a cloned `d2c_results_pixel_center.db`, preserving the original
evaluation database for traceability.

---

## 10. Dependencies

| Package    | Min version | Purpose                  |
|------------|-------------|--------------------------|
| `ezdxf`    | 1.4         | DXF read/write           |
| `svgwrite` | 1.4         | SVG generation           |

Both are pure Python, no native compilation needed. No PyTorch / DL dependency in this stage.
