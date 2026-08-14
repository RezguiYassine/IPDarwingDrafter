# PatentVec Synthetic Composition Roadmap

## Implementation plan for interaction-aware synthetic patent drawing generation

**Document status:** Codex Desktop execution roadmap

**Primary objective:** generate a large synthetic raster/vector dataset whose drawings are significantly more complex than SketchGraphs, CAD-VGDrawing, Drawing2CAD, ArchCAD, FloorPlanCAD, and related datasets, while preserving exact vector ground truth.

**Primary downstream target:** improve raster-to-vector models, especially the Puhachov/PolyVector pipeline, junction detection, primitive fitting, topology recovery, semantic stroke separation, and post-vectorization quality gating.

**Secondary target:** create a separate CAD-valid subset for Drawing2CAD, Free2CAD-style systems, and vector-to-CAD reconstruction.

**Core principle:** compose vector components through explicit geometric interactions rather than placing complete drawings side by side.

---

# 1. Mission

Build a synthetic-data generation system called **PatentVec Compose**.

The system must:

1. ingest several existing vector or CAD datasets;
2. convert all source samples into one canonical primitive-graph representation;
3. decompose source drawings into reusable components and motifs;
4. detect typed anchors and attachment opportunities;
5. sample a connected interaction graph;
6. place and connect 2–12 source components using valid geometric relationships;
7. update primitive topology exactly;
8. add patent-specific layers such as hatching, dashed centre lines, leaders, numerals, dimensions, text boxes, and scan artefacts;
9. render clean and degraded raster images;
10. retain exact visible, amodal, semantic, topology, and provenance ground truth;
11. reject unrealistic or ambiguous composites;
12. generate train, validation, and test releases without source leakage;
13. train and evaluate vectorization models against real patent drawings;
14. separately generate CAD-valid compositions for vector-to-CAD training.

The resulting synthetic samples must look like **single complex technical drawings**, not collages.

---

# 2. Key design decision

Do not merge full raster drawings directly.

The generator must operate on vector components:

```text
Source datasets
    -> canonical vector graphs
    -> component and motif extraction
    -> typed anchors
    -> interaction graph
    -> similarity-transform placement
    -> exact topology update
    -> patent semantic layers
    -> rasterization and degradation
    -> exact ground-truth package
```

Every donor component added to a composition must interact with at least one existing component through one or more of:

- endpoint connection;
- T-junction;
- tangent contact;
- concentric alignment;
- shared boundary;
- containment;
- insertion into a gap;
- connected crossing;
- disconnected crossing;
- partial occlusion;
- bridging two components;
- array or repeated-pattern relation.

Pure side-by-side placement is not an accepted composition operator.

---

# 3. Two separate generation tracks

## 3.1 Track A: PatentVec-Compose2D

Use for:

- Puhachov/PolyVector;
- endpoint and junction prediction;
- centreline extraction;
- primitive classification;
- line, arc, circle, ellipse, and Bézier fitting;
- topology recovery;
- object-versus-annotation separation;
- post-vectorization quality gates.

A Track-A sample must have exact 2D vector ground truth. It does not need to correspond to one physically valid 3D CAD model.

## 3.2 Track B: PatentVec-ComposeCAD

Use for:

- Drawing2CAD;
- Free2CAD-style systems;
- vector-to-CAD models;
- multi-view CAD reconstruction;
- CadQuery or FreeCAD code generation.

A Track-B sample must come from:

- valid CAD programs;
- valid 3D component assemblies;
- valid boolean operations;
- or valid sketch-and-feature constructions.

Do not assign an invented CAD program to an arbitrary Track-A 2D composition.

---

# 4. Expected final deliverables

Codex must produce:

## 4.1 Dataset builder

A reproducible command-line workflow that generates:

- clean SVG-like vector composites;
- clean raster renderings;
- degraded patent-style raster renderings;
- visible vector ground truth;
- amodal vector ground truth;
- semantic stroke labels;
- endpoint and junction labels;
- primitive-type labels;
- interaction metadata;
- source provenance;
- component and instance labels;
- quality and realism scores.

## 4.2 Dataset releases

At minimum:

```text
PatentVec-Compose2D-v1
PatentVec-ComposeCAD-v1
PatentVec-Compose2D-Hard-v1
PatentVec-Compose2D-Ablation-v1
```

## 4.3 Training integrations

Adapters for:

- Puhachov/PolyVector;
- PatentVec multitask vectorizer;
- Drawing2CAD;
- optional Free2CAD graph adapter.

## 4.4 Evaluation reports

Reports comparing:

- original source datasets only;
- patent-style degradation only;
- side-by-side composition;
- random overlap;
- interaction-aware composition;
- interaction-aware composition plus patent annotations;
- real-statistics-calibrated composition.

## 4.5 Deployment tools

- CLI;
- configuration files;
- Docker environment;
- regression tests;
- data manifests;
- montage viewer;
- sample inspector;
- release scripts;
- dataset card;
- provenance and license registry.

---

# 5. Recommended repository structure

Create or extend the PatentVec repository with:

```text
patentvec/
├── README.md
├── STATUS.md
├── DECISIONS.md
├── LICENSES.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── docker-compose.yml
├── configs/
│   ├── sources/
│   ├── extraction/
│   ├── anchors/
│   ├── interactions/
│   ├── composition/
│   ├── patent_layers/
│   ├── degradation/
│   ├── realism/
│   ├── release/
│   └── training/
├── patentvec/
│   ├── schema/
│   │   ├── primitive.py
│   │   ├── graph.py
│   │   ├── component.py
│   │   ├── anchor.py
│   │   ├── interaction.py
│   │   ├── sample.py
│   │   └── validation.py
│   ├── ingest/
│   │   ├── sketchgraphs.py
│   │   ├── cad_vgdrawing.py
│   │   ├── drawing2cad.py
│   │   ├── archcad.py
│   │   ├── floorplancad.py
│   │   ├── polyvector.py
│   │   └── cad_models.py
│   ├── components/
│   │   ├── connected.py
│   │   ├── constrained_subgraphs.py
│   │   ├── closed_profiles.py
│   │   ├── semantic_instances.py
│   │   ├── motif_index.py
│   │   └── descriptors.py
│   ├── anchors/
│   │   ├── point_anchors.py
│   │   ├── boundary_anchors.py
│   │   ├── region_anchors.py
│   │   ├── structural_anchors.py
│   │   ├── compatibility.py
│   │   └── scoring.py
│   ├── interactions/
│   │   ├── endpoint_join.py
│   │   ├── t_junction.py
│   │   ├── collinear.py
│   │   ├── tangent.py
│   │   ├── concentric.py
│   │   ├── containment.py
│   │   ├── insertion.py
│   │   ├── shared_edge.py
│   │   ├── crossings.py
│   │   ├── occlusion.py
│   │   ├── bridge.py
│   │   └── repetition.py
│   ├── composition/
│   │   ├── graph_sampler.py
│   │   ├── donor_retrieval.py
│   │   ├── transform_solver.py
│   │   ├── placement.py
│   │   ├── topology_update.py
│   │   ├── visibility.py
│   │   ├── provenance.py
│   │   └── generator.py
│   ├── patent_layers/
│   │   ├── line_styles.py
│   │   ├── centerlines.py
│   │   ├── hatching.py
│   │   ├── leaders.py
│   │   ├── numerals.py
│   │   ├── dimensions.py
│   │   ├── text_boxes.py
│   │   └── layout.py
│   ├── render/
│   │   ├── svg.py
│   │   ├── raster.py
│   │   ├── masks.py
│   │   ├── degradation.py
│   │   └── previews.py
│   ├── quality/
│   │   ├── geometry_gate.py
│   │   ├── interaction_gate.py
│   │   ├── topology_gate.py
│   │   ├── visibility_gate.py
│   │   ├── density_gate.py
│   │   ├── realism_gate.py
│   │   ├── roundtrip_gate.py
│   │   └── report.py
│   ├── realism/
│   │   ├── patent_statistics.py
│   │   ├── profile.py
│   │   ├── distance.py
│   │   └── synthetic_real_classifier.py
│   ├── cad_compose/
│   │   ├── solid_library.py
│   │   ├── assembly.py
│   │   ├── boolean_ops.py
│   │   ├── projections.py
│   │   ├── cad_targets.py
│   │   └── verification.py
│   ├── datasets/
│   │   ├── manifests.py
│   │   ├── splits.py
│   │   ├── dedup.py
│   │   └── release.py
│   ├── training/
│   │   ├── puhachov_adapter.py
│   │   ├── drawing2cad_adapter.py
│   │   ├── curriculum.py
│   │   └── loaders.py
│   ├── evaluation/
│   │   ├── vector_metrics.py
│   │   ├── topology_metrics.py
│   │   ├── semantic_metrics.py
│   │   ├── ablations.py
│   │   └── reports.py
│   └── cli.py
├── scripts/
│   ├── bootstrap.sh
│   ├── download_source_samples.py
│   ├── build_component_index.py
│   ├── generate_pilot.py
│   ├── validate_release.py
│   └── train_smoke.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
├── data/
│   ├── raw/
│   ├── source_manifests/
│   ├── components/
│   ├── synthetic/
│   ├── releases/
│   └── reports/
└── docs/
    ├── architecture.md
    ├── source_adapters.md
    ├── interaction_grammar.md
    ├── ground_truth.md
    ├── dataset_card.md
    └── troubleshooting.md
```

---

# 6. Engineering rules

1. Raw source data is immutable.
2. Source licenses must be registered before full ingestion.
3. Every generated primitive must have provenance.
4. Every transformation must be saved.
5. Similarity transforms are the default.
6. Nonuniform transforms require exact primitive conversion and explicit configuration.
7. Arbitrary image warping is forbidden before vector labels are updated.
8. Every component must interact with the composition.
9. Every interaction must be typed.
10. Every selected interaction must pass a geometric residual threshold.
11. Visible and amodal ground truth must be stored separately.
12. Coincident visible geometry must not be duplicated.
13. Unintended crossings must be detected and handled.
14. Source train, validation, and test pools must remain separate.
15. Different views of the same CAD object must remain in the same split.
16. Generation must be deterministic for a stored random seed.
17. Every rejection reason must be logged.
18. No large generation run before a small pilot passes.
19. Every release must include a manifest and checksums.
20. Track-A and Track-B samples must never be confused.

---

# 7. Phase 0 — Bootstrap and baseline setup

## 7.1 Create the repository structure

Codex must:

- create directories;
- initialize Git;
- configure Python;
- add linting, typing, and tests;
- add Docker;
- create `STATUS.md`;
- create `DECISIONS.md`;
- create `LICENSES.md`.

## 7.2 Required dependencies

Suggested:

- Python 3.11;
- NumPy;
- SciPy;
- Shapely;
- svgpathtools;
- NetworkX;
- OpenCV;
- Pillow;
- scikit-image;
- Pydantic;
- Typer;
- Hydra or typed YAML;
- PyTorch;
- pandas;
- pyarrow;
- matplotlib;
- pytest;
- DVC;
- MLflow.

Optional CAD stack:

- FreeCAD;
- CadQuery;
- OpenCascade/pythonOCC;
- trimesh.

## 7.3 Smoke tests

Before source ingestion, verify:

- parse one SVG;
- construct one line and one arc;
- calculate one line-line intersection;
- calculate one line-circle intersection;
- apply one similarity transform;
- rasterize one graph;
- export one SVG;
- serialize one sample;
- reload and validate it;
- build one simple CadQuery or FreeCAD solid.

Output:

```text
reports/bootstrap_report.json
```

---

# 8. Phase 1 — Define the canonical schema

## 8.1 Primitive graph

Each drawing or component is:

\[
G = (P, J, E, C, A, R, M)
\]

where:

- `P`: primitives;
- `J`: junctions;
- `E`: graph edges and geometric relations;
- `C`: components or instances;
- `A`: anchors;
- `R`: interaction relations;
- `M`: metadata and provenance.

## 8.2 Primitive types

Initial support:

- line;
- polyline;
- circle;
- circular arc;
- ellipse;
- elliptical arc;
- quadratic Bézier;
- cubic Bézier;
- spline;
- point.

## 8.3 Semantic layers

Use:

- `object_visible`;
- `object_hidden`;
- `object_center`;
- `object_section_boundary`;
- `construction`;
- `hatch`;
- `leader`;
- `dimension`;
- `arrowhead`;
- `reference_numeral`;
- `figure_label`;
- `text_box`;
- `diagram_connector`;
- `text`;
- `unknown`.

## 8.4 Junction types

Use:

- endpoint;
- sharp corner;
- smooth continuation;
- T-junction;
- X-junction;
- Y-junction;
- tangent contact;
- line-arc transition;
- arc-arc transition;
- connected crossing;
- disconnected crossing;
- leader contact;
- arrow tip;
- occlusion boundary.

## 8.5 Coordinate systems

Save:

- source coordinates;
- component-local coordinates;
- composite coordinates;
- raster pixel coordinates;
- normalized coordinates.

For each component:

```text
T_source_to_component
T_component_to_composite
T_composite_to_raster
T_source_to_composite
T_source_to_raster
```

## 8.6 Primitive provenance

Every primitive must contain:

```json
{
  "primitive_id": "cmp_p_001",
  "source_dataset": "SketchGraphs",
  "source_sample_id": "sg_1001",
  "source_component_id": "component_03",
  "source_primitive_id": "line_17",
  "parent_primitive_ids": [],
  "generated_by": "source_transform",
  "interaction_ids": ["rel_08"],
  "transform_id": "T_05",
  "visibility": "visible",
  "source_interval": [0.0, 1.0]
}
```

Generated patent annotation primitives use:

```text
generated_by = hatch_generator
generated_by = leader_generator
generated_by = dimension_generator
generated_by = text_box_generator
```

## 8.7 Sample schema

Each synthetic sample must include:

```json
{
  "schema_version": "1.0.0",
  "sample_id": "pvcomp_00000001",
  "track": "compose2d",
  "difficulty": "hard",
  "seed": 123456,
  "sources": [],
  "components": [],
  "anchors": [],
  "interactions": [],
  "primitives_visible": [],
  "primitives_amodal": [],
  "junctions_visible": [],
  "junctions_amodal": [],
  "semantic_layers": {},
  "images": {},
  "quality": {},
  "realism": {},
  "processing": {}
}
```

## 8.8 Schema gate

Tests must reject:

- missing source IDs;
- invalid primitive parameters;
- invalid transforms;
- broken primitive references;
- invalid interaction IDs;
- missing lineage;
- impossible visibility intervals;
- disconnected required components;
- duplicate primitive IDs;
- invalid normalized coordinates.

No source adapter work begins before the schema tests pass.

---

# 9. Phase 2 — Source ingestion

Implement source adapters incrementally.

## 9.1 Priority order

1. SketchGraphs;
2. CAD-VGDrawing;
3. ArchCAD;
4. FloorPlanCAD;
5. PolyVector data;
6. Drawing2CAD-specific source representations;
7. optional DeepCAD or Fusion 360 models for Track B.

## 9.2 Adapter contract

Every source adapter must provide:

```python
def load_source_sample(source_id) -> CanonicalDrawing
def list_source_samples(split=None) -> Iterable[SourceRecord]
def extract_license_metadata() -> LicenseRecord
def validate_source_sample(sample) -> ValidationReport
```

## 9.3 SketchGraphs adapter

Convert:

- lines;
- circles;
- arcs;
- points;
- constraints;
- coincidence relations;
- tangency;
- parallelism;
- perpendicularity;
- symmetry where available.

Do not infer every visual crossing as connected.

Use the original constraint graph.

## 9.4 CAD-VGDrawing adapter

Retain:

- front/top/right/isometric views;
- exact SVG primitives;
- view type;
- CAD target;
- model ID;
- source split;
- view relationships.

## 9.5 ArchCAD adapter

Use semantic and instance IDs to extract coherent motifs instead of arbitrary crops.

Preserve:

- lines;
- arcs;
- circles;
- instances;
- semantic labels;
- source drawing boundaries.

## 9.6 FloorPlanCAD adapter

Use:

- line-level semantics;
- symbols;
- rooms;
- architectural units;
- repeated structures;
- dense intersections.

## 9.7 Source validation report

For each source:

- total samples;
- valid samples;
- parser failures;
- primitive distribution;
- component distribution;
- licensing status;
- split policy;
- example montage.

---

# 10. Phase 3 — Component and motif extraction

## 10.1 Extraction units

Do not use complete source drawings only.

Create reusable units:

- connected components;
- connected constrained subgraphs;
- semantic instances;
- closed profiles;
- repeated patterns;
- circular motifs;
- line-arc chains;
- symbols;
- cavities;
- boundary groups;
- mechanical subassemblies;
- architectural units.

## 10.2 Connected components

Extract by graph connectivity, not raster connectivity alone.

Respect:

- connected versus disconnected crossings;
- source constraints;
- semantic instance IDs;
- shared primitives.

## 10.3 Closed-profile extraction

Detect:

- simple closed loops;
- nested loops;
- loops with holes;
- circular and elliptical enclosures;
- polygonal profiles.

Store:

- area;
- perimeter;
- interior mask;
- hierarchy;
- orientation;
- possible insertion region.

## 10.4 Constrained-subgraph extraction

For SketchGraphs:

- select connected subgraphs;
- preserve all required constraints;
- reject subgraphs whose geometry becomes under-constrained or invalid;
- store boundary anchors.

## 10.5 Component descriptors

For every component calculate:

- primitive count;
- primitive-type histogram;
- junction count;
- degree histogram;
- bounding box;
- oriented bounding box;
- convex hull;
- centroid;
- principal axes;
- symmetry candidates;
- closed-loop count;
- circle centres;
- dominant directions;
- boundary length;
- local density;
- aspect ratio;
- semantic category;
- source-domain category;
- complexity level.

## 10.6 Component library

Store components in a searchable index.

Recommended columns:

```text
component_id
source_dataset
source_sample_id
source_split
domain
semantic_type
primitive_count
junction_count
closed_loop_count
circle_count
arc_count
bbox_width
bbox_height
aspect_ratio
complexity
anchor_count
relative_path
sha256
```

Use Parquet.

## 10.7 Component extraction gate

Reject components with:

- no visible geometry;
- invalid primitives;
- extreme aspect ratio outside configured limits;
- near-zero scale;
- unresolved topology;
- ambiguous source split;
- duplicate hash;
- unsupported licensing.

---

# 11. Phase 4 — Anchor detection

## 11.1 Point anchors

Detect:

- endpoints;
- sharp corners;
- T-junctions;
- circle centres;
- arc centres;
- centroids;
- symmetry centres;
- intersection candidates.

## 11.2 Directional anchors

For every relevant point store:

- tangent direction;
- normal direction;
- dominant component axis;
- line direction;
- local curvature.

## 11.3 Boundary anchors

Sample:

- line midpoints;
- long-edge intervals;
- convex-hull boundary points;
- concave corners;
- contour points;
- cavity boundaries.

## 11.4 Region anchors

Detect:

- closed-loop interior;
- holes;
- free cavities;
- rectangular regions;
- circular regions;
- unoccupied interior regions;
- nearby exterior free regions.

## 11.5 Structural anchors

Infer:

- axis of shaft-like geometry;
- bolt-circle centre;
- open connector;
- gap entrance;
- repeated-pattern axis;
- socket-like region;
- wall opening;
- bridge-compatible endpoints.

## 11.6 Anchor record

```json
{
  "anchor_id": "a_001",
  "component_id": "c_002",
  "type": "boundary_point",
  "position": [0.3, 0.8],
  "direction": [1.0, 0.0],
  "normal": [0.0, 1.0],
  "primitive_ids": ["p_017"],
  "local_scale": 0.12,
  "compatible_relations": [
    "endpoint_to_curve",
    "tangent",
    "shared_edge"
  ]
}
```

## 11.7 Compatibility matrix

Implement an explicit matrix mapping host and donor anchor types to valid interaction operators.

No operator may use incompatible anchor pairs.

---

# 12. Phase 5 — Interaction graph generation

## 12.1 Graph definition

Create a composition interaction graph:

\[
H = (N, R)
\]

where:

- each node is a component;
- each relation is a typed interaction;
- the graph is connected.

## 12.2 Component-count curriculum

Use configurable ranges:

```yaml
easy:
  min_components: 2
  max_components: 3

medium:
  min_components: 3
  max_components: 5

hard:
  min_components: 5
  max_components: 8

very_hard:
  min_components: 8
  max_components: 12
```

Do not always use exactly 6, 7, or 8 components.

## 12.3 Interaction graph rules

- create a spanning tree first;
- add optional cycle or bridge relations;
- every node must have degree at least 1;
- at least one nontrivial interaction must exist;
- hard samples should have multiple interaction types;
- prevent too many leaves that merely touch the host weakly;
- restrict repeated use of the same operator;
- enforce source-domain compatibility profiles.

## 12.4 Domain modes

Implement:

- `same_domain`;
- `cross_domain_geometry`;
- `adversarial_complexity`.

Suggested initial default:

```yaml
same_domain: 0.65
cross_domain_geometry: 0.25
adversarial_complexity: 0.10
```

Treat these as configurable starting values.

## 12.5 Interaction distribution

Initial operator pool:

- endpoint join;
- endpoint-to-curve T-junction;
- collinear extension;
- perpendicular attachment;
- tangent attachment;
- concentric placement;
- containment;
- insertion into gap;
- shared edge;
- connected crossing;
- disconnected crossing;
- partial occlusion;
- bridge;
- repetition.

## 12.6 Graph acceptance

Reject sampled graphs with:

- disconnected nodes;
- incompatible required relations;
- impossible anchor demands;
- excessive repetition;
- too many occlusion-only relations;
- no cycle or bridge in very-hard mode;
- topology outside configured complexity bounds.

---

# 13. Phase 6 — Geometric placement

## 13.1 Default transform

Use:

\[
T(x) = s R x + t
\]

where:

- `s` is uniform scale;
- `R` is rotation;
- optional reflection is explicitly configured;
- `t` is translation.

## 13.2 Placement solver

For each interaction:

1. choose host anchor;
2. choose donor anchor;
3. solve transformation;
4. calculate geometric residual;
5. check clearance and overlap;
6. detect unintended intersections;
7. evaluate visibility;
8. accept or retry.

## 13.3 Scale constraints

Use relative-scale ranges based on:

- host local scale;
- target complexity;
- source component size;
- real patent distributions.

Avoid extremely tiny donors that contribute meaningless visible geometry.

Avoid donors so large that they obscure the host completely.

## 13.4 Rotation policy

Support:

- arbitrary rotation;
- dominant-angle snapping;
- 0/45/90-degree engineering orientations;
- alignment to host tangent;
- alignment to host normal;
- axis alignment.

Mix exact and noisy angle modes.

## 13.5 Reflection policy

Reflection may be useful for diversity, but:

- preserve primitive validity;
- record handedness change;
- disable for source types where semantics would become invalid;
- keep configurable by dataset.

## 13.6 Placement retries

For each donor:

- try several candidate components;
- try several anchor pairs;
- try several valid transformations;
- log all rejection reasons;
- abort the sample only after configurable exhaustion.

---

# 14. Phase 7 — Implement interaction operators

Implement and test each operator independently.

## 14.1 Endpoint-to-endpoint

Behavior:

- align donor endpoint with host endpoint;
- optionally align tangents;
- create smooth or sharp junction;
- merge coincident endpoints.

Acceptance:

- endpoint residual below tolerance;
- donor contributes visible geometry;
- no unintended duplicate stroke.

## 14.2 Endpoint-to-curve T-junction

Behavior:

- place donor endpoint on host curve interior;
- split host primitive at parameter `u`;
- create T-junction;
- preserve source intervals.

Acceptance:

- contact lies away from host endpoints unless allowed;
- junction degree correct;
- no unresolved near-overlap.

## 14.3 Collinear extension

Variants:

- exact continuation;
- small gap;
- partial overlap;
- dashed continuation.

Acceptance:

- direction difference below tolerance;
- lateral offset below tolerance;
- overlap policy satisfied.

## 14.4 Parallel or perpendicular attachment

Behavior:

- align donor structural edge to host edge;
- preserve configured spacing;
- optionally connect endpoints.

## 14.5 Tangent attachment

Support:

- line-circle;
- line-arc;
- arc-line;
- arc-arc.

Acceptance:

- tangent residual;
- contact residual;
- curvature compatibility;
- no self-intersection.

## 14.6 Concentric placement

Support:

- circle-circle;
- circular component around axis;
- ring structures;
- radial arrays.

## 14.7 Containment

Behavior:

- place donor inside host closed region;
- satisfy margin constraints;
- optional contact or tangency;
- preserve visibility.

## 14.8 Gap insertion

Behavior:

- detect opening;
- fit donor into opening;
- allow one or two boundary contacts;
- reject if donor must distort.

## 14.9 Shared edge

Behavior:

- align donor edge with host edge;
- deduplicate visible geometry;
- store multiple source parents;
- store shared-edge relation.

## 14.10 Connected crossing

Behavior:

- split both curves;
- create connected junction;
- update graph degrees.

## 14.11 Disconnected crossing

Behavior:

- preserve geometric intersection;
- do not create topology connection;
- optionally add bridge or gap convention.

## 14.12 Partial occlusion

Behavior:

- place donor at z-order;
- clip visible intervals;
- retain amodal geometry;
- create occlusion-boundary metadata.

## 14.13 Bridge interaction

Behavior:

- connect donor to two existing components;
- create a cycle or structural span;
- validate both contacts simultaneously.

## 14.14 Repetition

Support:

- linear array;
- circular array;
- grid;
- curve-following array;
- alternating orientation.

Store instance lineage.

## 14.15 Operator test suite

Each operator requires:

- unit geometry test;
- topology test;
- provenance test;
- render test;
- invalid-placement test;
- regression preview.

---

# 15. Phase 8 — Topology update and provenance

## 15.1 Intersection resolution

After every placement:

1. compute all new intersections;
2. distinguish intended from unintended intersections;
3. classify connectedness;
4. split primitives where necessary;
5. create junctions;
6. update visible intervals;
7. preserve parent-child mapping.

## 15.2 Parent-child intervals

When a primitive is split, store:

```json
{
  "source_primitive_id": "p_source_12",
  "children": [
    {
      "primitive_id": "p_new_31",
      "source_interval": [0.0, 0.42]
    },
    {
      "primitive_id": "p_new_32",
      "source_interval": [0.42, 1.0]
    }
  ]
}
```

## 15.3 Coincident geometry

If two source curves become identical:

- render only one visible primitive;
- retain all parent source IDs;
- retain interaction relation;
- avoid doubled line width.

## 15.4 Visible versus amodal graph

Save:

- complete transformed component geometry;
- visible geometry after occlusion;
- visible intervals;
- occluded intervals;
- z-order;
- occluder IDs.

The default Puhachov target is the visible graph.

## 15.5 Unintended intersections

Each unintended intersection must trigger one of:

- reject placement;
- classify as disconnected crossing;
- convert to intended connection if allowed;
- adjust transform.

Do not silently accept ambiguous topology.

---

# 16. Phase 9 — Patent-specific semantic layers

Geometric complexity alone is insufficient.

## 16.1 Hatching

Generate:

- single-angle hatching;
- cross-hatching;
- clipped hatching;
- hatching touching boundaries;
- hatching interrupted by numerals;
- varied spacing;
- varied stroke width;
- faded and broken hatch lines.

Store exact hatch vectors and masks.

## 16.2 Dashed and centre lines

Apply dash patterns to analytic primitives.

Store both:

- logical primitive;
- rendered visible dash segments.

Support:

- straight dashed lines;
- dashed circles;
- dashed arcs;
- dash-dot centre lines;
- irregular missing dashes.

## 16.3 Reference numerals

Generate:

- single digits;
- multi-digit labels;
- alphanumeric labels;
- varied fonts;
- small rotations;
- labels near or inside geometry;
- labels overlapping hatching.

Store text boxes and semantic masks.

## 16.4 Leaders and arrowheads

Generate:

- straight leaders;
- bent leaders;
- multi-segment leaders;
- open arrowheads;
- filled arrowheads;
- leader contact;
- near-contact;
- leader crossing object geometry.

## 16.5 Dimensions

Generate:

- linear dimensions;
- angular dimensions;
- radius dimensions;
- diameter dimensions;
- extension lines;
- measurement text;
- arrowheads;
- dimension arcs.

## 16.6 Text boxes

Generate:

- text inside rectangles;
- glyphs touching boundaries;
- nested boxes;
- boxes connected by lines;
- mixed diagram and object regions.

## 16.7 Semantic output

Render separate masks for:

- object visible;
- hidden/centre;
- hatch;
- leader/dimension;
- text/numerals;
- text boxes;
- background.

---

# 17. Phase 10 — Raster rendering and degradation

## 17.1 Clean rendering

Render:

- vector-perfect antialiased image;
- binary image;
- grayscale image;
- per-layer masks;
- primitive-ID map;
- component-ID map;
- junction heatmaps.

## 17.2 Resolution profiles

Support:

- 512;
- 768;
- 1024;
- 1536;
- 2048;
- native large-format.

## 17.3 Patent degradation

Apply seeded degradations:

- thresholding;
- blur;
- sharpening;
- one-bit conversion;
- JPEG artefacts;
- CCITT-like compression simulation;
- line dropout;
- broken strokes;
- local fading;
- duplicated contours;
- ink bleed;
- speckles;
- salt-and-pepper noise;
- skew;
- small perspective warp;
- nonuniform illumination;
- crop truncation;
- local occlusion;
- scanner background.

## 17.4 Ground-truth transforms

If the raster is geometrically warped:

- apply the same exact transform to vector labels;
- convert primitive type when required;
- or retain a sampled polyline target only in explicitly configured degradation experiments.

Default production data should avoid transformations that destroy analytic primitive identity.

---

# 18. Phase 11 — Real-patent statistical calibration

## 18.1 Build real target profiles

From the clean and audited patent corpus, compute:

- foreground ratio;
- stroke-width distribution;
- line-orientation histogram;
- local line density;
- connected-component distribution;
- component-size distribution;
- primitive count estimates;
- junction density;
- text ratio;
- hatch ratio;
- dash ratio;
- local clutter;
- whitespace distribution;
- image aspect ratio;
- semantic-layer prevalence.

## 18.2 Graph-level target profiles

Estimate:

- node count;
- edge count;
- degree histogram;
- connected-component count;
- loop count;
- crossing count;
- tangent count;
- concentric-circle count;
- isolation ratio;
- graph diameter;
- local clustering.

## 18.3 Interaction target profiles

Estimate or define:

- component count;
- number of contact interactions;
- containment frequency;
- crossing frequency;
- partial occlusion ratio;
- repeated-pattern frequency;
- relative scale distribution;
- visible-to-amodal ratio.

## 18.4 Realism score

Implement:

```text
realism_score =
    complexity_similarity
  + topology_similarity
  + density_similarity
  + semantic_style_similarity
  + interaction_similarity
  - invalidity_penalty
```

Weights must be configurable.

## 18.5 Synthetic-real classifier

Train a diagnostic classifier.

Use it to discover obvious synthetic artefacts such as:

- fixed component count;
- uniform whitespace;
- perfect line widths;
- unnatural intersections;
- repeated source patterns;
- excessive symmetry;
- uniform annotation placement.

Do not use the classifier as the only quality measure.

---

# 19. Phase 12 — Quality gates

Every sample must pass all mandatory gates.

## 19.1 Geometry gate

Reject:

- zero-length lines;
- invalid circles;
- invalid arc intervals;
- invalid Béziers;
- NaN or infinite coordinates;
- singular transforms;
- unresolved intersections.

## 19.2 Interaction gate

Reject:

- disconnected interaction graph;
- donor without meaningful interaction;
- unsatisfied relation residual;
- effectively side-by-side layout;
- excessive isolated components;
- repeated weak contacts only.

## 19.3 Topology gate

Reject:

- wrong junction degree;
- unintended connection;
- missing intended connection;
- invalid closed loops;
- unresolved crossing semantics;
- excessive dangling fragments.

## 19.4 Visibility gate

Reject or regenerate when:

- donor contributes less than minimum visible geometry;
- too much geometry is fully occluded;
- visible target becomes ambiguous;
- one component dominates nearly all foreground.

## 19.5 Density gate

Reject when:

- foreground nearly saturates region;
- primitive density is outside target range;
- annotations dominate object geometry;
- local congestion is unrealistic.

## 19.6 Round-trip gate

Rasterize the visible graph and compare with the clean raster.

Require:

- near-perfect foreground agreement;
- minimal Chamfer distance;
- no missing layer;
- no duplicated coincident stroke.

## 19.7 Provenance gate

Every primitive must trace to:

- source primitive;
- generated annotation;
- or explicit interaction-generated primitive.

## 19.8 Realism gate

Reject samples far outside configured real-patent profiles.

## 19.9 Quality report

Save per sample:

```json
{
  "accepted": true,
  "geometry_score": 1.0,
  "interaction_score": 0.93,
  "topology_score": 0.96,
  "visibility_score": 0.88,
  "density_score": 0.90,
  "roundtrip_score": 0.999,
  "realism_score": 0.84,
  "warnings": []
}
```

---

# 20. Phase 13 — Source splits and leakage prevention

## 20.1 Split sources before composition

Create:

```text
source_train_pool
source_validation_pool
source_test_pool
```

A synthetic sample must use components from only one source split.

## 20.2 Group related sources

Keep together:

- all views of one CAD model;
- related SketchGraphs variants;
- repeated ArchCAD instances from the same source design;
- CAD-VGDrawing views of the same object;
- duplicates and near duplicates.

## 20.3 Composite split assignment

Assign a composite based on its source pool.

Never randomly split generated composites after generation if their source components cross split boundaries.

## 20.4 Deduplication

Use:

- exact hashes;
- canonical graph hashes;
- component-ID combinations;
- perceptual hashes;
- rendered mask hashes;
- interaction-graph signatures.

---

# 21. Phase 14 — Generation curricula

## 21.1 Level 0: style-only

One source drawing plus:

- line-style variation;
- hatching;
- leaders;
- numerals;
- dimensions;
- degradation.

Purpose:

- patent appearance adaptation without composition.

## 21.2 Level 1: simple composition

2–3 components.

Allowed interactions:

- endpoint join;
- containment;
- concentric placement;
- collinear extension.

## 21.3 Level 2: medium composition

3–5 components.

Add:

- T-junction;
- tangent;
- shared edge;
- connected crossing;
- limited occlusion.

## 21.4 Level 3: hard composition

5–8 components.

Add:

- cycles;
- bridging;
- repeated patterns;
- nested containment;
- multiple semantic layers;
- dense local regions.

## 21.5 Level 4: very hard/adversarial

8–12 components.

Include:

- dashed bolt circles;
- hatching touching boundaries;
- text inside boxes;
- angle dimensions;
- leader networks;
- partial occlusion;
- dense PCB-like structures;
- scan degradation.

## 21.6 Curriculum manifest

Every sample must store:

- curriculum level;
- component count;
- interaction count;
- operator histogram;
- semantic-layer histogram;
- difficulty score.

---

# 22. Phase 15 — Track B CAD-valid composition

## 22.1 CAD component library

Build from:

- CAD-VGDrawing models;
- DeepCAD;
- Fusion 360 Gallery;
- simple generated CadQuery solids;
- validated STEP files.

## 22.2 Valid composition operations

Use:

- rigid assembly;
- union;
- subtraction;
- intersection where meaningful;
- hole insertion;
- repeated feature arrays;
- sketch addition;
- extrusion;
- revolve;
- component placement with assembly transforms.

## 22.3 Preserve targets

Store:

- original component CAD programs;
- assembly transforms;
- boolean operation history;
- final CAD program;
- final STEP;
- mesh preview;
- orthographic views;
- exact projected vectors.

## 22.4 Verification

Every Track-B sample must pass:

- CAD program execution;
- non-empty solid;
- valid topology;
- finite bounding box;
- successful STEP export;
- successful projection;
- vector/raster round-trip;
- multi-view consistency.

## 22.5 Patentization

Apply the same semantic and degradation layers to generated views.

Do not allow patent annotations to modify the CAD target.

---

# 23. Phase 16 — Pilot dataset

Do not start with millions of samples.

Generate:

```text
5,000 Level-0 samples
5,000 Level-1 samples
5,000 Level-2 samples
5,000 Level-3 samples
2,000 Level-4 samples
1,000 Track-B CAD-valid samples
```

Use smaller counts for the first smoke pilot if necessary.

## 23.1 Pilot inspection

Create HTML viewer with:

- source components;
- source transforms;
- interaction graph;
- clean raster;
- degraded raster;
- visible vector overlay;
- amodal overlay;
- semantic masks;
- primitive IDs;
- junction IDs;
- quality scores;
- rejection reason for failed candidates.

## 23.2 Manual audit

Audit at least:

- 100 accepted samples per level;
- 100 rejected samples;
- all interaction operators;
- all semantic layers;
- all source datasets;
- all difficulty levels.

## 23.3 Pilot acceptance

Proceed to larger generation only when:

- no side-by-side composites pass;
- topology errors are rare;
- provenance is complete;
- raster/vector round-trip is near perfect;
- interaction variety is visible;
- synthetic drawings resemble complex technical drawings;
- no source leakage is found.

---

# 24. Phase 17 — Puhachov integration

## 24.1 Training targets

Provide:

- raster image;
- endpoint heatmaps;
- corner heatmaps;
- T/X/Y junction heatmaps;
- tangent points;
- line-arc transition points;
- object mask;
- hidden/centre mask;
- hatch mask;
- leader/dimension mask;
- text mask;
- primitive-type map;
- visible vector graph;
- optional amodal graph.

## 24.2 Training mixture

Initial configurable mix:

```yaml
original_simple_data: 0.20
style_only_synthetic: 0.20
simple_composites: 0.15
medium_composites: 0.20
hard_composites: 0.15
real_clean_pseudo: 0.05
manual_real_hard: 0.05
```

Do not treat this as fixed.

## 24.3 Failure-targeted sampling

Oversample samples containing:

- dashed circles and arcs;
- hatch-boundary contact;
- text boxes;
- dimension arcs;
- dense detail;
- disconnected crossings;
- partial occlusion;
- shared edges.

## 24.4 Model-selection metrics

Use:

- endpoint F1;
- junction F1;
- primitive F1;
- topology F1;
- fragmentation count;
- isolation score;
- semantic-layer mIoU;
- raster Chamfer;
- invalid-output rate.

---

# 25. Phase 18 — Drawing2CAD and Free2CAD integration

## 25.1 Drawing2CAD

Use Track-B samples only for CAD supervision.

Input options:

- exact vectors;
- predicted vectors;
- noisy vectors;
- semantic-layer-aware vectors.

Train with controlled vectorization errors:

- missing primitives;
- duplicate primitives;
- endpoint noise;
- wrong layer;
- broken dashed structures;
- extra annotation strokes.

## 25.2 Free2CAD-style training

Use:

- valid CAD sketch or program sequences;
- simulated drawing orders;
- graph-based component representations;
- Track-B valid compositions.

Do not create fake operation sequences for Track-A composites.

## 25.3 Shared front-end

Track-A samples can still improve the shared raster-to-vector model used before CAD reconstruction.

---

# 26. Phase 19 — Required ablations

Run:

| ID | Data strategy |
|---|---|
| B0 | Original datasets only |
| B1 | Original + patent degradation |
| B2 | Side-by-side composition |
| B3 | Random overlap |
| B4 | Interaction-aware composition |
| B5 | Interaction-aware + patent layers |
| B6 | Interaction-aware + patent layers + real calibration |
| B7 | B6 + real pseudo-labels |
| B8 | B7 + manual hard positives |

## 26.1 Evaluation data

Use:

- frozen 100-sample pilot;
- accepted-sample audit;
- hard-fragmentation subset;
- invalid-content subset;
- larger C/D-inclusive holdout.

## 26.2 Metrics

Report:

- Stage-2 acceptance;
- postvec clean rate;
- invalid-content false acceptance;
- valid-drawing rejection;
- endpoint F1;
- junction F1;
- primitive F1;
- topology score;
- fragmentation;
- isolation;
- circle and arc accuracy;
- semantic-layer accuracy;
- runtime.

## 26.3 Key proof

The generator is successful only if interaction-aware composition outperforms:

- side-by-side composition;
- random overlap;
- patent degradation alone.

---

# 27. Phase 20 — CLI

Implement:

```bash
patentvec source ingest --dataset sketchgraphs
patentvec source ingest --dataset cad_vgdrawing
patentvec components build-index
patentvec anchors build
patentvec compose generate --config configs/composition/pilot.yaml
patentvec compose validate --manifest output/manifest.parquet
patentvec compose viewer --manifest output/manifest.parquet
patentvec compose release --version v1
patentvec cad-compose generate --config configs/composition/cad_v1.yaml
patentvec train puhachov --config configs/training/puhachov_compose.yaml
patentvec evaluate ablation --config configs/evaluation/compose_ablation.yaml
```

---

# 28. Phase 21 — Configuration examples

## 28.1 Composition config

```yaml
seed: 42

track: compose2d
difficulty: hard
count: 5000

component_count:
  min: 5
  max: 8

domain_mode:
  same_domain: 0.65
  cross_domain_geometry: 0.25
  adversarial_complexity: 0.10

interaction_weights:
  endpoint_join: 0.10
  t_junction: 0.12
  collinear_extension: 0.08
  tangent: 0.10
  concentric: 0.10
  containment: 0.12
  insertion: 0.08
  shared_edge: 0.06
  connected_crossing: 0.06
  disconnected_crossing: 0.04
  occlusion: 0.06
  bridge: 0.04
  repetition: 0.04

requirements:
  connected_graph: true
  minimum_cycle_count: 1
  minimum_visible_fraction_per_component: 0.20
  maximum_global_occlusion: 0.45
  forbid_side_by_side: true
```

## 28.2 Patent-layer config

```yaml
hatching:
  probability: 0.40

centerlines:
  probability: 0.35

leaders:
  probability: 0.55

reference_numerals:
  probability: 0.60

dimensions:
  probability: 0.30

text_boxes:
  probability: 0.20

degradation:
  profile: patent_scan_medium
```

---

# 29. Phase 22 — Testing

## 29.1 Unit tests

Test:

- transforms;
- intersections;
- primitive splitting;
- anchor compatibility;
- each interaction operator;
- visibility intervals;
- provenance;
- semantic generators;
- rasterization;
- quality gates.

## 29.2 Integration tests

Test:

- one SketchGraphs component plus one CAD-VGDrawing donor;
- one containment example;
- one T-junction example;
- one tangent example;
- one shared-edge example;
- one occlusion example;
- one hard patentized composite;
- one Track-B CAD assembly.

## 29.3 Regression fixtures

Include:

- dashed bolt circle;
- hatched section;
- text inside box;
- angle dimension;
- dense PCB-like region;
- connected crossing;
- disconnected crossing;
- bridge relation;
- multi-component hard composite.

---

# 30. Phase 23 — Dataset release

## 30.1 Release layout

```text
PatentVec-Compose2D-v1/
├── dataset_card.md
├── LICENSES/
├── manifests/
│   ├── train.parquet
│   ├── validation.parquet
│   ├── test.parquet
│   └── all.parquet
├── samples/
│   └── <prefix>/<sample_id>/
│       ├── sample.json
│       ├── clean.png
│       ├── degraded.png
│       ├── visible.svg
│       ├── amodal.svg
│       ├── masks.npz
│       ├── preview.png
│       └── interaction_graph.json
└── reports/
```

## 30.2 Dataset card

Document:

- purpose;
- source datasets;
- licenses;
- generation method;
- interaction grammar;
- transforms;
- ground-truth definitions;
- visible versus amodal labels;
- known synthetic biases;
- split policy;
- quality gates;
- intended uses;
- limitations;
- version history.

---

# 31. Milestones and gates

## M0 — Bootstrap

Deliver:

- repository;
- schema;
- tests;
- one rendered primitive graph.

Gate:

- schema and round-trip tests pass.

## M1 — Source adapters

Deliver:

- SketchGraphs;
- CAD-VGDrawing;
- ArchCAD;
- FloorPlanCAD sample adapters.

Gate:

- source montages and validation reports pass.

## M2 — Component library

Deliver:

- extraction;
- descriptors;
- component index;
- split-safe storage.

Gate:

- components are coherent and searchable.

## M3 — Anchors

Deliver:

- point;
- boundary;
- region;
- structural anchors;
- compatibility matrix.

Gate:

- anchor visualizations are correct.

## M4 — Core interactions

Deliver first:

- endpoint join;
- T-junction;
- containment;
- concentric placement;
- tangent attachment.

Gate:

- topology and provenance tests pass.

## M5 — Full composition engine

Deliver:

- graph sampler;
- donor retrieval;
- placement;
- bridge/cycle relations;
- visibility;
- rejection logic.

Gate:

- no side-by-side samples pass manual audit.

## M6 — Patent semantic layers

Deliver:

- hatching;
- dashes;
- leaders;
- numerals;
- dimensions;
- text boxes.

Gate:

- semantic masks and vector labels are exact.

## M7 — Realism calibration

Deliver:

- real patent profiles;
- realism score;
- synthetic-real diagnostic.

Gate:

- generator no longer has obvious fixed signatures.

## M8 — Pilot dataset

Deliver:

- at least 20,000 Track-A pilot samples;
- viewer;
- audit;
- validation report.

Gate:

- accepted manual audit and round-trip quality.

## M9 — Puhachov experiment

Deliver:

- training adapter;
- baseline;
- ablations;
- real-patent evaluation.

Gate:

- interaction-aware data improves real hard cases.

## M10 — Track-B CAD generation

Deliver:

- valid assemblies;
- projections;
- CAD targets;
- verification.

Gate:

- every accepted CAD sample executes and reprojects correctly.

## M11 — Release

Deliver:

- versioned datasets;
- documentation;
- checksums;
- model integration;
- final report.

---

# 32. First vertical slice

Codex should implement one complete example before scaling.

## Inputs

- one SketchGraphs component;
- one CAD-VGDrawing component.

## Composition

- detect anchors;
- create endpoint-to-curve T-junction;
- apply similarity transform;
- split host primitive;
- create junction;
- preserve provenance;
- add one dashed centre line;
- add one leader and numeral;
- add hatching to one closed region.

## Outputs

- clean SVG;
- degraded PNG;
- visible graph JSON;
- amodal graph JSON;
- semantic masks;
- interaction graph;
- preview;
- quality report.

## Gate

The vertical slice is complete only when:

- render matches vector labels;
- provenance is complete;
- topology is correct;
- no source primitive is lost;
- generated annotations are separate semantic layers.

---

# 33. Immediate Codex task list

- [ ] Create repository structure.
- [ ] Add schema models.
- [ ] Add schema tests.
- [ ] Implement primitive geometry utilities.
- [ ] Implement similarity transforms.
- [ ] Implement intersection and splitting utilities.
- [ ] Implement SketchGraphs sample adapter.
- [ ] Implement CAD-VGDrawing sample adapter.
- [ ] Extract connected components.
- [ ] Detect endpoints and boundary anchors.
- [ ] Implement anchor compatibility.
- [ ] Implement endpoint join.
- [ ] Implement endpoint-to-curve T-junction.
- [ ] Implement containment.
- [ ] Implement concentric placement.
- [ ] Implement tangent attachment.
- [ ] Add provenance tracking.
- [ ] Add visible/amodal graph support.
- [ ] Implement clean raster renderer.
- [ ] Implement dashed-line generator.
- [ ] Implement hatching generator.
- [ ] Implement leader and numeral generator.
- [ ] Build the first vertical slice.
- [ ] Create HTML viewer.
- [ ] Generate 100 pilot composites.
- [ ] Audit all 100.
- [ ] Add quality gates.
- [ ] Generate 1,000 pilot composites.
- [ ] Train a small Puhachov smoke experiment.
- [ ] Compare against degradation-only and random-overlap baselines.
- [ ] Update `STATUS.md`.
- [ ] Commit milestone M4 only after tests pass.

---

# 34. Final success criteria

The project succeeds when:

1. generated drawings are interaction-rich and not side-by-side collages;
2. every component has at least one meaningful relation;
3. all primitive ground truth remains exact;
4. topology is updated correctly;
5. visible and amodal labels are available;
6. patent annotations are separate semantic layers;
7. source provenance is complete;
8. source splits prevent leakage;
9. generated complexity approaches real patent statistics;
10. Puhachov trained with the synthetic data improves on real hard patent drawings;
11. dashed lines, hatching, text boxes, dimensions, and dense details show measurable improvement;
12. invalid-content acceptance does not increase;
13. CAD training uses only CAD-valid Track-B samples;
14. the entire pipeline is reproducible from committed configuration files.

---

# 35. Final implementation principle

The generator must not answer the question:

> How can several drawings be placed on one canvas?

It must answer:

> How can exact vector motifs from several sources be assembled into one connected technical structure through controlled geometric relations, while preserving every primitive, junction, semantic layer, and source transformation?

That principle should guide every design and implementation decision in PatentVec Compose.
