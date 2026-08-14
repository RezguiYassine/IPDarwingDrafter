# Source license registry

Upstream terms were rechecked on 2026-07-31. Code licensing and data licensing
must remain separate:

| Source | Local role | Verified upstream evidence | Release action |
|---|---|---|---|
| SketchGraphs | Track-A vector components | Repository code is [MIT](https://github.com/PrincetonLIPS/SketchGraphs/blob/master/LICENSE). Its [dataset README](https://github.com/PrincetonLIPS/SketchGraphs#data) says original sketch creators retain copyright and points to Onshape Terms of Use section 1.g.ii. | Keep generated derivatives local to the research workspace. Obtain legal/written clearance before dataset redistribution or commercial release. |
| Drawing2CAD / CAD-VGDrawing | Track-A vector components and views | Repository code is [MIT](https://github.com/lllssc/Drawing2CAD/blob/main/LICENSE). The [dataset README](https://github.com/lllssc/Drawing2CAD#-dataset) says its CAD models come from DeepCAD; no separate dataset-specific license grant was found. | Keep generated derivatives local to the research workspace. Confirm with the dataset authors or counsel before redistribution or commercial release. |
| DeepCAD / Onshape public documents | Upstream source of CAD-VGDrawing models | DeepCAD repository code is MIT and its README says the data were parsed from Onshape public documents. [Onshape terms](https://www.onshape.com/en/legal/terms-of-use) grant broad rights to qualifying Public Documents but retain creator ownership and include account/date/license-tab conditions. | Do not infer a blanket source-by-source redistribution grant from repository code licenses. |

Generated samples retain source dataset, sample ID, split, view, component ID,
primitive ID, and all applied transforms. This registry is engineering metadata,
not a legal conclusion. A release is blocked until the applicable upstream terms
are reviewed and recorded in the dataset card. Local generation for internal
research and unrestricted release are separate decisions; this registry does not
authorize either commercial use or redistribution.
