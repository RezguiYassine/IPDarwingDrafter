"""Patent sheets of independent figures, without invented cross-figure joins."""

import copy
from dataclasses import replace

import numpy as np

from .geometry import apply_points, transform_primitive
from .render import DEFAULT_STROKE_WIDTH, render_clean, render_masks
from .schema import CanonicalDrawing, Primitive, Component, SourceRecord, validate_drawing
from .quality import evaluate_quality
from .training import free2cad_arrays


def compose_sheet(drawings, sample_id, seed, canvas):
    if len(drawings) != 4 or len({d.split for d in drawings}) != 1:
        raise ValueError("A sheet requires four figures from one partition")
    sheet = CanonicalDrawing(sample_id=sample_id, track="compose2d", difficulty=drawings[0].difficulty,
                             seed=seed, split=drawings[0].split, canvas=[canvas, canvas],
                             sources=[], components=[], transforms=[], anchors=[], interactions=[],
                             primitives_visible=[], primitives_amodal=[], junctions_visible=[],
                             junctions_amodal=[], semantic_layers={})
    figures = []
    for number, source in enumerate(drawings):
        prefix = f"f{number+1}_"
        def pid(value):
            return prefix + value if value is not None else None
        matrix = np.array([[.45, 0, .02 + .5*(number % 2)],
                           [0, .45, .02 + .5*(number // 2)], [0, 0, 1.]])
        def point(value):
            return apply_points(np.asarray(value), matrix).tolist()
        known_ids = {x.primitive_id for x in source.primitives_visible + source.primitives_amodal}
        def metadata(value):
            if isinstance(value, str):
                return pid(value) if value in known_ids else value
            if isinstance(value, list):
                return [metadata(v) for v in value]
            if isinstance(value, dict):
                return {k: metadata(v) for k, v in value.items()}
            return value
        for layer in ("primitives_visible", "primitives_amodal"):
            for original in getattr(source, layer):
                item = transform_primitive(original, matrix, primitive_id=pid(original.primitive_id),
                                           component_id=pid(original.component_id), transform_id=pid(original.transform_id))
                item.parent_primitive_ids = [pid(v) for v in original.parent_primitive_ids]
                item.interaction_ids = [pid(v) for v in original.interaction_ids]
                item.style["stroke_width"] = .45 * original.style.get("stroke_width", DEFAULT_STROKE_WIDTH.get(original.semantic, .0015))
                if item.style.get("dasharray"):
                    item.style["dasharray"] = [v*.45 for v in item.style["dasharray"]]
                getattr(sheet, layer).append(item)
        for item in source.components:
            sheet.components.append(replace(item, component_id=pid(item.component_id),
                primitive_ids=[pid(v) for v in item.primitive_ids], transform_id=pid(item.transform_id),
                parent_component_id=pid(item.parent_component_id), descriptor=copy.deepcopy(item.descriptor)))
        for item in source.transforms:
            target = item.target_frame
            source_frame = item.source_frame
            for component in source.components:
                source_frame = source_frame.replace(component.component_id + ":local", pid(component.component_id) + ":local")
                target = target.replace(component.component_id + ":local", pid(component.component_id) + ":local")
            transformed = matrix @ np.asarray(item.matrix) if item.target_frame == "composite_normalized" else item.matrix
            sheet.transforms.append(replace(item, transform_id=pid(item.transform_id), matrix=np.asarray(transformed).tolist(),
                                             source_frame=source_frame, target_frame=target))
        for item in source.anchors:
            sheet.anchors.append(replace(item, anchor_id=pid(item.anchor_id), component_id=pid(item.component_id),
                                         position=point(item.position), primitive_ids=[pid(v) for v in item.primitive_ids],
                                         local_scale=item.local_scale*.45))
        for item in source.interactions:
            sheet.interactions.append(replace(item, interaction_id=pid(item.interaction_id),
                host_component_id=pid(item.host_component_id), donor_component_id=pid(item.donor_component_id),
                host_anchor_id=pid(item.host_anchor_id), donor_anchor_id=pid(item.donor_anchor_id),
                intended_contacts=[point(v) for v in item.intended_contacts], residual=item.residual*.45,
                metadata=metadata(item.metadata)))
        for layer in ("junctions_visible", "junctions_amodal"):
            for item in getattr(source, layer):
                getattr(sheet, layer).append(replace(item, junction_id=pid(item.junction_id), position=point(item.position),
                    primitive_ids=[pid(v) for v in item.primitive_ids], component_ids=[pid(v) for v in item.component_ids],
                    interaction_id=pid(item.interaction_id)))
        sheet.sources.extend(copy.deepcopy(source.sources))
        figures.append({"figure_id": prefix[:-1], "source_sample_id": source.sample_id,
                        "source_component_ids": [pid(c.component_id) for c in source.components if c.source_dataset != "generated"],
                        "matrix": matrix.tolist(), "processing_frame": "figure_local_normalized",
                        "processing": copy.deepcopy(source.processing)})
        label_id = prefix + "figure_label"
        component_id = prefix + "caption"
        caption = Primitive(primitive_id=label_id, kind="text", semantic="figure_label", component_id=component_id,
            source_dataset="generated", source_sample_id=sample_id, source_component_id=component_id,
            source_primitive_id=label_id, generated_by="patent_sheet_caption",
            geometry={"position": [.25 + .5*(number % 2), .49 + .5*(number // 2) - .008],
                      "text": f"FIG. {number+1}", "size": .009, "anchor": "middle"})
        sheet.components.append(Component(component_id, "generated", sample_id, component_id, sheet.split, [label_id]))
        sheet.primitives_visible.append(caption)
        sheet.primitives_amodal.append(copy.deepcopy(caption))
    for item in sheet.primitives_visible:
        sheet.semantic_layers.setdefault(item.semantic, []).append(item.primitive_id)
    sheet.processing = {"layout": "independent_four_figure_sheet", "figures": figures,
                        "physical_multiview_correspondence": False,
                        "generation_attempt": sum(d.processing.get("generation_attempt", 0) for d in drawings)}
    if validate_drawing(sheet):
        raise ValueError(validate_drawing(sheet))
    return sheet


class SheetGenerator:
    def __init__(self, source_pool, canvas):
        from .generator import ComplexPilotGenerator
        self.generator = ComplexPilotGenerator(source_pool, canvas=1024)
        self.canvas = canvas

    def generate_prepared(self, sample_id, seed, difficulty, max_attempts):
        """Compose an accepted four-figure sheet, resampling on sheet-level rejection.

        Per-figure generation already retries under ``max_attempts``, but a sheet
        can still fail a gate that only exists once the four figures share one
        canvas -- annotation/text collisions are the observed case. Raising on the
        first such rejection aborts the whole run, so sheets retry with a
        perturbed seed exactly as single drawings do. The attempt count is
        recorded, so a high rejection rate stays visible as evidence that
        annotation placement needs fixing rather than resampling.
        """
        from .generator import PreparedDrawing
        failures: list = []
        rejection_reasons: dict = {}
        attempts = max(1, int(max_attempts))
        for attempt in range(attempts):
            offset = seed + attempt * 7919
            try:
                figures = [self.generator.generate_prepared(f"{sample_id}_figure{i+1}",
                           offset + i*10000019, difficulty, max_attempts).drawing
                           for i in range(4)]
            except Exception as exc:
                # A single figure that exhausts its own attempts must not abort the
                # whole sheet: resample the quadruple instead, matching how one
                # rejected single drawing is retried rather than ending the run.
                failures = [f"figure generation: {type(exc).__name__}: {exc}"]
                rejection_reasons["figure generation"] = rejection_reasons.get("figure generation", 0) + 1
                continue
            drawing = compose_sheet(figures, sample_id, offset, self.canvas)
            clean, masks = render_clean(drawing), render_masks(drawing)
            report = evaluate_quality(drawing, clean=clean, masks=masks, difficulty_gate=difficulty)
            if report["accepted"]:
                drawing.processing["sheet_attempts"] = attempt + 1
                drawing.processing["sheet_rejection_reasons"] = rejection_reasons
                targets = free2cad_arrays(drawing, source_index=0, object_mask=masks["object"])
                return PreparedDrawing(drawing, clean, masks, report, targets)
            failures = report["failures"]
            # Keep why sheets were rejected, so the dominant gate stays measurable
            # instead of being hidden by the retry.
            for item in failures:
                reason = str(item).split(":")[0].strip()
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
        raise ValueError(f"Sheet failed quality after {attempts} attempts: {failures}")
