"""Per-source-item accounting for SVG/DXF exports, independent of entity count."""

import hashlib
import io
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def primitive_budget_cost(primitive: dict) -> int:
    """Packaging disconnected strokes must not bypass the primitive-count gate."""
    return len(primitive["strokes"]) if primitive.get("type") == "hatch_strokes" else 1


def stroke_digest(strokes) -> str:
    coordinates = [[[round(float(x), 6), round(float(y), 6)] for x, y in stroke]
                   for stroke in strokes]
    return hashlib.sha256(json.dumps(coordinates, separators=(",", ":")).encode()).hexdigest()


def validate_primitive(primitive: dict) -> None:
    def points(values, minimum=1):
        if len(values) < minimum or any(
            len(p) != 2 or not all(math.isfinite(float(v)) for v in p) for p in values
        ):
            raise ValueError("Missing or nonfinite primitive points")

    kind = primitive["type"]
    if kind == "line":
        points([primitive["p1"], primitive["p2"]])
        if primitive["p1"] == primitive["p2"]:
            raise ValueError("Zero-length line")
    elif kind in {"circle", "arc", "ellipse"}:
        points([primitive["center"]])
        for key in (("a", "b") if kind == "ellipse" else ("radius",)):
            if not math.isfinite(float(primitive[key])) or float(primitive[key]) <= 0:
                raise ValueError("Invalid primitive radius")
        for key in (("start_angle", "end_angle") if kind == "arc" else ("angle",)):
            if not math.isfinite(float(primitive.get(key, 0))):
                raise ValueError("Nonfinite primitive angle")
        if kind == "arc" and (primitive["end_angle"] - primitive["start_angle"]) % 360 == 0:
            raise ValueError("Empty arc sweep; full circles need a circle primitive")
    elif kind in {"polyline", "polygon", "bezier"}:
        values = primitive.get("points") or []
        points(values, {"polyline": 2, "polygon": 3, "bezier": 4}[kind])
        if kind == "bezier" and (len(values) - 1) % 3:
            raise ValueError("Incomplete cubic Bezier control points")
    elif kind == "path":
        if not primitive.get("segments"):
            raise ValueError("Empty compound path")
        for segment in primitive["segments"]:
            if segment.get("type") not in {"line", "arc", "bezier"}:
                raise ValueError("Unsupported compound path segment")
            validate_primitive(segment)
    elif kind == "hatch_strokes":
        strokes = primitive.get("strokes") or []
        owners = primitive.get("stroke_source_indices") or []
        if primitive.get("style") != "hachure" or not strokes or len(strokes) != len(owners):
            raise ValueError("Invalid explicit hatch stroke mapping")
        for stroke in strokes:
            points(stroke, 2)
            if all(p == stroke[0] for p in stroke):
                raise ValueError("Zero-length hatch stroke")
        flat = [index for group in owners for index in group]
        if (any(not group for group in owners) or any(type(i) is not int or i < 0 for i in flat)
                or len(set(flat)) != len(flat)
                or sorted(flat) != sorted(primitive.get("source_hachure_indices", []))):
            raise ValueError("Incomplete or repeated explicit hatch ownership")
    elif kind == "hatch":
        points(primitive.get("boundary") or [], 3)
        spacing = float(primitive.get("spacing", 0))
        angles = primitive.get("angles") or []
        if not math.isfinite(spacing) or spacing <= 0 or not angles or not all(
            math.isfinite(float(angle)) for angle in angles
        ):
            raise ValueError("Invalid hatch pattern")
    else:
        raise ValueError(f"Unsupported primitive type: {kind}")


def start_item(audit: dict | None, collection: str, index: int) -> dict:
    item = {"index": index, "complete": False, "entity_ids": [], "errors": []}
    if audit is not None:
        audit.setdefault(collection, []).append(item)
    return item


def capture_entities(item: dict, entities: list, *, svg: bool, prefix: str) -> None:
    if svg:
        for number, entity in enumerate(entities):
            entity["id"] = f"{prefix}_{item['index']}_{number}"
        item["entity_ids"] = [entity["id"] for entity in entities]
    else:
        item["entity_ids"] = [entity.dxf.handle for entity in entities]


def finish_annotation(item: dict, annotation: dict) -> None:
    leaders = annotation.get("leader_lines") or []
    item["leaders_expected"] = len(leaders) or int("leader_to" in annotation)
    item["label_required"] = bool(annotation.get("text") or annotation.get("image_path")
                                  or annotation.get("source") == "stage0_references")
    item["complete"] = (
        not item["errors"] and item["leaders_written"] == item["leaders_expected"]
        and (not item["label_required"] or item["label"] in {"text", "crop"})
    )


def verify_serialized(format_name: str, path: Path, audit: dict) -> None:
    """Verify that every recorded entity survived serialization, not just creation."""
    if format_name == "svg":
        root = ET.parse(path).getroot()
        nodes = list(root.iter())
        ids = {node.get("id") for node in nodes if node.get("id")}
        if len(ids) != sum(bool(node.get("id")) for node in nodes):
            raise ValueError("Duplicate SVG entity IDs")
        import cairosvg
        from PIL import Image
        _, _, width, height = map(float, root.attrib["viewBox"].split())
        if not all(math.isfinite(v) and v > 0 for v in (width, height)):
            raise ValueError("Invalid SVG viewport")
        scale = 512 / max(width, height)
        png = cairosvg.svg2png(
            url=str(path), output_width=max(1, round(width * scale)),
            output_height=max(1, round(height * scale)), background_color="white",
        )
        with Image.open(io.BytesIO(png)) as image:
            audit["render_ink_pixels"] = sum(image.convert("L").histogram()[:255])
        if audit.get("primitives") and not audit["render_ink_pixels"]:
            raise ValueError("SVG rendered blank despite declared primitives")
        audit["render_valid"] = True
    else:
        import ezdxf
        document = ezdxf.readfile(path)
        checked = document.audit()
        if checked.errors or checked.fixes:
            raise ValueError("DXF required audit repairs or has errors")
        ids = {entity.dxf.handle for entity in document.modelspace()}
    recorded = []
    for collection in ("primitives", "annotations"):
        for item in audit.get(collection, []):
            recorded.extend(item["entity_ids"])
            if not set(item["entity_ids"]).issubset(ids):
                raise ValueError("Recorded export entities are missing from the file")
            if "stroke_geometry_sha256" in item:
                if format_name == "svg":
                    node_by_id = {node.get("id"): node for node in nodes}
                    path_data = node_by_id[item["entity_ids"][0]].get("d", "")
                    digest = hashlib.sha256(path_data.encode()).hexdigest()
                else:
                    strokes = []
                    for handle in item["entity_ids"]:
                        entity = document.entitydb[handle]
                        if (entity.dxftype() != "LWPOLYLINE" or entity.closed
                                or any(b[0] != 0 for b in entity.get_points("b"))):
                            raise ValueError("Explicit hatch stroke changed representation")
                        strokes.append(entity.get_points("xy"))
                    digest = stroke_digest(strokes)
                    if len(strokes) != item["stroke_count"]:
                        raise ValueError("Explicit hatch stroke missing from DXF")
                if digest != item["stroke_geometry_sha256"]:
                    raise ValueError("Serialized explicit hatch geometry differs from source")
    if len(set(recorded)) != len(recorded):
        raise ValueError("Export entity assigned to multiple source items")
    audit["serialized_valid"] = True
    audit["sha256"] = file_digest(path)
