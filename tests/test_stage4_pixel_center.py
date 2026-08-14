import io
import json
import xml.etree.ElementTree as ET

import cairosvg
import cv2
import ezdxf
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

from tools import d2c_eval
from tools.rescore_d2c_stage4 import _rescore_one

_export_dxf_basic = d2c_eval.stage4_export._export_dxf_basic
_export_svg = d2c_eval.stage4_export._export_svg
_svg_path_continuous = d2c_eval.stage4_export._svg_path_continuous
_rasterize_svg_to_binary = d2c_eval._rasterize_svg_to_binary


def _line_data():
    return {
        "sketch_id": "pixel_center",
        "image_size": [32, 32],
        "stroke_width": 4.1,
        "primitives": [
            {
                "edge_id": 0,
                "type": "line",
                "p1": [4.0, 16.0],
                "p2": [28.0, 16.0],
                "confidence": 1.0,
            }
        ],
        "annotations": [],
    }


def test_svg_exports_raster_indices_at_pixel_centers(tmp_path):
    output = tmp_path / "line.svg"
    data = _line_data()

    assert _export_svg(data, output, default_sw=data["stroke_width"]) == 1

    root = ET.parse(output).getroot()
    geometry = next(
        child for child in root
        if child.tag.endswith("g") and "translate" in child.attrib.get("transform", "")
    )
    assert geometry.attrib["transform"] == "translate(0.5 0.5)"

    png = cairosvg.svg2png(
        url=str(output), output_width=32, output_height=32,
        background_color="white",
    )
    gray = np.array(Image.open(io.BytesIO(png)).convert("L"))
    binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    ys = np.where(skeletonize(binary < 128))[0]

    assert float(np.median(ys)) == 16.0


def test_dxf_uses_the_same_pixel_center_conversion(tmp_path):
    output = tmp_path / "line.dxf"

    assert _export_dxf_basic(_line_data(), output) == 1

    line = next(iter(ezdxf.readfile(output).modelspace().query("LINE")))
    assert tuple(line.dxf.start) == (4.5, 15.5, 0.0)
    assert tuple(line.dxf.end) == (28.5, 15.5, 0.0)


def test_stage4_only_rescore_uses_archived_primitives(tmp_path):
    folder_id = "0000_00000000"
    sketch_id = f"{folder_id}_Front"
    work_dir = tmp_path / folder_id
    primitives_dir = work_dir / "primitives"
    primitives_dir.mkdir(parents=True)
    data = _line_data()
    data["sketch_id"] = sketch_id
    (primitives_dir / f"{sketch_id}_primitives.json").write_text(
        json.dumps(data)
    )

    gt_svg = work_dir / "ground_truth.svg"
    gt_svg.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" '
        'viewBox="0 0 32 32"><rect width="32" height="32" fill="white"/>'
        '<line x1="4.5" y1="16.5" x2="28.5" y2="16.5" stroke="black" '
        'stroke-width="4.1" fill="none"/></svg>'
    )
    gt_raster = work_dir / f"{sketch_id}_input.png"
    _rasterize_svg_to_binary(gt_svg, gt_raster)

    result = _rescore_one(
        (str(tmp_path), "0000/00000000", "Front", str(gt_raster), None)
    )

    assert result["error"] is None
    assert result["metrics"]["chamfer_sym"] == 0.0


def test_compound_path_orients_first_arc_toward_following_segment():
    path = _svg_path_continuous([
        {
            "type": "arc",
            "center": [0.0, 0.0],
            "radius": 10.0,
            "start_angle": 0.0,
            "end_angle": 180.0,
        },
        {"type": "line", "p1": [10.0, 0.0], "p2": [20.0, 0.0]},
    ])

    assert path.startswith("M -10.000 0.000 A 10.000 10.000 0 0 0 10.000 0.000")
    assert "L -10.000" not in path
    assert path.endswith("L 20.000 0.000")
