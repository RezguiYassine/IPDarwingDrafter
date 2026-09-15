import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import ezdxf

from tools.batch_run import stage0_handle_references as s0
from tools.batch_run import stage4_export as s4


def _reference_document(tmp_path, text="12", kind="ocr_numeral"):
    gray = np.full((64, 64), 255, dtype=np.uint8)
    gray[12:24, 12:20] = 0
    return s0._build_reference_doc(
        sketch_id="refs", source_path=tmp_path / "input.tif",
        gray=gray, ink=gray < 128,
        labels=[{
            "bbox": [10, 10, 20, 20], "centroid": [20, 20],
            "text": text, "kind": kind,
            "ref_class": "numeral" if text else None,
            "confidence": 0.91 if text else None,
        }],
        mask_path=tmp_path / "mask.png",
        reference_free_path=tmp_path / "norefs.png",
        crop_dir=tmp_path / "crops", cfg={"crop_pad": 2},
        active_removal=True, removed_ink_ratio=0.01, repair_pixels=0,
        n_iterations=1, iteration_summaries=[], flagged=False, reason="",
    )


def _export_data(tmp_path, document):
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps(document))
    primitives = tmp_path / "primitives.json"
    primitives.write_text(json.dumps({
        "sketch_id": "refs", "image_size": [64, 64],
        "primitives": [], "annotations": [],
    }))
    attached = s0.attach_references_to_primitives(primitives, refs)
    return json.loads(attached.read_text())


@pytest.mark.parametrize("text,kind", [("12", "ocr_numeral"), ("A2", "ocr_reference")])
def test_ocr_reference_identity_survives_json_and_dxf(tmp_path, text, kind):
    doc = _reference_document(tmp_path, text, kind)
    doc["reference_labels"][0]["ref_class"] = "numeral" if text == "12" else "alphanumeric"
    data = _export_data(tmp_path, doc)
    annotation = data["annotations"][0]
    assert annotation["text"] == text
    assert annotation["ref_class"] == doc["reference_labels"][0]["ref_class"]
    assert annotation["confidence"] == 0.91
    output = tmp_path / "refs.dxf"
    s4._export_dxf_patent(data, output)
    texts = list(ezdxf.readfile(output).modelspace().query("MTEXT"))
    assert len(texts) == 1
    assert texts[0].plain_text() == text
    assert texts[0].dxf.char_height == 16
    assert texts[0].dxf.attachment_point == 5
    assert tuple(texts[0].dxf.insert) == (20.5, 43.5, 0.0)


def test_svg_prefers_crop_without_drawing_the_same_text_twice(tmp_path):
    data = _export_data(tmp_path, _reference_document(tmp_path))
    output = tmp_path / "refs.svg"
    s4._export_svg(data, output)
    root = ET.parse(output).getroot()
    assert len(root.findall(".//{*}image")) == 1
    assert root.findall(".//{*}text") == []


@pytest.mark.parametrize("explicit_text", [False, True])
def test_svg_can_render_text_without_a_duplicate_crop(tmp_path, explicit_text):
    data = _export_data(tmp_path, _reference_document(tmp_path))
    if explicit_text:
        data["annotations"][0]["svg_render_mode"] = "text"
    else:
        data["annotations"][0]["image_path"] = str(tmp_path / "missing.png")
    output = tmp_path / "refs.svg"
    s4._export_svg(data, output)
    root = ET.parse(output).getroot()
    assert root.findall(".//{*}image") == []
    assert [node.text for node in root.findall(".//{*}text")] == ["12"]


def test_unknown_reference_is_preserved_as_crop_not_invented_text(tmp_path):
    doc = _reference_document(tmp_path, text="", kind="cc_recovered")
    data = _export_data(tmp_path, doc)
    assert data["annotations"][0]["text"] == ""
    assert data["annotations"][0]["confidence"] is None
    data["annotations"][0]["svg_render_mode"] = "text"
    output = tmp_path / "refs.svg"
    s4._export_svg(data, output)
    root = ET.parse(output).getroot()
    assert len(root.findall(".//{*}image")) == 1
    assert root.findall(".//{*}text") == []
    dxf = tmp_path / "refs.dxf"
    s4._export_dxf_patent(data, dxf)
    assert len(ezdxf.readfile(dxf).modelspace().query("MTEXT")) == 0
