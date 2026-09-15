import json
import sqlite3

import ezdxf
import pytest

from tools.audit_preservation_run import audit


@pytest.fixture
def saved_run(tmp_path):
    with sqlite3.connect(tmp_path / "results.db") as connection:
        connection.execute("CREATE TABLE results (patent_id TEXT, sketch_id TEXT, status TEXT, s1_model_used TEXT)")
        connection.execute("INSERT INTO results VALUES ('EP1', 'F1', 'ok', 'passthrough_binary')")
    root = tmp_path / "EP1"
    for directory in ("primitives", "graphs", "references", "vectors"):
        (root / directory).mkdir(parents=True)
    (root / "graphs/F1_graph.json").write_text(json.dumps({"removed_hachures": [{"pixels": [[0, 0], [1, 1]]}]}))
    document = {"primitives": [{"style": "hachure", "source_hachure_indices": [0]}],
                "quality_metrics": {"hachure_coverage": {"source_edges": 1, "represented_edges": 1,
                                                           "unrepresented_indices": []}}}
    (root / "primitives/F1_primitives.json").write_text(json.dumps(document))
    (root / "references/F1_references.json").write_text(json.dumps({"active_removal": True, "reference_labels": []}))
    (root / "vectors/F1.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    ezdxf.new().saveas(root / "vectors/F1.dxf")
    return tmp_path


def test_preservation_audit_accepts_complete_unique_ownership(saved_run):
    report = audit(saved_run)
    assert report["errors"] == []
    assert report["totals"]["represented_hatch_edges"] == 1


@pytest.mark.parametrize("indices", [[], [0, 0], [1]])
def test_preservation_audit_catches_missing_duplicate_or_invalid_ownership(saved_run, indices):
    path = saved_run / "EP1/primitives/F1_primitives.json"
    document = json.loads(path.read_text())
    document["primitives"][0]["source_hachure_indices"] = indices
    path.write_text(json.dumps(document))
    assert any("ownership" in error for error in audit(saved_run)["errors"])


def test_preservation_audit_does_not_reinject_unremoved_labels(saved_run):
    crop = saved_run / "crop.png"
    crop.write_bytes(b"crop exists")
    path = saved_run / "EP1/references/F1_references.json"
    path.write_text(json.dumps({"active_removal": False,
                               "reference_labels": [{"text": "12", "crop_path": str(crop)}]}))
    assert audit(saved_run)["errors"] == []
