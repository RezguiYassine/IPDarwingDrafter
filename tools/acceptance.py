"""Fail-closed acceptance records for training-target selection.

Execution success is not content or geometry validation. Geometry is measured
against source artifacts; content still requires an explicit versioned decision.
"""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tools import content_routing, geometry_validation


SCHEMA = "ap3-acceptance-v1"
POLICY_VERSION = "2026-09-19.1"
REQUIRED_CHECKS = {"execution", "stage_flags", "deployment", "artifacts", "exports",
                   "references", "hachures", "content", "geometry"}
ROW_FIELDS = ("patent_id", "sketch_id", "input_path", "status", "error",
              "s0_flagged", "s1_flagged", "s2_flagged", "s3_flagged", "s4_flagged",
              "s4_n_in", "s4_n_out")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def value_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def record_path(output_dir: Path, sketch_id: str) -> Path:
    return output_dir / "acceptance" / f"{sketch_id}_acceptance.json"


def _check(status: str = "pass", *reasons: str) -> dict:
    return {"status": status, "reason_codes": list(reasons)}


def decision(checks: dict) -> tuple[str, list[str]]:
    reasons = []
    states = []
    for name in sorted(REQUIRED_CHECKS):
        check = checks.get(name, _check("pending", f"{name}_validation_missing"))
        status = check.get("status")
        if status not in {"pass", "pending", "review", "fail", "error"}:
            return "error", [f"{name}_invalid_check_status"]
        states.append(status)
        if status != "pass":
            reasons.extend(check.get("reason_codes") or [f"{name}_{status}"])
    status = ("error" if "error" in states else "rejected" if "fail" in states
              else "review" if any(s in {"review", "pending"} for s in states) else "accepted")
    return status, sorted(set(reasons))


def _versioned_validation(name: str, validations: dict) -> dict:
    result = validations.get(name)
    if not isinstance(result, dict) or not result.get("validator") or not result.get("version"):
        return _check("pending", f"{name}_validation_pending")
    return dict(result)


def _check_export(report: dict, document: dict, artifacts: dict) -> dict:
    failures, reviews = [], []
    n_primitives = len(document["primitives"])
    annotations = document.get("annotations") or []
    if (report.get("schema") != "ap3-export-report-v1"
            or report.get("input_sha256") != artifacts["export_input"]["sha256"]
            or report.get("expected_primitives") != n_primitives
            or report.get("expected_annotations") != len(annotations)):
        return _check("error", "export_report_input_mismatch")
    for name in ("svg", "dxf"):
        fmt = report.get("formats", {}).get(name, {})
        if (not fmt.get("serialized_valid") or fmt.get("errors")
                or (name == "svg" and not fmt.get("render_valid"))
                or fmt.get("sha256") != artifacts[name]["sha256"]):
            failures.append(f"{name}_invalid_export")
        primitives = fmt.get("primitives", [])
        if ([p.get("index") for p in primitives] != list(range(n_primitives))
                or any(not p.get("complete") or p.get("errors") or not p.get("entity_ids")
                       for p in primitives)):
            failures.append(f"{name}_incomplete_primitives")
        output_annotations = fmt.get("annotations", [])
        if [a.get("index") for a in output_annotations] != list(range(len(annotations))):
            failures.append(f"{name}_incomplete_annotations")
            continue
        for item, annotation in zip(output_annotations, annotations):
            leaders = len(annotation.get("leader_lines") or []) or int("leader_to" in annotation)
            if (item.get("errors") or item.get("leaders_written") != leaders
                    or item.get("leaders_expected") != leaders):
                failures.append(f"{name}_incomplete_leaders")
            if annotation.get("text") and item.get("label") not in {"text", "crop"}:
                failures.append(f"{name}_missing_reference_label")
            elif (name == "dxf" and not annotation.get("text")
                  and annotation.get("kind") != UNIDENTIFIED
                  and annotation.get("source") == "stage0_references"):
                reviews.append("unknown_reference_text")
            elif not item.get("complete"):
                failures.append(f"{name}_incomplete_annotations")
    if failures:
        return _check("error", *sorted(set(failures + reviews)))
    return _check("review", *sorted(set(reviews))) if reviews else _check()


# A mark Stage 0 removed without identifying. It is not a reference, so it
# carries no text and none is demanded of it; the crop still preserves it.
UNIDENTIFIED = "unidentified_mark"


def _check_references(references: dict, document: dict, bind) -> dict:
    labels = references["reference_labels"]
    expected = labels if references.get("active_removal") else []
    annotations = [a for a in document.get("annotations", []) if a.get("source") == "stage0_references"]
    if len(expected) != len(annotations):
        return _check("error", "reference_annotation_count_mismatch")
    for index, (reference, annotation) in enumerate(zip(expected, annotations)):
        bind(f"reference_crop_{index}", Path(reference["crop_path"]))
        if Path(annotation.get("image_path", "")).resolve() != Path(reference["crop_path"]).resolve():
            return _check("error", "reference_crop_mismatch")
        for key in ("id", "text", "position", "bbox", "crop_bbox", "ref_class", "confidence"):
            if reference.get(key) != annotation.get(key):
                return _check("error", "reference_annotation_mismatch")
        leaders = [{k: leader.get(k) for k in ("p1", "p2", "leader_to")}
                   for leader in reference.get("leader_lines", []) if leader.get("removed", True)]
        if leaders != annotation.get("leader_lines", []):
            return _check("error", "reference_leader_mismatch")
    if labels and not references.get("active_removal"):
        return _check("review", "references_not_removed")
    if references.get("flagged") is not False:
        return _check("review", "reference_document_flagged_or_unverified")
    reviews = []
    # Only a label that claims to be a reference owes the record its text.
    claimed = [l for l in expected if l.get("kind") != UNIDENTIFIED]
    # Demotion trades a wrong label for a missing one, which is the safer
    # direction but not a free one: the mark is removed from the drawing and
    # contributes no text-to-drawing link. Counting it keeps that cost visible
    # in the record instead of letting a clean acceptance imply clean reading.
    demoted = sum(1 for l in expected if l.get("demoted_reason"))
    if any(not label.get("text") for label in claimed):
        reviews.append("unknown_reference_text")
    # A numeral the patent's own description never names is a misreading far
    # more often than a reference the drafter forgot: measured over the
    # curated cohort, 92.1% of readings that match the patent's numbering
    # series are named in the text against 24.7% of those that do not. The
    # contract could previously see a missing numeral but never a wrong one.
    if any(label.get("in_vocabulary") == "absent" for label in claimed):
        reviews.append("reference_not_in_description")
    result = _check("review", *reviews) if reviews else _check()
    result["labels_total"] = len(expected)
    result["labels_demoted_unreadable"] = demoted
    result["labels_unidentified"] = sum(1 for l in expected if l.get("kind") == UNIDENTIFIED)
    return result


def _check_hachures(graph: dict, document: dict) -> dict:
    edges = graph.get("removed_hachures") or []
    coverage = document.get("quality_metrics", {}).get("hachure_coverage")
    if not isinstance(coverage, dict):
        return _check("review", "hachure_coverage_missing")
    indices = [index for p in document["primitives"] if p.get("style") == "hachure"
               for index in p.get("source_hachure_indices", [])]
    if (any(type(index) is not int for index in indices)
            or Counter(indices) != Counter(range(len(edges)))
            or coverage.get("source_edges") != len(edges)
            or coverage.get("represented_edges") != len(edges)
            or coverage.get("unrepresented_indices") != []):
        return _check("error", "hachure_coverage_incomplete")
    return _check()


def record(row: dict, output_dir: Path, deployment_identity: dict | None,
           *, validations: dict | None = None, stage0_enabled: bool = True,
           content_manifest: dict | None = None) -> tuple[dict, Path]:
    """Bind decisions to exact artifacts; callers cannot turn missing checks into passes."""
    sketch = row["sketch_id"]
    checks = {name: _check("pending", f"{name}_validation_missing") for name in REQUIRED_CHECKS}
    checks["deployment"] = _check() if deployment_identity else _check("review", "deployment_unverified")
    checks["content"] = _versioned_validation("content", validations or {})
    status = row.get("status")
    checks["execution"] = (_check() if status == "ok" else
                           _check("fail", str(status)) if str(status).startswith("quality_gate_") else
                           _check("review", "pipeline_incomplete") if str(status).startswith("ok_stage") else
                           _check("error", f"execution_{status}"))
    flag_fields = [f"s{i}_flagged" for i in range(0 if stage0_enabled else 1, 5)]
    flags = [field for field in flag_fields if row.get(field)]
    missing_flags = [field for field in flag_fields if row.get(field) is None]
    checks["stage_flags"] = (_check("review", *(flags + [f"{k}_missing" for k in missing_flags]))
                             if flags or missing_flags else _check())
    artifacts, missing = {}, []

    def bind(name: str, path: Path):
        path = path.resolve()
        if not path.is_file():
            missing.append(f"missing_{name}")
            return
        artifacts[name] = {"path": str(path), "sha256": file_digest(path)}

    bind("source", Path(row["input_path"]))
    if status == "ok":
        locations = {
            "skeleton": output_dir / "cleaned" / f"{sketch}_skeleton.png",
            "cleaned": output_dir / "cleaned" / f"{sketch}_cleaned.png",
            "graph": output_dir / "graphs" / f"{sketch}_graph.json",
            "primitives": output_dir / "primitives" / f"{sketch}_primitives.json",
            "export_input": output_dir / "primitives" / f"{sketch}_primitives{'_with_refs' if stage0_enabled else ''}.json",
            "export_report": output_dir / "vectors" / f"{sketch}_export_report.json",
            "svg": output_dir / "vectors" / f"{sketch}.svg",
            "dxf": output_dir / "vectors" / f"{sketch}.dxf",
        }
        if stage0_enabled:
            locations.update({
                "references": output_dir / "references" / f"{sketch}_references.json",
                "reference_free": output_dir / "references" / f"{sketch}_norefs.png",
                "reference_mask": output_dir / "references" / f"{sketch}_references_mask.png",
            })
        for name, path in locations.items():
            bind(name, path)
        if not missing:
            try:
                def read(name):
                    return json.loads(Path(artifacts[name]["path"]).read_text())
                document, primitives, graph = read("export_input"), read("primitives"), read("graph")
                geometry_path = output_dir / "geometry" / f"{sketch}_geometry_report.json"
                geometry_report = geometry_validation.run(
                    locations["graph"], locations["primitives"], locations["skeleton"], geometry_path,
                )
                bind("geometry_report", geometry_path)
                checks["geometry"] = geometry_validation.acceptance_check(geometry_report)
                if document["primitives"] != primitives["primitives"]:
                    raise ValueError("Export input changed main/hatch primitives")
                checks["exports"] = _check_export(read("export_report"), document, artifacts)
                checks["references"] = (_check_references(read("references"), document, bind)
                                         if stage0_enabled else _check())
                checks["hachures"] = _check_hachures(graph, document)
                if not document["primitives"]:
                    checks["exports"] = _check("fail", "empty_geometry")
            except (KeyError, TypeError, ValueError, OSError) as exc:
                checks["exports"] = _check("error", "invalid_acceptance_evidence")
                checks["exports"]["detail"] = str(exc)
    if not (validations or {}).get("content"):
        # Ink routing needs the channel artifacts; the class decision needs only
        # the source, so a gated or crashed figure still records why its content
        # was or was not in scope.
        channels = {name: Path(artifacts[name]["path"])
                    for name in ("cleaned", "graph", "reference_mask")
                    if name in artifacts} if status == "ok" and not missing else None
        content_path = output_dir / "content" / f"{sketch}_content_report.json"
        try:
            content_report = content_routing.run(
                row["patent_id"], sketch, Path(row["input_path"]), content_path,
                manifest=content_manifest, channels=channels,
            )
            bind("content_report", content_path)
            checks["content"] = content_routing.acceptance_check(content_report)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            checks["content"] = _check("error", "content_validation_failed")
            checks["content"]["detail"] = str(exc)
    checks["artifacts"] = _check("error", *missing) if missing else _check()
    verdict, reasons = decision(checks)
    result = {
        "schema": SCHEMA, "policy_version": POLICY_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "patent_id": row["patent_id"], "sketch_id": sketch,
        "execution_status": status, "acceptance_status": verdict,
        "training_eligible": verdict == "accepted", "reason_codes": reasons,
        "deployment_identity": deployment_identity,
        "row_sha256": value_digest({field: row.get(field) for field in ROW_FIELDS}),
        "artifacts": artifacts, "checks": checks,
    }
    path = record_path(output_dir, sketch)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    return result, path


def eligibility_reason(row: dict, run_output: Path, selected_identity: dict | None) -> str | None:
    """Verify the sealed acceptance record before allowing a training manifest row."""
    if not selected_identity:
        return "acceptance_deployment_unverified"
    path = record_path(run_output / row["patent_id"], row["sketch_id"])
    if not path.is_file():
        return "acceptance_missing"
    try:
        if file_digest(path) != row.get("acceptance_sha256"):
            return "acceptance_record_hash_mismatch"
        report = json.loads(path.read_text())
        if report.get("schema") != SCHEMA or report.get("policy_version") != POLICY_VERSION:
            return "acceptance_policy_mismatch"
        if report.get("deployment_identity") != selected_identity:
            return "acceptance_deployment_mismatch"
        if report.get("row_sha256") != value_digest({field: row.get(field) for field in ROW_FIELDS}):
            return "acceptance_row_mismatch"
        verdict, reasons = decision(report["checks"])
        if (report.get("acceptance_status") != verdict or report.get("reason_codes") != reasons
                or report.get("training_eligible") is not (verdict == "accepted")):
            return "acceptance_inconsistent_decision"
        if verdict != "accepted":
            return f"acceptance_{verdict}"
        if (row.get("acceptance_status") != "accepted" or row.get("training_eligible") != 1
                or row.get("acceptance_policy_version") != POLICY_VERSION):
            return "acceptance_database_mismatch"
        for name in ("content", "geometry"):
            if _versioned_validation(name, report["checks"])["status"] != "pass":
                return f"acceptance_{name}_unverified"
        geometry = json.loads(Path(report["artifacts"]["geometry_report"]["path"]).read_text())
        if (geometry.get("schema") != geometry_validation.SCHEMA
                or geometry.get("validator") != geometry_validation.VALIDATOR
                or geometry.get("version") != geometry_validation.VERSION
                or geometry_validation.acceptance_check(geometry) != report["checks"]["geometry"]
                or any(geometry["inputs"][name] != report["artifacts"][name]
                       for name in ("graph", "primitives", "skeleton"))):
            return "acceptance_geometry_unverified"
        if "content_report" in report["artifacts"]:
            content = json.loads(Path(report["artifacts"]["content_report"]["path"]).read_text())
            if (content.get("schema") != content_routing.SCHEMA
                    or content.get("validator") != content_routing.VALIDATOR
                    or content.get("version") != content_routing.VERSION
                    or content_routing.acceptance_check(content) != report["checks"]["content"]
                    or content["inputs"]["source"] != report["artifacts"]["source"]
                    or content["class_decision"].get("routing") != "in_scope"):
                return "acceptance_content_unverified"
        if not report.get("artifacts"):
            return "acceptance_artifacts_missing"
        for artifact in report["artifacts"].values():
            if file_digest(Path(artifact["path"])) != artifact["sha256"]:
                return "acceptance_artifact_changed"
    except (OSError, ValueError, TypeError, KeyError):
        return "acceptance_invalid_record"
    return None
