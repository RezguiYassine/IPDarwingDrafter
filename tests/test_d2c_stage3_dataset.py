import numpy as np

from tools.d2c_stage3_dataset import (
    PrepareConfig,
    SourceSegment,
    _edge_points,
    classify_edge,
    parse_svg_segments,
)


def _edge(points, closed=False):
    return {"pixels": np.asarray(points).tolist(), "smooth_pts": [], "is_closed": closed}


def _segment(path_id, order, command, points):
    return SourceSegment(path_id, order, command, np.asarray(points, dtype=float))


def test_svg_parser_preserves_line_and_cubic_commands(tmp_path):
    svg = tmp_path / "drawing.svg"
    svg.write_text(
        '<svg viewBox="0 0 100 100"><path d="M 0,0 L 10,0 '
        'C 20,0 20,10 30,10"/></svg>'
    )

    segments = parse_svg_segments(svg, render_width=100)

    assert [segment.command for segment in segments] == ["L", "C"]
    np.testing.assert_allclose(segments[0].samples[[0, -1]], [[0, 0], [10, 0]])
    np.testing.assert_allclose(segments[1].samples[[0, -1]], [[10, 0], [30, 10]])


def test_multiple_linear_source_segments_label_a_bent_edge_polyline():
    first = np.column_stack([np.linspace(0, 20, 21), np.zeros(21)])
    second = np.column_stack([np.full(21, 20), np.linspace(0, 20, 21)])
    points = np.vstack([first, second[1:]])
    segments = [_segment(0, 0, "L", first), _segment(0, 1, "L", second)]

    command, _ = classify_edge(_edge(points), segments, PrepareConfig())

    assert command == "POLYLINE"


def test_circular_svg_cubics_are_not_mislabelled_bezier():
    angles = np.linspace(0, 2 * np.pi, 129)
    points = np.column_stack([50 + 20 * np.cos(angles), 50 + 20 * np.sin(angles)])
    segments = [_segment(0, i, "C", part) for i, part in enumerate(np.array_split(points, 4))]

    command, _ = classify_edge(_edge(points, closed=True), segments, PrepareConfig())

    assert command == "CIRCLE"


def test_non_circular_cubic_is_labelled_bezier():
    x = np.linspace(0, 50, 101)
    points = np.column_stack([x, 12 * np.sin(x / 8)])
    segments = [_segment(0, 0, "C", points)]

    command, _ = classify_edge(_edge(points), segments, PrepareConfig())

    assert command == "BEZIER"


def test_closed_scanline_pixels_are_reordered_into_a_local_walk():
    angles = np.linspace(0, 2 * np.pi, 65, endpoint=False)
    loop = np.unique(np.round(np.column_stack([
        50 + 20 * np.cos(angles), 50 + 20 * np.sin(angles)
    ])).astype(int), axis=0)
    scanline = loop[np.lexsort((loop[:, 0], loop[:, 1]))]

    ordered = _edge_points(_edge(scanline, closed=True))

    distances = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    assert np.quantile(distances, 0.90) < 3.0
