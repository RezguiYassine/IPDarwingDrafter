"""Style tags must stay aligned with the shapes they describe.

The labeller saves "polygons" and a parallel "styles" list. If an undo pops a
polygon without its style, or a circle is saved in a different order from its
tag, every later region on that figure carries the wrong style -- silently,
because the file still loads. These tests drive the session through each edit
path and check the alignment the save path depends on.
"""
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from tools import hatch_mask_label as hml


@pytest.fixture
def session():
    fig, ax = plt.subplots()
    s = hml._Session(ax, (200, 200), np.zeros((200, 200), bool))
    s._img, s._title = np.zeros((200, 200, 3), np.uint8), "t"
    yield s
    plt.close(fig)


def key(s, k):
    s.on_key(SimpleNamespace(key=k))


def polygon(s, offset=0):
    s.current = [(10 + offset, 10), (40 + offset, 10), (40 + offset, 40)]


def test_each_polygon_takes_the_style_current_when_it_closed(session):
    polygon(session); key(session, "enter")                  # thin (default)
    key(session, "2"); polygon(session, 50); key(session, "enter")   # bold
    key(session, "3"); polygon(session, 100); key(session, "enter")  # shading
    assert session.shape_styles() == ["thin", "bold", "shading"]
    assert len(session.shape_styles()) == len(session.polygons)


def test_undo_removes_the_style_with_its_polygon(session):
    polygon(session); key(session, "enter")
    key(session, "4"); polygon(session, 50); key(session, "enter")
    key(session, "z")
    assert session.shape_styles() == ["thin"]
    assert len(session.polygons) == 1


def test_reset_clears_styles(session):
    key(session, "2"); polygon(session); key(session, "enter")
    key(session, "r")
    assert session.shape_styles() == [] and session.polygons == []


def test_done_and_quit_close_the_open_polygon_with_its_style(session):
    key(session, "3"); polygon(session); key(session, "n")
    assert session.shape_styles() == ["shading"]


def test_circles_carry_their_own_style_after_polygons(session):
    """The save path concatenates polygons then circles; tags must match that order."""
    key(session, "2"); polygon(session); key(session, "enter")
    session.circles.append({"center": (100, 100), "radius": 20, "style": "solid"})
    assert session.shape_styles() == ["bold", "solid"]


def test_unknown_key_leaves_style_alone(session):
    key(session, "2"); key(session, "x")
    assert session.style == "bold"


def test_legacy_circle_without_a_style_defaults_to_thin(session):
    session.circles.append({"center": (100, 100), "radius": 20})
    assert session.shape_styles() == ["thin"]
