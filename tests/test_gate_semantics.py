"""A quality gate must mean what the config says it means.

The previous convention disabled a gate on any falsy value, so
`max_low_conf_ratio: 0` -- the strictest-looking setting a reader could write
-- silently turned the gate off, and a ratio gate was disabled by setting it
to 1.01, above the maximum a ratio can reach, which reads stricter still.
config_deploy.yaml therefore asserted the opposite of its own behaviour on
four of its nine gates. These tests pin the replacement: `null` disables, a
number is literal, and the old idioms raise.
"""

import pytest
import yaml

from tools.batch_run import stage2_stroke_extract as s2

gate_bound = s2.gate_bound


# ── reading a bound ─────────────────────────────────────────────────────────

def test_null_disables_the_gate():
    assert gate_bound({"max_edges": None}, "max_edges", 5000) is None


def test_zero_is_a_literal_bound_not_an_off_switch():
    """The whole point: 0 now means zero tolerance."""
    assert gate_bound({"max_low_conf_ratio": 0}, "max_low_conf_ratio", None) == 0.0


def test_absent_key_keeps_the_documented_default():
    assert gate_bound({}, "max_edges", 5000) == 5000.0
    assert gate_bound({}, "max_primitives", None) is None


def test_a_number_is_returned_as_given():
    assert gate_bound({"max_edges": 2500}, "max_edges", 5000) == 2500.0


@pytest.mark.parametrize("value", [1.01, 2.0, 100])
def test_a_ratio_gate_above_its_maximum_is_refused(value):
    """The 1.01 idiom cannot come back silently."""
    with pytest.raises(ValueError, match="never fire|maximum"):
        gate_bound({"max_micro_edge_ratio": value}, "max_micro_edge_ratio",
                   0.5, maximum=1.0)


def test_a_ratio_gate_at_its_maximum_is_allowed():
    assert gate_bound({"r": 1.0}, "r", 0.5, maximum=1.0) == 1.0


def test_an_unbounded_gate_accepts_large_values():
    assert gate_bound({"max_edges": 1_000_000}, "max_edges", 5000) == 1_000_000.0


# ── the shipped configs say what they do ────────────────────────────────────

RATIO_GATES = ("max_low_conf_ratio", "max_low_conf_ratio_after_hachure",
               "max_micro_edge_ratio", "max_short_edge_ratio")


@pytest.mark.parametrize("path", ["config_deploy.yaml", "config_deploy_recal.yaml",
                                  "config_bench_armA.yaml", "config_bench_armB.yaml"])
def test_shipped_configs_disable_gates_explicitly(path):
    config = yaml.safe_load(open(path))
    blocks = [config.get("pipeline", {}).get("quality_gates", {}),
              config.get("stage2", {}).get("fragmentation", {})]
    for block in blocks:
        for key in RATIO_GATES:
            if key not in block:
                continue
            value = block[key]
            assert value is None or 0 <= float(value) <= 1.0, (
                f"{path}: {key}={value} is outside the range of a ratio; "
                "a gate that cannot fire must be written as null")


def test_canonical_config_gate_bounds_load():
    """Every gate in the canonical config reads without raising."""
    config = yaml.safe_load(open("config_deploy.yaml"))
    gates = config["pipeline"]["quality_gates"]
    fragmentation = config["stage2"]["fragmentation"]
    assert gate_bound(gates, "max_low_conf_ratio", None, maximum=1.0) is None
    assert gate_bound(gates, "max_primitives", None) == 50000.0
    assert gate_bound(fragmentation, "max_micro_edge_ratio", 0.5, maximum=1.0) is None
    assert gate_bound(fragmentation, "max_edges", 5000) == 2500.0
