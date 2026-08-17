"""Publication-layout and claim-boundary tests for method_extensions."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_hierarchy_and_formal_decisions() -> None:
    hierarchy = (ROOT / "BENCHMARK_HIERARCHY.md").read_text(encoding="utf-8")
    assert "Basic / Original Classical Benchmark" in hierarchy
    assert "Historical benchmark" not in hierarchy
    assert "Strengthened benchmark 1" in hierarchy
    assert "CN + Policy Iteration" in hierarchy
    assert "Strengthened benchmark 2" in hierarchy
    assert "CN + Projected LU" in hierarchy

    policy = json.loads(
        (
            ROOT
            / "results/07_method_extensions/02_warmstart/method_decision.json"
        ).read_text(encoding="utf-8")
    )
    projected = json.loads(
        (ROOT / "results/13_projected_lu/method_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["selected_method"] == "policy_iteration_previous_slice"
    assert projected["status"] == "GO_PROJECTED_LU_NUMERICALLY_CERTIFIED"

    poster = json.loads(
        (ROOT / "results/14_poster_unified_comparison/method_decision.json").read_text(
            encoding="utf-8"
        )
    )
    assert poster["strict_solver_status"] == "STRICT_THREE_CONFIRMED"
    assert poster["penalty_newton_status"] == "FAILED_CORRECTNESS"
    assert (
        poster["benchmark_roles"]["psor"]
        == "Basic / Original Classical Benchmark"
    )


def test_unfinished_neural_methods_publish_no_result_directories() -> None:
    assert not (ROOT / "results/08_pinn_gap").exists()
    assert not (ROOT / "results/12_positive_premium_deeponet").exists()
    for relative in (
        "neural_methods/02_pinn_arms_c_d_e/NO_FORMAL_RESULTS.md",
        "neural_methods/03_positive_premium_deeponet/NO_FORMAL_RESULTS.md",
    ):
        assert "No Formal" in (ROOT / relative).read_text(encoding="utf-8")


def test_public_text_contains_no_local_home_path() -> None:
    suffixes = {".md", ".json", ".csv", ".py", ".txt"}
    offenders = []
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            text = path.read_text(encoding="utf-8")
            if "/Users/" + "xrx" in text or "/private/" + "var" in text:
                offenders.append(str(path.relative_to(ROOT)))
    assert not offenders
