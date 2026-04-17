"""Smoke tests for GCCE gap attribution + routing.

These tests use only stdlib + the in-repo gcce package; no heavy RL deps
are exercised, so they are safe to run in CI on a CPU-only box.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agentevolver.module.gcce import (
    GCCERouter,
    GapAttributor,
    OracleProbe,
    TeacherCache,
)


class _StubSnapshot:
    """Minimal TOCFStats-compatible stub."""

    def __init__(self, categories: dict[str, dict[str, float]]):
        self._categories = categories

    def snapshot(self, window: bool = True):  # noqa: ARG002
        return {"categories": dict(self._categories)}


def test_router_sums_to_one():
    router = GCCERouter(
        config={
            "gcce": {
                "enable": True,
                "router": {
                    "eta": 1e-4,
                    "alpha_env": 1.0,
                    "alpha_policy": 1.0,
                    "min_weight": 0.5,
                    "max_weight": 2.5,
                },
            }
        }
    )
    # Build two fake gaps: env-dominated, policy-dominated.
    from agentevolver.module.gcce.gap_attributor import CategoryGap

    gaps = {
        "multi_turn_miss_func": CategoryGap(
            category="multi_turn_miss_func",
            count=50,
            current_success=0.20,
            oracle_success=0.90,
            teacher_success=0.30,
            delta_e=0.70,
            delta_pi=0.10,
        ),
        "multi_turn_base": CategoryGap(
            category="multi_turn_base",
            count=50,
            current_success=0.30,
            oracle_success=0.40,
            teacher_success=0.90,
            delta_e=0.10,
            delta_pi=0.60,
        ),
    }
    decision = router.route(gaps)

    # r_E + r_pi = 1 for every category
    for cat in gaps:
        assert abs(decision.r_env[cat] + decision.r_policy[cat] - 1.0) < 1e-6

    # env-dominated category is routed primarily to the environment
    assert decision.r_env["multi_turn_miss_func"] > 0.8
    # policy-dominated category is routed primarily to the policy
    assert decision.r_policy["multi_turn_base"] > 0.8

    # advantage_weight respects range and monotonicity with r_pi
    w_miss = router.advantage_weight("multi_turn_miss_func")
    w_base = router.advantage_weight("multi_turn_base")
    assert w_base > w_miss  # policy-dominated gets stronger advantage boost


def test_attributor_bayesian_shrinkage(tmp_path: Path):
    cache_path = tmp_path / "teacher.json"
    cache_path.write_text(
        json.dumps(
            {
                "meta": {"teacher_model": "test"},
                "scores": {
                    "multi_turn_base_0": {"success": 1.0, "reward": 1.0},
                    "multi_turn_miss_func_0": {"success": 0.5, "reward": 0.5},
                    "multi_turn_miss_func_1": {"success": 0.7, "reward": 0.7},
                },
            }
        )
    )
    cache = TeacherCache(path=str(cache_path), enabled=True)
    cache.load()
    assert cache.get("multi_turn_miss_func_0").success == 0.5

    config = {
        "gcce": {
            "enable": True,
            "attribution": {"min_samples": 1, "prior_strength": 0.0},
        }
    }
    oracle = OracleProbe(config={"gcce": {"oracle_env": {"enable": True, "oracle_success_prior": 1.0}}})
    attributor = GapAttributor(config=config, teacher_cache=cache, oracle_probe=oracle)

    categories = {
        "multi_turn_miss_func": {"count": 10, "success_rate": 0.2, "reward_mean": 0.2, "partial_count": 0, "success": 2},
        "multi_turn_base": {"count": 10, "success_rate": 0.1, "reward_mean": 0.1, "partial_count": 0, "success": 1},
    }
    snap = _StubSnapshot(categories)

    gaps = attributor.attribute(snap, tasks=None, env_type="bfcl")
    assert set(gaps) == {"multi_turn_miss_func", "multi_turn_base"}
    # miss_func: teacher mean = 0.6, current = 0.2 -> Delta_pi ~= 0.4
    assert 0.3 < gaps["multi_turn_miss_func"].delta_pi < 0.5
    # base: teacher mean = 1.0, current = 0.1 -> Delta_pi ~= 0.9
    assert gaps["multi_turn_base"].delta_pi > 0.7


if __name__ == "__main__":
    test_router_sums_to_one()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_attributor_bayesian_shrinkage(Path(d))
    print("GCCE router/attributor tests OK")
