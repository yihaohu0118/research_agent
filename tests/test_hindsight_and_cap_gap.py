"""Unit tests for the hindsight attributor and cap-gap explorer.

These are pure-Python tests: they stub the ``llm_chat_fn`` and never
touch torch or the environment service. Run with::

    python3 -m pytest tests/test_hindsight_and_cap_gap.py -q

or::

    python3 tests/test_hindsight_and_cap_gap.py
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from agentevolver.module.gcce.hindsight_attributor import (
    DEFAULT_REFLECTION_PROMPT,
    AttributionResult,
    GapType,
    HindsightAttributionRouter,
    compute_attribution_metrics,
)
from agentevolver.module.seal_ext.cap_gap_explorer import (
    CapGapExplorer,
    _build_variants,
)


def _fake_reflection(response_text: str):
    """Factory: returns an llm_chat_fn that always returns the same text."""

    def _fn(messages, request_id="", custom_sampling_params=None):
        return {"role": "assistant", "content": response_text}

    return _fn


class TestHindsightAttribution(unittest.TestCase):
    def test_success_skips_llm_call(self):
        calls = {"n": 0}

        def _fn(msgs, **kw):
            calls["n"] += 1
            return ""

        router = HindsightAttributionRouter(llm_chat_fn=_fn)
        res = router.attribute(
            trajectory_messages=[{"role": "user", "content": "q"}],
            score=1.0,
            category="multi_turn_base",
        )
        self.assertEqual(res.gap_type, GapType.UNDETERMINED)
        self.assertEqual(calls["n"], 0)

    def test_env_block_declared(self):
        router = HindsightAttributionRouter(
            llm_chat_fn=_fake_reflection("[ENV_BLOCK]")
        )
        res = router.attribute(
            trajectory_messages=[{"role": "user", "content": "q"}],
            score=0.0,
            category="multi_turn_miss_func",
        )
        self.assertEqual(res.gap_type, GapType.ENVIRONMENT_GAP)
        self.assertTrue(res.env_block_declared)
        rates = router.category_rates()
        self.assertIn("multi_turn_miss_func", rates)
        self.assertAlmostEqual(
            rates["multi_turn_miss_func"]["env_gap_rate"], 1.0
        )
        self.assertAlmostEqual(
            rates["multi_turn_miss_func"]["cap_gap_rate"], 0.0
        )

    def test_valid_corrected_action_is_capability_gap(self):
        action = (
            '<tool_call>\n{"name": "get_weather", '
            '"arguments": {"city": "Tokyo"}}\n</tool_call>'
        )
        router = HindsightAttributionRouter(
            llm_chat_fn=_fake_reflection(action)
        )
        res = router.attribute(
            trajectory_messages=[{"role": "user", "content": "q"}],
            score=0.0,
            category="multi_turn_base",
        )
        self.assertEqual(res.gap_type, GapType.CAPABILITY_GAP)
        self.assertIsNotNone(res.corrected_action)
        parsed = json.loads(res.corrected_action)
        self.assertEqual(parsed["name"], "get_weather")

    def test_invalid_parse_collapses_to_env_gap(self):
        # No tool_call tag, no env block -> safe default is ENVIRONMENT_GAP.
        router = HindsightAttributionRouter(
            llm_chat_fn=_fake_reflection("I think I would have succeeded.")
        )
        res = router.attribute(
            trajectory_messages=[{"role": "user", "content": "q"}],
            score=0.0,
            category="multi_turn_long_context",
        )
        self.assertEqual(res.gap_type, GapType.ENVIRONMENT_GAP)

    def test_category_rates_mix(self):
        # Route via the FIRST user message (original trajectory) — the
        # reflection prompt appended by the router always contains the word
        # "capability" (the legitimate default prompt discusses "API
        # capability"), so using the last message would incorrectly mask
        # the trajectory marker.
        def _fn(messages, request_id="", custom_sampling_params=None):
            first_user = next(
                (m["content"] for m in messages if m.get("role") == "user"),
                "",
            )
            if "MARK_CAP" in first_user:
                return (
                    '<tool_call>\n{"name":"f","arguments":{}}\n</tool_call>'
                )
            return "[ENV_BLOCK]"

        router = HindsightAttributionRouter(llm_chat_fn=_fn)
        for marker in ["MARK_CAP", "MARK_CAP", "MARK_ENV"]:
            router.attribute(
                trajectory_messages=[
                    {"role": "user", "content": marker}
                ],
                score=0.0,
                category="A",
            )
        rates = router.category_rates()
        self.assertIn("A", rates)
        self.assertAlmostEqual(rates["A"]["cap_gap_rate"], 2 / 3, places=3)
        self.assertAlmostEqual(rates["A"]["env_gap_rate"], 1 / 3, places=3)

    def test_reset_window_clears(self):
        router = HindsightAttributionRouter(
            llm_chat_fn=_fake_reflection("[ENV_BLOCK]")
        )
        router.attribute(
            trajectory_messages=[{"role": "user", "content": "q"}],
            score=0.0,
            category="A",
        )
        self.assertTrue(router.category_rates())
        router.reset_window()
        self.assertEqual(router.category_rates(), {})

    def test_batch_respects_max_cap(self):
        router = HindsightAttributionRouter(
            llm_chat_fn=_fake_reflection("[ENV_BLOCK]"),
            max_attributions_per_step=2,
        )
        # All failures (score=0). max=2 ⇒ only the first two are classified,
        # the remaining three are dropped to keep LM cost bounded. This is
        # the current contract; if we ever decide to emit UNDETERMINED
        # placeholders for dropped failures we also need to update the
        # downstream cap-gap buffer push to ignore them explicitly.
        items = [
            {"score": 0.0, "messages": [{"role": "user", "content": "x"}],
             "category": "A", "task_id": f"t{i}"}
            for i in range(5)
        ]
        results = router.batch_attribute(items)
        self.assertEqual(len(results), 2)
        self.assertEqual(router._total_attributions, 2)

    def test_batch_keeps_all_successes(self):
        router = HindsightAttributionRouter(
            llm_chat_fn=_fake_reflection("[ENV_BLOCK]"),
            max_attributions_per_step=1,
        )
        items = [
            {"score": 1.0, "messages": [], "category": "A", "task_id": "s1"},
            {"score": 1.0, "messages": [], "category": "A", "task_id": "s2"},
            {"score": 0.0, "messages": [{"role": "user", "content": "x"}],
             "category": "A", "task_id": "f1"},
        ]
        results = router.batch_attribute(items)
        # 2 successes (never dropped) + 1 failure under cap = 3 records.
        self.assertEqual(len(results), 3)
        self.assertEqual(router._total_attributions, 1)

    def test_compute_attribution_metrics_shape(self):
        results = [
            AttributionResult(gap_type=GapType.CAPABILITY_GAP),
            AttributionResult(gap_type=GapType.ENVIRONMENT_GAP),
            AttributionResult(gap_type=GapType.UNDETERMINED),
        ]
        m = compute_attribution_metrics(results)
        for key in (
            "gcce/hindsight/batch_total",
            "gcce/hindsight/batch_capability_gap_rate",
            "gcce/hindsight/batch_environment_gap_rate",
            "gcce/hindsight/batch_undetermined_rate",
        ):
            self.assertIn(key, m)
        self.assertAlmostEqual(m["gcce/hindsight/batch_total"], 3.0)


class TestCapGapExplorer(unittest.TestCase):
    def test_disabled_noop(self):
        exp = CapGapExplorer({"enable": False})
        exp.push(
            task_id="t1", first_query="q", messages=[], corrected_action=None
        )
        self.assertEqual(exp.buffer_size(), 0)
        self.assertFalse(exp.should_explore(0))

    def test_enabled_push_and_sample(self):
        exp = CapGapExplorer(
            {"enable": True, "max_seeds_per_step": 2, "sample_frequency": 1}
        )
        for i in range(4):
            exp.push(
                task_id=f"t{i}",
                first_query=f"query_{i}",
                messages=[{"role": "user", "content": f"query_{i}"}],
                corrected_action=(
                    '{"name":"f","arguments":{}}' if i % 2 == 0 else None
                ),
                category="A",
            )
        self.assertEqual(exp.buffer_size(), 4)
        self.assertTrue(exp.should_explore(0))
        seeds = exp.sample_seeds()
        self.assertEqual(len(seeds), 2)

    def test_build_exploration_tasks_yields_variants(self):
        exp = CapGapExplorer({"enable": True, "variants_per_seed": 2})
        seeds = [
            {
                "task_id": "abc",
                "first_query": "Please find the weather.",
                "messages": [],
                "corrected_action": json.dumps(
                    {"name": "get_weather", "arguments": {"city": "Tokyo"}}
                ),
                "category": "A",
                "metadata": {},
            }
        ]
        tasks = exp.build_exploration_tasks(seeds, env_type="bfcl")
        self.assertEqual(len(tasks), 2)
        for t in tasks:
            self.assertEqual(t.env_type, "bfcl")
            self.assertTrue(t.task_id.startswith("abc::cap_gap_explore::"))
            self.assertIn("Please find the weather.", t.query or "")

    def test_variants_without_correction_fallback(self):
        variants = _build_variants("do X", corrected_action=None)
        self.assertEqual(len(variants), 2)
        self.assertTrue(all(v.startswith("do X") for v in variants))

    def test_variants_with_correction_include_hint(self):
        corrected = json.dumps(
            {"name": "book_flight", "arguments": {"origin": "NYC"}}
        )
        variants = _build_variants("find flight", corrected_action=corrected)
        self.assertEqual(len(variants), 2)
        self.assertIn("book_flight", variants[0])
        self.assertIn("origin", variants[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
