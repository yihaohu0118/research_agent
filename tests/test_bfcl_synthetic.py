from pathlib import Path
import importlib.util
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "bfcl_synthetic",
    ROOT / "agentevolver" / "module" / "tocf" / "bfcl_synthetic.py",
)
bfcl_synthetic = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = bfcl_synthetic
spec.loader.exec_module(bfcl_synthetic)

compare_tool_turns = bfcl_synthetic.compare_tool_turns
bfcl_synthetic_env_params = bfcl_synthetic.bfcl_synthetic_env_params
build_bfcl_synthetic_case_overlay = bfcl_synthetic.build_bfcl_synthetic_case_overlay
extract_observed_tool_turns = bfcl_synthetic.extract_observed_tool_turns
materialize_bfcl_synthetic_case = bfcl_synthetic.materialize_bfcl_synthetic_case
normalize_tool_turns = bfcl_synthetic.normalize_tool_turns
normalize_turns_for_tool_schema = bfcl_synthetic.normalize_turns_for_tool_schema
parse_tool_schema_from_prompt = bfcl_synthetic.parse_tool_schema_from_prompt
replay_tool_turns_in_bfcl_env = bfcl_synthetic.replay_tool_turns_in_bfcl_env
serialize_tool_turns = bfcl_synthetic.serialize_tool_turns
synthetic_bfcl_case_id = bfcl_synthetic.synthetic_bfcl_case_id
tool_schemas_from_prompt = bfcl_synthetic.tool_schemas_from_prompt
tool_turns_to_tool_call_content = bfcl_synthetic.tool_turns_to_tool_call_content
tool_turns_to_strings = bfcl_synthetic.tool_turns_to_strings
validate_bfcl_synthetic_alignment = bfcl_synthetic.validate_bfcl_synthetic_alignment
write_materialized_bfcl_synthetic_case = bfcl_synthetic.write_materialized_bfcl_synthetic_case


def test_normalizes_step_action_sequence():
    gt = """# step0
cd(folder='document')
mkdir(dir_name='temp')
mv(source='final_report.pdf', destination='temp')

# step1
grep(file_name='final_report.pdf', pattern='budget analysis')
"""
    turns = normalize_tool_turns(gt)

    assert tool_turns_to_strings(turns) == [
        [
            "cd(folder='document')",
            "mkdir(dir_name='temp')",
            "mv(source='final_report.pdf', destination='temp')",
        ],
        ["grep(file_name='final_report.pdf', pattern='budget analysis')"],
    ]


def test_observed_tool_calls_match_serialized_gt():
    expected = normalize_tool_turns(
        serialize_tool_turns(normalize_tool_turns(["sort('final_report.pdf')"]))
    )
    observed = extract_observed_tool_turns(
        [
            {"role": "system", "content": "# Tools"},
            {"role": "user", "content": "Sort the final report."},
            {
                "role": "assistant",
                "content": (
                    "<tool_call>\n"
                    '{"name":"sort","arguments":{"file_name":"final_report.pdf"}}'
                    "\n</tool_call>"
                ),
            },
            {"role": "user", "content": "<tool_response>\nok\n</tool_response>"},
        ]
    )
    comparison = compare_tool_turns(expected, observed)

    assert comparison["success"]
    assert comparison["score"] == 1.0


def test_positional_gt_maps_to_bfcl_tool_payload():
    prompt = """
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"name":"sort","parameters":{"type":"object","properties":{"file_name":{"type":"string"}},"required":["file_name"]}}
</tools>
"""
    turns = normalize_tool_turns("sort('final_report.pdf')")
    schemas = tool_schemas_from_prompt(prompt)
    content = tool_turns_to_tool_call_content(turns, schemas)
    normalized = normalize_turns_for_tool_schema(turns, schemas)

    assert parse_tool_schema_from_prompt(prompt) == {"sort": ["file_name"]}
    assert content is not None
    assert '"name":"sort"' in content
    assert '"file_name":"final_report.pdf"' in content
    assert tool_turns_to_strings(normalized) == [
        ["sort(file_name='final_report.pdf')"]
    ]


def test_replay_tool_turns_uses_bfcl_payload_format():
    class FakeEnv:
        def __init__(self):
            self.actions = []

        def create_instance(self, env_type, task_id, instance_id=None, params=None):
            assert env_type == "bfcl"
            return {
                "state": [
                    {
                        "role": "system",
                        "content": (
                            "<tools>\n"
                            '{"name":"sort","parameters":{"type":"object",'
                            '"properties":{"file_name":{"type":"string"}},'
                            '"required":["file_name"]}}\n'
                            "</tools>"
                        ),
                    },
                    {"role": "user", "content": "Sort the report."},
                ]
            }

        def step(self, instance_id, action):
            self.actions.append(action["content"])
            return {"state": [{"role": "user", "content": "<tool_response>\nok\n</tool_response>"}]}

        def release_instance(self, instance_id):
            return True

    env = FakeEnv()
    ok, reason = replay_tool_turns_in_bfcl_env(
        env,
        "bfcl_test",
        normalize_tool_turns("sort('final_report.pdf')"),
    )

    assert ok, reason
    assert '"name":"sort"' in env.actions[0]
    assert '"file_name":"final_report.pdf"' in env.actions[0]


def test_builds_single_turn_synthetic_overlay():
    overlay, reason = build_bfcl_synthetic_case_overlay(
        task_id="bfcl_test",
        query="Sort the report.",
        ground_truth="sort('final_report.pdf')",
    )

    assert reason == "ok"
    assert overlay["parent_task_id"] == "bfcl_test"
    assert overlay["question"] == [[{"role": "user", "content": "Sort the report."}]]
    assert overlay["ground_truth_turns"] == [["sort('final_report.pdf')"]]


def test_materializes_overlay_as_independent_bfcl_case_and_answer():
    parent = {
        "id": "multi_turn_base_001",
        "question": [[{"role": "user", "content": "Parent query."}]],
        "function": [{"name": "sort"}],
        "initial_config": {"files": []},
        "involved_classes": [],
    }
    overlay, reason = build_bfcl_synthetic_case_overlay(
        task_id="multi_turn_base_001",
        query="Sort the final report.",
        ground_truth="sort('final_report.pdf')",
    )
    assert reason == "ok"

    case, answer = materialize_bfcl_synthetic_case(parent, overlay)

    assert case["id"].startswith("multi_turn_base_syn")
    assert case["id"] == answer["id"]
    assert case["question"] == [[{"role": "user", "content": "Sort the final report."}]]
    assert answer["ground_truth"] == [["sort('final_report.pdf')"]]
    assert case["metadata"]["synthetic_case_overlay"]["materialized"] is True
    assert case["metadata"]["synthetic_case_overlay"]["possible_answer"] == answer
    assert synthetic_bfcl_case_id("multi_turn_base_001", overlay["objective_hash"]) == case["id"]


def test_writes_materialized_case_files():
    parent = {
        "id": "multi_turn_base_001",
        "question": [[{"role": "user", "content": "Parent query."}]],
        "function": [{"name": "sort"}],
        "initial_config": {},
        "involved_classes": [],
    }
    overlay, _ = build_bfcl_synthetic_case_overlay(
        task_id="multi_turn_base_001",
        query="Sort the final report.",
        ground_truth="sort('final_report.pdf')",
    )
    case, answer = materialize_bfcl_synthetic_case(parent, overlay)

    with tempfile.TemporaryDirectory() as tmpdir:
        files = write_materialized_bfcl_synthetic_case(case, answer, tmpdir)

        assert Path(files["case_path"]).exists()
        assert Path(files["answer_path"]).exists()
        assert Path(files["case_jsonl_path"]).exists()
        assert Path(files["answer_jsonl_path"]).exists()
        assert Path(files["manifest_path"]).exists()


def test_rejects_multiturn_overlay_without_question_schedule():
    overlay, reason = build_bfcl_synthetic_case_overlay(
        task_id="bfcl_test",
        query="Do two things.",
        ground_truth="# step0\nsort('a.txt')\n# step1\nsort('b.txt')",
    )

    assert overlay is None
    assert reason == "synthetic_multiturn_unaligned"


def test_replay_tool_turns_checks_overlay_init_query():
    class FakeEnv:
        def create_instance(self, env_type, task_id, instance_id=None, params=None):
            return {
                "state": [
                    {
                        "role": "system",
                        "content": (
                            "<tools>\n"
                            '{"name":"sort","parameters":{"type":"object",'
                            '"properties":{"file_name":{"type":"string"}},'
                            '"required":["file_name"]}}\n'
                            "</tools>"
                        ),
                    },
                    {"role": "user", "content": "Parent query."},
                ]
            }

        def step(self, instance_id, action):
            return {"state": [{"role": "user", "content": "<tool_response>\nok\n</tool_response>"}]}

        def release_instance(self, instance_id):
            return True

    overlay, reason = build_bfcl_synthetic_case_overlay(
        task_id="bfcl_test",
        query="Synthetic query.",
        ground_truth="sort('final_report.pdf')",
    )
    assert reason == "ok"

    ok, replay_reason = replay_tool_turns_in_bfcl_env(
        FakeEnv(),
        "bfcl_test",
        normalize_tool_turns("sort('final_report.pdf')"),
        params={"synthetic_case_overlay": overlay},
    )

    assert not ok
    assert replay_reason == "synthetic_state_mismatch"


def test_env_params_adds_overlay_for_task_like_object():
    class TaskLike:
        task_id = "bfcl_test"
        query = "Sort the report."
        ground_truth = "sort('final_report.pdf')"
        metadata = {}

    params, reason = bfcl_synthetic_env_params(TaskLike(), {"is_open_query": True})

    assert reason == "ok"
    assert params["is_open_query"] is True
    assert params["synthetic_case_overlay"]["question"][0][0]["content"] == "Sort the report."


def test_multiturn_overlay_sets_closed_query_and_replays_turns():
    class TaskLike:
        task_id = "bfcl_test"
        query = "Sort the first report."
        ground_truth = "# step0\nsort('a.txt')\n# step1\nsort('b.txt')"
        metadata = {
            "bfcl_synthetic_question_schedule": [
                "Sort the first report.",
                "Now sort the second report.",
            ]
        }

    class FakeEnv:
        def __init__(self):
            self.actions = []

        def create_instance(self, env_type, task_id, instance_id=None, params=None):
            assert params["is_open_query"] is False
            return {
                "state": [
                    {
                        "role": "system",
                        "content": (
                            "<tools>\n"
                            '{"name":"sort","parameters":{"type":"object",'
                            '"properties":{"file_name":{"type":"string"}},'
                            '"required":["file_name"]}}\n'
                            "</tools>"
                        ),
                    },
                    {"role": "user", "content": "Sort the first report."},
                ]
            }

        def step(self, instance_id, action):
            self.actions.append(action["content"])
            if "<tool_call>" in action["content"]:
                return {"state": [{"role": "user", "content": "<tool_response>\nok\n</tool_response>"}]}
            return {"state": [{"role": "user", "content": "Now sort the second report."}]}

        def release_instance(self, instance_id):
            return True

    params, reason = bfcl_synthetic_env_params(TaskLike(), {"is_open_query": True})
    assert reason == "ok"
    assert params["is_open_query"] is False

    env = FakeEnv()
    ok, replay_reason = replay_tool_turns_in_bfcl_env(
        env,
        "bfcl_test",
        normalize_tool_turns(TaskLike.ground_truth),
        params=params,
    )

    assert ok, replay_reason
    assert len([action for action in env.actions if "<tool_call>" in action]) == 2
    assert "Done." in env.actions


def test_semantic_alignment_catches_unanchored_gt():
    aligned = validate_bfcl_synthetic_alignment(
        "Sort the final report.",
        normalize_tool_turns("sort('final_report.pdf')"),
    )
    unanchored = validate_bfcl_synthetic_alignment(
        "Tell me a joke.",
        normalize_tool_turns("sort('final_report.pdf')"),
    )

    assert aligned["ok"]
    assert not unanchored["ok"]
    assert unanchored["reason"] == "synthetic_semantic_unanchored"


def test_compare_tool_turns_reports_specific_mismatch_tags():
    expected = normalize_tool_turns("sort('a.txt')")

    assert compare_tool_turns(expected, normalize_tool_turns("grep('a.txt')"))[
        "failure_tag"
    ] == "synthetic_tool_name_mismatch"
    assert compare_tool_turns(expected, normalize_tool_turns("sort('b.txt')"))[
        "failure_tag"
    ] == "synthetic_arg_mismatch"
    assert compare_tool_turns(expected, [])["failure_tag"] == "synthetic_missing_call"
    assert compare_tool_turns(expected, normalize_tool_turns(["sort('a.txt')", "grep('a.txt')"]))[
        "failure_tag"
    ] == "synthetic_extra_call"


if __name__ == "__main__":
    test_normalizes_step_action_sequence()
    test_observed_tool_calls_match_serialized_gt()
    test_positional_gt_maps_to_bfcl_tool_payload()
    test_replay_tool_turns_uses_bfcl_payload_format()
    test_builds_single_turn_synthetic_overlay()
    test_materializes_overlay_as_independent_bfcl_case_and_answer()
    test_writes_materialized_case_files()
    test_rejects_multiturn_overlay_without_question_schedule()
    test_replay_tool_turns_checks_overlay_init_query()
    test_env_params_adds_overlay_for_task_like_object()
    test_multiturn_overlay_sets_closed_query_and_replays_turns()
    test_semantic_alignment_catches_unanchored_gt()
    test_compare_tool_turns_reports_specific_mismatch_tags()
