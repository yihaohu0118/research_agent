from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from env_service.environments.bfcl.bfcl_env import (
        parse_assistant_content_to_tool_calls,
        tool_message_to_qwen_text,
        tools_schema_to_llama31_official_prompt,
        tools_schema_to_toolace_official_prompt,
        tools_schema_to_qwen_prompt,
    )
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("env_service"):
        raise
    parse_assistant_content_to_tool_calls = None
    tool_message_to_qwen_text = None
    tools_schema_to_qwen_prompt = None
    tools_schema_to_llama31_official_prompt = None
    tools_schema_to_toolace_official_prompt = None

try:
    from env_service.environments.bfcl.env_handler import EnvHandler
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("env_service"):
        raise
    EnvHandler = None


def _require_bfcl_eval() -> bool:
    return EnvHandler is not None and parse_assistant_content_to_tool_calls is not None


def _require_bfcl_helpers() -> bool:
    return (
        parse_assistant_content_to_tool_calls is not None
        and tool_message_to_qwen_text is not None
        and tools_schema_to_qwen_prompt is not None
        and tools_schema_to_llama31_official_prompt is not None
        and tools_schema_to_toolace_official_prompt is not None
    )


def _tool_call(name: str, arguments: dict) -> dict:
    return {
        "id": f"{name}_1",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def test_strict_parser_rejects_malformed_tool_tags():
    if not _require_bfcl_eval():
        return

    parsed = parse_assistant_content_to_tool_calls(
        {
            "role": "assistant",
            "content": "<tool_call>not json</tool_call>",
        },
        strict=True,
    )

    assert parsed["tool_calls"] == []
    assert "_bfcl_parse_error" in parsed


def test_unavailable_tool_is_rejected_before_execution():
    if not _require_bfcl_eval():
        return

    handler = object.__new__(EnvHandler)

    try:
        handler._convert_tool_calls_to_execution_format(
            [_tool_call("sort", {"file_name": "final_report.pdf"})],
            available_function_names={"cd", "grep"},
        )
    except ValueError as exc:
        assert "not available" in str(exc)
    else:
        raise AssertionError("Unavailable tools should be rejected before execution.")


def test_rejected_tool_calls_are_not_extracted_for_eval():
    if not _require_bfcl_eval():
        return

    handler = object.__new__(EnvHandler)
    rejected = _tool_call("sort", {"file_name": "final_report.pdf"})
    accepted = _tool_call("cd", {"folder": "document"})

    messages = [
        {"role": "user", "content": "Do the task."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
            "_bfcl_rejected_tool_calls": [rejected],
        },
        {
            "role": "env",
            "content": "[ERROR] Invalid tool call format: Tool 'sort' is not available.",
        },
        {"role": "assistant", "content": "", "tool_calls": [accepted]},
        {"role": "tool", "content": "ok", "tool_call_id": "cd_1"},
        {"role": "assistant", "content": "done", "tool_calls": []},
        {"role": "user", "content": "Next turn."},
    ]

    assert handler._extract_multi_turn_responses(messages)[0] == [
        "cd(folder='document')"
    ]


def test_bfcl_prompt_defaults_to_qwen_fc_protocol():
    if not _require_bfcl_helpers():
        return

    prompt = tools_schema_to_qwen_prompt(
        [
            {
                "name": "noop",
                "description": "No-op tool.",
                "parameters": {"type": "dict", "properties": {}, "required": []},
            }
        ]
    )
    tool_text = tool_message_to_qwen_text(
        {"role": "tool", "content": {"ok": True}, "tool_call_id": "noop_1"}
    )

    assert "# Tools" in prompt
    assert "Your response must always start" not in prompt
    assert "within <tool_call></tool_call> XML tags" in prompt
    assert "<tool_response>" in tool_text


def test_bfcl_t3rl_protocol_stays_available():
    if not _require_bfcl_helpers():
        return

    prompt = tools_schema_to_qwen_prompt(
        [
            {
                "name": "noop",
                "description": "No-op tool.",
                "parameters": {"type": "dict", "properties": {}, "required": []},
            }
        ],
        prompt_mode="t3rl_text",
    )
    tool_text = tool_message_to_qwen_text(
        {"role": "tool", "content": {"ok": True}, "tool_call_id": "noop_1"},
        result_mode="plain_user",
    )

    assert "Your response must always start" in prompt
    assert "<tool_response>" not in tool_text
    assert "<tool_call>" not in tool_text


def test_llama31_official_parser_accepts_raw_parameters_json():
    if not _require_bfcl_helpers():
        return

    parsed = parse_assistant_content_to_tool_calls(
        {
            "role": "assistant",
            "content": '{"name": "get_weather", "parameters": {"city": "Paris"}}',
        },
        strict=True,
        parser_mode="llama31_official_fc",
    )

    assert parsed["content"] == ""
    assert parsed["tool_calls"] == [_tool_call("get_weather", {"city": "Paris"})]


def test_llama31_official_prompt_uses_raw_json_protocol():
    if not _require_bfcl_helpers():
        return

    prompt = tools_schema_to_llama31_official_prompt(
        [
            {
                "name": "noop",
                "description": "No-op tool.",
                "parameters": {"type": "dict", "properties": {}, "required": []},
            }
        ],
        "Do the task.",
    )
    tool_text = tool_message_to_qwen_text(
        {"role": "tool", "content": {"ok": True}, "tool_call_id": "noop_1"},
        result_mode="llama31_official",
    )

    assert '"parameters"' in prompt
    assert "<tool_call>\n{" not in prompt
    assert "<tool_response>" not in tool_text


def test_toolace_official_prompt_uses_bfcl_python_protocol():
    if not _require_bfcl_helpers():
        return

    parsed = parse_assistant_content_to_tool_calls(
        {
            "role": "assistant",
            "content": '[search(query="value", top_k=3)]',
        },
        strict=True,
        parser_mode="toolace_official_prompt",
    )
    parsed_bare = parse_assistant_content_to_tool_calls(
        {
            "role": "assistant",
            "content": 'search(query="value")',
        },
        strict=True,
        parser_mode="toolace_official_prompt",
    )
    prompt = tools_schema_to_toolace_official_prompt(
        [
            {
                "name": "search",
                "description": "Search tool.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
    )
    tool_text = tool_message_to_qwen_text(
        {"role": "tool", "content": {"ok": True}, "tool_call_id": "search_1"},
        result_mode="toolace_official_prompt",
    )

    assert parsed["tool_calls"] == [_tool_call("search", {"query": "value", "top_k": 3})]
    assert parsed_bare["tool_calls"] == [_tool_call("search", {"query": "value"})]
    assert "[func_name1(" in prompt
    assert "<tools>" not in prompt
    assert "<tool_call>" not in prompt
    assert "<tool_response>" not in tool_text


def test_toolace_official_parser_accepts_json_style_fallback():
    if not _require_bfcl_helpers():
        return

    parsed = parse_assistant_content_to_tool_calls(
        {
            "role": "assistant",
            "content": '{"name": "get_stock_info", "parameters": {"symbol": "NVDA"}}',
        },
        strict=True,
        parser_mode="toolace_official_prompt",
    )
    parsed_unquoted_name = parse_assistant_content_to_tool_calls(
        {
            "role": "assistant",
            "content": '{"name": get_watchlist, "parameters": {}}',
        },
        strict=True,
        parser_mode="toolace_official_prompt",
    )
    parsed_json_list = parse_assistant_content_to_tool_calls(
        {
            "role": "assistant",
            "content": (
                '[{"name": "ls", "parameters": {"a": true}}, '
                '{"name": "cat", "arguments": {"file_name": "report.txt"}}]'
            ),
        },
        strict=True,
        parser_mode="toolace_official_prompt",
    )

    assert parsed["content"] == ""
    assert parsed["tool_calls"] == [_tool_call("get_stock_info", {"symbol": "NVDA"})]
    assert parsed_unquoted_name["tool_calls"] == [_tool_call("get_watchlist", {})]
    assert parsed_json_list["tool_calls"] == [
        _tool_call("ls", {"a": True}),
        {
            **_tool_call("cat", {"file_name": "report.txt"}),
            "id": "cat_2",
        },
    ]


def test_clean_trajectory_flags_tool_errors():
    if not _require_bfcl_eval():
        return

    handler = object.__new__(EnvHandler)
    messages = [
        {"role": "user", "content": "Do the task."},
        {"role": "assistant", "content": "", "tool_calls": [_tool_call("cat", {})]},
        {"role": "tool", "content": {"error": "cat: invalid path"}, "tool_call_id": "cat_1"},
        {"role": "assistant", "content": "Recovered.", "tool_calls": []},
    ]

    diagnostics = handler._diagnose_trajectory(messages, completed=True)
    assert diagnostics["has_tool_error"]
    assert not diagnostics["clean"]


if __name__ == "__main__":
    test_strict_parser_rejects_malformed_tool_tags()
    test_unavailable_tool_is_rejected_before_execution()
    test_rejected_tool_calls_are_not_extracted_for_eval()
    test_bfcl_prompt_defaults_to_qwen_fc_protocol()
    test_bfcl_t3rl_protocol_stays_available()
    test_llama31_official_parser_accepts_raw_parameters_json()
    test_llama31_official_prompt_uses_raw_json_protocol()
    test_toolace_official_prompt_uses_bfcl_python_protocol()
    test_toolace_official_parser_accepts_json_style_fallback()
    test_clean_trajectory_flags_tool_errors()
