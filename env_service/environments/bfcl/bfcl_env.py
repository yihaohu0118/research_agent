# -*- coding: utf-8 -*-
"""
This file is part of https://github.com/ShishirPatil/gorilla

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# environments/bfcl_env.py
from __future__ import annotations
import json, os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List
import re
import uuid

from env_service.base import BaseEnv
from env_service.registry import Registry
from env_service.trajectory import StateMessage, ActionMessage, ToolCall


# 默认路径，可用环境变量覆盖
os.environ.setdefault("BFCL_DATA_PATH", "./bfcl_data/multiturn_data.jsonl")
os.environ.setdefault("BFCL_ANSWER_PATH", "./bfcl_eval/possible_answer")

__all__ = ["BfclEnv"]


T3RL_BFCL_SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

Your response must always start with your step-by-step reasoning process enclosed in `<thinking></thinking>` XML tags. This is for you to outline your plan and justify your chosen actions.

After the thinking block, you will perform one of the following actions:

1. If tool calls are necessary: For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags.
Examples:
<tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>

2. If no tool calls are necessary or possible: Directly provide a user-facing response in plain text. This applies if none of the functions can be used, or if the given question lacks the parameters required by the function.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task."""


def parse_assistant_content_to_tool_calls(
    msg: Dict[str, Any], strict: bool = False
) -> Dict[str, Any]:
    """
    从 assistant 的 content 中解析出 tool_calls，并返回新的消息结构。
    支持 Qwen 的 <tool_call>...<tool_call> 工具调用格式。

    Args:
        msg (dict): 原始 assistant 消息，包含 'content' 字段

    Returns:
        dict: 包含 'content' 和 'tool_calls' 的新消息结构
    """
    content = msg.get("content", "") or ""
    if not isinstance(content, str):
        content = str(content)

    tool_calls = []
    remaining_content = content
    call_id_counter = 1

    # 正则匹配 <tool_call> ... asdf ... asdf ...
    pattern = r'<tool_call>\s*\n?({.*?})\s*\n?\</tool_call>'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    parse_errors: list[str] = []
    has_tool_tag = "<tool_call" in content or "</tool_call>" in content

    if not matches:
        result = {
            "role": "assistant",
            "content": content.strip(),
            "tool_calls": []
        }
        if strict and has_tool_tag:
            result["_bfcl_parse_error"] = (
                "Found tool_call tag text, but no valid <tool_call>{...}</tool_call> block."
            )
        return result

    # 提取所有匹配的 JSON 字符串
    for match in matches:
        json_str = match.group(1).strip()
        try:
            data = json.loads(json_str)
            if not isinstance(data, dict):
                continue
            if "name" not in data or "arguments" not in data:
                parse_errors.append("Tool call JSON must contain both 'name' and 'arguments'.")
                continue
            
            func_name = data["name"]
            arguments = data["arguments"]
            if strict:
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        parse_errors.append(
                            f"Tool '{func_name}' has non-JSON string arguments."
                        )
                        continue
                if not isinstance(arguments, dict):
                    parse_errors.append(
                        f"Tool '{func_name}' arguments must be a JSON object."
                    )
                    continue
            tool_call = {
                "id": f"{func_name}_{call_id_counter}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": arguments
                }
            }
            tool_calls.append(tool_call)
            call_id_counter += 1
        except json.JSONDecodeError as e:
            print(f"JSON 解析失败: {json_str[:50]}... -> {e}")
            parse_errors.append(f"Invalid tool call JSON: {e}")
            continue

    # 移除所有 tool call 部分，得到纯文本 content
    cleaned_content = re.sub(pattern, '', content, flags=re.DOTALL).strip()
    # 可选：清理多余的空白
    cleaned_content = re.sub(r'\n\s*\n', '\n\n', cleaned_content).strip()

    result = {
        "role": "assistant",
        "content": cleaned_content,
        "tool_calls": tool_calls
    }
    if strict and parse_errors:
        result["_bfcl_parse_error"] = "; ".join(parse_errors)

    return result

def tools_schema_to_qwen_prompt(tools_schema, prompt_mode: str = "bfcl_qwen_fc"):
    """
    将 tools_schema 转换为符合 Qwen 模型 chat_template 的工具描述 prompt。

    Args:
        tools_schema (list): 工具列表，格式如下：
            [
                {
                    "name": "tool_name",
                    "description": "工具描述",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [ ... ]
                    }
                }
            ]

    Returns:
        str: 包含 <tools> 标签的完整 system 工具描述 prompt
    """
    if not tools_schema:
        return ""

    lines = []
    if prompt_mode == "bfcl_qwen_fc":
        lines.append("# Tools\n")
        lines.append("You may call one or more functions to assist with the user query.\n")
        lines.append("You are provided with function signatures within <tools></tools> XML tags:")
    elif prompt_mode == "t3rl_text":
        lines.append(T3RL_BFCL_SYSTEM_PROMPT.strip())
        lines.append("\n\n# Tools\n")
        lines.append("You are provided with function signatures within <tools></tools> XML tags:")
    else:
        lines.append("\n\n# Tools\n")
        lines.append("You may call one or more functions to assist with the user query.\n")
        lines.append("You are provided with function signatures within <tools></tools> XML tags:")
    lines.append("<tools>")
    for tool in tools_schema:
        if prompt_mode == "bfcl_qwen_fc":
            # Match BFCL's QwenFCHandler prompt style as closely as possible.
            tool_json = json.dumps(tool)
        else:
            tool_json = json.dumps(
                tool,
                ensure_ascii=False,
                separators=(',', ':')
            )
        lines.append(tool_json)
    lines.append("</tools>\n")
    if prompt_mode == "bfcl_qwen_fc":
        lines.append("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:")
        lines.append("<tool_call>")
        lines.append('{"name": <function-name>, "arguments": <args-json-object>}')
        lines.append("</tool_call>")
    elif prompt_mode != "t3rl_text":
        lines.append("Important: Always use only the latest tool list provided, ignoring any functions mentioned in previous messages.")
        lines.append("For each function call, return a json object with function name and arguments within <tool_call> and <tool_call> XML tags:")
        lines.append("<tool_call>")
        lines.append('{\"name\": <function-name>, \"arguments\": <args-json-object>}')
        lines.append("</tool_call>")

    return "\n".join(lines)

def tool_message_to_qwen_text(tool_messages, result_mode: str = "bfcl_tool_response"):
    """
    将 role 为 'tool' 的消息列表转换为符合 Qwen chat_template 格式的字符串。
    支持单个或多个连续的 tool 消息。

    Args:
        tool_messages (list or dict): 一个或多个 tool 消息字典

    Returns:
        str: 符合 Qwen 模板的文本表示，包含 <|im_start|>user ... <|im_end|>
    """
    if isinstance(tool_messages, dict):
        tool_messages = [tool_messages]

    if not tool_messages:
        return ""

    # AgentEvolver feeds tool observations back as user-role text. The
    # bfcl_tool_response branch mirrors BFCL QwenFCHandler's tool rendering.
    tool_entries = []
    for msg in tool_messages:
        if msg.get("role") != "tool":
            raise ValueError("All messages must have role 'tool'")

        content = msg.get("content", "")
        tool_call_id = msg.get("tool_call_id", "")
        # NOTICE: yunpeng - bfcl 不返回toolname，用id代替
        name = msg.get("name", tool_call_id)  # 工具名称 

        if not name:
            raise ValueError("Missing 'name' in tool message.")

        if result_mode == "bfcl_tool_response":
            if isinstance(content, str):
                content_text = content
            else:
                content_text = json.dumps(content, ensure_ascii=False)
            tool_entries.append(f"<tool_response>\n{content_text}\n</tool_response>")
            continue

        # 确保 content 是 JSON 可序列化对象
        try:
            if isinstance(content, str):
                parsed_content = json.loads(content) if content.strip().startswith(('{', '[')) else content
            else:
                parsed_content = content
        except Exception:
            parsed_content = content

        # 构造工具返回的标准结构：{"name": "...", "content": ...}
        entry = {
            "name": name,
            "content": parsed_content
        }
        if result_mode == "plain_user":
            if isinstance(parsed_content, str):
                content_text = parsed_content
            else:
                content_text = json.dumps(parsed_content, ensure_ascii=False)
            tool_entries.append(f"Tool result from {name}:\n{content_text}")
        else:
            tool_entries.append(f'<tool_call>\n{json.dumps(entry, ensure_ascii=False)}\n</tool_call>')

    # 合并所有 tool entry，用换行连接
    inner_text = "\n".join(tool_entries) + "\n"

    return inner_text

@Registry.register("bfcl")
class BfclEnv(BaseEnv):
    """Berkeley-Function-Calling-Leaderboard 多轮对话环境"""

    # ------------------------------------------------------------------ #
    # 初始化
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        task_id: str | None = None,
        instance_id: str | None = None,
        params: Dict[str, Any] | None = None,
    ):
        self.task_id, self.instance_id = task_id, instance_id
        self.params: Dict[str, Any] = params or {}

        self.data_path = self.params.get("data_path", os.getenv("BFCL_DATA_PATH"))
        self.answer_path = self.params.get("answer_path", os.getenv("BFCL_ANSWER_PATH"))
        self.model_name = self.params.get("model_name", "env_handler")

        # runtime
        self.test_entry: Dict[str, Any] | None = None
        self.original_test_entry: Dict[str, Any] | None = None
        self.env_handler = None
        self.conversation_history: list[Dict[str, Any]] = []
        self.current_turn = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.tools_info = ""

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    def get_init_state(self, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """载入测试用例并返回首条 user 消息"""
        if params:
            self.params.update(params)
        self.test_entry = self._load_test_case(self.data_path, self.task_id)
        self.original_test_entry = self.test_entry

        # 必须成功实例化真实 EnvHandler
        from env_service.environments.bfcl.env_handler import EnvHandler

        self.env_handler = EnvHandler(
            model_name=self.model_name, answer_path=Path(self.answer_path)
        )

        # 初始历史
        self.conversation_history = self.test_entry.get("question", [[]])[0].copy()
        self.current_turn = 0

        # 工具信息
        tools = self.test_entry.get("function", [])
        presented_tools = self._apply_observation_lite(tools)
        # print("tools:", tools)
        self.tools_info = "Available tools:\n" + "\n".join(
            f"- {t.get('function', {}).get('name', 'unknown')}" for t in tools
        )

        first_query = (
            self.conversation_history[0]["content"] if self.conversation_history else ""
        )

        # add system prompt: czy-0709
        # from bfcl_eval.constants.default_prompts import DEFAULT_SYSTEM_PROMPT
        # from bfcl_eval.model_handler.utils import func_doc_language_specific_pre_processing
        # system_prompt_template = DEFAULT_SYSTEM_PROMPT
        # functions = self.original_test_entry["function"]
        # test_category = self.test_entry["id"].rsplit("_", 1)[0]
        # function_docs = func_doc_language_specific_pre_processing(functions, test_category)
        # system_prompt = system_prompt_template.format(functions=function_docs)
        tool_prompt = tools_schema_to_qwen_prompt(
            presented_tools,
            prompt_mode=self.params.get("tool_prompt_mode", "bfcl_qwen_fc"),
        )
        if self._observation_lite_enabled():
            tool_prompt = f"{tool_prompt}\n\n{self._observation_lite_instruction()}"
        return {
            # system_prompt + "\n\n" + first_query
            "state": [
                {"role": "system", "content": tool_prompt},
                {"role": "user", "content": first_query}
                ],
            "info": {
                "instance_id": self.instance_id,
                "task_id": self.task_id,
                "test_id": self.test_entry.get("id", "unknown"),
                "tools_count": len(tools),
                "questions_count": len(self.original_test_entry.get("question", [])),
                "observation_lite": bool(self._observation_lite_enabled()),
            },
        }

    def step(
        self, action: Dict[str, Any], params: Dict[str, Any] | None = None
    ) -> Dict[str, Any]:
        # action: {'content': '<think>\n\n</think>\n\n', 'role': 'assistant', 'tool_calls': [{'id': 'chatcmpl-tool-xxx', 'function': {'arguments': '{..}', 'name': '...'}, 'type': 'function'},{...}]
        ### change by czy0712
        # action_msg = ActionMessage(**action)
        cur_turn=self.current_turn
        state_msg = self.transition(action, params=params or {}) # change by czy0712
        # state_msg: role=<Role.USER: 'user'> content='' reasoning_content='' tool_calls=[ToolCall(...)] timestamp='2025-xxx' metadata={} tool_call_id=''
        terminated = self._is_terminated(state_msg.simple_dict["content"]) # change by czy0721
        # if new query is ready to be sent but single_turn is True
        if cur_turn!=self.current_turn and (self.params.get('is_open_query',False)==True):
            # terminate the trajectory
            terminated=True
        reward = self.evaluate(params={"sparse": True}) if terminated else 0.0
        # print('state_msg.simple_dict',state_msg.simple_dict)
        return {
            "state": [state_msg.simple_dict],
            "reward": reward,
            "is_terminated": terminated,
            "info": {},
        }

    def transition(
        self, assistant_entry: Dict[str, Any], params: Dict[str, Any]
    ) -> StateMessage:
        """执行一次 assistant 行为并让 EnvHandler 给出回应"""
        # change by czy0712
        # action_msg: ActionMessage -> assistant_entry: Dict[str, Any]

        # 记录 assistant 消息
        # assistant_entry: Dict[str, Any] = {
        #     "role": "assistant",
        #     "content": action_msg.content or "",
        # }
        # if action_msg.tool_calls:
        #     assistant_entry["tool_calls"] = [
        #         {
        #             "id": tc.id or f"call_{tc.index}",
        #             "type": "function",
        #             "function": {"name": tc.name, "arguments": tc.arguments},
        #         }
        #         for tc in action_msg.tool_calls
        #     ]

        # content to toolcalls
        assistant_entry = parse_assistant_content_to_tool_calls(
            assistant_entry,
            strict=bool(self.params.get("strict_tool_parser", True)),
        )
        parse_error = assistant_entry.pop("_bfcl_parse_error", None)

        self.conversation_history.append(assistant_entry) ### change by czy0712

        if parse_error:
            assistant_entry["_bfcl_rejected_tool_calls"] = assistant_entry.get(
                "tool_calls", []
            )
            assistant_entry["tool_calls"] = []
            return StateMessage(
                role="user",
                content=f"[ERROR] Invalid tool call format: {parse_error}",
            )

        # ➜ 必须有已初始化的 handler
        if self.env_handler is None or self.original_test_entry is None:
            raise RuntimeError(
                "EnvHandler not initialised – call get_init_state() first."
            )
        # 与环境交互后env_response有两种返回情况: 
        # 1. 触发query, 附带着available tools列表, {"messages": [{"role": "user", "content": user_query}], "tools": tools} 
        # 2. 返回工具调用结果, {"messages": [{"role": "tool", "content": {<execution_results>}, 'tool_call_id': 'chatcmpl-tool-xxx'}]}
        #    <execution_results>: 正确执行时返回结果dict, e.g., {"travel_cost_list": [1140.0]}, 错误时返回error信息, e.g., {"error": "cd: temporary: No such directory. You cannot use path to change directory."}
        env_resp = self.env_handler.interact(
            self.conversation_history,
            self.original_test_entry,
            enforce_available_tools=bool(
                self.params.get("enforce_available_tools", True)
            ),
        )
        # print('env_resp in bfcl_env.py', env_resp)
        if env_resp.get("invalid_tool_call"):
            assistant_entry["_bfcl_rejected_tool_calls"] = assistant_entry.get(
                "tool_calls", []
            )
            assistant_entry["tool_calls"] = []

        return self._append_env_response(env_resp)

    def _append_env_response(
        self,
        env_resp: Dict[str, Any],
    ) -> StateMessage:
        """Append EnvHandler response to history and build the next state."""
        new_tool_calls: list[ToolCall] = []
        next_msg_content = ""

        # FIXME: yunpeng - messages 可能是多个吗
        for idx, msg in enumerate(env_resp.get("messages", [])):
            self.conversation_history.append(msg)
            if msg["role"] == "tool":
                # FIXME 改成一次性传入所有tool messages
                next_msg_content += tool_message_to_qwen_text(
                    msg,
                    result_mode=self.params.get("tool_result_mode", "bfcl_tool_response"),
                )
            elif msg["role"] == "user":
                next_msg_content = msg.get("content", "")
                # FIXME: yunpeng 更新新的tool schema到user msg里
                self.current_turn += 1
            elif msg["role"] == "env":
                # two situations:
                # 1. [{"role": "env", "content": "[CONVERSATION_COMPLETED]"}]
                # 2. [{"role": "env", "content": f"[ERROR] {error_message}"}]
                next_msg_content = msg.get("content", "")

        return (
            StateMessage(role="user",content=next_msg_content)
             ###### changed by czy0718, without tool_role, trajectory.steps cannot be converted by tool_execution_results
        )

    def _train_only_enabled(self, cfg: Dict[str, Any]) -> bool:
        if not bool(cfg.get("enable", False)):
            return False
        mode = str(self.params.get("rollout_mode", "") or "").lower()
        if mode in {"validate", "validation", "val", "test"} and not bool(
            cfg.get("apply_to_validation", False)
        ):
            return False
        categories = cfg.get("categories", None)
        if isinstance(categories, str):
            categories = [categories]
        if categories:
            category = self._test_category()
            if category not in set(str(item) for item in categories):
                return False
        return True

    def _observation_lite_config(self) -> Dict[str, Any]:
        cfg = self.params.get("observation_lite")
        return cfg if isinstance(cfg, dict) else {}

    def _observation_lite_enabled(self) -> bool:
        return self._train_only_enabled(self._observation_lite_config())

    def _observation_lite_instruction(self) -> str:
        cfg = self._observation_lite_config()
        return str(
            cfg.get(
                "instruction",
                "Observation note: parameters marked [required] must be "
                "provided exactly when calling a tool; do not invent tools "
                "outside the current schema.",
            )
        ).strip()

    def _apply_observation_lite(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self._observation_lite_enabled():
            return tools

        cfg = self._observation_lite_config()
        add_required = bool(cfg.get("add_required_hint", True))
        add_type = bool(cfg.get("add_type_hint", False))
        add_enum = bool(cfg.get("add_enum_hint", True))
        hide_optional = bool(cfg.get("hide_optional", False))

        patched_tools = deepcopy(tools)
        for tool in patched_tools:
            func = tool.get("function", tool)
            params = func.get("parameters", {}) or {}
            props = params.get("properties", {}) or {}
            required = set(params.get("required", []) or [])

            if hide_optional:
                for key in list(props.keys()):
                    if key not in required:
                        props.pop(key, None)

            for name, spec in props.items():
                if not isinstance(spec, dict):
                    continue
                desc = str(spec.get("description", "") or "")
                additions: list[str] = []
                if add_required and name in required and "[required]" not in desc:
                    additions.append("[required]")
                if add_type and spec.get("type") and "[type:" not in desc:
                    additions.append(f"[type: {spec.get('type')}]")
                enum_values = spec.get("enum")
                if add_enum and enum_values and "Allowed values:" not in desc:
                    additions.append(
                        "Allowed values: " + ", ".join(str(v) for v in enum_values)
                    )
                if additions:
                    spec["description"] = f"{desc} {' '.join(additions)}".strip()

        return patched_tools

    def _test_category(self) -> str:
        entry = self.test_entry or self.original_test_entry or {}
        test_id = str(entry.get("id") or self.task_id or "")
        return test_id.rsplit("_", 1)[0] if "_" in test_id else test_id

    def evaluate(
        self,
        messages: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ):
        """调用 EnvHandler 评估对话"""
        if self.env_handler is None:
            raise RuntimeError("EnvHandler not initialised – cannot evaluate.")

        conv_result = {
            "test_id": self.test_entry.get("id", "unknown"),
            "messages": self.conversation_history,
            "turn_count": self.current_turn,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "completed": self._is_terminated(self.conversation_history[-1]["content"]),  ### changed by czy0721
            "original_test_entry": self.original_test_entry,
        }
        sparse = (params or {}).get("sparse", False)
        result = self.env_handler.evaluate(conv_result)
        return result.get("accuracy", 0.0) if sparse else result

    def get_info(
        self,
        messages: Dict[str, Any] | None = None,
        params: Dict[str, Any] | None = None,
    ) -> str:
        return self.tools_info

    def close(self):  # Ray actor cleanup hook
        self.conversation_history.clear()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _is_terminated(self, env_content) -> bool:
        # return self.original_test_entry is not None and self.current_turn >= len(
        #     self.original_test_entry.get("question", [])
        # ) 
        # changed by czy0721, since current_turn is 0-based index
        return env_content == "[CONVERSATION_COMPLETED]"

    @staticmethod
    def _load_test_case(data_path: str, test_id: str | None) -> Dict[str, Any]:
        """按 ID / 行号加载单条 JSONL 测试用例。找不到就抛错。"""
        if not Path(data_path).exists():
            raise FileNotFoundError(f"BFCL data file '{data_path}' not found")

        if test_id is None:
            raise ValueError("task_id is required")

        with open(data_path, "r", encoding="utf-8") as f:
            if str(test_id).isdigit():
                idx = int(test_id)
                for line_no, line in enumerate(f):
                    if line_no == idx:
                        return json.loads(line)
                raise ValueError(f"Test case index {idx} not found in {data_path}")
            else:
                for line in f:
                    data = json.loads(line)
                    if data.get("id") == test_id:
                        return data
                raise ValueError(f"Test case id '{test_id}' not found in {data_path}")

    # 静态接口给 env_service 用
    @staticmethod
    def get_query_list(split: str = "train", params={"category": ["multi_turn_base"]}):
        """
        Get query list from preprocessed dataset.
        
        Args:
            split: Dataset split, either 'train' or 'test'
            params: Parameters to filter dataset (currently supports 'category')
        
        Returns:
            List of query id
        """

        path = os.getenv("BFCL_SPLID_ID_PATH")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)[split]
            # return [json.loads(l)["id"] for l in f]
