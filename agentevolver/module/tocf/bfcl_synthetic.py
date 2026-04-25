from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_BUILTIN_NAMES = {
    "bool",
    "dict",
    "float",
    "int",
    "len",
    "list",
    "print",
    "set",
    "str",
    "tuple",
}


@dataclass(frozen=True)
class NormalizedToolCall:
    name: str
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()

    @property
    def kwargs_dict(self) -> dict[str, Any]:
        return dict(self.kwargs)

    def canonical(self) -> str:
        pieces = [repr(value) for value in self.args]
        pieces.extend(f"{key}={repr(value)}" for key, value in self.kwargs)
        return f"{self.name}({', '.join(pieces)})"


def _json_ready(value: Any) -> Any:
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.unparse(node) if hasattr(ast, "unparse") else repr(node)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _normalize_call_node(node: ast.Call) -> NormalizedToolCall | None:
    name = _call_name(node.func)
    if not name or name in _BUILTIN_NAMES:
        return None
    args = tuple(_literal(arg) for arg in node.args)
    kwargs = tuple(
        (str(keyword.arg), _literal(keyword.value))
        for keyword in node.keywords
        if keyword.arg is not None
    )
    return NormalizedToolCall(name=name, args=args, kwargs=kwargs)


def parse_tool_call_expr(expr: str) -> NormalizedToolCall | None:
    text = str(expr or "").strip().rstrip(",")
    if not text:
        return None

    try:
        parsed = ast.parse(text, mode="eval")
        if isinstance(parsed.body, ast.Call):
            return _normalize_call_node(parsed.body)
    except SyntaxError:
        pass

    try:
        module = ast.parse(text, mode="exec")
    except SyntaxError:
        return None

    for node in ast.walk(module):
        if isinstance(node, ast.Call):
            call = _normalize_call_node(node)
            if call is not None:
                return call
    return None


def _iter_balanced_call_exprs(text: str) -> Iterable[str]:
    source = str(text or "")
    index = 0
    while index < len(source):
        match = re.search(r"[A-Za-z_][\w\.]*\s*\(", source[index:])
        if not match:
            break
        start = index + match.start()
        pos = index + match.end() - 1
        depth = 0
        quote = ""
        escaped = False
        end = None
        while pos < len(source):
            ch = source[pos]
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = ""
            else:
                if ch in ("'", '"'):
                    quote = ch
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end = pos + 1
                        break
            pos += 1
        if end is None:
            break
        yield source[start:end]
        index = end


def _calls_from_tool_tags(text: str) -> list[NormalizedToolCall]:
    calls: list[NormalizedToolCall] = []
    for match in _TOOL_CALL_RE.finditer(str(text or "")):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name") or payload.get("function", {}).get("name")
        arguments = payload.get("arguments")
        if arguments is None:
            arguments = payload.get("function", {}).get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict) or not name:
            continue
        calls.append(
            NormalizedToolCall(
                name=str(name),
                kwargs=tuple((str(key), value) for key, value in arguments.items()),
            )
        )
    return calls


def calls_from_text(text: str) -> list[NormalizedToolCall]:
    tagged = _calls_from_tool_tags(text)
    if tagged:
        return tagged

    calls: list[NormalizedToolCall] = []
    seen: set[str] = set()
    for expr in _iter_balanced_call_exprs(text):
        call = parse_tool_call_expr(expr)
        if call is None:
            continue
        key = call.canonical()
        if key in seen:
            continue
        seen.add(key)
        calls.append(call)
    return calls


def _coerce_turns(value: Any) -> list[list[str]]:
    value = _json_ready(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            sections = re.split(r"(?im)^\s*#\s*step\s*\d+\s*$", stripped)
            blocks = [section.strip() for section in sections if section.strip()]
            return [[block] for block in (blocks or [stripped])]
        return _coerce_turns(parsed)
    if isinstance(value, list):
        if not value:
            return []
        if all(isinstance(item, str) for item in value):
            return [[str(item) for item in value]]
        turns: list[list[str]] = []
        for item in value:
            if isinstance(item, list):
                turns.extend(_coerce_turns(item))
            elif isinstance(item, str):
                turns.append([item])
            else:
                turns.extend(_coerce_turns(item))
        return turns
    return _coerce_turns(str(value))


def normalize_tool_turns(raw: Any) -> list[list[NormalizedToolCall]]:
    turns: list[list[NormalizedToolCall]] = []
    for block_or_calls in _coerce_turns(raw):
        calls: list[NormalizedToolCall] = []
        for item in block_or_calls:
            calls.extend(calls_from_text(item))
        if calls:
            turns.append(calls)
    return turns


def serialize_tool_turns(turns: list[list[NormalizedToolCall]]) -> str:
    return json.dumps(
        [[call.canonical() for call in turn] for turn in turns],
        ensure_ascii=False,
    )


def tool_turns_to_strings(turns: list[list[NormalizedToolCall]]) -> list[list[str]]:
    return [[call.canonical() for call in turn] for turn in turns]


def _first_user_content(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def _normalize_question_schedule(value: Any) -> list[list[dict[str, str]]] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return _normalize_question_schedule(json.loads(stripped))
        except json.JSONDecodeError:
            return [[{"role": "user", "content": stripped}]]
    if not isinstance(value, list) or not value:
        return None

    if all(isinstance(item, str) for item in value):
        return [[{"role": "user", "content": str(item)}] for item in value]
    if all(isinstance(item, dict) for item in value):
        value = [value]

    schedule: list[list[dict[str, str]]] = []
    for turn in value:
        if isinstance(turn, str):
            messages = [{"role": "user", "content": turn}]
        elif isinstance(turn, dict):
            messages = [turn]
        elif isinstance(turn, list):
            messages = [message for message in turn if isinstance(message, dict)]
        else:
            return None

        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role != "user":
                return None
            normalized_messages.append({"role": role, "content": content})
        if not normalized_messages or not _first_user_content(normalized_messages).strip():
            return None
        schedule.append(normalized_messages)
    return schedule or None


def _schedule_user_queries(schedule: list[list[dict[str, str]]] | None) -> list[str]:
    if not schedule:
        return []
    return [_first_user_content(turn).strip() for turn in schedule]


def _semantic_tokens(text: Any) -> set[str]:
    raw = str(text or "").lower().replace("_", " ").replace("-", " ")
    tokens = set(re.findall(r"[a-z0-9]+", raw))
    return {token for token in tokens if len(token) >= 3 or token.isdigit()}


def _iter_literal_strings(value: Any) -> Iterable[str]:
    value = _json_ready(value)
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, (str, int, float)):
        text = str(value).strip()
        if text:
            yield text
        return
    if isinstance(value, list):
        for item in value:
            yield from _iter_literal_strings(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_literal_strings(item)


_TOOL_ALIASES: dict[str, set[str]] = {
    "cd": {"cd", "change", "open", "enter", "folder", "directory"},
    "grep": {"grep", "find", "search", "match", "contains", "pattern"},
    "mkdir": {"mkdir", "create", "make", "folder", "directory"},
    "mv": {"mv", "move", "rename", "place", "transfer"},
    "sort": {"sort", "order", "arrange", "rank"},
    "cat": {"cat", "read", "show", "display", "view"},
    "ls": {"list", "show", "files", "folders", "directory"},
    "rm": {"remove", "delete"},
    "cp": {"copy", "duplicate"},
}


def validate_bfcl_synthetic_alignment(
    query: str | None,
    turns: list[list[NormalizedToolCall]],
    metadata: dict[str, Any] | None = None,
    *,
    min_score: float = 0.34,
) -> dict[str, Any]:
    schedule = _metadata_question_schedule(metadata)
    if schedule is None:
        schedule = [[{"role": "user", "content": str(query or "")}]]
    queries = _schedule_user_queries(schedule)
    flat_calls = flatten_turns(turns)
    if not queries or not flat_calls:
        return {
            "ok": False,
            "score": 0.0,
            "reason": "semantic_missing_query_or_gt",
            "grounded_calls": 0,
            "total_calls": len(flat_calls),
            "grounded_values": 0,
            "total_values": 0,
        }

    grounded_calls = 0
    total_values = 0
    grounded_values = 0
    for turn_idx, turn in enumerate(turns):
        query_tokens = _semantic_tokens(queries[min(turn_idx, len(queries) - 1)])
        for call in turn:
            name_tokens = _semantic_tokens(call.name)
            aliases = set(_TOOL_ALIASES.get(call.name, set())) | name_tokens
            op_grounded = bool(query_tokens & aliases)
            value_grounded = False
            values = list(call.args) + [value for _, value in call.kwargs]
            for value in values:
                value_strings = list(_iter_literal_strings(value))
                if not value_strings:
                    continue
                total_values += 1
                value_tokens = set()
                for text in value_strings:
                    value_tokens.update(_semantic_tokens(text))
                if value_tokens and (query_tokens & value_tokens):
                    grounded_values += 1
                    value_grounded = True
            if op_grounded or value_grounded:
                grounded_calls += 1

    call_score = grounded_calls / max(1, len(flat_calls))
    value_score = grounded_values / total_values if total_values else call_score
    score = 0.6 * call_score + 0.4 * value_score
    ok = score >= min_score
    return {
        "ok": ok,
        "score": score,
        "reason": "ok" if ok else "synthetic_semantic_unanchored",
        "grounded_calls": grounded_calls,
        "total_calls": len(flat_calls),
        "grounded_values": grounded_values,
        "total_values": total_values,
    }


def _metadata_question_schedule(metadata: dict[str, Any] | None) -> list[list[dict[str, str]]] | None:
    metadata = metadata or {}
    candidates = [
        metadata.get("bfcl_synthetic_question_schedule"),
        ((metadata.get("tocf") or {}).get("bfcl_synthetic") or {}).get("question_schedule"),
        ((metadata.get("tocf") or {}).get("coevo") or {}).get("question_schedule"),
    ]
    for candidate in candidates:
        schedule = _normalize_question_schedule(candidate)
        if schedule:
            return schedule
    return None


def build_bfcl_synthetic_case_overlay(
    *,
    task_id: str,
    query: str | None,
    ground_truth: Any,
    metadata: dict[str, Any] | None = None,
    turns: list[list[NormalizedToolCall]] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Build a BFCL case overlay that keeps prompt state aligned with synthetic GT."""
    query_text = str(query or "").strip()
    if not query_text:
        return None, "synthetic_query_missing"

    normalized_turns = turns if turns is not None else normalize_tool_turns(ground_truth)
    if not normalized_turns:
        return None, "synthetic_gt_unparseable"

    schedule = _metadata_question_schedule(metadata)
    if schedule is None:
        if len(normalized_turns) > 1:
            return None, "synthetic_multiturn_unaligned"
        schedule = [[{"role": "user", "content": query_text}]]

    if len(schedule) != len(normalized_turns):
        return None, "synthetic_turn_mismatch"

    first_user = _first_user_content(schedule[0]).strip()
    if first_user != query_text:
        return None, "synthetic_state_mismatch"

    semantic_alignment = validate_bfcl_synthetic_alignment(
        query_text,
        normalized_turns,
        metadata,
    )
    serialized_gt = serialize_tool_turns(normalized_turns)
    objective_hash = hashlib.sha1(
        f"{task_id}\n{query_text}\n{serialized_gt}".encode("utf-8")
    ).hexdigest()[:12]
    return (
        {
            "version": 1,
            "source": "bfcl_synthetic_overlay",
            "parent_task_id": str(task_id),
            "objective_hash": objective_hash,
            "question": schedule,
            "ground_truth_turns": tool_turns_to_strings(normalized_turns),
            "turn_count": len(schedule),
            "semantic_alignment": semantic_alignment,
        },
        "ok",
    )


def bfcl_synthetic_env_params(
    task: Any,
    params: dict[str, Any] | None = None,
    *,
    turns: list[list[NormalizedToolCall]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Return BFCL env params with a synthetic overlay when the task is synthetic."""
    merged = dict(params or {})
    overlay, reason = build_bfcl_synthetic_case_overlay(
        task_id=str(getattr(task, "task_id", "")),
        query=getattr(task, "query", None),
        ground_truth=getattr(task, "ground_truth", None),
        metadata=getattr(task, "metadata", None),
        turns=turns,
    )
    if overlay is not None:
        merged["synthetic_case_overlay"] = overlay
        merged["is_open_query"] = bool(len(overlay.get("question") or []) <= 1)
    return merged, reason


def synthetic_bfcl_case_id(parent_id: str, objective_hash: str | None) -> str:
    parent_id = str(parent_id or "bfcl")
    category = parent_id.rsplit("_", 1)[0] if "_" in parent_id else parent_id
    suffix = re.sub(r"[^A-Za-z0-9]", "", str(objective_hash or ""))[:12] or uuid.uuid4().hex[:12]
    return f"{category}_syn{suffix}"


def possible_answer_from_overlay(
    overlay: dict[str, Any],
    *,
    case_id: str | None = None,
) -> dict[str, Any]:
    gt_turns = overlay.get("ground_truth_turns") or []
    if not isinstance(gt_turns, list):
        gt_turns = []
    parent_id = str(overlay.get("parent_task_id") or "")
    category = parent_id.rsplit("_", 1)[0] if "_" in parent_id else parent_id
    ground_truth: Any
    if category.startswith("multi_turn") or len(gt_turns) > 1:
        ground_truth = gt_turns
    elif len(gt_turns) == 1:
        ground_truth = gt_turns[0] if gt_turns else []
    else:
        ground_truth = []
    return {
        "id": str(case_id or overlay.get("synthetic_id") or ""),
        "ground_truth": ground_truth,
    }


def materialize_bfcl_synthetic_case(
    parent_entry: dict[str, Any],
    overlay: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(parent_entry, dict):
        raise ValueError("parent_entry must be a dict")
    if not isinstance(overlay, dict):
        raise ValueError("synthetic overlay must be a dict")

    question = _normalize_question_schedule(overlay.get("question"))
    if not question:
        raise ValueError("synthetic overlay question must be a non-empty schedule")

    parent_id = str(parent_entry.get("id") or overlay.get("parent_task_id") or "")
    synthetic_id = synthetic_bfcl_case_id(parent_id, overlay.get("objective_hash"))
    materialized = copy.deepcopy(parent_entry)
    materialized["id"] = synthetic_id
    materialized["question"] = question

    metadata = dict(materialized.get("metadata") or {})
    synthetic_meta = {
        "version": overlay.get("version", 1),
        "source": overlay.get("source", "bfcl_synthetic_overlay"),
        "parent_task_id": overlay.get("parent_task_id", parent_id),
        "objective_hash": overlay.get("objective_hash"),
        "synthetic_id": synthetic_id,
        "turn_count": len(question),
        "ground_truth_turns": overlay.get("ground_truth_turns"),
        "semantic_alignment": overlay.get("semantic_alignment"),
        "materialized": True,
        "case_type": "parent_state_synthetic_question",
        "state_mode": "inherited_parent_initial_config",
    }
    answer_entry = possible_answer_from_overlay(overlay, case_id=synthetic_id)
    synthetic_meta["possible_answer"] = answer_entry
    metadata["synthetic_case_overlay"] = synthetic_meta
    materialized["metadata"] = metadata
    return materialized, answer_entry


def write_materialized_bfcl_synthetic_case(
    case: dict[str, Any],
    answer: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir)
    cases_dir = root / "cases"
    data_dir = root / "data"
    answers_dir = root / "possible_answer"
    cases_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    answers_dir.mkdir(parents=True, exist_ok=True)

    case_id = str(case.get("id") or answer.get("id") or uuid.uuid4().hex)
    category = case_id.rsplit("_", 1)[0] if "_" in case_id else case_id
    case_path = cases_dir / f"{case_id}.json"
    answer_path = answers_dir / f"{case_id}.json"
    case_jsonl_path = data_dir / "synthetic_cases.jsonl"
    answer_jsonl_path = answers_dir / f"{category}.jsonl"
    manifest_path = root / "manifest.jsonl"
    first_write = not case_path.exists()

    for path, payload in ((case_path, case), (answer_path, answer)):
        tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)

    if first_write:
        with case_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
        with answer_jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(answer, ensure_ascii=False) + "\n")
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "id": case_id,
                        "parent_task_id": (
                            (case.get("metadata") or {})
                            .get("synthetic_case_overlay", {})
                            .get("parent_task_id")
                        ),
                        "case_path": str(case_path),
                        "answer_path": str(answer_path),
                        "case_jsonl_path": str(case_jsonl_path),
                        "answer_jsonl_path": str(answer_jsonl_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    return {
        "case_path": str(case_path),
        "answer_path": str(answer_path),
        "case_jsonl_path": str(case_jsonl_path),
        "answer_jsonl_path": str(answer_jsonl_path),
        "manifest_path": str(manifest_path),
    }


def _iter_tool_schema_objects(tool_block: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    text = str(tool_block or "").strip()
    if not text:
        return

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        yield parsed
        return
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                yield item
        return

    index = 0
    while index < len(text):
        next_obj = text.find("{", index)
        if next_obj < 0:
            break
        try:
            value, end = decoder.raw_decode(text[next_obj:])
        except json.JSONDecodeError:
            index = next_obj + 1
            continue
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    yield item
        index = next_obj + end


def tool_schemas_from_prompt(prompt: str) -> dict[str, dict[str, Any]]:
    schemas: dict[str, dict[str, Any]] = {}
    for match in re.finditer(r"<tools>\s*(.*?)\s*</tools>", str(prompt or ""), re.DOTALL):
        for schema in _iter_tool_schema_objects(match.group(1)):
            name = (
                schema.get("name")
                or schema.get("function", {}).get("name")
                or schema.get("function", {}).get("function", {}).get("name")
            )
            if name:
                schemas[str(name)] = schema
    return schemas


def _schema_parameter_order(schema: dict[str, Any] | None) -> list[str]:
    schema = schema or {}
    parameters = schema.get("parameters") or schema.get("function", {}).get("parameters") or {}
    if not isinstance(parameters, dict):
        return []
    properties = parameters.get("properties") or {}
    required = [str(item) for item in parameters.get("required") or []]
    order = list(required)
    for key in properties:
        key = str(key)
        if key not in order:
            order.append(key)
    return order


def parse_tool_schema_from_prompt(prompt: str) -> dict[str, list[str]]:
    return {
        name: _schema_parameter_order(schema)
        for name, schema in tool_schemas_from_prompt(prompt).items()
    }


def call_to_tool_payload(
    call: NormalizedToolCall,
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    arguments = dict(call.kwargs)
    if call.args:
        order = _schema_parameter_order((tool_schemas or {}).get(call.name))
        available = [name for name in order if name not in arguments]
        if len(available) < len(call.args):
            return None
        for key, value in zip(available, call.args):
            arguments[key] = value
    return {"name": call.name, "arguments": arguments}


def call_to_tool_call_payload(
    call: NormalizedToolCall,
    schema_args: list[str] | None = None,
) -> dict[str, Any] | None:
    arguments = dict(call.kwargs)
    if call.args:
        if not schema_args:
            return None
        available = [name for name in schema_args if name not in arguments]
        if len(available) < len(call.args):
            return None
        for key, value in zip(available, call.args):
            arguments[key] = value
    return {"name": call.name, "arguments": arguments}


def normalize_turns_for_tool_schema(
    turns: list[list[NormalizedToolCall]],
    tool_schemas: dict[str, dict[str, Any]] | None,
) -> list[list[NormalizedToolCall]] | None:
    normalized_turns: list[list[NormalizedToolCall]] = []
    for turn in turns:
        normalized_calls: list[NormalizedToolCall] = []
        for call in turn:
            payload = call_to_tool_payload(call, tool_schemas)
            if payload is None:
                return None
            arguments = payload.get("arguments") or {}
            if not isinstance(arguments, dict):
                return None
            normalized_calls.append(
                NormalizedToolCall(
                    name=str(payload.get("name") or call.name),
                    kwargs=tuple((str(key), value) for key, value in arguments.items()),
                )
            )
        if normalized_calls:
            normalized_turns.append(normalized_calls)
    return normalized_turns


def tool_turns_to_tool_call_content(
    turns: list[list[NormalizedToolCall]],
    tool_schemas: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    blocks: list[str] = []
    for call in flatten_turns(turns):
        payload = call_to_tool_payload(call, tool_schemas)
        if payload is None:
            return None
        blocks.append(
            "<tool_call>\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n</tool_call>"
        )
    if not blocks:
        return None
    return "\n".join(blocks)


def _state_messages(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("state"), list):
            return [item for item in payload["state"] if isinstance(item, dict)]
        if isinstance(payload.get("state"), dict):
            return [payload["state"]]
        if payload.get("role"):
            return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _first_system_prompt(messages: Iterable[dict[str, Any]]) -> str:
    for message in messages or []:
        if message.get("role") == "system":
            return str(message.get("content") or "")
    return ""


def replay_tool_turns_in_bfcl_env(
    env: Any,
    task_id: str,
    turns: list[list[NormalizedToolCall]],
    *,
    params: dict[str, Any] | None = None,
    instance_prefix: str = "bfcl_gt",
) -> tuple[bool, str]:
    if not turns:
        return False, "synthetic_gt_unparseable"

    instance_id = f"{instance_prefix}_{uuid.uuid4().hex}"
    transcript: list[dict[str, Any]] = []
    try:
        init_response = env.create_instance(
            env_type="bfcl",
            task_id=str(task_id),
            instance_id=instance_id,
            params=params or {},
        )
        init_messages = _state_messages(init_response)
        transcript.extend(init_messages)
        overlay = (params or {}).get("synthetic_case_overlay")
        schedule = None
        if isinstance(overlay, dict):
            schedule = _normalize_question_schedule(overlay.get("question"))
            expected_query = _first_user_content(schedule[0]) if schedule else ""
            actual_query = _first_user_content(init_messages)
            if expected_query.strip() and actual_query.strip() != expected_query.strip():
                return False, "synthetic_state_mismatch"
            if schedule and len(schedule) != len(turns):
                return False, "synthetic_turn_mismatch"
        tool_schemas = tool_schemas_from_prompt(_first_system_prompt(init_messages))

        for turn_idx, turn in enumerate(turns):
            content = tool_turns_to_tool_call_content([turn], tool_schemas)
            if content is None:
                return False, "synthetic_gt_payload_unmappable"

            step_response = env.step(
                instance_id,
                {"role": "assistant", "content": content},
            )
            step_messages = _state_messages(step_response)
            transcript.extend(step_messages)
            if trajectory_has_tool_error(transcript):
                return False, "synthetic_gt_tool_error"

            if turn_idx >= len(turns) - 1:
                continue

            advance_response = env.step(
                instance_id,
                {"role": "assistant", "content": "Done."},
            )
            advance_messages = _state_messages(advance_response)
            transcript.extend(advance_messages)
            if trajectory_has_tool_error(transcript):
                return False, "synthetic_gt_tool_error"
            if schedule:
                expected_next_query = _first_user_content(schedule[turn_idx + 1]).strip()
                actual_next_query = _first_user_content(advance_messages).strip()
                if expected_next_query and actual_next_query != expected_next_query:
                    return False, "synthetic_state_mismatch"
        return True, "ok"
    except Exception as exc:
        return False, f"synthetic_gt_replay_error:{type(exc).__name__}:{str(exc)[:240]}"
    finally:
        try:
            env.release_instance(instance_id)
        except Exception:
            pass


def extract_observed_tool_turns(messages: Iterable[dict[str, Any]]) -> list[list[NormalizedToolCall]]:
    turns: list[list[NormalizedToolCall]] = []
    current: list[NormalizedToolCall] = []
    seen_user_query = False

    for message in messages or []:
        role = message.get("role")
        content = str(message.get("content", "") or "")
        if role == "user":
            is_tool_observation = (
                content.lstrip().startswith("<tool_response>")
                or content.lstrip().startswith("[ERROR]")
                or content.lstrip().startswith("[No content provided")
            )
            if seen_user_query and not is_tool_observation:
                if current:
                    turns.append(current)
                    current = []
            seen_user_query = True
            continue
        if role != "assistant":
            continue
        current.extend(calls_from_text(content))

    if current:
        turns.append(current)
    return turns


def _same_literal(left: Any, right: Any) -> bool:
    return _json_ready(left) == _json_ready(right)


def calls_equivalent(expected: NormalizedToolCall, observed: NormalizedToolCall) -> bool:
    if expected.name != observed.name:
        return False
    if expected.args == observed.args and expected.kwargs == observed.kwargs:
        return True
    if expected.kwargs_dict == observed.kwargs_dict and len(expected.args) == len(observed.args):
        return all(_same_literal(a, b) for a, b in zip(expected.args, observed.args))
    if expected.args and not expected.kwargs and not observed.args and observed.kwargs:
        return [repr(value) for value in expected.args] == [
            repr(value) for _, value in observed.kwargs
        ]
    if observed.args and not observed.kwargs and not expected.args and expected.kwargs:
        return [repr(value) for _, value in expected.kwargs] == [
            repr(value) for value in observed.args
        ]
    return False


def flatten_turns(turns: list[list[NormalizedToolCall]]) -> list[NormalizedToolCall]:
    return [call for turn in turns for call in turn]


def trajectory_has_tool_error(messages: Iterable[dict[str, Any]]) -> bool:
    error_re = re.compile(
        r"\[ERROR\]|Error during execution|Invalid tool call format|"
        r"not available in the current tool list|unexpected keyword argument|"
        r"No such file or directory|No such directory|does not exist|"
        r"Invalid character|[\"']error[\"']\s*:",
        re.IGNORECASE,
    )
    for message in messages or []:
        content = str(message.get("content", "") or "")
        if error_re.search(content):
            return True
    return False


def compare_tool_turns(
    expected_turns: list[list[NormalizedToolCall]],
    observed_turns: list[list[NormalizedToolCall]],
) -> dict[str, Any]:
    expected = flatten_turns(expected_turns)
    observed = flatten_turns(observed_turns)
    if not expected:
        return {
            "score": 0.0,
            "matched": 0,
            "expected": 0,
            "observed": len(observed),
            "success": False,
            "failure_tag": "synthetic_gt_unparseable",
        }

    matched = 0
    first_mismatch = ""
    for expected_call, observed_call in zip(expected, observed):
        if calls_equivalent(expected_call, observed_call):
            matched += 1
            continue
        if not first_mismatch:
            if expected_call.name != observed_call.name:
                first_mismatch = "synthetic_tool_name_mismatch"
            else:
                first_mismatch = "synthetic_arg_mismatch"

    denominator = max(len(expected), len(observed), 1)
    success = matched == len(expected) and len(observed) == len(expected)
    if success:
        failure_tag = "pass"
    elif len(expected_turns) != len(observed_turns) and observed:
        failure_tag = "synthetic_turn_mismatch"
    elif first_mismatch:
        failure_tag = first_mismatch
    elif len(observed) < len(expected):
        failure_tag = "synthetic_missing_call"
    elif len(observed) > len(expected):
        failure_tag = "synthetic_extra_call"
    else:
        failure_tag = "synthetic_gt_mismatch"
    return {
        "score": float(matched) / float(denominator),
        "matched": matched,
        "expected": len(expected),
        "observed": len(observed),
        "success": success,
        "failure_tag": failure_tag,
    }
