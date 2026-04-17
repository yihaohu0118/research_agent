#!/usr/bin/env python3
"""Summarize official vs clean BFCL validation scores from a dumped jsonl log."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path


ERROR_RE = re.compile(
    r"("
    r"\[ERROR\]|"
    r"Error during execution|"
    r"Invalid tool call format|"
    r"not available in the current tool list|"
    r"path not allowed|"
    r"Invalid character|"
    r"No such file or directory|"
    r"unexpected keyword argument|"
    r"\"error\"\s*:"
    r")",
    re.IGNORECASE,
)


def prompt_mode(input_text: str) -> str:
    if "Your response must always start" in input_text:
        return "t3rl_text"
    if "You may call one or more functions" in input_text:
        return "legacy"
    return "unknown"


def pct(num: float, den: int) -> str:
    return f"{100.0 * num / den:.2f}%" if den else "0.00%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="Path to validation_log/*.jsonl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.jsonl.open(encoding="utf-8")]
    if not rows:
        raise SystemExit("empty jsonl")

    by_source: dict[str, list[float]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    prompt_modes = collections.Counter()

    official_success = 0
    clean_success = 0
    dirty_success = 0
    any_error = 0

    for row in rows:
        score = float(row.get("score", 0.0))
        output = str(row.get("output", ""))
        has_error = ERROR_RE.search(output) is not None
        source = row.get("data_source") or "unknown"

        prompt_modes[prompt_mode(str(row.get("input", "")))] += 1
        official_success += int(score >= 1.0)
        any_error += int(has_error)
        dirty_success += int(score >= 1.0 and has_error)
        clean_success += int(score >= 1.0 and not has_error)

        stats = by_source[source]
        stats[0] += 1
        stats[1] += int(score >= 1.0)
        stats[2] += int(score >= 1.0 and not has_error)
        stats[3] += int(score >= 1.0 and has_error)

    total = len(rows)
    print(f"file: {args.jsonl}")
    print(f"rows: {total}")
    print(f"prompt_modes: {dict(prompt_modes)}")
    print(f"official_success: {official_success}/{total} = {pct(official_success, total)}")
    print(f"clean_success:    {clean_success}/{total} = {pct(clean_success, total)}")
    print(f"dirty_success:    {dirty_success}/{total} = {pct(dirty_success, total)}")
    print(f"any_error:        {any_error}/{total} = {pct(any_error, total)}")
    print()
    print("source official clean dirty")
    for source in sorted(by_source):
        n, official, clean, dirty = by_source[source]
        print(
            f"{source} "
            f"{int(official)}/{int(n)}={pct(official, int(n))} "
            f"{int(clean)}/{int(n)}={pct(clean, int(n))} "
            f"{int(dirty)}/{int(n)}={pct(dirty, int(n))}"
        )


if __name__ == "__main__":
    main()
