#!/usr/bin/env python3
"""
Generates sabotaged traces from valid traces to test and calibrate the Referee.
Creates three mutation types:
1. Valid Control Trace (untouched valid trace, expected_verdict = APPROVED).
2. Phantom Tool Execution (deletes tool call/output, leaves assistant success claim).
3. Silent Tool Failure (replaces tool output with error, leaves assistant success claim).
"""
import argparse
import copy
import json
from pathlib import Path
from typing import Any


def mutate_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    mutated: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = record.get("messages", [])

    control: dict[str, Any] = copy.deepcopy(record)
    control["mutation_type"] = "control_valid"
    control["expected_verdict"] = "APPROVED"
    mutated.append(control)

    has_tools: bool = any(m.get("role") == "tool" for m in messages)
    if has_tools:
        phantom: dict[str, Any] = copy.deepcopy(record)
        new_msgs: list[dict[str, Any]] = []
        for m in phantom["messages"]:
            if m.get("role") == "tool":
                continue
            if m.get("role") == "assistant" and "tool_calls" in m:
                continue
            new_msgs.append(m)
        phantom["messages"] = new_msgs
        phantom["mutation_type"] = "sabotage_phantom_tool"
        phantom["expected_verdict"] = "REJECTED"
        mutated.append(phantom)

        silent_fail: dict[str, Any] = copy.deepcopy(record)
        for m in silent_fail["messages"]:
            if m.get("role") == "tool":
                m["content"] = json.dumps({
                    "status": "error",
                    "ok": False,
                    "error": "Execution failed: timeout or permission denied."
                })
        silent_fail["mutation_type"] = "sabotage_silent_tool_failure"
        silent_fail["expected_verdict"] = "REJECTED"
        mutated.append(silent_fail)

    return mutated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate adversarial mutations to calibrate the Referee.")
    parser.add_argument("--input", required=True, help="Valid traces JSONL")
    parser.add_argument("--output", required=True, help="Output mutated_calibration.jsonl")
    parser.add_argument("--limit", type=int, default=15, help="Max source traces to mutate")
    args = parser.parse_args()

    out_records: list[dict[str, Any]] = []
    with Path(args.input).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= args.limit or not line.strip():
                break
            rec: dict[str, Any] = json.loads(line)
            out_records.extend(mutate_record(rec))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r) + "\n")

    print(f"[OK] Generated {len(out_records)} calibration traces (valid + sabotaged) at {out_path}")


if __name__ == "__main__":
    main()
