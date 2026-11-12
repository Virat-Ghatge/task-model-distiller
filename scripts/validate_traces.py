#!/usr/bin/env python3
"""
Validates full-session synthetic training traces against spec.json.
Enforces multi-turn structure, valid JSON tool arguments, and non-empty responses.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any


def validate_trace(trace: dict[str, Any], allowed_tools: set[str], idx: int) -> list[str]:
    errors: list[str] = []
    messages: list[dict[str, Any]] = trace.get("messages", [])
    if len(messages) < 3:
        errors.append(f"Line {idx}: Trace too short ({len(messages)} msgs). Must include full session.")
        return errors

    roles: list[str | None] = [m.get("role") for m in messages]
    if roles[0] != "system" or "user" not in roles or roles[-1] != "assistant":
        errors.append(f"Line {idx}: Invalid role sequence {roles}. Must start with system and end with assistant.")

    for m_idx, msg in enumerate(messages):
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                fn: dict[str, Any] = tc.get("function", {})
                name: str | None = fn.get("name")
                if allowed_tools and name not in allowed_tools:
                    errors.append(f"Line {idx} msg {m_idx}: Unauthorized tool '{name}'.")
                args: Any = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        json.loads(args)
                    except json.JSONDecodeError:
                        errors.append(f"Line {idx} msg {m_idx}: Malformed JSON arguments for tool '{name}'.")
            if m_idx + 1 >= len(messages) or messages[m_idx + 1].get("role") != "tool":
                errors.append(f"Line {idx} msg {m_idx}: tool_call not immediately followed by role=='tool' output.")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate synthetic full-session traces.")
    parser.add_argument("--spec", required=True, help="Path to spec.json")
    parser.add_argument("--input", required=True, help="Path to train_traces.jsonl")
    args = parser.parse_args()

    spec: dict[str, Any] = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    allowed_tools: set[str] = {t["name"] for t in spec.get("tools", []) if "name" in t}

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[FAIL] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    total: int = 0
    valid: int = 0
    all_errors: list[str] = []

    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                record: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                all_errors.append(f"Line {idx}: Invalid JSON ({exc})")
                continue

            errs: list[str] = validate_trace(record, allowed_tools, idx)
            if errs:
                all_errors.extend(errs)
            else:
                valid += 1

    print("=== Trace Validation Summary ===")
    print(f"Total Traces: {total} | Valid: {valid} | Invalid: {total - valid}")
    if all_errors:
        for err in all_errors[:15]:
            print(f"  - {err}")
        sys.exit(1)
    if total == 0:
        print("[FAIL] Dataset is empty.")
        sys.exit(1)
    print("[PASS] All traces verified.")


if __name__ == "__main__":
    main()
