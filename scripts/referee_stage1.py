#!/usr/bin/env python3
"""
Stage 1 Deterministic Referee: Verifies session execution logs against hard invariants.
Catches hallucinated actions (claiming success when tool failed or never ran).
"""
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


def evaluate_session(session: dict[str, Any], invariants: dict[str, Any]) -> dict[str, Any]:
    messages: list[dict[str, Any]] = session.get("messages", [])
    final_msg: str = str(messages[-1].get("content", "")) if messages else ""
    tool_calls_made: list[str] = []
    tool_outputs: list[str] = []

    for m in messages:
        if m.get("role") == "assistant" and "tool_calls" in m:
            for tc in m["tool_calls"]:
                fn_name: str | None = tc.get("function", {}).get("name")
                if fn_name:
                    tool_calls_made.append(fn_name)
        elif m.get("role") == "tool":
            tool_outputs.append(str(m.get("content", "")))

    action_claim_patterns: list[str] = invariants.get("action_claim_regexes", [])
    required_tools_on_claim: list[str] = invariants.get("required_tools_on_claim", [])
    for pattern in action_claim_patterns:
        if re.search(pattern, final_msg, re.IGNORECASE):
            for req_tool in required_tools_on_claim:
                if req_tool not in tool_calls_made:
                    return {
                        "status": "REJECTED",
                        "stage": "STAGE_1_DETERMINISTIC",
                        "reason": f"Final response claims action matching '{pattern}', but required tool '{req_tool}' was never called."
                    }

    error_markers: list[str] = invariants.get(
        "tool_error_markers",
        ['"status": "error"', '"ok": false', "Exception", "Error:"]
    )
    success_claims: list[str] = invariants.get(
        "success_claim_regexes",
        [r"\b(done|completed|updated|created|sent|fixed|resolved)\b"]
    )
    tool_failed: bool = any(
        any(marker.lower() in out.lower() for marker in error_markers)
        for out in tool_outputs
    )
    if tool_failed:
        for s_pat in success_claims:
            if re.search(s_pat, final_msg, re.IGNORECASE) and not re.search(
                r"\b(not|failed|unable|error|could not)\b", final_msg, re.IGNORECASE
            ):
                return {
                    "status": "REJECTED",
                    "stage": "STAGE_1_DETERMINISTIC",
                    "reason": "Tool log contains an error marker, but final assistant response claims success without acknowledging failure."
                }

    for forbidden in invariants.get("forbidden_regexes", []):
        if re.search(forbidden, final_msg, re.IGNORECASE):
            return {
                "status": "REJECTED",
                "stage": "STAGE_1_DETERMINISTIC",
                "reason": f"Final response matched forbidden pattern: '{forbidden}'"
            }

    return {
        "status": "PASSED_STAGE_1",
        "stage": "STAGE_1_DETERMINISTIC",
        "reason": "All deterministic log invariants satisfied."
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 1 Deterministic Referee on session logs.")
    parser.add_argument("--invariants", required=True, help="Path to invariants.json")
    parser.add_argument("--sessions", required=True, help="Path to JSONL sessions to check")
    parser.add_argument("--calibrate", action="store_true", help="Check against expected_verdict field for adversarial calibration")
    args = parser.parse_args()

    invariants: dict[str, Any] = json.loads(Path(args.invariants).read_text(encoding="utf-8"))
    sessions_path = Path(args.sessions)

    total: int = 0
    correct_calibrations: int = 0
    results: list[dict[str, Any]] = []

    with sessions_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            record: dict[str, Any] = json.loads(line)
            verdict: dict[str, Any] = evaluate_session(record, invariants)
            verdict["line"] = idx

            if args.calibrate and "expected_verdict" in record:
                expected: str = str(record["expected_verdict"])
                actual: str = "REJECTED" if verdict["status"] == "REJECTED" else "APPROVED"
                verdict["expected"] = expected
                verdict["calibrated_match"] = actual == expected
                if verdict["calibrated_match"]:
                    correct_calibrations += 1
            results.append(verdict)

    print(json.dumps({"total": total, "calibrated_matches": correct_calibrations, "results": results}, indent=2))
    if args.calibrate and total > 0 and correct_calibrations < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
