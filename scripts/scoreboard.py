#!/usr/bin/env python3
"""
The Scoreboard: Evaluates a local Ollama model against benchmark_locked.jsonl
and outputs a failure breakdown to drive targeted synthetic data generation.
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from referee_stage1 import evaluate_session


def query_ollama(model: str, messages: list[dict[str, Any]], host: str = "http://localhost:11434") -> str:
    payload: bytes = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.0}
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return str(data.get("message", {}).get("content", ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run private benchmark scoreboard against an Ollama model.")
    parser.add_argument("--model", required=True, help="Ollama model tag (e.g., qwen2.5:0.5b or task-tuned:latest)")
    parser.add_argument("--benchmark", required=True, help="Path to benchmark_locked.jsonl")
    parser.add_argument("--invariants", required=True, help="Path to referee/invariants.json")
    parser.add_argument("--report-out", required=True, help="Path to write scoreboard report JSON")
    parser.add_argument("--ollama-host", default="http://localhost:11434", help="Ollama server URL")
    args = parser.parse_args()

    invariants: dict[str, Any] = json.loads(Path(args.invariants).read_text(encoding="utf-8"))
    passed: int = 0
    total: int = 0
    failures: list[dict[str, Any]] = []

    with Path(args.benchmark).open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            case: dict[str, Any] = json.loads(line)
            prompt_msgs: list[dict[str, Any]] = case["messages"][:-1]
            expected_patterns: list[str] = case.get("expected_regexes", [])

            reply: str = query_ollama(args.model, prompt_msgs, host=args.ollama_host)
            session_copy: dict[str, Any] = {
                "messages": prompt_msgs + [{"role": "assistant", "content": reply}]
            }

            ref_verdict: dict[str, Any] = evaluate_session(session_copy, invariants)
            if ref_verdict["status"] == "REJECTED":
                failures.append({
                    "case_id": case.get("id", idx),
                    "category": case.get("category", "general"),
                    "reason": ref_verdict["reason"],
                    "model_reply": reply
                })
                continue

            missing: list[str] = [p for p in expected_patterns if not re.search(p, reply, re.IGNORECASE)]
            if missing:
                failures.append({
                    "case_id": case.get("id", idx),
                    "category": case.get("category", "general"),
                    "reason": f"Missing required benchmark patterns: {missing}",
                    "model_reply": reply
                })
                continue

            passed += 1

    score_pct: float = round((passed / total) * 100, 2) if total else 0.0
    summary: dict[str, Any] = {
        "model": args.model,
        "total_cases": total,
        "passed": passed,
        "failed": len(failures),
        "score_percent": score_pct,
        "failures": failures
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"=== Scoreboard: {args.model} -> {passed}/{total} ({score_pct}%) ===")
    if failures:
        print("Top Failure Categories to feed back into Phase 1 Synthesis:")
        cats: dict[str, int] = {}
        for fail in failures:
            c: str = str(fail["category"])
            cats[c] = cats.get(c, 0) + 1
        for c, count in sorted(cats.items(), key=lambda x: -x[1]):
            print(f"  - {c}: {count} failures")


if __name__ == "__main__":
    main()
