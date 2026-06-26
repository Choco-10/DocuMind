import re
import sys
import time
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm.llm_model import generate_answer, stream_answer


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def run_non_stream_checks() -> List[Dict[str, str]]:
    checks = []

    t0 = time.perf_counter()
    ans = generate_answer(
        question="What is the capital of France?",
        context="Paris is the capital of France.",
        history="",
    )
    t1 = time.perf_counter()

    ok = "paris" in normalize(ans)
    checks.append(
        {
            "name": "grounded_answer",
            "status": "PASS" if ok else "FAIL",
            "detail": f"answer={ans!r}, latency_s={t1 - t0:.2f}",
        }
    )

    t0 = time.perf_counter()
    ans = generate_answer(
        question="Who won FIFA World Cup 2022?",
        context="This context is only about machine learning and neural networks.",
        history="",
    )
    t1 = time.perf_counter()

    ok = normalize(ans) == "i don't know."
    checks.append(
        {
            "name": "unknown_fallback",
            "status": "PASS" if ok else "FAIL",
            "detail": f"answer={ans!r}, latency_s={t1 - t0:.2f}",
        }
    )

    t0 = time.perf_counter()
    ans = generate_answer(
        question="Give transaction id only.",
        context="A receipt contains transaction id TX123.",
        history="",
    )
    t1 = time.perf_counter()

    leaked = bool(re.search(r"<think>|</think>|\bassistant\b|Context:|Question:", ans, flags=re.IGNORECASE))
    ok = not leaked
    checks.append(
        {
            "name": "clean_non_stream_output",
            "status": "PASS" if ok else "FAIL",
            "detail": f"answer={ans!r}, latency_s={t1 - t0:.2f}",
        }
    )

    return checks


def run_stream_check() -> Dict[str, str]:
    t0 = time.perf_counter()
    parts = []

    for tok in stream_answer(
        question="What is the capital of France?",
        context="Paris is the capital of France.",
        history="",
    ):
        if tok:
            parts.append(tok)

    t1 = time.perf_counter()
    text = "".join(parts).strip()

    leaked = bool(re.search(r"<think>|</think>|\bassistant\b|Context:|Question:", text, flags=re.IGNORECASE))
    ok = not leaked and len(text) > 0

    return {
        "name": "stream_cleanliness",
        "status": "PASS" if ok else "FAIL",
        "detail": f"stream_text={text!r}, latency_s={t1 - t0:.2f}",
    }


def main() -> None:
    all_checks = []
    all_checks.extend(run_non_stream_checks())
    all_checks.append(run_stream_check())

    passed = sum(1 for c in all_checks if c["status"] == "PASS")
    total = len(all_checks)

    print("LLM smoke check results")
    print("=" * 60)
    for c in all_checks:
        print(f"[{c['status']}] {c['name']}: {c['detail']}")

    print("=" * 60)
    print(f"Summary: {passed}/{total} passed")


if __name__ == "__main__":
    main()