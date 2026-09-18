#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness battery for a live DSv41 server.

Three independent checks:

1. exact-answer tasks with verifiable expected strings across several domains;
2. determinism: the same greedy prompt must return byte-identical output
   across repeated calls, and after intervening unrelated traffic;
3. positional retrieval: a needle at varying depth in a long document, which
   depends on the sparse indexer actually selecting the right positions.
"""

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request

FILLER = (
    "Record {i}: standard log entry, no code present, nothing unusual to note."
)


def ask(base, prompt, max_tokens, timeout, seed=None, temperature=0.0):
    payload = {
        "model": "dsv41",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"thinking": False},
    }
    if seed is not None:
        payload["seed"] = seed
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    choice = body["choices"][0]
    return (
        choice["message"]["content"].strip(),
        choice["finish_reason"],
        body["usage"],
        time.monotonic() - started,
    )


EXACT_CASES = [
    ("arith_mul", "What is 137 times 24? Reply with only the integer.", "3288"),
    ("arith_div", "What is 1728 divided by 16? Reply with only the integer.", "108"),
    ("arith_chain", "Compute ((7 + 5) * 3) - 4. Reply with only the integer.", "32"),
    ("reverse", "Reverse the string 'sdrawdeht'. Reply with only the "
                "reversed string.", "theadwards"),
    ("count_letters", "How many times does the letter 'r' appear in "
                      "'strawberry'? Reply with only the number.", "3"),
    ("capital", "What is the capital of Australia? Reply with only the city name.",
                "Canberra"),
    ("unit_math", "If a train travels 180 km in 2.5 hours, what is its average "
                  "speed in km/h? Reply with only the number.", "72"),
    ("list_max", "What is the largest number in the list 14, 9, 27, 3, 22? "
                 "Reply with only the number.", "27"),
    ("base_convert", "What is the binary representation of the decimal number "
                     "13? Reply with only the binary digits.", "1101"),
    ("json_keys", "Reply with only the JSON object {\"a\": 1, \"b\": 2}.",
                  '{"a": 1, "b": 2}'),
]


def build_retrieval(needle, target_tokens, depth, chars_per_token=4):
    total_chars = target_tokens * chars_per_token
    line_chars = len(FILLER.format(i=0)) + 1
    lines = max(2, total_chars // line_chars)
    split = max(0, min(lines, int(lines * depth)))
    head = "\n".join(FILLER.format(i=i) for i in range(split))
    tail = "\n".join(FILLER.format(i=i) for i in range(split, lines))
    return (
        head
        + f"\nThe access code is {needle}.\n"
        + tail
        + "\n\nWhat access code appeared in the document above? Reply with "
        "only the code."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:30100")
    parser.add_argument("--jsonl")
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    sink = open(args.jsonl, "a") if args.jsonl else None
    failures = []

    def emit(rec):
        line = json.dumps(rec, sort_keys=True)
        print(line, flush=True)
        if sink:
            sink.write(line + "\n")
            sink.flush()

    # 1. exact answers
    for name, prompt, expected in EXACT_CASES:
        try:
            text, finish, usage, elapsed = ask(
                args.base, prompt, 32, args.timeout
            )
            normalized = text.strip().strip(".").strip()
            ok = normalized == expected and finish == "stop"
            emit(
                {
                    "kind": "exact",
                    "case": name,
                    "expected": expected,
                    "got": text,
                    "finish_reason": finish,
                    "prompt_tokens": usage["prompt_tokens"],
                    "elapsed_s": round(elapsed, 3),
                    "passed": ok,
                }
            )
            if not ok:
                failures.append(f"exact:{name}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"exact:{name}")
            emit({"kind": "exact", "case": name, "status": "error",
                  "error": f"{type(exc).__name__}: {exc}"[:200]})
        if sink:
            sink.flush()

    # 2. determinism across repeated calls and intervening traffic
    det_prompt = "What is 137 times 24? Reply with only the integer."
    try:
        first, _, _, _ = ask(args.base, det_prompt, 32, args.timeout)
        # interleave unrelated traffic so KV/prefix state changes
        ask(args.base, "Name a colour. Reply with one word.", 8, args.timeout)
        ask(
            args.base,
            build_retrieval("NOISE-1234", 3000, 0.5),
            16,
            args.timeout,
        )
        second, _, _, _ = ask(args.base, det_prompt, 32, args.timeout)
        third, _, _, _ = ask(args.base, det_prompt, 32, args.timeout)
        ok = first == second == third
        emit(
            {
                "kind": "determinism",
                "case": "greedy_repeat",
                "outputs": [first, second, third],
                "passed": ok,
            }
        )
        if not ok:
            failures.append("determinism:greedy_repeat")
    except Exception as exc:  # noqa: BLE001
        failures.append("determinism:greedy_repeat")
        emit({"kind": "determinism", "status": "error",
              "error": f"{type(exc).__name__}: {exc}"[:200]})

    # 3. positional retrieval at several depths
    for index, (target, depth) in enumerate(
        ((2048, 0.02), (2048, 0.5), (4096, 0.85), (6000, 0.5), (7000, 0.3))
    ):
        needle = f"CODE-{index}-{target}"
        prompt = build_retrieval(needle, target, depth)
        case = f"depth{index}:{target}@{depth}"
        try:
            text, finish, usage, elapsed = ask(
                args.base, prompt, 24, args.timeout
            )
            ok = needle in text and finish == "stop"
            emit(
                {
                    "kind": "retrieval",
                    "case": case,
                    "depth": depth,
                    "needle": needle,
                    "got": text[:80],
                    "prompt_tokens": usage["prompt_tokens"],
                    "elapsed_s": round(elapsed, 3),
                    "passed": ok,
                }
            )
            if not ok:
                failures.append(f"retrieval:{case}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"retrieval:{case}")
            emit({"kind": "retrieval", "case": case, "status": "error",
                  "error": f"{type(exc).__name__}: {exc}"[:200]})

    emit(
        {
            "kind": "summary",
            "status": "fail" if failures else "pass",
            "exact": len(EXACT_CASES),
            "retrieval": 5,
            "failures": failures,
        }
    )
    if sink:
        sink.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())