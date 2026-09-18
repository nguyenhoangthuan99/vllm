#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Long-context retrieval sweep against a live server.

Places a unique needle at several depths in a long filler document and checks
the model returns it. Depth matters more than length here: the sparse indexer
selects by position, so a needle near the start exercises a different path than
one near the end.
"""

import argparse
import json
import sys
import time
import urllib.request

FILLER = (
    "Archive entry {i}: routine inventory record. No access code is stored "
    "in this line, and nothing here identifies the requester."
)


def one_request(base, prompt, timeout):
    payload = {
        "model": "dsv41",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 24,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    return body, time.monotonic() - started


def build_prompt(needle, target_tokens, depth, chars_per_token=4):
    """Filler sized to ~target_tokens, with `needle` at fraction `depth`."""
    total_chars = target_tokens * chars_per_token
    line_chars = len(FILLER.format(i=0)) + 1
    lines = max(2, total_chars // line_chars)
    split = int(lines * depth)
    head = [FILLER.format(i=i) for i in range(split)]
    tail = [FILLER.format(i=i) for i in range(split, lines)]
    return (
        "\n".join(head)
        + f"\nThe access code is {needle}.\n"
        + "\n".join(tail)
        + "\n\nWhat access code appeared in the document above? Reply with "
        "only the code."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:30100")
    parser.add_argument("--jsonl")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument(
        "--cases",
        default="2048:0.5,8192:0.5,16384:0.5,16384:0.05,32768:0.5,32768:0.9",
        help="comma-separated target_tokens:depth",
    )
    args = parser.parse_args()
    sink = open(args.jsonl, "a") if args.jsonl else None

    def emit(rec):
        line = json.dumps(rec, sort_keys=True)
        print(line, flush=True)
        if sink:
            sink.write(line + "\n")
            sink.flush()

    failures = []
    for index, spec in enumerate(args.cases.split(",")):
        target, depth = spec.split(":")
        needle = f"NEEDLE-{index}-{int(float(target))}"
        prompt = build_prompt(needle, int(target), float(depth))
        try:
            body, elapsed = one_request(args.base, prompt, args.timeout)
            choice = body["choices"][0]
            text = choice["message"]["content"].strip()
            usage = body["usage"]
            ok = needle in text and choice["finish_reason"] == "stop"
            emit(
                {
                    "case": spec,
                    "depth": float(depth),
                    "needle": needle,
                    "found": needle in text,
                    "text": text[:80],
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "finish_reason": choice["finish_reason"],
                    "elapsed_s": elapsed,
                    "passed": ok,
                }
            )
            if not ok:
                failures.append(spec)
        except Exception as exc:  # noqa: BLE001 - report every case
            failures.append(spec)
            emit(
                {
                    "case": spec,
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                }
            )
    emit(
        {
            "kind": "summary",
            "status": "fail" if failures else "pass",
            "failures": failures,
        }
    )
    if sink:
        sink.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())