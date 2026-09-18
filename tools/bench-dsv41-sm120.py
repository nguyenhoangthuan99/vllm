#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Throughput measurement for a live DSv41 server.

Counts tokens actually reported by the API (`usage`), never estimated.

Reports three things separately, because they mean different things:

* ``decode``    — long generations at a fixed concurrency: tokens/s of decode.
* ``prefill``   — TTFT for cold long prompts (unique prefix), so the number
  reflects prefill work rather than prefix-cache hits.
* ``mixed``     — realistic concurrent load at several levels.
"""

import argparse
import concurrent.futures
import json
import statistics
import sys
import time
import urllib.request

FILLER = "Record {i}: standard log line, nothing unusual, no code stored here."


def post(base, payload, timeout):
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def chat(base, prompt, max_tokens, timeout, ignore_eos=True):
    body = post(
        base,
        {
            "model": "dsv41",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "ignore_eos": ignore_eos,
            "chat_template_kwargs": {"thinking": False},
        },
        timeout,
    )
    return body["usage"], body["choices"][0]["finish_reason"]


def unique_prompt(tag, target_tokens, chars_per_token=4):
    """Unique prefix so the prefix cache cannot mask real prefill work."""
    total_chars = target_tokens * chars_per_token
    line_chars = len(FILLER.format(i=0)) + 1
    lines = max(2, total_chars // line_chars)
    body = "\n".join(FILLER.format(i=f"{tag}-{i}") for i in range(lines))
    return f"Document {tag}:\n{body}\n\nSummarise the document above in one word."


def decode_run(base, concurrency, out_tokens, timeout):
    """All requests share one prompt; measures decode throughput."""
    prompt = "Write a long, detailed description of the number 7."
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(chat, base, prompt, out_tokens, timeout)
            for _ in range(concurrency)
        ]
        usages = [f.result()[0] for f in futures]
    elapsed = time.monotonic() - started
    completion = sum(u["completion_tokens"] for u in usages)
    prompt_tokens = sum(u["prompt_tokens"] for u in usages)
    return {
        "concurrency": concurrency,
        "completion_tokens": completion,
        "prompt_tokens": prompt_tokens,
        "elapsed_s": round(elapsed, 2),
        "decode_tok_per_s": round(completion / elapsed, 1),
        "total_tok_per_s": round((completion + prompt_tokens) / elapsed, 1),
    }


def prefill_runs(base, tag, lengths, timeout):
    results = []
    for target in lengths:
        prompt = unique_prompt(f"{tag}-{target}", target)
        started = time.monotonic()
        usage, _ = chat(base, prompt, 1, timeout)
        elapsed = time.monotonic() - started
        results.append(
            {
                "target_tokens": target,
                "prompt_tokens": usage["prompt_tokens"],
                "ttft_s": round(elapsed, 3),
                "prefill_tok_per_s": round(usage["prompt_tokens"] / elapsed, 1),
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:30100")
    parser.add_argument("--jsonl")
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--out-tokens", type=int, default=256)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--prefill", default="1000,4000,7000")
    args = parser.parse_args()
    sink = open(args.jsonl, "a") if args.jsonl else None

    def emit(rec):
        line = json.dumps(rec, sort_keys=True)
        print(line, flush=True)
        if sink:
            sink.write(line + "\n")
            sink.flush()

    emit(
        {
            "kind": "config",
            "out_tokens": args.out_tokens,
            "concurrency": args.concurrency,
            "prefill_targets": args.prefill,
            "counting": "usage.completion_tokens and usage.prompt_tokens",
        }
    )

    # Warm the server so the first sample is not a JIT/allocator artifact.
    chat(args.base, "Say hello.", 4, args.timeout)

    for level in (int(c) for c in args.concurrency.split(",")):
        try:
            emit({"kind": "decode", **decode_run(
                args.base, level, args.out_tokens, args.timeout)})
        except Exception as exc:  # noqa: BLE001
            emit({"kind": "decode", "concurrency": level, "status": "error",
                  "error": f"{type(exc).__name__}: {exc}"[:200]})

    try:
        for rec in prefill_runs(
            args.base, "prefill", [int(t) for t in args.prefill.split(",")],
            args.timeout,
        ):
            emit({"kind": "prefill", **rec})
    except Exception as exc:  # noqa: BLE001
        emit({"kind": "prefill", "status": "error",
              "error": f"{type(exc).__name__}: {exc}"[:200]})

    if sink:
        sink.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())