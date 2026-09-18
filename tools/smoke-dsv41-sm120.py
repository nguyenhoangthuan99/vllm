# SPDX-License-Identifier: Apache-2.0
"""Exercise short generation, long prefill, and concurrent requests over HTTP."""

import argparse
import concurrent.futures
import json
import time
import urllib.request


def request(base, prompt):
    payload = {
        "model": "dsv41",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"thinking": False},
    }
    started = time.monotonic()
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as response:
        result = json.load(response)
    return result, time.monotonic() - started


def check(base, name, prompt, expected, min_prompt_tokens=0):
    result, elapsed = request(base, prompt)
    choice = result["choices"][0]
    text = choice["message"]["content"].strip()
    usage = result["usage"]
    passed = (
        text == expected
        and choice["finish_reason"] == "stop"
        and usage["prompt_tokens"] >= min_prompt_tokens
        and usage["completion_tokens"] > 0
    )
    print(
        json.dumps(
            {
                "case": name,
                "passed": passed,
                "text": text,
                "expected": expected,
                "usage": usage,
                "elapsed_s": elapsed,
                "finish_reason": choice["finish_reason"],
            }
        ),
        flush=True,
    )
    if not passed:
        raise AssertionError(f"{name}: unexpected generation or token accounting")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:30100")
    args = parser.parse_args()
    with urllib.request.urlopen(args.base + "/health", timeout=10) as response:
        assert response.status == 200
    check(
        args.base,
        "multiply",
        "What is 17 times 23? Reply with only the integer.",
        "391",
    )
    check(
        args.base,
        "divide",
        "What is 144 divided by 12? Reply with only the integer.",
        "12",
    )
    filler = "\n".join(
        f"Archive entry {i}: ordinary inventory record; no access code is stored here."
        for i in range(220)
    )
    prompt = (
        "The access code is MARBLE-7291.\n"
        + filler
        + "\nWhat access code appeared at the start? Reply with only that code."
    )
    check(args.base, "long_prefill_retrieval", prompt, "MARBLE-7291", 2048)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                check,
                args.base,
                f"concurrent_{i}",
                (
                    f"The access code is COBALT-{4100 + i}.\n"
                    + filler
                    + "\nWhat access code appeared at the start? Reply with only that code."
                    if i % 2 == 0
                    else f"Reply with exactly this string and nothing else: COBALT-{4100 + i}"
                ),
                f"COBALT-{4100 + i}",
                2048 if i % 2 == 0 else 0,
            )
            for i in range(4)
        ]
        for future in futures:
            future.result()
    print(json.dumps({"status": "passed", "requests": 7}), flush=True)


if __name__ == "__main__":
    main()
