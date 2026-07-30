import argparse
import json
import time
import requests
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

F_TTFT = 100.0      # ms
C_TTFT = 1500.0     # ms

F_TPOT = 20.0       # ms
C_TPOT = 45.0       # ms

GAMMA = 2.0
WEIGHT = 0.5

RESULT_FILE = "result.jsonl"

# Thread-safe locks
stats_lock = threading.Lock()
result_lock = threading.Lock()


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def score_ttft(ttft_ms):
    x = clamp((C_TTFT - ttft_ms) / (C_TTFT - F_TTFT))
    return x ** GAMMA


def score_tpot(tpot_ms):
    x = clamp((C_TPOT - tpot_ms) / (C_TPOT - F_TPOT))
    return x ** GAMMA


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a vLLM request trace from a data.jsonl file "
            "using multiple threads."
        )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to data.jsonl",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8000",
        help="vLLM server address",
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="OpenAI endpoint",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="HTTP timeout (seconds)",
    )

    parser.add_argument(
        "--no-timing",
        action="store_true",
        help="Ignore timestamp_ms and send requests immediately.",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=128,
        help="Maximum number of concurrent worker threads.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=f"results/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Output folder",
    )

    return parser.parse_args()


def load_requests(path):
    requests_data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                requests_data.append(json.loads(line))

    requests_data.sort(key=lambda x: x["timestamp_ms"])

    return requests_data


def write_result(result, result_file):
    """
    Thread-safe JSONL writer.
    Each request is written as one JSON object.
    """
    with result_lock:
        result_file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )
        result_file.flush()


def send_request(
    req,
    url,
    timeout,
    global_stats,
    result_file,
):
    # Time when the HTTP request is sent.
    t0 = time.perf_counter()

    # Copy original request body.
    body = dict(req["body"])

    # Force streaming.
    body["stream"] = True

    # Force model.
    body["model"] = "Qwen/Qwen3.5-0.8B"

    # Accumulate generated text.
    generated_text = []

    # Token timing.
    first_token_time = None
    last_token_time = None

    token_count = 0
    token_intervals = []

    # Token usage.
    input_tokens = None
    output_tokens = None

    try:
        response = requests.post(
            url,
            json=body,
            stream=True,
            timeout=timeout,
        )

        response.raise_for_status()

        for line in response.iter_lines(
            decode_unicode=True
        ):
            if not line or not line.startswith("data: "):
                continue

            payload = line[6:]

            if payload == "[DONE]":
                break

            try:
                obj = json.loads(payload)
            except Exception:
                continue

            # --------------------------------------------------
            # Extract usage information.
            #
            # vLLM/OpenAI-compatible streaming responses may
            # include usage in a final chunk.
            # --------------------------------------------------
            usage = obj.get("usage")

            if usage is not None:
                if usage.get("prompt_tokens") is not None:
                    input_tokens = usage["prompt_tokens"]

                if usage.get("completion_tokens") is not None:
                    output_tokens = usage["completion_tokens"]

            choices = obj.get("choices", [])

            if not choices:
                continue

            delta = choices[0].get("delta", {})

            content = delta.get("content")

            if not content:
                continue

            # Accumulate generation.
            generated_text.append(content)

            now = time.perf_counter()

            # First generated content chunk.
            if first_token_time is None:
                first_token_time = now

            # Subsequent generated chunks.
            else:
                token_intervals.append(
                    now - last_token_time
                )

            last_token_time = now
            token_count += 1

        # --------------------------------------------------
        # Request completion time.
        # --------------------------------------------------
        end_time = time.perf_counter()

        total_time_sec = end_time - t0

        # --------------------------------------------------
        # Calculate TTFT.
        # --------------------------------------------------
        if first_token_time is None:
            ttft_ms = None
        else:
            ttft_ms = (
                first_token_time - t0
            ) * 1000

        # --------------------------------------------------
        # Calculate TPOT.
        # --------------------------------------------------
        if token_count == 0:
            tpot_ms = None

        elif token_intervals:
            tpot_ms = (
                sum(token_intervals)
                / len(token_intervals)
            ) * 1000

        else:
            # Only one content chunk.
            tpot_ms = 0.0

        # --------------------------------------------------
        # If usage is not provided by the server, fall back
        # to the number of generated content chunks.
        #
        # Note:
        # token_count here is NOT necessarily the number of
        # output tokens. A streaming chunk can contain multiple
        # tokens.
        # --------------------------------------------------
        if output_tokens is None:
            output_tokens = token_count

        # --------------------------------------------------
        # Calculate input/output token throughput.
        #
        # These are effective end-to-end throughput values:
        #
        # tokens / request wall-clock time
        #
        # They are NOT pure GPU prefill/decode throughput.
        # --------------------------------------------------
        if total_time_sec > 0 and input_tokens is not None:
            input_tokens_per_sec = (
                input_tokens / total_time_sec
            )
        else:
            input_tokens_per_sec = None

        if total_time_sec > 0 and output_tokens is not None:
            output_tokens_per_sec = (
                output_tokens / total_time_sec
            )
        else:
            output_tokens_per_sec = None

        # --------------------------------------------------
        # Score.
        # --------------------------------------------------
        if ttft_ms is None or tpot_ms is None:
            request_score = 0.0

        else:
            request_score = (
                WEIGHT * score_ttft(ttft_ms)
                + (1 - WEIGHT) * score_tpot(tpot_ms)
            )

        # --------------------------------------------------
        # Combine generated chunks.
        # --------------------------------------------------
        generation = "".join(generated_text)

        # --------------------------------------------------
        # Preserve original request information.
        # --------------------------------------------------
        result = {
            "request_id": req["request_id"],
            "status": "success",

            "prediction": generation,

            "token_count": token_count,

            "input_tokens": input_tokens,
            "output_tokens": output_tokens,

            "input_tokens_per_sec": input_tokens_per_sec,
            "output_tokens_per_sec": output_tokens_per_sec,

            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,

            "total_time_ms": total_time_sec * 1000,

            "score": request_score,
        }

        # --------------------------------------------------
        # Update statistics.
        # --------------------------------------------------
        with stats_lock:
            global_stats["success"] += 1

            global_stats["scores"].append(
                request_score
            )

            if ttft_ms is not None:
                global_stats["ttfts"].append(
                    ttft_ms
                )

            if tpot_ms is not None:
                global_stats["tpots"].append(
                    tpot_ms
                )

            if input_tokens is not None:
                global_stats["input_tokens"].append(
                    input_tokens
                )

                if total_time_sec > 0:
                    global_stats[
                        "input_tokens_per_sec"
                    ].append(
                        input_tokens
                        / total_time_sec
                    )

            if output_tokens is not None:
                global_stats["output_tokens"].append(
                    output_tokens
                )

                if total_time_sec > 0:
                    global_stats[
                        "output_tokens_per_sec"
                    ].append(
                        output_tokens
                        / total_time_sec
                    )

            print(
                f"[{req['request_id']:4d}] "
                f"in={input_tokens if input_tokens is not None else -1:6d} "
                f"out={output_tokens if output_tokens is not None else -1:4d} "
                f"TTFT="
                f"{ttft_ms if ttft_ms is not None else -1:7.1f} ms "
                f"TPOT="
                f"{tpot_ms if tpot_ms is not None else -1:6.1f} ms "
                f"InTok/s="
                f"{input_tokens_per_sec if input_tokens_per_sec is not None else -1:8.1f} "
                f"OutTok/s="
                f"{output_tokens_per_sec if output_tokens_per_sec is not None else -1:8.1f} "
                f"Score={request_score:.3f}"
            )

        # --------------------------------------------------
        # Write result to JSONL.
        # --------------------------------------------------
        write_result(
            result,
            result_file,
        )

    except Exception as e:

        # Save failed request as well.
        result = {
            "request_id": req["request_id"],
            "status": "failed",

            "prediction": "".join(generated_text),

            "token_count": token_count,

            "input_tokens": input_tokens,
            "output_tokens": output_tokens,

            "input_tokens_per_sec": None,
            "output_tokens_per_sec": None,

            "ttft_ms": None,
            "tpot_ms": None,

            "total_time_ms": None,

            "score": 0.0,

            "error": str(e),
        }

        with stats_lock:
            global_stats["failed"] += 1

            global_stats["scores"].append(
                0.0
            )

            print(
                f"[{req['request_id']:4d}] "
                f"FAILED {e}"
            )

        # Write failed request to JSONL.
        write_result(
            result,
            result_file,
        )


def main():
    args = parse_args()

    url = (
        args.host.rstrip("/")
        + args.endpoint
    )

    data = load_requests(args.input)

    if not data:
        print("No requests found.")
        return

    print(
        f"Loaded {len(data)} requests"
    )

    print(
        f"Sending to {url} "
        f"using up to {args.max_workers} threads..."
    )

    print(
        f"Saving results to {args.output}"
    )

    global_stats = {
        "success": 0,
        "failed": 0,

        "scores": [],

        "ttfts": [],
        "tpots": [],

        "input_tokens": [],
        "output_tokens": [],

        "input_tokens_per_sec": [],
        "output_tokens_per_sec": [],
    }

    start_wall = time.perf_counter()

    start_trace = data[0]["timestamp_ms"]

    # Open output file once.
    os.makedirs(
        args.output,
        exist_ok=True,
    )

    json_file_path = os.path.join(
        args.output,
        RESULT_FILE,
    )

    with open(
        json_file_path,
        "w",
        encoding="utf-8",
    ) as result_file:

        with ThreadPoolExecutor(
            max_workers=args.max_workers
        ) as executor:

            futures = []

            for req in data:

                if not args.no_timing:

                    target = (
                        req["timestamp_ms"]
                        - start_trace
                    ) / 1000.0

                    # Wait until this request's
                    # original timestamp.
                    while True:

                        elapsed = (
                            time.perf_counter()
                            - start_wall
                        )

                        if elapsed >= target:
                            break

                        time.sleep(
                            min(
                                0.001,
                                target - elapsed,
                            )
                        )

                # Dispatch request immediately.
                futures.append(
                    executor.submit(
                        send_request,
                        req,
                        url,
                        args.timeout,
                        global_stats,
                        result_file,
                    )
                )

            # Wait for all requests.
            for _ in as_completed(futures):
                pass

    # ==============================
    # Summary
    # ==============================

    SYSTEM_SUMMARY_FILE = os.path.join(
        args.output,
        "system_summary.txt",
    )

    summary_lines = [
        "========== Summary ==========",
        f"Total requests : {len(data)}",
        "",
        f"Success        : {global_stats['success']}",
        f"Failed         : {global_stats['failed']}",
    ]

    if global_stats["scores"]:
        ers = (
            sum(global_stats["scores"])
            / len(global_stats["scores"])
        )

        summary_lines.append(
            f"ERS            : {ers:.4f}"
        )

    # --------------------------------------------------
    # Token statistics.
    # --------------------------------------------------
    if global_stats["input_tokens"]:

        total_input_tokens = sum(
            global_stats["input_tokens"]
        )

        mean_input_tokens = (
            total_input_tokens
            / len(global_stats["input_tokens"])
        )

        summary_lines.extend([
            "",
            f"Total Input Tokens  : {total_input_tokens}",
            f"Mean Input Tokens   : {mean_input_tokens:.2f}",
        ])

    if global_stats["output_tokens"]:

        total_output_tokens = sum(
            global_stats["output_tokens"]
        )

        mean_output_tokens = (
            total_output_tokens
            / len(global_stats["output_tokens"])
        )

        summary_lines.extend([
            f"Total Output Tokens : {total_output_tokens}",
            f"Mean Output Tokens  : {mean_output_tokens:.2f}",
        ])

    # --------------------------------------------------
    # TTFT statistics.
    # --------------------------------------------------
    if global_stats["ttfts"]:

        summary_lines.extend([
            "",
            f"Mean TTFT      : "
            f"{sum(global_stats['ttfts']) / len(global_stats['ttfts']):.2f} ms",

            f"Min TTFT       : "
            f"{min(global_stats['ttfts']):.2f} ms",

            f"Max TTFT       : "
            f"{max(global_stats['ttfts']):.2f} ms",
        ])

    # --------------------------------------------------
    # TPOT statistics.
    # --------------------------------------------------
    if global_stats["tpots"]:

        summary_lines.extend([
            f"Mean TPOT      : "
            f"{sum(global_stats['tpots']) / len(global_stats['tpots']):.2f} ms",

            f"Min TPOT       : "
            f"{min(global_stats['tpots']):.2f} ms",

            f"Max TPOT       : "
            f"{max(global_stats['tpots']):.2f} ms",
        ])

    # --------------------------------------------------
    # Input throughput statistics.
    # --------------------------------------------------
    if global_stats["input_tokens_per_sec"]:

        mean_input_tokens_per_sec = (
            sum(
                global_stats[
                    "input_tokens_per_sec"
                ]
            )
            / len(
                global_stats[
                    "input_tokens_per_sec"
                ]
            )
        )

        summary_lines.append(
            f"Mean Input Tokens/sec  : "
            f"{mean_input_tokens_per_sec:.2f}"
        )

    # --------------------------------------------------
    # Output throughput statistics.
    # --------------------------------------------------
    if global_stats["output_tokens_per_sec"]:

        mean_output_tokens_per_sec = (
            sum(
                global_stats[
                    "output_tokens_per_sec"
                ]
            )
            / len(
                global_stats[
                    "output_tokens_per_sec"
                ]
            )
        )

        summary_lines.append(
            f"Mean Output Tokens/sec : "
            f"{mean_output_tokens_per_sec:.2f}"
        )

    summary = "\n".join(
        summary_lines
    )

    print(
        f"\n{summary}"
    )

    with open(
        SYSTEM_SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            summary + "\n"
        )

    print(
        f"\nResults saved to: {args.output}"
    )


if __name__ == "__main__":
    main()