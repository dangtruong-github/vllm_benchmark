import argparse
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

F_TTFT = 100.0      # ms
C_TTFT = 1500.0     # ms

F_TPOT = 20.0       # ms
C_TPOT = 45.0       # ms

GAMMA = 2.0
WEIGHT = 0.5

# Thread-safe locks for printing and accumulating statistics
stats_lock = threading.Lock()

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
        description="Replay a vLLM request trace from a data.jsonl file using multiple threads."
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

    return parser.parse_args()


def load_requests(path):
    requests_data = []

    with open(path, "r") as f:
        for line in f:
            if line.strip():
                requests_data.append(json.loads(line))

    requests_data.sort(key=lambda x: x["timestamp_ms"])
    return requests_data


def send_request(req, url, timeout, global_stats):
    t0 = time.perf_counter()
    body = dict(req["body"])
    body["stream"] = True

    try:
        response = requests.post(
            url,
            json=body,
            stream=True,
            timeout=timeout,
        )
        response.raise_for_status()

        first_token_time = None
        last_token_time = None
        token_count = 0
        token_intervals = []

        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue

            payload = line[6:]
            if payload == "[DONE]":
                break

            try:
                obj = json.loads(payload)
            except Exception:
                continue

            choices = obj.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            if not delta.get("content"):
                continue

            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            else:
                token_intervals.append(now - last_token_time)

            last_token_time = now
            token_count += 1

        if token_count == 0:
            request_score = 0.0
            ttft_ms = None
            tpot_ms = None
        else:
            ttft_ms = (first_token_time - t0) * 1000
            if token_intervals:
                tpot_ms = (sum(token_intervals) / len(token_intervals)) * 1000
            else:
                tpot_ms = 0.0

            request_score = (
                WEIGHT * score_ttft(ttft_ms)
                + (1 - WEIGHT) * score_tpot(tpot_ms)
            )

        with stats_lock:
            global_stats["success"] += 1
            global_stats["scores"].append(request_score)
            if ttft_ms is not None:
                global_stats["ttfts"].append(ttft_ms)
            if tpot_ms is not None:
                global_stats["tpots"].append(tpot_ms)
            
            print(
                f"[{req['request_id']:4d}] "
                f"tokens={token_count:4d} "
                f"TTFT={ttft_ms if ttft_ms else -1:7.1f} ms "
                f"TPOT={tpot_ms if tpot_ms else -1:6.1f} ms "
                f"Score={request_score:.3f}"
            )

    except Exception as e:
        with stats_lock:
            global_stats["failed"] += 1
            global_stats["scores"].append(0.0)
            print(f"[{req['request_id']:4d}] FAILED {e}")


def main():
    args = parse_args()
    url = args.host.rstrip("/") + args.endpoint
    data = load_requests(args.input)

    print(f"Loaded {len(data)} requests")
    print(f"Sending to {url} using up to {args.max_workers} threads...")

    global_stats = {
        "success": 0,
        "failed": 0,
        "scores": [],
        "ttfts": [],
        "tpots": []
    }

    start_wall = time.perf_counter()
    start_trace = data[0]["timestamp_ms"]

    # Use a ThreadPoolExecutor to spawn threads dynamically
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        
        for req in data:
            if not args.no_timing:
                target = (req["timestamp_ms"] - start_trace) / 1000.0

                # Main thread handles pacing and sleeps until next request timestamp is due
                while True:
                    elapsed = time.perf_counter() - start_wall
                    if elapsed >= target:
                        break
                    time.sleep(min(0.001, target - elapsed))

            # Dispatch the request to a worker thread immediately
            futures.append(
                executor.submit(send_request, req, url, args.timeout, global_stats)
            )

        # Wait for all remaining requests to finish processing
        for _ in as_completed(futures):
            pass

    print("\n========== Summary ==========")
    print(f"Total requests : {len(data)}")
    print(f"Success        : {global_stats['success']}")
    print(f"Failed         : {global_stats['failed']}")

    if global_stats["scores"]:
        ers = sum(global_stats["scores"]) / len(global_stats["scores"])
        print(f"ERS            : {ers:.4f}")

    if global_stats["ttfts"]:
        print(f"Mean TTFT      : {sum(global_stats['ttfts'])/len(global_stats['ttfts']):.2f} ms")
        print(f"Min TTFT       : {min(global_stats['ttfts']):.2f} ms")
        print(f"Max TTFT       : {max(global_stats['ttfts']):.2f} ms")

    if global_stats["tpots"]:
        print(f"Mean TPOT      : {sum(global_stats['tpots'])/len(global_stats['tpots']):.2f} ms")
        print(f"Min TPOT       : {min(global_stats['tpots']):.2f} ms")
        print(f"Max TPOT       : {max(global_stats['tpots']):.2f} ms")


if __name__ == "__main__":
    main()
