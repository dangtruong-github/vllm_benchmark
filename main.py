import argparse
import os
import signal
import shutil
import subprocess
import sys
import time

import requests
from datetime import datetime


def wait_for_server(
    host: str,
    process: subprocess.Popen,
    timeout: float,
    interval: float,
) -> bool:
    """Wait until the vLLM server is ready."""

    health_url = f"{host.rstrip('/')}/v1/models"
    start_time = time.time()

    print(f"Waiting for vLLM server at {health_url} ...")

    while True:
        elapsed = time.time() - start_time

        if elapsed >= timeout:
            print(
                f"\nERROR: vLLM server was not ready "
                f"after {timeout:.1f} seconds."
            )
            return False

        return_code = process.poll()

        if return_code is not None:
            print(
                f"\nERROR: vLLM process exited before becoming ready "
                f"(return code: {return_code})."
            )
            return False

        try:
            response = requests.get(
                health_url,
                timeout=5,
            )

            if response.status_code == 200:
                print(
                    f"\nvLLM server is ready "
                    f"(after {elapsed:.1f} seconds)."
                )
                return True

        except requests.exceptions.RequestException:
            pass

        time.sleep(interval)


def terminate_process(process: subprocess.Popen):
    """Terminate the entire vLLM process group."""

    if process.poll() is not None:
        print(
            f"vLLM launcher already exited "
            f"(return code: {process.returncode})."
        )
        return

    try:
        pgid = os.getpgid(process.pid)

        print(
            f"Stopping vLLM process group "
            f"(PID={process.pid}, PGID={pgid})..."
        )

        # Send SIGTERM to the entire process group.
        os.killpg(pgid, signal.SIGTERM)

        try:
            process.wait(timeout=10)
            print("vLLM process group terminated gracefully.")

        except subprocess.TimeoutExpired:
            print(
                "vLLM process group did not terminate "
                "after 10 seconds. Sending SIGKILL..."
            )

            # Force kill the entire process group.
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(
                    "WARNING: vLLM launcher process "
                    "still has not exited."
                )

    except ProcessLookupError:
        print("vLLM process group no longer exists.")

    except Exception as e:
        print(f"ERROR while terminating vLLM: {e}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Start vLLM, wait until it is ready, "
            "then run benchmark.py."
        )
    )

    parser.add_argument(
        "--serve-script",
        type=str,
        required=True,
        help="Bash script used to start vLLM.",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="http://localhost:8000",
        help="vLLM server address.",
    )

    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=300,
        help="Maximum time to wait for vLLM to become ready.",
    )

    parser.add_argument(
        "--wait-interval",
        type=float,
        default=2,
        help="Seconds between vLLM readiness checks.",
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to data.jsonl.",
    )

    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/chat/completions",
        help="OpenAI endpoint.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=300,
        help="HTTP timeout for benchmark requests.",
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
        default=(
            f"results/result_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ),
        help="Output folder",
    )

    args = parser.parse_args()

    print("=" * 60)
    print(f"Starting vLLM using: {args.serve_script}")
    print("=" * 60)

    os.makedirs(args.output, exist_ok=True)

    stdout_file = os.path.join(args.output, "vllm_out.log")
    stderr_file = os.path.join(args.output, "vllm_err.log")

    with open(stdout_file, "w") as vllm_stdout, \
         open(stderr_file, "w") as vllm_stderr:

        vllm_process = subprocess.Popen(
            ["bash", args.serve_script],

            # Critical:
            # Create a new session/process group.
            start_new_session=True,

            stdout=vllm_stdout,
            stderr=vllm_stderr,
        )

        try:
            ready = wait_for_server(
                host=args.host,
                process=vllm_process,
                timeout=args.wait_timeout,
                interval=args.wait_interval,
            )

            if not ready:
                terminate_process(vllm_process)
                sys.exit(1)

            benchmark_cmd = [
                sys.executable,
                "benchmark.py",
                "--input",
                args.input,
                "--host",
                args.host,
                "--endpoint",
                args.endpoint,
                "--timeout",
                str(args.timeout),
                "--max-workers",
                str(args.max_workers),
                "--output",
                args.output,
            ]

            if args.no_timing:
                benchmark_cmd.append("--no-timing")

            print("\n" + "=" * 60)
            print("vLLM is ready. Starting benchmark...")
            print("=" * 60)

            print("Command:")
            print(" ".join(benchmark_cmd))
            print()

            benchmark_process = subprocess.run(benchmark_cmd)

            if benchmark_process.returncode != 0:
                print(
                    f"\nBenchmark failed with exit code "
                    f"{benchmark_process.returncode}."
                )
                sys.exit(benchmark_process.returncode)

            print("\n" + "=" * 60)
            print("Benchmark completed successfully.")
            print("=" * 60)

            save_bash_file = os.path.join(
                args.output,
                "serve.sh",
            )

            shutil.copy2(
                args.serve_script,
                save_bash_file,
            )

            print(
                f"Saved vLLM serve script to: "
                f"{save_bash_file}"
            )

        except KeyboardInterrupt:
            print("\nInterrupted.")

        finally:
            print("Shutting down vLLM...")
            terminate_process(vllm_process)


if __name__ == "__main__":
    main()
