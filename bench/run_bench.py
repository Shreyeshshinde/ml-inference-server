"""Benchmark driver. Starts a server, warms it up, runs Locust, records results, stops it.

Usage (from the repo root, inside the venv):
  python bench/run_bench.py --experiment all          # everything (about 45 minutes)
  python bench/run_bench.py --experiment batch_sweep --duration 20 --repeats 1   # quick look

Experiments (all on resnet50, FP32):
  cpp_vs_python  C++ server vs FastAPI+PyTorch baseline at 100 users
  batch_sweep    C++ with max_batch_size 1, 4, 8, 16 at 100 users
  load_levels    C++ at 10, 50, 100, 200 users with queue_limit 64 (to see 503s)

Rules: 50 warm-up requests before each run, 60 s per run, 3 repeats, server pinned to
CPUs 0-11 and Locust to CPUs 12-15. Every run is one CSV row; plot_results.py averages.
"""
import argparse
import copy
import csv
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = Path("/tmp/mis_bench")  # Locust's own CSVs (scratch, not committed)
PY = sys.executable
MODEL = "resnet50"
WARMUP_REQUESTS = 50


# ---------------------------------------------------------------- servers

def write_cpp_config(**overrides) -> Path:
    """Copy of config/server.json with only resnet50, logging off, and overrides."""
    cfg = json.loads((ROOT / "config" / "server.json").read_text())
    cfg["models"] = [copy.deepcopy(m) for m in cfg["models"] if m["name"] == MODEL]
    cfg["log_requests"] = False
    # httplib keeps one thread per open connection, so it needs at least as many
    # threads as concurrent users or connections wait before reaching our code.
    cfg["http_threads"] = 256
    cfg["models"][0].update(overrides)
    path = RAW / "bench_config.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


def start_cpp(config: Path, cores: str) -> subprocess.Popen:
    log = open(RAW / "cpp_server.log", "w")
    return subprocess.Popen(["taskset", "-c", cores, str(ROOT / "build" / "inference_server"),
                             "--config", str(config)], cwd=ROOT, stderr=log)


def start_python(threads: int, cores: str) -> subprocess.Popen:
    log = open(RAW / "py_server.log", "w")
    return subprocess.Popen(["taskset", "-c", cores, PY, "python/baseline_server.py",
                             "--port", "8000", "--model", MODEL, "--threads", str(threads)],
                            cwd=ROOT, stderr=log)


def wait_healthy(port: int, proc: subprocess.Popen, timeout_s: float = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server on port {port} exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"server on port {port} did not become healthy")


def stop(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGINT)  # graceful shutdown path
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def warmup(port: int) -> None:
    """50 sequential requests: loads lazy state, fills caches, triggers ORT's first runs."""
    body_file = sorted((ROOT / "data/imagenette2-320/val/n02102040").glob("*.JPEG"))[0]
    for _ in range(WARMUP_REQUESTS):
        subprocess.run(["curl", "-s", "-o", "/dev/null", "-F", f"image=@{body_file}",
                        f"http://localhost:{port}/v1/models/{MODEL}/predict"], check=True)


def server_counters(port: int) -> dict:
    """Counters from the server's /metrics. Works for both servers: the C++ one labels
    its metrics with model="...", the Python baseline does not."""
    with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=5) as r:
        text = r.read().decode()

    def value(name):
        m = re.search(rf'^{name}(?:{{model="{MODEL}"}})? (\S+)$', text, re.M)
        return float(m.group(1)) if m else 0.0

    return {k: value(f"mis_{k}_total")
            for k in ("requests", "errors", "batches", "batched_requests")}


# ---------------------------------------------------------------- locust

def run_locust(port: int, users: int, duration: int, client_cores: str, tag: str) -> dict:
    prefix = RAW / tag
    env = dict(os.environ, MODEL=MODEL)
    cmd = ["taskset", "-c", client_cores, PY, "-m", "locust", "-f", str(ROOT / "bench/locustfile.py"),
           "--headless", "-u", str(users), "-r", str(users), "-t", f"{duration}s",
           "--host", f"http://localhost:{port}", "--csv", str(prefix), "--only-summary"]
    # Single Locust process: the first full run used --processes 2 and its reported
    # requests/s did not match its own request counts.
    subprocess.run(cmd, cwd=ROOT, env=env, check=False,
                   stdout=subprocess.DEVNULL, stderr=open(RAW / f"{tag}_locust.log", "w"))

    with open(f"{prefix}_stats.csv") as f:
        agg = next(r for r in csv.DictReader(f) if r["Name"] == "Aggregated")
    status_counts = {}
    with open(f"{prefix}_failures.csv") as f:
        for r in csv.DictReader(f):
            m = re.search(r"HTTP (\d+)", r["Error"])
            key = m.group(1) if m else "other"
            status_counts[key] = status_counts.get(key, 0) + int(r["Occurrences"])

    total = int(agg["Request Count"])
    failures = int(agg["Failure Count"])
    return {
        "requests": total,
        "failures": failures,
        "http_503": status_counts.get("503", 0),
        "http_504": status_counts.get("504", 0),
        "other_errors": failures - status_counts.get("503", 0) - status_counts.get("504", 0),
        "p50_ms": float(agg["50%"]),
        "p90_ms": float(agg["90%"]),
        "p99_ms": float(agg["99%"]),
        "avg_ms": round(float(agg["Average Response Time"]), 2),
    }


# ---------------------------------------------------------------- experiments

COOLDOWN_S = 20  # idle time between runs so a hot laptop CPU affects runs more evenly


def measured_run(port: int, proc, args, users: int, tag: str) -> dict:
    """Warm up, run Locust, and compute throughput as successful responses the CLIENT
    received / run length. (Locust's own "Requests/s" field was unreliable.) The
    server's success count is kept as a cross-check; it is a bit higher because the
    server also finishes requests that were in flight when Locust stopped."""
    wait_healthy(port, proc)
    warmup(port)
    c0 = server_counters(port)
    row = run_locust(port, users, args.duration, args.client_cores, tag)
    c1 = server_counters(port)
    ok = row["requests"] - row["failures"]
    row["successes"] = ok
    row["images_per_s"] = round(ok / args.duration, 2)
    row["server_successes"] = int((c1["requests"] - c1["errors"]) - (c0["requests"] - c0["errors"]))
    batches = c1["batches"] - c0["batches"]
    row["avg_batch_size"] = (round((c1["batched_requests"] - c0["batched_requests"]) / batches, 2)
                             if batches > 0 else 1.0)  # Python baseline: no batching
    return row


def one_cpp_run(args, users, tag, **overrides) -> dict:
    proc = start_cpp(write_cpp_config(**overrides), args.server_cores)
    try:
        return measured_run(8080, proc, args, users, tag)
    finally:
        stop(proc)
        time.sleep(COOLDOWN_S)


def one_python_run(args, users, threads, tag) -> dict:
    proc = start_python(threads, args.server_cores)
    try:
        return measured_run(8000, proc, args, users, tag)
    finally:
        stop(proc)
        time.sleep(COOLDOWN_S)


def append_rows(path: Path, rows: list[dict]) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def log(row: dict) -> None:
    print(json.dumps(row), flush=True)


def tune_python_threads(args) -> int:
    """Short runs to give the baseline its BEST torch thread count, so the comparison
    is against Python at its best, not at a bad default."""
    best, best_ips, rows = None, -1.0, []
    for threads in (4, 8, 12):
        r = one_python_run(argparse.Namespace(**{**vars(args), "duration": 20}),
                           100, threads, f"tune_py_t{threads}")
        rows.append({"torch_threads": threads, "duration_s": 20, **r})
        log(rows[-1])
        if r["images_per_s"] > best_ips:
            best, best_ips = threads, r["images_per_s"]
    append_rows(RESULTS / "bench_python_tuning.csv", rows)
    return best


def exp_cpp_vs_python(args) -> None:
    py_threads = tune_python_threads(args)
    for rep in range(1, args.repeats + 1):
        # Alternate which server goes first, so slow drift (heat) hits both equally.
        order = ["cpp", "py"] if rep % 2 else ["py", "cpp"]
        for which in order:
            if which == "cpp":
                r = one_cpp_run(args, 100, f"cvp_cpp_{rep}")
                row = {"server": "cpp", "users": 100, "repeat": rep, **r}
            else:
                r = one_python_run(args, 100, py_threads, f"cvp_py_{rep}")
                row = {"server": f"python_t{py_threads}", "users": 100, "repeat": rep, **r}
            log(row)
            append_rows(RESULTS / "bench_cpp_vs_python.csv", [row])


def rotated(values: tuple, rep: int) -> list:
    """Forward order on odd repeats, reversed on even ones (spreads thermal drift)."""
    return list(values) if rep % 2 else list(reversed(values))


def exp_batch_sweep(args) -> None:
    for rep in range(1, args.repeats + 1):
        for bs in rotated((1, 4, 8, 16), rep):
            r = one_cpp_run(args, 100, f"bs{bs}_{rep}", max_batch_size=bs)
            row = {"max_batch_size": bs, "users": 100, "repeat": rep, **r}
            log(row)
            append_rows(RESULTS / "bench_batch_sweep.csv", [row])


def exp_load_levels(args) -> None:
    for rep in range(1, args.repeats + 1):
        for users in rotated((10, 50, 100, 200), rep):
            r = one_cpp_run(args, users, f"load{users}_{rep}", queue_limit=64)
            row = {"users": users, "queue_limit": 64, "repeat": rep, **r}
            log(row)
            append_rows(RESULTS / "bench_load_levels.csv", [row])


def write_hardware(args) -> None:
    cpu = subprocess.run("lscpu | grep 'Model name' | cut -d: -f2", shell=True,
                         capture_output=True, text=True).stdout.strip()
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    (RESULTS / "bench_hardware.txt").write_text(
        f"cpu: {cpu}\nlogical_cpus: {os.cpu_count()}\nram_gb: {mem_gb:.1f}\n"
        f"os: {platform.platform()}\nserver_cpus: {args.server_cores}\n"
        f"client_cpus: {args.client_cores}\nrun_seconds: {args.duration}\n"
        f"repeats: {args.repeats}\nwarmup_requests: {WARMUP_REQUESTS}\n"
        f"model: {MODEL} (FP32)\nload_generator: locust (FastHttpUser, closed loop, 1 process)\n"
        f"throughput: server-side successful responses / run_seconds\n"
        f"cooldown_between_runs_s: {COOLDOWN_S}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="all",
                   choices=["all", "cpp_vs_python", "batch_sweep", "load_levels"])
    p.add_argument("--duration", type=int, default=60)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--server-cores", default="0-11")
    p.add_argument("--client-cores", default="12-15")
    args = p.parse_args()

    RAW.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    write_hardware(args)
    todo = ["cpp_vs_python", "batch_sweep", "load_levels"] if args.experiment == "all" \
        else [args.experiment]
    for name in todo:
        # Fresh CSV per experiment so old runs never mix with new ones.
        for f in {"cpp_vs_python": ["bench_cpp_vs_python.csv", "bench_python_tuning.csv"],
                  "batch_sweep": ["bench_batch_sweep.csv"],
                  "load_levels": ["bench_load_levels.csv"]}[name]:
            (RESULTS / f).unlink(missing_ok=True)
        print(f"=== {name}", flush=True)
        globals()[f"exp_{name}"](args)


if __name__ == "__main__":
    main()
