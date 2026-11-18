"""FP32 vs INT8 comparison on the Imagenette validation split.

Usage: python python/eval_accuracy.py [--threads 4] [--latency-runs 200]
Writes results/accuracy.csv and results/hardware.txt.

For each model file it measures:
- size_mb: file size on disk
- latency: batch-1 inference time (median and p90 over many runs, after warm-up),
  ONNX Runtime only, preprocessing excluded
- top1 / top5: over all 1000 ImageNet classes (standard), on every val image
"""
import argparse
import csv
import os
import platform
import subprocess
import time

import numpy as np
import onnxruntime as ort

from common import MODEL_NAMES, MODELS_DIR, ROOT, load_image, preprocess_bgr
from imagenette import check_mapping, list_split

RESULTS = ROOT / "results"


def make_session(path, threads: int) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def measure_latency(sess, x: np.ndarray, runs: int) -> tuple[float, float]:
    for _ in range(20):  # warm-up: first runs pay one-time allocation costs
        sess.run(None, {"input": x})
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times)), float(np.percentile(times, 90))


def measure_accuracy(sess, items: list, batch: int = 64):
    """items: (path, target) pairs. Images are loaded and preprocessed one chunk at a
    time, so peak memory is one batch (about 40 MB), not the whole val set (2.4 GB)."""
    top1 = top5 = 0
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        x = np.stack([preprocess_bgr(load_image(p)) for p, _ in chunk])
        t = np.array([target for _, target in chunk])
        logits = sess.run(None, {"input": x})[0]
        top5_ids = np.argsort(-logits, axis=1)[:, :5]
        top1 += int((top5_ids[:, 0] == t).sum())
        top5 += int((top5_ids == t[:, None]).any(axis=1).sum())
    return top1 / len(items), top5 / len(items)


def write_hardware(threads: int) -> None:
    cpu = subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | cut -d: -f2"],
                         capture_output=True, text=True).stdout.strip()
    mem_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    (RESULTS / "hardware.txt").write_text(
        f"cpu: {cpu}\nlogical_cpus: {os.cpu_count()}\nram_gb: {mem_gb:.1f}\n"
        f"os: {platform.platform()}\nonnxruntime: {ort.__version__}\n"
        f"intra_op_threads: {threads}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--latency-runs", type=int, default=200)
    args = parser.parse_args()
    check_mapping()
    RESULTS.mkdir(exist_ok=True)

    val = list_split("val")
    one_image = preprocess_bgr(load_image(val[0][0]))[None]  # fixed input for latency
    print(f"evaluating on {len(val)} validation images ...")

    rows = []
    for name in MODEL_NAMES:
        for precision, fname in [("fp32", f"{name}.onnx"), ("int8", f"{name}_int8.onnx")]:
            path = MODELS_DIR / fname
            if not path.exists():
                print(f"skip {fname} (not found; run quantize.py)")
                continue
            sess = make_session(path, args.threads)
            p50, p90 = measure_latency(sess, one_image, args.latency_runs)
            top1, top5 = measure_accuracy(sess, val)
            row = {
                "model": name, "precision": precision,
                "size_mb": round(path.stat().st_size / 1e6, 1),
                "latency_ms_p50": round(p50, 2), "latency_ms_p90": round(p90, 2),
                "top1": round(top1 * 100, 2), "top5": round(top5 * 100, 2),
                "num_images": len(val), "intra_op_threads": args.threads,
            }
            rows.append(row)
            print(row)

    out = RESULTS / "accuracy.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_hardware(args.threads)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
