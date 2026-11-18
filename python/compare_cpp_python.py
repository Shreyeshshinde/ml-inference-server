"""Phase 1 acceptance check: C++ top-5 must match Python top-5 on the same 20 images.

Python side: OpenCV preprocessing (python/common.py) + ONNX Runtime + numpy softmax.
C++ side:    build/classify --json (src/preprocess.cpp + src/model_runner.cpp + src/postprocess.cpp).
Pass rule:   identical top-5 class ids in identical order, and every score within 1e-4.
"""
import json
import subprocess
import sys

import numpy as np
import onnxruntime as ort

from common import (LABELS_PATH, MODEL_NAMES, MODELS_DIR, ROOT, load_image, preprocess_bgr,
                    sample_val_images)

SCORE_TOL = 1e-4
CLASSIFY_BIN = ROOT / "build" / "classify"


def python_probs(sess, path) -> np.ndarray:
    x = preprocess_bgr(load_image(path))[None]
    logits = sess.run(None, {"input": x})[0][0].astype(np.float64)
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def top5_matches(cpp_preds, probs) -> tuple[bool, float]:
    """Rank by rank, C++ and Python must agree on the score; the class id must match
    too, unless several classes share exactly that score (a tie). INT8 models produce
    many exact ties, and C++ partial_sort and NumPy argsort order ties differently.
    Returns (match, worst score difference)."""
    py_ids = np.argsort(-probs)[:5]
    worst, ok = 0.0, True
    for k, p in enumerate(cpp_preds):
        py_score = probs[py_ids[k]]
        diff = abs(p["score"] - py_score)
        worst = max(worst, diff)
        same_class = p["class_id"] == py_ids[k]
        tie = abs(probs[p["class_id"]] - py_score) < SCORE_TOL
        ok &= diff < SCORE_TOL and (same_class or tie)
    return ok, worst


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--suffix", default="", help='model file suffix, e.g. "_int8"')
    args = parser.parse_args()

    images = sample_val_images(20)
    all_ok = True
    for name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{name}{args.suffix}.onnx"
        name = model_path.stem
        # Same settings as the C++ classify tool (1 intra-op thread, full graph
        # optimization). Thread count changes the order of floating-point sums, and in
        # INT8 models those last-bit differences can flip a rounding step.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

        out = subprocess.run([str(CLASSIFY_BIN), str(model_path), str(LABELS_PATH),
                              *map(str, images), "--json"],
                             capture_output=True, text=True, check=True)
        cpp_results = [json.loads(line) for line in out.stdout.splitlines()]
        assert len(cpp_results) == len(images)

        matches, worst = 0, 0.0
        for path, cpp in zip(images, cpp_results):
            probs = python_probs(sess, path)
            ok, diff = top5_matches(cpp["predictions"], probs)
            worst = max(worst, diff)
            if ok:
                matches += 1
            else:
                cpp_ids = [p["class_id"] for p in cpp["predictions"]]
                print(f"  MISMATCH {path.name}: cpp={cpp_ids} "
                      f"py={np.argsort(-probs)[:5].tolist()}")
        ok = matches == len(images)
        all_ok &= ok
        print(f"{name:20s} top5_match={matches}/{len(images)}  "
              f"max_score_diff={worst:.2e}  {'PASS' if ok else 'FAIL'}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
