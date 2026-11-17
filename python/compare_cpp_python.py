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


def python_top5(sess, path):
    x = preprocess_bgr(load_image(path))[None]
    logits = sess.run(None, {"input": x})[0][0].astype(np.float64)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    ids = np.argsort(-probs)[:5]
    return [(int(i), float(probs[i])) for i in ids]


def main() -> None:
    images = sample_val_images(20)
    all_ok = True
    for name in MODEL_NAMES:
        model_path = MODELS_DIR / f"{name}.onnx"
        sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        out = subprocess.run([str(CLASSIFY_BIN), str(model_path), str(LABELS_PATH),
                              *map(str, images), "--json"],
                             capture_output=True, text=True, check=True)
        cpp_results = [json.loads(line) for line in out.stdout.splitlines()]
        assert len(cpp_results) == len(images)

        matches, worst = 0, 0.0
        for path, cpp in zip(images, cpp_results):
            py = python_top5(sess, path)
            cpp_ids = [p["class_id"] for p in cpp["predictions"]]
            diffs = [abs(p["score"] - s) for p, (_, s) in zip(cpp["predictions"], py)]
            worst = max(worst, max(diffs))
            if cpp_ids == [i for i, _ in py] and max(diffs) < SCORE_TOL:
                matches += 1
            else:
                print(f"  MISMATCH {path.name}: cpp={cpp_ids} py={[i for i, _ in py]}")
        ok = matches == len(images)
        all_ok &= ok
        print(f"{name:20s} top5_match={matches}/{len(images)}  "
              f"max_score_diff={worst:.2e}  {'PASS' if ok else 'FAIL'}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
