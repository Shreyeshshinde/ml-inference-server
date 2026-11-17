"""Python baseline: FastAPI + PyTorch, same models, preprocessing and JSON as the C++ server.

Run from the repo root:
    python python/baseline_server.py --port 8000 [--model resnet50] [--threads 4]

Fairness choices (so benchmarks compare architectures, not a crippled baseline):
- Model loaded once at startup, eval mode, torch.inference_mode() (no autograd bookkeeping).
- Identical OpenCV preprocessing (python/common.py), verified to match the C++ code.
- The endpoint is a plain `def`, so FastAPI runs it in a thread pool. PyTorch releases
  the GIL inside its kernels, so several requests can compute at the same time.
- No dynamic batching: that is the C++ server's feature being measured.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import LABELS_PATH, ROOT, preprocess_bgr  # noqa: E402

MAX_BODY_BYTES = 10 * 1024 * 1024


def create_app(model_name: str, config_path: Path) -> FastAPI:
    cfg = json.loads(config_path.read_text())
    mcfg = next((m for m in cfg["models"] if m["name"] == model_name), None)
    if mcfg is None:
        raise SystemExit(f"model {model_name} not in {config_path}")

    model = getattr(torchvision.models, model_name)(weights="IMAGENET1K_V2").eval()
    labels = LABELS_PATH.read_text().splitlines()
    resize_short = mcfg.get("resize_short", 256)
    crop = mcfg.get("input_size", 224)
    counters = {"requests": 0, "errors": 0}

    app = FastAPI(title="baseline")

    def error(status: int, msg: str):
        counters["errors"] += 1
        raise HTTPException(status_code=status, detail=msg)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        return (f"mis_requests_total {counters['requests']}\n"
                f"mis_errors_total {counters['errors']}\n")

    @app.post("/v1/models/{name}/predict")
    def predict(name: str, image: UploadFile = File(...), top_k: int = Query(5)):
        t_start = time.perf_counter()
        counters["requests"] += 1
        if name != model_name:
            error(404, f"unknown model: {name}")
        if not 1 <= top_k <= len(labels):
            error(400, "top_k out of range")
        data = image.file.read()
        if len(data) > MAX_BODY_BYTES:
            error(413, "payload too large")

        t_pre = time.perf_counter()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
        if img is None:
            error(400, "could not decode image")
        x = torch.from_numpy(preprocess_bgr(img, resize_short, crop)).unsqueeze(0)

        t_inf = time.perf_counter()
        with torch.inference_mode():
            logits = model(x)[0]

        t_post = time.perf_counter()
        probs = torch.softmax(logits, dim=0)
        scores, ids = torch.topk(probs, top_k)
        preds = [{"class_id": int(i), "label": labels[int(i)], "score": float(s)}
                 for s, i in zip(scores, ids)]
        t_end = time.perf_counter()

        ms = lambda a, b: (b - a) * 1000.0  # noqa: E731
        return JSONResponse({
            "model": model_name,
            "version": mcfg.get("version", 1),
            "predictions": preds,
            "batch_size": 1,
            "latency_ms": ms(t_start, t_end),
            "timings_ms": {"queue_wait": 0.0, "preprocess": ms(t_pre, t_inf),
                           "inference": ms(t_inf, t_post), "postprocess": ms(t_post, t_end)},
        })

    # Same error body shape as the C++ server: {"error": "..."}
    @app.exception_handler(HTTPException)
    def http_error(_, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    # FastAPI answers 422 for a missing/invalid field by default; the spec says 400.
    @app.exception_handler(RequestValidationError)
    def validation_error(_, exc: RequestValidationError):
        counters["errors"] += 1
        return JSONResponse({"error": "bad request: " + str(exc.errors()[:1])}, status_code=400)

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--config", default=str(ROOT / "config" / "server.json"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=4, help="torch intra-op threads")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    app = create_app(args.model, Path(args.config))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
