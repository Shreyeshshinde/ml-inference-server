#!/usr/bin/env bash
# Starts the C++ server (port 8080) and the Python baseline (port 8000), sends the same
# image to both, prints both responses, and checks error codes. Run from the repo root.
#   bash scripts/smoke_test.sh [image.jpg]
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/venvs/mis/bin/python}
default_imgs=(data/imagenette2-320/val/n02102040/*.JPEG)
IMG=${1:-${default_imgs[0]}}

./build/inference_server --config config/server.json 2>/tmp/cpp_server.log &
CPP_PID=$!
"$PY" python/baseline_server.py --port 8000 2>/tmp/py_server.log &
PY_PID=$!
trap 'kill -INT $CPP_PID $PY_PID 2>/dev/null || true' EXIT

for port in 8080 8000; do
    for _ in $(seq 1 60); do curl -sf "localhost:$port/health" >/dev/null && break; sleep 0.5; done
done

echo "image: $IMG"
for port in 8080 8000; do
    echo "--- port $port"
    curl -s -F "image=@$IMG" "localhost:$port/v1/models/resnet50/predict?top_k=3"
    echo
done

echo "--- error codes (expected: 404 400 400)"
for port in 8080 8000; do
    a=$(curl -s -o /dev/null -w '%{http_code}' -F "image=@$IMG" "localhost:$port/v1/models/nope/predict")
    b=$(dd if=/dev/urandom of=/tmp/rand.bin bs=2000 count=1 2>/dev/null; curl -s -o /dev/null -w '%{http_code}' -F "image=@/tmp/rand.bin" "localhost:$port/v1/models/resnet50/predict")
    c=$(curl -s -o /dev/null -w '%{http_code}' -X POST "localhost:$port/v1/models/resnet50/predict")
    echo "port $port: $a $b $c"
done
