#!/usr/bin/env bash
# Shows dynamic batching on the real server: fires N concurrent requests, then prints
# the batch sizes the workers actually ran, the metrics, and the graceful shutdown log.
#   bash scripts/batch_demo.sh [num_requests] [concurrency]
set -euo pipefail
cd "$(dirname "$0")/.."
N=${1:-200}
C=${2:-32}
imgs=(data/imagenette2-320/val/n02102040/*.JPEG)
IMG=${imgs[0]}
LOG=/tmp/batch_demo_server.log

./build/inference_server --config config/server.json 2>"$LOG" &
PID=$!
for _ in $(seq 1 60); do curl -sf localhost:8080/health >/dev/null && break; sleep 0.5; done

echo "sending $N requests with $C in parallel ..."
start=$(date +%s.%N)
seq 1 "$N" | xargs -P "$C" -I{} curl -s -o /dev/null -w '%{http_code}\n' \
    -F "image=@$IMG" localhost:8080/v1/models/resnet50/predict | sort | uniq -c
end=$(date +%s.%N)
echo "wall time: $(echo "$end - $start" | bc) s (rough; real benchmarks come in Phase 6)"

echo "--- batch sizes the workers ran (count x size)"
grep '^batch ' "$LOG" | sed -E 's/.*size=([0-9]+).*/\1/' | sort -n | uniq -c

echo "--- metrics"
curl -s localhost:8080/metrics | grep -E '^mis_(requests|rejected|batches|avg_batch|queue)'

echo "--- graceful shutdown (SIGINT)"
kill -INT $PID
wait $PID || true
tail -3 "$LOG"
