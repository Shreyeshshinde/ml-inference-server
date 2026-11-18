#!/usr/bin/env bash
# Zero-downtime reload on the real server: sends continuous traffic to BOTH models,
# reloads resnet50 twice in the middle, then reports status codes and versions seen.
#   bash scripts/reload_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."
imgs=(data/imagenette2-320/val/n02102040/*.JPEG)
IMG=${imgs[0]}
LOG=/tmp/reload_demo_server.log
OUT=/tmp/reload_demo_responses.txt
: > "$OUT"

./build/inference_server --config config/server.json 2>"$LOG" &
PID=$!
trap 'kill -INT $PID 2>/dev/null || true' EXIT
for _ in $(seq 1 120); do curl -sf localhost:8080/health >/dev/null && break; sleep 0.5; done

echo "--- GET /v1/models"
curl -s localhost:8080/v1/models | python3 -c \
    'import json,sys; [print(" ", m["name"], "v%d" % m["version"], m["status"], m["path"]) for m in json.load(sys.stdin)["models"]]'

# Each client loops until the stop file appears, writing "model status version" per request.
client() {
    local model=$1
    while [ ! -f /tmp/reload_demo_stop ]; do
        curl -s -w ' %{http_code}\n' -F "image=@$IMG" "localhost:8080/v1/models/$model/predict?top_k=1" |
            python3 -c 'import json,sys
line=sys.stdin.read().rsplit(" ",1); body,code=line[0],line[1].strip()
v=json.loads(body).get("version","-") if code=="200" else "-"
print(sys.argv[1], code, v)' "$model" >> "$OUT"
    done
}
rm -f /tmp/reload_demo_stop
for _ in 1 2 3 4 5 6; do client resnet50 & done
for _ in 1 2; do client mobilenet_v3_large & done

sleep 3
echo "--- reload resnet50 (same file, new version) while traffic is running"
curl -s -X POST localhost:8080/v1/models/resnet50/reload; echo
sleep 3
curl -s -X POST -H 'Content-Type: application/json' \
    -d '{"path": "models/resnet50.onnx", "version": 10}' localhost:8080/v1/models/resnet50/reload; echo
sleep 3
touch /tmp/reload_demo_stop
wait $(jobs -p | grep -v "^$PID$") 2>/dev/null || true

echo "--- responses (count, model, http status, version that answered)"
sort "$OUT" | uniq -c
