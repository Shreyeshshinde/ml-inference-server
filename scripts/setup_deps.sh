#!/usr/bin/env bash
# Installs everything needed to build the C++ server on Ubuntu 22.04 (or WSL2).
# Usage: sudo bash scripts/setup_deps.sh
set -euo pipefail

ORT_DIR="/opt/onnxruntime"
HTTPLIB_VERSION="v0.15.3"
JSON_VERSION="v3.11.3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Installing system packages (compiler, CMake, OpenCV)"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake pkg-config git curl ca-certificates unzip \
    clang-format libopencv-dev

# ONNX Runtime is not in apt; fetch Microsoft's prebuilt CPU library (hash-checked).
if [ ! -f "$ORT_DIR/lib/libonnxruntime.so" ]; then
    echo "==> Installing ONNX Runtime to $ORT_DIR"
    bash "$SCRIPT_DIR/fetch_onnxruntime.sh" "$ORT_DIR"
else
    echo "==> ONNX Runtime already present in $ORT_DIR"
fi

# Single-header libraries are committed in third_party/ at pinned versions; fetch
# them only if they are missing.
mkdir -p "$ROOT_DIR/third_party"
if [ ! -f "$ROOT_DIR/third_party/httplib.h" ]; then
    curl -fsSL -o "$ROOT_DIR/third_party/httplib.h" \
        "https://raw.githubusercontent.com/yhirose/cpp-httplib/${HTTPLIB_VERSION}/httplib.h"
fi
if [ ! -f "$ROOT_DIR/third_party/json.hpp" ]; then
    curl -fsSL -o "$ROOT_DIR/third_party/json.hpp" \
        "https://github.com/nlohmann/json/releases/download/${JSON_VERSION}/json.hpp"
fi

echo "==> Done. Build with:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build"
