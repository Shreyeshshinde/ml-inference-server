#!/usr/bin/env bash
# Installs everything needed to build the C++ server on Ubuntu 22.04 (or WSL2).
# Usage: sudo bash scripts/setup_deps.sh
set -euo pipefail

ORT_VERSION="1.17.3"
ORT_DIR="/opt/onnxruntime"
HTTPLIB_VERSION="v0.15.3"
JSON_VERSION="v3.11.3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==> Installing system packages (compiler, CMake, OpenCV)"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential cmake pkg-config git curl ca-certificates \
    clang-format libopencv-dev

# ONNX Runtime is not in apt, so we use the official prebuilt CPU release.
if [ ! -f "$ORT_DIR/lib/libonnxruntime.so" ]; then
    echo "==> Downloading ONNX Runtime $ORT_VERSION to $ORT_DIR"
    tmp="$(mktemp -d)"
    curl -fsSL -o "$tmp/ort.tgz" \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    tar -xzf "$tmp/ort.tgz" -C "$tmp"
    rm -rf "$ORT_DIR"
    mv "$tmp/onnxruntime-linux-x64-${ORT_VERSION}" "$ORT_DIR"
    rm -rf "$tmp"
else
    echo "==> ONNX Runtime already present in $ORT_DIR"
fi

# Single-header libraries are vendored into third_party/ at pinned versions.
echo "==> Fetching single-header libraries into third_party/"
mkdir -p "$ROOT_DIR/third_party"
curl -fsSL -o "$ROOT_DIR/third_party/httplib.h" \
    "https://raw.githubusercontent.com/yhirose/cpp-httplib/${HTTPLIB_VERSION}/httplib.h"
curl -fsSL -o "$ROOT_DIR/third_party/json.hpp" \
    "https://github.com/nlohmann/json/releases/download/${JSON_VERSION}/json.hpp"

echo "==> Done. Build with:"
echo "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build"
