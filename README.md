# ML Inference Server ("Mini Triton")

A C++17 server that keeps ONNX models loaded in memory and serves image classification over HTTP, with dynamic batching, per-model queues, backpressure, and zero-downtime model reload.

> Status: work in progress (Phase 0, project setup).

## Headline result

To be filled in with measured numbers after benchmarking.

## Architecture

To be added.

## Quick start (Linux / WSL2)

```bash
sudo bash scripts/setup_deps.sh          # compiler, CMake, OpenCV, ONNX Runtime, header libs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/inference_server
```

## API

To be added.

## Benchmarks

To be added (measured results only).

## Design decisions

To be added.

## Limitations and future work

To be added.
