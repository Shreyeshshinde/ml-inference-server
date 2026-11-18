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

### Export models and verify

```bash
python3 -m venv ~/venvs/mis
~/venvs/mis/bin/pip install -r python/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
bash scripts/download_data.sh                    # Imagenette, about 340 MB
~/venvs/mis/bin/python python/export_onnx.py     # models/*.onnx + data/imagenet_labels.txt
~/venvs/mis/bin/python python/verify_onnx.py     # PyTorch vs ONNX Runtime, max abs diff < 1e-4
~/venvs/mis/bin/python python/compare_cpp_python.py   # C++ top-5 vs Python top-5 on 20 images
./build/classify models/resnet50.onnx data/imagenet_labels.txt path/to/image.jpg
```

### Run the server

```bash
./build/inference_server --config config/server.json
curl -F "image=@dog.jpg" "http://localhost:8080/v1/models/resnet50/predict?top_k=5"
bash scripts/smoke_test.sh     # C++ server and Python baseline side by side
bash scripts/batch_demo.sh     # concurrent load, prints the batch sizes that formed
```

## API

To be added.

## Benchmarks

To be added (measured results only).

## Design decisions

To be added.

## Limitations and future work

To be added.
