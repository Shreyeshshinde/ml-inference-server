# Multi-stage build. Each stage starts clean and copies in only what it needs, so the
# final image has the server, its shared libraries and the models, and nothing else
# (no compiler, no PyTorch).
#
#   docker build -t mini-triton .
#   docker run --rm -p 8080:8080 mini-triton
#
# Models are produced in stage 1 from torchvision's public pretrained weights, so they
# are never committed to Git. Only the FP32 models are built here; the INT8 files need
# the Imagenette dataset for calibration (see python/quantize.py).

# ---------- Stage 1: export ONNX models with PyTorch ----------
FROM python:3.10-slim AS export
WORKDIR /work
# CPU-only PyTorch, pinned to the +cpu builds. Plain "torch==2.5.1" from PyPI is the
# CUDA build, which drags in gigabytes of GPU libraries. PyPI stays available as an
# extra index for torch's ordinary dependencies; the +cpu pins keep the CUDA build out.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.5.1+cpu torchvision==0.20.1+cpu
RUN pip install --no-cache-dir "numpy<2" onnx==1.17.0 onnxruntime==1.17.3 \
        onnxsim==0.4.36 opencv-python-headless==4.10.0.84
COPY python/common.py python/export_onnx.py python/
RUN mkdir -p models data && python python/export_onnx.py

# ---------- Stage 2: compile the C++ server ----------
FROM ubuntu:22.04 AS build
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential cmake curl ca-certificates unzip libopencv-dev \
    && rm -rf /var/lib/apt/lists/*
# ONNX Runtime 1.17.3 from Microsoft's NuGet package, SHA-256 checked by the script.
COPY scripts/fetch_onnxruntime.sh /tmp/
RUN bash /tmp/fetch_onnxruntime.sh /opt/onnxruntime
WORKDIR /src
COPY CMakeLists.txt ./
COPY include include
COPY src src
COPY tools tools
COPY third_party third_party
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMIS_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)" --target inference_server

# ---------- Stage 3: slim runtime ----------
FROM ubuntu:22.04 AS runtime
# Only the OpenCV runtime libraries we link (core, imgproc, imgcodecs), not -dev
# packages. curl is for the HEALTHCHECK.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libopencv-core4.5d libopencv-imgproc4.5d libopencv-imgcodecs4.5d curl \
    && rm -rf /var/lib/apt/lists/*
# Run as an unprivileged user: a bug in the server should not mean root in the container.
RUN useradd --system --uid 10001 mis
# Same path as in the build stage: the binary's RPATH points here.
COPY --from=build /opt/onnxruntime/lib/ /opt/onnxruntime/lib/
WORKDIR /app
# --chown sets ownership while copying. A separate "RUN chown -R" would store a second
# full copy of the 124 MB of models in a new layer.
COPY --chown=mis --from=build /src/build/inference_server ./
COPY --chown=mis --from=export /work/models/ models/
COPY --chown=mis --from=export /work/data/imagenet_labels.txt data/
COPY --chown=mis config/server.json config/
USER mis
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s \
    CMD curl -fs http://localhost:8080/health || exit 1
CMD ["./inference_server", "--config", "config/server.json"]
