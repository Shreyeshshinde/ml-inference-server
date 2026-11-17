#!/usr/bin/env bash
# Downloads Imagenette (320px version, about 340 MB) into data/imagenette2-320/
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR/data"
if [ ! -d imagenette2-320 ]; then
    curl -fsSL -o imagenette2-320.tgz https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
    tar -xzf imagenette2-320.tgz
    rm imagenette2-320.tgz
fi
echo "Imagenette ready in data/imagenette2-320"
