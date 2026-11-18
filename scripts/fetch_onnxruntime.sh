#!/usr/bin/env bash
# Downloads the ONNX Runtime 1.17.3 CPU library for Linux x64 into <dest> with the
# layout CMakeLists.txt expects:  <dest>/include/*.h  and  <dest>/lib/libonnxruntime.so*
#
#   bash scripts/fetch_onnxruntime.sh /opt/onnxruntime
#
# Source: Microsoft's official NuGet package Microsoft.ML.OnnxRuntime on nuget.org.
# (The GitHub release tarball this project first used, under microsoft/onnxruntime,
# returned HTTP 404 when checked on 2026-11-18.) The library inside is byte-identical
# to that tarball's; the SHA-256 check below enforces it.
set -euo pipefail

DEST=${1:?usage: fetch_onnxruntime.sh <dest>}
VERSION=1.17.3
LIB_SHA256=8bdcd79ab25e38d1d7646948e07a7d87ed95c2164a04e43dd685147fd5c86b4c
URL="https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime/${VERSION}"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
echo "downloading $URL"
curl -fsSL -o "$tmp/ort.nupkg" "$URL"

# A .nupkg is a zip file. Take only the C/C++ headers and the Linux x64 library.
mkdir -p "$tmp/x"
unzip -q "$tmp/ort.nupkg" 'build/native/include/*' 'runtimes/linux-x64/native/libonnxruntime.so' -d "$tmp/x"

echo "$LIB_SHA256  $tmp/x/runtimes/linux-x64/native/libonnxruntime.so" | sha256sum -c --quiet -

rm -rf "$DEST"
mkdir -p "$DEST/include" "$DEST/lib"
cp "$tmp"/x/build/native/include/*.h "$DEST/include/"
# The library's SONAME is libonnxruntime.so.1.17.3, so binaries look for that exact
# file name at runtime; the unversioned name is what the linker looks for at build time.
cp "$tmp/x/runtimes/linux-x64/native/libonnxruntime.so" "$DEST/lib/libonnxruntime.so.${VERSION}"
ln -s "libonnxruntime.so.${VERSION}" "$DEST/lib/libonnxruntime.so"
echo "ONNX Runtime ${VERSION} installed in ${DEST}"
