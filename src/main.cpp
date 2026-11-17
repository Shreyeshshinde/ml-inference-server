#include <iostream>

#include "version.h"

// Phase 0 placeholder: proves the binary links against ONNX Runtime and OpenCV.
// The real HTTP server replaces this in Phase 2.
int main() {
    std::cout << "ml-inference-server (hello world)\n"
              << "  ONNX Runtime: " << mis::onnxruntime_version() << "\n"
              << "  OpenCV:       " << mis::opencv_version() << "\n";
    return 0;
}
