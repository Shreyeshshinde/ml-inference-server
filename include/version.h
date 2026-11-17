#pragma once

#include <string>

namespace mis {

// Versions of the libraries we linked against. Printed at startup so logs show
// exactly which runtime produced a given benchmark number.
std::string onnxruntime_version();
std::string opencv_version();

}  // namespace mis
