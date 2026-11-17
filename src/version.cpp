#include "version.h"

#include <onnxruntime_cxx_api.h>
#include <opencv2/core/version.hpp>

namespace mis {

std::string onnxruntime_version() { return Ort::GetVersionString(); }

std::string opencv_version() { return CV_VERSION; }

}  // namespace mis
