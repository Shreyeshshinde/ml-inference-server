#pragma once

#include <array>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace mis {

// Per-model preprocessing settings. Defaults match torchvision ImageNet models.
struct PreprocessConfig {
    int resize_short = 256;  // shorter side is resized to this, aspect ratio kept
    int crop = 224;          // then a crop x crop square is cut from the center
    std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};  // RGB order
    std::array<float, 3> std = {0.229f, 0.224f, 0.225f};   // RGB order
};

// Decodes JPEG/PNG/etc. bytes into a BGR image. Returns an empty Mat if the bytes
// are not a valid image (never throws), so the server can answer 400 on bad input.
cv::Mat decode_image(const std::string& bytes);

// Number of floats one preprocessed image occupies: 3 * crop * crop.
inline size_t tensor_size(const PreprocessConfig& cfg) {
    return static_cast<size_t>(3) * cfg.crop * cfg.crop;
}

// Resize, center crop, BGR->RGB, normalize, HWC->CHW.
// Writes tensor_size(cfg) floats to `out`. Writing into a caller-owned buffer lets the
// batcher fill one big [N,3,H,W] buffer without extra copies.
void preprocess_into(const cv::Mat& bgr, const PreprocessConfig& cfg, float* out);

// Convenience wrapper that allocates the output.
std::vector<float> preprocess(const cv::Mat& bgr, const PreprocessConfig& cfg);

}  // namespace mis
