#include "preprocess.h"

#include <cmath>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace mis {

cv::Mat decode_image(const std::string& bytes) {
    if (bytes.empty()) return {};
    // cv::Mat header over the existing bytes: no copy. imdecode only reads it.
    cv::Mat raw(1, static_cast<int>(bytes.size()), CV_8UC1,
                const_cast<char*>(bytes.data()));
    try {
        return cv::imdecode(raw, cv::IMREAD_COLOR);  // empty Mat on failure
    } catch (const cv::Exception&) {
        return {};  // some corrupt inputs make OpenCV throw instead
    }
}

void preprocess_into(const cv::Mat& bgr, const PreprocessConfig& cfg, float* out) {
    if (bgr.empty() || bgr.type() != CV_8UC3) {
        throw std::invalid_argument("preprocess expects a non-empty 8-bit 3-channel image");
    }

    // 1. Resize shorter side to resize_short. Long side truncated, same as torchvision.
    int h = bgr.rows, w = bgr.cols;
    int new_h, new_w;
    if (h < w) {
        new_h = cfg.resize_short;
        new_w = static_cast<int>(static_cast<double>(cfg.resize_short) * w / h);
    } else {
        new_w = cfg.resize_short;
        new_h = static_cast<int>(static_cast<double>(cfg.resize_short) * h / w);
    }
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 2. Center crop. A ROI is just a view into `resized`, no pixel copy.
    // Offsets use round-half-to-even (std::nearbyint), matching Python's round() that
    // torchvision's CenterCrop uses. std::round would turn 50.5 into 51 instead of 50,
    // shifting the crop by one pixel on odd-sized images.
    int top = static_cast<int>(std::nearbyint((new_h - cfg.crop) / 2.0));
    int left = static_cast<int>(std::nearbyint((new_w - cfg.crop) / 2.0));
    cv::Mat cropped = resized(cv::Rect(left, top, cfg.crop, cfg.crop));

    // 3+4+5. BGR->RGB, scale to [0,1], normalize, and HWC->CHW, all in one pass.
    // OpenCV stores pixels interleaved (B,G,R,B,G,R...). The model wants planes:
    // all R values, then all G, then all B. Channel c of pixel (y,x) goes to
    // out[c*H*W + y*W + x]. RGB channel c is BGR index (2 - c).
    const int hw = cfg.crop * cfg.crop;
    for (int y = 0; y < cfg.crop; ++y) {
        const cv::Vec3b* row = cropped.ptr<cv::Vec3b>(y);
        for (int x = 0; x < cfg.crop; ++x) {
            const int idx = y * cfg.crop + x;
            for (int c = 0; c < 3; ++c) {
                float v = row[x][2 - c] / 255.0f;
                out[c * hw + idx] = (v - cfg.mean[c]) / cfg.std[c];
            }
        }
    }
}

std::vector<float> preprocess(const cv::Mat& bgr, const PreprocessConfig& cfg) {
    std::vector<float> out(tensor_size(cfg));
    preprocess_into(bgr, cfg, out.data());
    return out;
}

}  // namespace mis
