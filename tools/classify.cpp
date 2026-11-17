// CLI: classify images with an ONNX model and print the top-k predictions.
//
//   classify <model.onnx> <labels.txt> <image> [<image> ...] [--top_k N] [--json]
//
// With --json, prints one JSON object per image (used by python/compare_cpp_python.py).
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <json.hpp>

#include "model_runner.h"
#include "postprocess.h"
#include "preprocess.h"

namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> positional;
    int top_k = 5;
    bool json_out = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--top_k") == 0 && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (std::strcmp(argv[i], "--json") == 0) {
            json_out = true;
        } else {
            positional.emplace_back(argv[i]);
        }
    }
    if (positional.size() < 3) {
        std::cerr << "usage: classify <model.onnx> <labels.txt> <image>... [--top_k N] [--json]\n";
        return 2;
    }

    try {
        mis::ModelRunner runner(positional[0]);
        const auto labels = mis::load_labels(positional[1]);
        const mis::PreprocessConfig cfg;  // ImageNet defaults

        int failures = 0;
        for (size_t i = 2; i < positional.size(); ++i) {
            const std::string& path = positional[i];
            cv::Mat img = mis::decode_image(read_file(path));
            if (img.empty()) {
                std::cerr << "error: cannot decode " << path << "\n";
                ++failures;
                continue;
            }
            auto tensor = mis::preprocess(img, cfg);
            auto logits = runner.run(tensor.data(), 1, cfg.crop, cfg.crop);
            auto probs = mis::softmax(logits.data(), logits.size());
            auto preds = mis::top_k(probs, top_k, labels);

            if (json_out) {
                nlohmann::json j;
                j["image"] = path;
                for (const auto& p : preds) {
                    j["predictions"].push_back(
                        {{"class_id", p.class_id}, {"label", p.label}, {"score", p.score}});
                }
                std::cout << j.dump() << "\n";
            } else {
                std::cout << path << "\n";
                for (const auto& p : preds) {
                    std::cout << "  " << p.class_id << "\t" << p.score << "\t" << p.label << "\n";
                }
            }
        }
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
