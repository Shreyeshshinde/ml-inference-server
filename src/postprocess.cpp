#include "postprocess.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace mis {

std::vector<std::string> load_labels(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open labels file: " + path);
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // tolerate CRLF files
        labels.push_back(line);
    }
    return labels;
}

std::vector<float> softmax(const float* logits, size_t n) {
    std::vector<float> probs(n);
    if (n == 0) return probs;
    // exp(x - max) gives the same result as exp(x) after normalizing,
    // but the largest exponent is exp(0) = 1, so it can never overflow.
    const float max_logit = *std::max_element(logits, logits + n);
    double sum = 0.0;  // double accumulator: less rounding error over 1000 terms
    for (size_t i = 0; i < n; ++i) {
        probs[i] = std::exp(logits[i] - max_logit);
        sum += probs[i];
    }
    for (auto& p : probs) p = static_cast<float>(p / sum);
    return probs;
}

std::vector<Prediction> top_k(const std::vector<float>& probs, int k,
                              const std::vector<std::string>& labels) {
    const int n = static_cast<int>(probs.size());
    k = std::clamp(k, 0, n);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    // partial_sort only fully sorts the first k: O(n log k) instead of O(n log n).
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int a, int b) { return probs[a] > probs[b]; });

    std::vector<Prediction> result;
    result.reserve(k);
    for (int i = 0; i < k; ++i) {
        const int id = idx[i];
        std::string label =
            id < static_cast<int>(labels.size()) ? labels[id] : "class_" + std::to_string(id);
        result.push_back({id, std::move(label), probs[id]});
    }
    return result;
}

}  // namespace mis
