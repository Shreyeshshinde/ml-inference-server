#pragma once

#include <string>
#include <vector>

namespace mis {

struct Prediction {
    int class_id;
    std::string label;
    float score;  // softmax probability in [0, 1]
};

// Reads one class name per line; line index is the class id.
// Throws std::runtime_error if the file can't be opened.
std::vector<std::string> load_labels(const std::string& path);

// Numerically stable softmax: subtracts the max logit first so exp() can't overflow.
std::vector<float> softmax(const float* logits, size_t n);

// Indices of the k highest probabilities, highest first. k is clamped to n.
// `labels` may be empty (labels then become "class_<id>").
std::vector<Prediction> top_k(const std::vector<float>& probs, int k,
                              const std::vector<std::string>& labels);

}  // namespace mis
