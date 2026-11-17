#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace mis {

// Wraps one ONNX Runtime session. The model is loaded once in the constructor;
// run() can then be called many times, and from several threads at once
// (Ort::Session::Run is thread-safe).
class ModelRunner {
public:
    // intra_op_threads: threads ORT uses inside one Run() call (e.g. to split a matmul).
    ModelRunner(const std::string& model_path, int intra_op_threads = 1);

    // Runs a batch. `input` holds batch * 3 * height * width floats in NCHW order.
    // Returns logits flattened as [batch, num_classes].
    std::vector<float> run(const float* input, int64_t batch, int64_t height, int64_t width);

    int64_t num_classes() const { return num_classes_; }
    const std::string& path() const { return path_; }

private:
    // One Env per process is enough; it owns ORT's logging and global thread state.
    static Ort::Env& env();

    std::string path_;
    std::unique_ptr<Ort::Session> session_;
    std::string input_name_;
    std::string output_name_;
    int64_t num_classes_ = 0;
};

}  // namespace mis
