#include "model_runner.h"

#include <array>
#include <stdexcept>

namespace mis {

Ort::Env& ModelRunner::env() {
    // Function-local static: created on first use, thread-safe initialization in C++11+.
    static Ort::Env instance(ORT_LOGGING_LEVEL_WARNING, "mis");
    return instance;
}

ModelRunner::ModelRunner(const std::string& model_path, int intra_op_threads)
    : path_(model_path) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(intra_op_threads);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    // Throws Ort::Exception if the file is missing or not a valid model.
    session_ = std::make_unique<Ort::Session>(env(), model_path.c_str(), opts);

    if (session_->GetInputCount() != 1 || session_->GetOutputCount() != 1) {
        throw std::runtime_error("expected a model with exactly 1 input and 1 output: " +
                                 model_path);
    }
    // Read names from the model instead of hard-coding "input"/"logits".
    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_->GetInputNameAllocated(0, alloc).get();
    output_name_ = session_->GetOutputNameAllocated(0, alloc).get();

    // Output shape is [batch, num_classes]; batch is dynamic (-1), classes is fixed.
    auto out_shape = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (out_shape.size() != 2 || out_shape[1] <= 0) {
        throw std::runtime_error("expected output shape [batch, classes]: " + model_path);
    }
    num_classes_ = out_shape[1];
}

std::vector<float> ModelRunner::run(const float* input, int64_t batch, int64_t height,
                                    int64_t width) {
    const std::array<int64_t, 4> shape = {batch, 3, height, width};
    const size_t count = static_cast<size_t>(batch * 3 * height * width);

    // Wraps our buffer as a tensor without copying. ORT only reads it.
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(input), count,
                                                    shape.data(), shape.size());

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, 1);

    const float* logits = outputs[0].GetTensorData<float>();
    return std::vector<float>(logits, logits + batch * num_classes_);
}

}  // namespace mis
