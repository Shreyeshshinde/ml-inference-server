#include <gtest/gtest.h>

#include <filesystem>

#include "model_runner.h"

// These tests need an exported model. They are skipped (not failed) when it's missing,
// so CI without models still passes. MIS_MODELS_DIR is set by CMake.
namespace {
const std::string kModel = std::string(MIS_MODELS_DIR) + "/mobilenet_v3_large.onnx";

bool have_model() { return std::filesystem::exists(kModel); }
}  // namespace

TEST(ModelRunner, MissingFileThrows) {
    EXPECT_ANY_THROW(mis::ModelRunner("/nonexistent/model.onnx"));
}

TEST(ModelRunner, RunsBatchOfOne) {
    if (!have_model()) GTEST_SKIP() << "model not exported: " << kModel;
    mis::ModelRunner runner(kModel);
    EXPECT_EQ(runner.num_classes(), 1000);
    std::vector<float> input(3 * 224 * 224, 0.f);
    auto out = runner.run(input.data(), 1, 224, 224);
    EXPECT_EQ(out.size(), 1000u);
}

TEST(ModelRunner, BatchedRowsEqualSingleRuns) {
    // Dynamic batch axis check: running [img_a, img_b] as one batch must give the
    // same logits as running each image alone. This is what makes batching safe.
    if (!have_model()) GTEST_SKIP() << "model not exported: " << kModel;
    mis::ModelRunner runner(kModel);
    const size_t n = 3 * 224 * 224;
    std::vector<float> batch(2 * n);
    for (size_t i = 0; i < n; ++i) {
        batch[i] = static_cast<float>((i % 255) / 255.0 - 0.5);
        batch[n + i] = static_cast<float>(((i * 7) % 255) / 255.0 - 0.5);
    }
    auto both = runner.run(batch.data(), 2, 224, 224);
    auto a = runner.run(batch.data(), 1, 224, 224);
    auto b = runner.run(batch.data() + n, 1, 224, 224);
    ASSERT_EQ(both.size(), 2000u);
    for (size_t i = 0; i < 1000; ++i) {
        EXPECT_NEAR(both[i], a[i], 1e-3);
        EXPECT_NEAR(both[1000 + i], b[i], 1e-3);
    }
}
