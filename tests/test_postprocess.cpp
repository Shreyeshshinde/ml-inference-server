#include <gtest/gtest.h>

#include <cmath>
#include <numeric>

#include "postprocess.h"

TEST(Softmax, SumsToOneAndKeepsOrder) {
    std::vector<float> logits = {1.f, 2.f, 3.f};
    auto p = mis::softmax(logits.data(), logits.size());
    EXPECT_NEAR(std::accumulate(p.begin(), p.end(), 0.0), 1.0, 1e-6);
    EXPECT_LT(p[0], p[1]);
    EXPECT_LT(p[1], p[2]);
    EXPECT_NEAR(p[2], std::exp(3.f) / (std::exp(1.f) + std::exp(2.f) + std::exp(3.f)), 1e-6);
}

TEST(Softmax, StableWithHugeLogits) {
    // Naive exp(1000) overflows to inf; the max-subtraction trick avoids that.
    std::vector<float> logits = {1000.f, 1000.f};
    auto p = mis::softmax(logits.data(), logits.size());
    EXPECT_FLOAT_EQ(p[0], 0.5f);
    EXPECT_FLOAT_EQ(p[1], 0.5f);
}

TEST(TopK, ReturnsHighestFirstWithLabels) {
    std::vector<float> probs = {0.1f, 0.5f, 0.05f, 0.35f};
    std::vector<std::string> labels = {"a", "b", "c", "d"};
    auto top = mis::top_k(probs, 2, labels);
    ASSERT_EQ(top.size(), 2u);
    EXPECT_EQ(top[0].class_id, 1);
    EXPECT_EQ(top[0].label, "b");
    EXPECT_EQ(top[1].class_id, 3);
    EXPECT_FLOAT_EQ(top[1].score, 0.35f);
}

TEST(TopK, ClampsKAndHandlesMissingLabels) {
    std::vector<float> probs = {0.7f, 0.3f};
    auto top = mis::top_k(probs, 10, {});
    ASSERT_EQ(top.size(), 2u);
    EXPECT_EQ(top[0].label, "class_0");
    EXPECT_TRUE(mis::top_k(probs, -1, {}).empty());
}

TEST(Labels, MissingFileThrows) {
    EXPECT_THROW(mis::load_labels("/nonexistent/labels.txt"), std::runtime_error);
}
