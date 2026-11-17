#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>

#include "version.h"

// Sanity checks that the toolchain is wired correctly before real code exists.

TEST(Setup, OnnxRuntimeIsAtLeast117) {
    std::string v = mis::onnxruntime_version();  // e.g. "1.17.3"
    int major = 0, minor = 0;
    ASSERT_EQ(std::sscanf(v.c_str(), "%d.%d", &major, &minor), 2);
    EXPECT_TRUE(major > 1 || (major == 1 && minor >= 17)) << "got " << v;
}

TEST(Setup, CanCreateOrtEnv) {
    // Creating an Env exercises the shared library at runtime, not just at link time.
    EXPECT_NO_THROW({ Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test"); });
}

TEST(Setup, OpenCvMatWorks) {
    cv::Mat m = cv::Mat::ones(2, 3, CV_32F);
    EXPECT_EQ(m.rows, 2);
    EXPECT_FLOAT_EQ(cv::sum(m)[0], 6.0);
}
