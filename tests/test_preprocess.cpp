#include <gtest/gtest.h>

#include <opencv2/imgcodecs.hpp>

#include "preprocess.h"

using mis::PreprocessConfig;

TEST(Preprocess, OutputHasChwSize) {
    PreprocessConfig cfg;
    cv::Mat img(300, 400, CV_8UC3, cv::Scalar(10, 20, 30));
    auto t = mis::preprocess(img, cfg);
    EXPECT_EQ(t.size(), 3u * 224 * 224);
}

TEST(Preprocess, SolidColorIsNormalizedPerChannelInRgbOrder) {
    // A solid BGR (0, 128, 255) image means R=255, G=128, B=0.
    // After normalization every pixel of plane c must equal (v/255 - mean[c]) / std[c].
    PreprocessConfig cfg;
    cv::Mat img(240, 320, CV_8UC3, cv::Scalar(0, 128, 255));
    auto t = mis::preprocess(img, cfg);
    const size_t hw = 224 * 224;
    const float rgb[3] = {255.f, 128.f, 0.f};
    for (int c = 0; c < 3; ++c) {
        float expected = (rgb[c] / 255.f - cfg.mean[c]) / cfg.std[c];
        EXPECT_NEAR(t[c * hw], expected, 1e-5) << "channel " << c;
        EXPECT_NEAR(t[c * hw + hw - 1], expected, 1e-5) << "channel " << c;
    }
}

TEST(Preprocess, LayoutIsChwNotHwc) {
    // Left half black, right half white. In CHW, row 0 of plane 0 should go from
    // low values (black) on the left to high values (white) on the right.
    PreprocessConfig cfg;
    cfg.resize_short = 224;  // no resize distortion: image is already 224x224
    cv::Mat img(224, 224, CV_8UC3, cv::Scalar(0, 0, 0));
    img(cv::Rect(112, 0, 112, 224)).setTo(cv::Scalar(255, 255, 255));
    auto t = mis::preprocess(img, cfg);
    EXPECT_LT(t[0], 0.f);    // plane 0, x=0: black -> negative after normalization
    EXPECT_GT(t[223], 0.f);  // plane 0, x=223: white -> positive
}

TEST(Preprocess, CenterCropTakesTheMiddle) {
    // A 224x448 image: only the middle 224 columns are red. After resize_short=224
    // (no-op) and center crop, every pixel should be red.
    PreprocessConfig cfg;
    cfg.resize_short = 224;
    cv::Mat img(224, 448, CV_8UC3, cv::Scalar(255, 0, 0));         // blue
    img(cv::Rect(112, 0, 224, 224)).setTo(cv::Scalar(0, 0, 255));  // red middle
    auto t = mis::preprocess(img, cfg);
    float red_r = (1.f - cfg.mean[0]) / cfg.std[0];
    EXPECT_NEAR(t[0], red_r, 1e-5);
    EXPECT_NEAR(t[223], red_r, 1e-5);
}

TEST(Preprocess, CenterCropRoundsHalfToEvenLikeTorchvision) {
    // Height 325 after resize -> offset (325-224)/2 = 50.5, which must round to 50
    // (Python round), not 51. Only row 50 is white, so crop row 0 is white
    // exactly when the offset is 50. (The resize is a no-op at this size.)
    PreprocessConfig cfg;
    cfg.resize_short = 256;
    cv::Mat img(325, 256, CV_8UC3, cv::Scalar(0, 0, 0));
    img.row(50).setTo(cv::Scalar(255, 255, 255));
    auto t = mis::preprocess(img, cfg);
    EXPECT_GT(t[0], 0.f);  // white row landed at crop row 0
}

TEST(Preprocess, RejectsEmptyImage) {
    PreprocessConfig cfg;
    EXPECT_THROW(mis::preprocess(cv::Mat(), cfg), std::invalid_argument);
}

TEST(Decode, GarbageBytesGiveEmptyMat) {
    EXPECT_TRUE(mis::decode_image("").empty());
    EXPECT_TRUE(mis::decode_image("definitely not a jpeg").empty());
    std::string random_bytes(1000, '\0');
    for (size_t i = 0; i < random_bytes.size(); ++i) random_bytes[i] = char(i * 37 + 11);
    EXPECT_TRUE(mis::decode_image(random_bytes).empty());
}

TEST(Decode, RoundTripsAPng) {
    cv::Mat img(10, 20, CV_8UC3, cv::Scalar(1, 2, 3));
    std::vector<uchar> buf;
    ASSERT_TRUE(cv::imencode(".png", img, buf));
    cv::Mat out = mis::decode_image(std::string(buf.begin(), buf.end()));
    ASSERT_FALSE(out.empty());
    EXPECT_EQ(out.size(), img.size());
    EXPECT_EQ(cv::norm(out, img, cv::NORM_INF), 0.0);
}
