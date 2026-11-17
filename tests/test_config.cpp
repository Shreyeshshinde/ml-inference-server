#include <gtest/gtest.h>

#include "config.h"

namespace {
const char* kMinimal = R"({
  "port": 9000,
  "models": [ { "name": "m", "path": "m.onnx", "max_batch_size": 4 } ]
})";
}

TEST(Config, ParsesFieldsAndAppliesDefaults) {
    auto cfg = mis::parse_config(kMinimal);
    EXPECT_EQ(cfg.port, 9000);
    EXPECT_EQ(cfg.host, "0.0.0.0");  // default
    ASSERT_EQ(cfg.models.size(), 1u);
    EXPECT_EQ(cfg.models[0].name, "m");
    EXPECT_EQ(cfg.models[0].max_batch_size, 4);
    EXPECT_EQ(cfg.models[0].preprocess.crop, 224);  // default
    EXPECT_FLOAT_EQ(cfg.models[0].preprocess.mean[0], 0.485f);
}

TEST(Config, RepoConfigLoads) {
    auto cfg = mis::load_config(std::string(MIS_SOURCE_DIR) + "/config/server.json");
    EXPECT_GE(cfg.models.size(), 1u);
}

TEST(Config, RejectsBadInput) {
    EXPECT_THROW(mis::parse_config("{not json"), std::runtime_error);
    EXPECT_THROW(mis::parse_config(R"({"models": []})"), std::runtime_error);
    EXPECT_THROW(mis::parse_config(R"({"models": [{"path": "x"}]})"), std::runtime_error);
    EXPECT_THROW(mis::parse_config(
                     R"({"models": [{"name": "a", "path": "x", "max_batch_size": 0}]})"),
                 std::runtime_error);
    EXPECT_THROW(mis::parse_config(
                     R"({"models": [{"name": "a", "path": "x"}, {"name": "a", "path": "y"}]})"),
                 std::runtime_error);
    EXPECT_THROW(mis::load_config("/nonexistent.json"), std::runtime_error);
}
