#include <gtest/gtest.h>

#include <filesystem>

#include "model_registry.h"

namespace {

const std::string kModel = std::string(MIS_MODELS_DIR) + "/mobilenet_v3_large.onnx";

class RegistryTest : public ::testing::Test {
  protected:
    void SetUp() override {
        if (!std::filesystem::exists(kModel)) GTEST_SKIP() << "model not exported";
        mis::ServerConfig cfg;
        cfg.models_dir = MIS_MODELS_DIR;
        for (const char* name : {"a", "b"}) {
            mis::ModelConfig m;
            m.name = name;
            m.path = kModel;
            cfg.models.push_back(m);
        }
        registry_ = std::make_unique<mis::ModelRegistry>(cfg, nullptr);
    }
    std::unique_ptr<mis::ModelRegistry> registry_;
};

}  // namespace

TEST_F(RegistryTest, LoadsEveryModelInConfigOrder) {
    EXPECT_EQ(registry_->names(), (std::vector<std::string>{"a", "b"}));
    ASSERT_NE(registry_->find("a"), nullptr);
    EXPECT_NE(registry_->find("a"), registry_->find("b"));  // separate queues and workers
    EXPECT_EQ(registry_->find("missing"), nullptr);
}

TEST_F(RegistryTest, ReloadBumpsVersionOrUsesGivenOne) {
    auto r = registry_->reload("a", "", std::nullopt);
    EXPECT_EQ(r.http_status, 200) << r.error;
    EXPECT_EQ(r.old_version, 1);
    EXPECT_EQ(r.new_version, 2);

    r = registry_->reload("a", kModel, 10);
    EXPECT_EQ(r.http_status, 200) << r.error;
    EXPECT_EQ(registry_->find("a")->batcher->current_model()->version, 10);
    // Model b is untouched.
    EXPECT_EQ(registry_->find("b")->batcher->current_model()->version, 1);
}

TEST_F(RegistryTest, ReloadRejectsBadRequestsAndKeepsOldVersion) {
    EXPECT_EQ(registry_->reload("missing", "", std::nullopt).http_status, 404);
    // Path traversal out of the models folder.
    EXPECT_EQ(
        registry_->reload("a", std::string(MIS_MODELS_DIR) + "/../CMakeLists.txt", std::nullopt)
            .http_status,
        400);
    // Inside the folder but not a model.
    EXPECT_EQ(
        registry_->reload("a", std::string(MIS_MODELS_DIR) + "/.gitkeep", std::nullopt).http_status,
        400);
    EXPECT_EQ(registry_->reload("a", "", 0).http_status, 400);
    EXPECT_EQ(registry_->find("a")->batcher->current_model()->version, 1);
}
