#include <gtest/gtest.h>

#include <filesystem>
#include <thread>

#include <opencv2/imgproc.hpp>

#include "batcher.h"
#include "preprocess.h"

// Batcher tests run the real mobilenet model; skipped if it has not been exported.

namespace {

using namespace std::chrono_literals;

const std::string kModel = std::string(MIS_MODELS_DIR) + "/mobilenet_v3_large.onnx";

// Distinct synthetic images so each request has its own expected answer.
cv::Mat make_image(int i) {
    cv::Mat img(240 + 10 * i, 320, CV_8UC3, cv::Scalar(20 * i, 255 - 25 * i, 40 + 30 * i));
    cv::circle(img, {60 + 20 * i, 120}, 30 + 5 * i, cv::Scalar(255, 255 - 20 * i, 0), -1);
    cv::rectangle(img, {10, 10}, {50 + 15 * i, 60}, cv::Scalar(0, 0, 255), -1);
    return img;
}

mis::ModelConfig small_config() {
    mis::ModelConfig cfg;
    cfg.name = "mobilenet_v3_large";
    cfg.path = kModel;
    cfg.max_batch_size = 8;
    cfg.max_wait_ms = 20;
    cfg.num_workers = 2;
    cfg.queue_limit = 256;
    return cfg;
}

auto far_deadline() { return mis::Clock::now() + 30s; }

class BatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(kModel)) GTEST_SKIP() << "model not exported";
        runner_ = std::make_shared<mis::ModelRunner>(kModel);
    }
    std::shared_ptr<mis::ModelRunner> runner_;
};

}  // namespace

TEST_F(BatcherTest, FiftyParallelRequestsEachGetTheirOwnCorrectResult) {
    // Expected answers: run each image alone, without the batcher.
    constexpr int kDistinct = 5, kRequests = 50;
    mis::PreprocessConfig pcfg;
    std::vector<std::vector<mis::Prediction>> expected(kDistinct);
    for (int i = 0; i < kDistinct; ++i) {
        auto t = mis::preprocess(make_image(i), pcfg);
        auto logits = runner_->run(t.data(), 1, 224, 224);
        expected[i] = mis::top_k(mis::softmax(logits.data(), logits.size()), 5, {});
    }

    mis::Metrics metrics;
    mis::Batcher batcher(small_config(), runner_, {}, &metrics);
    batcher.start();

    // 50 threads submit at the same time; request r uses image r % 5.
    std::vector<mis::InferenceResult> results(kRequests);
    std::vector<std::thread> clients;
    for (int r = 0; r < kRequests; ++r) {
        clients.emplace_back([&, r] {
            auto fut = batcher.submit(make_image(r % kDistinct), 5, far_deadline());
            ASSERT_TRUE(fut.has_value());
            results[r] = fut->get();
        });
    }
    for (auto& t : clients) t.join();

    int max_batch = 0;
    for (int r = 0; r < kRequests; ++r) {
        const auto& want = expected[r % kDistinct];
        const auto& got = results[r].predictions;
        ASSERT_EQ(got.size(), 5u);
        for (int k = 0; k < 5; ++k) {
            EXPECT_EQ(got[k].class_id, want[k].class_id) << "request " << r;
            EXPECT_NEAR(got[k].score, want[k].score, 1e-4) << "request " << r;
        }
        max_batch = std::max(max_batch, results[r].batch_size);
    }
    EXPECT_GT(max_batch, 1) << "requests were never batched together";
    EXPECT_EQ(metrics.batched_items(), static_cast<uint64_t>(kRequests));
}

TEST_F(BatcherTest, FullQueueRejects) {
    auto cfg = small_config();
    cfg.queue_limit = 2;
    mis::Batcher batcher(cfg, runner_, {});
    // Workers not started, so nothing drains the queue.
    auto a = batcher.submit(make_image(0), 5, far_deadline());
    auto b = batcher.submit(make_image(1), 5, far_deadline());
    auto c = batcher.submit(make_image(2), 5, far_deadline());
    EXPECT_TRUE(a.has_value());
    EXPECT_TRUE(b.has_value());
    EXPECT_FALSE(c.has_value());
    // Shutdown with no workers must fail the queued requests, not leave them hanging.
    batcher.shutdown();
    EXPECT_THROW(a->get(), std::runtime_error);
    EXPECT_THROW(b->get(), std::runtime_error);
}

TEST_F(BatcherTest, ModelFailureFailsEveryRequestInTheBatch) {
    // Crop 100 makes a [N,3,100,100] tensor; the model expects 224x224, so Run() throws.
    auto cfg = small_config();
    cfg.preprocess.resize_short = 120;
    cfg.preprocess.crop = 100;
    cfg.max_wait_ms = 50;
    mis::Batcher batcher(cfg, runner_, {});
    std::vector<std::future<mis::InferenceResult>> futs;
    for (int i = 0; i < 6; ++i) futs.push_back(*batcher.submit(make_image(i), 5, far_deadline()));
    batcher.start();
    for (auto& f : futs) {
        ASSERT_EQ(f.wait_for(10s), std::future_status::ready) << "a client would hang";
        EXPECT_ANY_THROW(f.get());
    }
}

TEST_F(BatcherTest, ShutdownDrainsQueuedRequests) {
    mis::Batcher batcher(small_config(), runner_, {});
    std::vector<std::future<mis::InferenceResult>> futs;
    for (int i = 0; i < 20; ++i) futs.push_back(*batcher.submit(make_image(i % 5), 5, far_deadline()));
    batcher.start();
    batcher.shutdown();  // returns only after workers emptied the queue
    for (auto& f : futs) {
        ASSERT_EQ(f.wait_for(0s), std::future_status::ready);
        EXPECT_EQ(f.get().predictions.size(), 5u);
    }
    EXPECT_FALSE(batcher.submit(make_image(0), 5, far_deadline()).has_value());
}

TEST_F(BatcherTest, ExpiredRequestsAreSkipped) {
    mis::Metrics metrics;
    mis::Batcher batcher(small_config(), runner_, {}, &metrics);
    auto past = mis::Clock::now() - 1ms;
    auto fut = batcher.submit(make_image(0), 5, past);
    batcher.start();
    ASSERT_EQ(fut->wait_for(10s), std::future_status::ready);
    EXPECT_THROW(fut->get(), std::runtime_error);
    EXPECT_EQ(metrics.batches(), 0u);  // never reached the model
}
