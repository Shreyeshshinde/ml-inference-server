#include <gtest/gtest.h>

#include <filesystem>
#include <random>
#include <thread>

#include <json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "postprocess.h"
#include "server.h"

// End-to-end tests: start a real server on a free port and talk to it over HTTP.
// Uses mobilenet (fast); skipped if the model has not been exported.

namespace {

using nlohmann::json;

const std::string kModelPath = std::string(MIS_MODELS_DIR) + "/mobilenet_v3_large.onnx";
const std::string kLabels = std::string(MIS_SOURCE_DIR) + "/data/imagenet_labels.txt";

class ServerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(kModelPath)) GTEST_SKIP() << "model not exported";
        start_server(default_model_config());
    }

    void TearDown() override { stop_server(); }

    static mis::ModelConfig default_model_config() {
        mis::ModelConfig m;
        m.name = "mobilenet_v3_large";
        m.path = kModelPath;
        m.max_batch_size = 8;
        m.max_wait_ms = 5;
        m.num_workers = 1;
        return m;
    }

    // Tests that need different batching settings call this again with their own config.
    void start_server(const mis::ModelConfig& m, int http_threads = 8) {
        stop_server();
        mis::ServerConfig cfg;
        cfg.http_threads = http_threads;
        cfg.max_body_bytes = 1024 * 1024;  // 1 MB so the 413 test stays small
        server_ = std::make_unique<mis::InferenceServer>(
            cfg, mis::ServedModel{m, std::make_shared<mis::ModelRunner>(kModelPath)},
            mis::load_labels(kLabels));
        port_ = server_->bind_to_any_port();
        ASSERT_GT(port_, 0);
        thread_ = std::thread([this] { server_->listen_after_bind(); });
        server_->wait_until_ready();
        client_ = std::make_unique<httplib::Client>("127.0.0.1", port_);
    }

    void stop_server() {
        if (server_) server_->stop();
        if (thread_.joinable()) thread_.join();
        server_.reset();
    }

    httplib::Result post_image(const std::string& bytes, const std::string& query = "",
                               const std::string& model = "mobilenet_v3_large") {
        httplib::MultipartFormDataItems items = {{"image", bytes, "img.jpg", "image/jpeg"}};
        return client_->Post("/v1/models/" + model + "/predict" + query, items);
    }

    static std::string encoded_test_image() {
        // A synthetic image is enough to check the response shape and status.
        cv::Mat img(300, 400, CV_8UC3, cv::Scalar(30, 120, 200));
        cv::circle(img, {200, 150}, 80, cv::Scalar(10, 200, 40), -1);
        std::vector<uchar> buf;
        cv::imencode(".jpg", img, buf);
        return std::string(buf.begin(), buf.end());
    }

    std::unique_ptr<mis::InferenceServer> server_;
    std::unique_ptr<httplib::Client> client_;
    std::thread thread_;
    int port_ = 0;
};

}  // namespace

TEST_F(ServerTest, Health) {
    auto res = client_->Get("/health");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 200);
    EXPECT_EQ(json::parse(res->body)["status"], "ok");
}

TEST_F(ServerTest, PredictReturnsWellFormedJson) {
    auto res = post_image(encoded_test_image(), "?top_k=3");
    ASSERT_TRUE(res);
    ASSERT_EQ(res->status, 200) << res->body;
    auto j = json::parse(res->body);
    EXPECT_EQ(j["model"], "mobilenet_v3_large");
    EXPECT_EQ(j["version"], 1);
    EXPECT_EQ(j["batch_size"], 1);
    ASSERT_EQ(j["predictions"].size(), 3u);
    float prev = 2.f;
    for (const auto& p : j["predictions"]) {
        float s = p["score"];
        EXPECT_LE(s, prev);  // sorted descending
        prev = s;
        EXPECT_FALSE(p["label"].get<std::string>().empty());
    }
    for (const char* k : {"queue_wait", "preprocess", "inference", "postprocess"}) {
        EXPECT_TRUE(j["timings_ms"].contains(k)) << k;
    }
    EXPECT_GT(j["latency_ms"].get<double>(), 0.0);
}

TEST_F(ServerTest, UnknownModelIs404) {
    auto res = post_image(encoded_test_image(), "", "no_such_model");
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 404);
}

TEST_F(ServerTest, MissingImageFieldIs400) {
    httplib::MultipartFormDataItems items = {{"file", "abc", "a.jpg", "image/jpeg"}};
    auto res = client_->Post("/v1/models/mobilenet_v3_large/predict", items);
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 400);
    auto raw = client_->Post("/v1/models/mobilenet_v3_large/predict", "xyz", "text/plain");
    ASSERT_TRUE(raw);
    EXPECT_EQ(raw->status, 400);
}

TEST_F(ServerTest, BadTopKIs400) {
    for (const char* q : {"?top_k=abc", "?top_k=0", "?top_k=5000", "?top_k=3x"}) {
        auto res = post_image(encoded_test_image(), q);
        ASSERT_TRUE(res);
        EXPECT_EQ(res->status, 400) << q;
    }
}

TEST_F(ServerTest, TooLargeIs413) {
    auto res = post_image(std::string(2 * 1024 * 1024, 'x'));
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 413);
}

TEST_F(ServerTest, RandomBytesNeverCrashTheServer) {
    // Fuzz: random payloads, including truncated real JPEGs. Every reply must be 400,
    // and the server must still be healthy afterwards.
    std::mt19937 rng(42);
    const std::string jpg = encoded_test_image();
    for (int i = 0; i < 100; ++i) {
        std::string payload;
        if (i % 2 == 0) {
            payload.resize(1 + rng() % 5000);
            for (auto& c : payload) c = static_cast<char>(rng());
        } else {
            payload = jpg.substr(0, rng() % jpg.size());  // truncated JPEG
        }
        auto res = post_image(payload);
        ASSERT_TRUE(res) << "no response on iteration " << i;
        // A heavily truncated JPEG can still decode (partial image), so 200 is allowed.
        EXPECT_TRUE(res->status == 400 || res->status == 200) << res->status;
    }
    auto health = client_->Get("/health");
    ASSERT_TRUE(health);
    EXPECT_EQ(health->status, 200);
}

TEST_F(ServerTest, FullQueueReturns503AndEveryOtherRequestSucceeds) {
    // Queue of 1, batches of 1, one worker: a burst of 40 parallel requests cannot all
    // fit, so some must be rejected with 503. Nothing else (no 500, no hang) is allowed.
    auto m = default_model_config();
    m.queue_limit = 1;
    m.max_batch_size = 1;
    m.max_wait_ms = 0;
    start_server(m, /*http_threads=*/48);

    const std::string img = encoded_test_image();
    std::vector<int> statuses(40, 0);
    std::vector<std::thread> clients;
    for (int i = 0; i < 40; ++i) {
        clients.emplace_back([&, i] {
            httplib::Client c("127.0.0.1", port_);
            httplib::MultipartFormDataItems items = {{"image", img, "img.jpg", "image/jpeg"}};
            auto res = c.Post("/v1/models/mobilenet_v3_large/predict", items);
            statuses[i] = res ? res->status : -static_cast<int>(res.error());
        });
    }
    for (auto& t : clients) t.join();

    int ok = 0, rejected = 0;
    for (int s : statuses) {
        EXPECT_TRUE(s == 200 || s == 503) << "unexpected status " << s;
        ok += s == 200;
        rejected += s == 503;
    }
    EXPECT_GT(ok, 0);
    EXPECT_GT(rejected, 0);
}

TEST_F(ServerTest, SlowInferenceReturns504) {
    auto m = default_model_config();
    m.request_timeout_ms = 1;  // less than one mobilenet run
    start_server(m);
    auto res = post_image(encoded_test_image());
    ASSERT_TRUE(res);
    EXPECT_EQ(res->status, 504);
}

TEST_F(ServerTest, MetricsCountsRequestsAndErrors) {
    post_image(encoded_test_image());
    post_image("garbage");
    auto res = client_->Get("/metrics");
    ASSERT_TRUE(res);
    EXPECT_NE(res->body.find("mis_requests_total 2"), std::string::npos) << res->body;
    EXPECT_NE(res->body.find("mis_errors_total 1"), std::string::npos) << res->body;
}
