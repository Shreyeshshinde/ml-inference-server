#pragma once

#include <memory>
#include <string>
#include <vector>

#include <httplib.h>

#include "batcher.h"
#include "config.h"
#include "metrics.h"
#include "model_runner.h"

namespace mis {

// One loaded model plus its settings.
struct ServedModel {
    ModelConfig config;
    std::shared_ptr<ModelRunner> runner;
};

// HTTP front end. Phase 3: one model, requests go through a dynamic batcher.
class InferenceServer {
public:
    InferenceServer(ServerConfig config, ServedModel model, std::vector<std::string> labels);
    ~InferenceServer();

    // Blocks serving on config.host:config.port until stop() is called.
    bool listen();

    // For tests: bind to a free port (returned), then serve with listen_after_bind().
    int bind_to_any_port(const std::string& host = "127.0.0.1");
    bool listen_after_bind();
    void wait_until_ready() const { http_.wait_until_ready(); }

    // Thread-safe. Stops accepting connections; listen() returns once in-flight
    // HTTP requests have finished (the batcher keeps running so they can finish).
    void stop() { http_.stop(); }

    // Second half of graceful shutdown: drain the batch queue and join workers.
    // Call after listen() has returned.
    void drain() { batcher_->shutdown(); }

    const Metrics& metrics() const { return metrics_; }

private:
    void setup_routes();
    void handle_predict(const httplib::Request& req, httplib::Response& res);

    ServerConfig config_;
    ServedModel model_;
    Metrics metrics_;  // declared before batcher_ because the batcher points at it
    std::unique_ptr<Batcher> batcher_;
    httplib::Server http_;
};

}  // namespace mis
